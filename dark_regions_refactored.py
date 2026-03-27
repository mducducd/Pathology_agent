import os
from typing import Any, Dict, List

import numpy as np
import openslide

from .config import DEBUG_ROOT_DIR
from .slide_utils import _read_region_rgb, _resize_to_max_dim


def _mean_filter3(x: np.ndarray) -> np.ndarray:
    padded = np.pad(x, 1, mode="edge")
    acc = np.zeros_like(x, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            acc += padded[dy : dy + x.shape[0], dx : dx + x.shape[1]]
    return acc / 9.0


def _percentile_from_hist(hist: List[int], pct: float) -> int:
    total = sum(hist)
    if total <= 0:
        return 255
    cutoff = total * (pct / 100.0)
    running = 0
    for i, count in enumerate(hist):
        running += count
        if running >= cutoff:
            return i
    return 255


def _find_connected_components(mask: List[bool], w: int, h: int, min_area: int) -> List[Dict[str, int]]:
    visited = bytearray(w * h)
    boxes: List[Dict[str, int]] = []

    for idx in range(w * h):
        if visited[idx] or not mask[idx]:
            continue

        visited[idx] = 1
        stack = [idx]
        area = 0
        minx, miny, maxx, maxy = w, h, 0, 0

        while stack:
            i = stack.pop()
            x = i % w
            y = i // w
            area += 1
            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x)
            maxy = max(maxy, y)

            for n in ((i - 1) if x > 0 else None,
                      (i + 1) if x + 1 < w else None,
                      (i - w) if y > 0 else None,
                      (i + w) if y + 1 < h else None):
                if n is not None and mask[n] and not visited[n]:
                    visited[n] = 1
                    stack.append(n)

        if area >= min_area:
            boxes.append({
                "x": minx,
                "y": miny,
                "w": maxx - minx + 1,
                "h": maxy - miny + 1,
                "area": area,
            })

    return boxes


def _expand_box(box: Dict[str, int], width: int, height: int, pad: int) -> Dict[str, int]:
    x0 = max(0, int(box["x"]) - pad)
    y0 = max(0, int(box["y"]) - pad)
    x1 = min(width, int(box["x"]) + int(box["w"]) + pad)
    y1 = min(height, int(box["y"]) + int(box["h"]) + pad)
    return {
        "x": x0,
        "y": y0,
        "w": max(1, x1 - x0),
        "h": max(1, y1 - y0),
        "area": int(max(1, box.get("area", 0))),
    }


def _grow_mask_within(base_mask: np.ndarray, seed_mask: np.ndarray, steps: int) -> np.ndarray:
    grown = seed_mask.astype(bool, copy=True)
    allowed = base_mask.astype(bool, copy=False)
    if not np.any(grown) or not np.any(allowed):
        return np.zeros_like(allowed, dtype=bool)

    for _ in range(max(0, int(steps))):
        nxt = allowed & (_mean_filter3(grown.astype(np.float32, copy=False)) > 0.0)
        if np.array_equal(nxt, grown):
            break
        grown = nxt
    return grown


def _normalize_by_percentile(x: np.ndarray, mask: np.ndarray, pct: float = 95.0) -> np.ndarray:
    denom = max(float(np.percentile(x[mask], pct)), 0.01)
    return np.clip(x / denom, 0.0, 1.0).astype(np.float32, copy=False)


def _compute_roi_score(rgb: np.ndarray, gray: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ch_max = np.maximum(r, np.maximum(g, b))
    ch_min = np.minimum(r, np.minimum(g, b))
    chroma = ch_max - ch_min

    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    edge_mag = np.pad(grad_x, ((0, 0), (0, 1)), mode="constant") + np.pad(grad_y, ((0, 1), (0, 0)), mode="constant")

    chroma_gate = np.clip(chroma / 32.0, 0.0, 1.0)
    texture_gate = np.clip(edge_mag / 12.0, 0.0, 1.0)
    mid_darkness = np.clip((185.0 - gray) / 70.0, 0.0, 1.0) * np.clip((gray - 55.0) / 45.0, 0.0, 1.0)

    purple_blue = np.clip((b - r + 25.0) / 55.0, 0.0, 1.0) * chroma_gate * mid_darkness * (0.35 + 0.65 * texture_gate)
    chromatic_cellular = chroma_gate * mid_darkness * texture_gate * np.clip((np.maximum(r, b) - g + 18.0) / 60.0, 0.0, 1.0)
    red_smooth = np.clip((r - b) / 60.0, 0.0, 1.0) * chroma_gate * np.clip(1.0 - texture_gate, 0.0, 1.0)
    density = _mean_filter3(tissue_mask.astype(np.float32, copy=False))

    artifact_dark = np.clip((60.0 - gray) / 60.0, 0.0, 1.0) * np.clip(1.0 - chroma / 22.0, 0.0, 1.0)
    gray_black_penalty = np.clip((95.0 - gray) / 55.0, 0.0, 1.0) * np.clip((18.0 - chroma) / 18.0, 0.0, 1.0)
    light_penalty = np.clip((gray - 160.0) / 40.0, 0.0, 1.0)

    purple_blue_norm = _normalize_by_percentile(purple_blue, tissue_mask)
    chromatic_norm = _normalize_by_percentile(chromatic_cellular, tissue_mask)
    red_smooth_norm = _normalize_by_percentile(red_smooth, tissue_mask)

    return (
        0.56 * purple_blue_norm
        + 0.08 * chromatic_norm
        + 0.15 * density
        + 0.08 * texture_gate
        - 0.18 * red_smooth_norm
        - 0.14 * artifact_dark
        - 0.18 * gray_black_penalty
        - 0.22 * light_penalty
    ).astype(np.float32, copy=False)


def _select_dark_core_boxes(
    *,
    score: np.ndarray,
    tissue_mask: np.ndarray,
    threshold_pct: int,
    out_w: int,
    out_h: int,
    min_area: int,
    max_regions: int,
) -> List[Dict[str, int]]:
    core_pct = min(99.5, max(float(threshold_pct) + 12.0, 95.0))
    core_mask = tissue_mask & (score >= float(np.percentile(score[tissue_mask], core_pct)))
    core_boxes = _find_connected_components(
        core_mask.reshape(-1).tolist(),
        out_w,
        out_h,
        min_area=max(24, int(min_area // 6)),
    )

    if not core_boxes:
        base_mask = tissue_mask & (score >= float(np.percentile(score[tissue_mask], float(threshold_pct))))
        base_boxes = _find_connected_components(base_mask.reshape(-1).tolist(), out_w, out_h, min_area=min_area)
        return base_boxes[:max_regions]

    base_mask = tissue_mask & (score >= float(np.percentile(score[tissue_mask], float(max(threshold_pct, 75)))))
    grown = _grow_mask_within(base_mask, core_mask, steps=max(4, int(round(min(out_w, out_h) * 0.008))))
    region_mask = grown if np.any(grown) else core_mask

    boxes = _find_connected_components(
        region_mask.reshape(-1).tolist(),
        out_w,
        out_h,
        min_area=max(32, int(min_area // 4)),
    ) or core_boxes

    pad = max(2, int(round(min(out_w, out_h) * 0.005)))
    expanded = [_expand_box(box, out_w, out_h, pad) for box in boxes]
    expanded.sort(key=lambda b: b["area"], reverse=True)
    return expanded[:max_regions]


def detect_dark_regions(
    slide_path: str,
    run_id: str,
    max_dim: int = 1024,
    threshold_pct: int = 85,
    min_area: int = 800,
    max_regions: int = 30,
) -> Dict[str, Any]:
    slide = openslide.open_slide(slide_path)
    try:
        level = slide.level_count - 1
        level_w, level_h = slide.level_dimensions[level]
        region = _read_region_rgb(slide, 0, 0, level, (level_w, level_h))
        region, out_w, out_h = _resize_to_max_dim(region, max_dim=max_dim)

        rgb = np.asarray(region.convert("RGB"), dtype=np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        tissue_mask = gray < 242.0

        if np.any(tissue_mask):
            score = _compute_roi_score(rgb, gray, tissue_mask)
            threshold = float(np.percentile(score[tissue_mask], float(threshold_pct)))
            boxes = _select_dark_core_boxes(
                score=score,
                tissue_mask=tissue_mask,
                threshold_pct=threshold_pct,
                out_w=out_w,
                out_h=out_h,
                min_area=min_area,
                max_regions=max_regions,
            )
        else:
            threshold = float(_percentile_from_hist(region.convert("L").histogram(), float(threshold_pct)))
            mask_np = (gray <= threshold).reshape(-1).tolist()
            boxes = _find_connected_components(mask_np, out_w, out_h, min_area=min_area)
            boxes.sort(key=lambda b: b["area"], reverse=True)
            boxes = boxes[:max_regions]

        base_w0, base_h0 = slide.level_dimensions[0]
        scale_x = base_w0 / float(out_w)
        scale_y = base_h0 / float(out_h)
        boxes_level0 = [{
            "x0": int(round(b["x"] * scale_x)),
            "y0": int(round(b["y"] * scale_y)),
            "w": int(round(b["w"] * scale_x)),
            "h": int(round(b["h"] * scale_y)),
            "area": int(b["area"]),
        } for b in boxes]

        out_dir = os.path.join(DEBUG_ROOT_DIR, run_id, "dark")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "overview.jpg")
        region.save(out_path, format="JPEG", quality=90)

        return {
            "image_path": out_path,
            "image_dims": [out_w, out_h],
            "threshold": threshold,
            "threshold_pct": int(threshold_pct),
            "boxes": boxes,
            "boxes_level0": boxes_level0,
        }
    finally:
        try:
            slide.close()
        except Exception:
            pass
