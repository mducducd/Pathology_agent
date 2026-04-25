import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openslide
from PIL import Image

from .config import DEBUG_ROOT_DIR
from .slide_utils import _read_region_rgb, _resize_to_max_dim

Box = Dict[str, int]


def _percentile_from_hist(hist: List[int], pct: float) -> int:
    total = sum(hist)
    if total <= 0:
        return 255
    target = total * (pct / 100.0)
    running = 0
    for i, count in enumerate(hist):
        running += count
        if running >= target:
            return i
    return 255


def _mean_filter3(x: np.ndarray) -> np.ndarray:
    padded = np.pad(x, 1, mode="edge")
    acc = np.zeros_like(x, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            acc += padded[dy : dy + x.shape[0], dx : dx + x.shape[1]]
    return acc / 9.0


def _find_connected_components(mask: List[bool], w: int, h: int, min_area: int) -> List[Box]:
    visited = bytearray(w * h)
    boxes: List[Box] = []

    for idx in range(w * h):
        if not mask[idx] or visited[idx]:
            continue
        stack = [idx]
        visited[idx] = 1
        area = 0
        minx, miny = w, h
        maxx, maxy = 0, 0

        while stack:
            i = stack.pop()
            x, y = i % w, i // w
            area += 1
            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x)
            maxy = max(maxy, y)

            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < w and 0 <= ny < h:
                    n = ny * w + nx
                    if mask[n] and not visited[n]:
                        visited[n] = 1
                        stack.append(n)

        if area >= min_area:
            boxes.append({"x": minx, "y": miny, "w": maxx - minx + 1, "h": maxy - miny + 1, "area": area})

    return boxes


def _box_bounds(box: Box) -> Tuple[int, int, int, int]:
    x0, y0 = int(box["x"]), int(box["y"])
    return x0, y0, x0 + int(box["w"]), y0 + int(box["h"])


def _box_from_bounds(x0: int, y0: int, x1: int, y1: int, *, area: int) -> Box:
    return {
        "x": int(x0),
        "y": int(y0),
        "w": max(1, int(x1) - int(x0)),
        "h": max(1, int(y1) - int(y0)),
        "area": int(max(1, area)),
    }


def _expand_box(box: Box, width: int, height: int, pad: int) -> Box:
    x0, y0, x1, y1 = _box_bounds(box)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(width, x1 + pad)
    y1 = min(height, y1 + pad)
    return _box_from_bounds(x0, y0, x1, y1, area=int(max(1, box.get("area", 0))))


def _trim_box_to_tissue(
    box: Box,
    tissue_mask: np.ndarray,
    *,
    line_threshold: float = 0.72,
    min_side: int = 6,
) -> Optional[Box]:
    x0, y0, x1, y1 = _box_bounds(box)
    h, w = tissue_mask.shape
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))

    changed = True
    while changed and (x1 - x0) >= min_side and (y1 - y0) >= min_side:
        changed = False
        if np.mean(tissue_mask[y0, x0:x1]) < line_threshold:
            y0 += 1
            changed = True
        if np.mean(tissue_mask[y1 - 1, x0:x1]) < line_threshold:
            y1 -= 1
            changed = True
        if x1 - x0 < min_side or y1 - y0 < min_side:
            break
        if np.mean(tissue_mask[y0:y1, x0]) < line_threshold:
            x0 += 1
            changed = True
        if np.mean(tissue_mask[y0:y1, x1 - 1]) < line_threshold:
            x1 -= 1
            changed = True

    if (x1 - x0) < min_side or (y1 - y0) < min_side:
        return None
    return _box_from_bounds(x0, y0, x1, y1, area=min(int(box.get("area", 1)), (x1 - x0) * (y1 - y0)))


def _box_touches_tissue_edge(box: Box, tissue_mask: np.ndarray) -> bool:
    x0, y0, x1, y1 = _box_bounds(box)
    box_mask = tissue_mask[y0:y1, x0:x1]
    if box_mask.size == 0:
        return True
    if float(np.mean(box_mask)) < 0.94:
        return True

    h, w = box_mask.shape
    band = min(max(1, int(round(min(h, w) * 0.12))), max(1, min(h, w) // 2))

    sides = [box_mask[:band, :], box_mask[h - band:, :], box_mask[:, :band], box_mask[:, w - band:]]
    if np.any(np.array([np.mean(s) for s in sides], dtype=np.float32) < 0.86):
        return True

    corners = [box_mask[:band, :band], box_mask[:band, w - band:], box_mask[h - band:, :band], box_mask[h - band:, w - band:]]
    if np.any(np.array([np.mean(c) for c in corners], dtype=np.float32) < 0.82):
        return True

    border_mask = np.zeros_like(box_mask, dtype=bool)
    border_mask[:band, :] = border_mask[h - band:, :] = border_mask[:, :band] = border_mask[:, w - band:] = True
    return float(np.mean(box_mask[border_mask])) < 0.90


def _boxes_touch_or_near(a: Box, b: Box, *, gap: int) -> bool:
    ax0, ay0, ax1, ay1 = _box_bounds(a)
    bx0, by0, bx1, by1 = _box_bounds(b)
    return not (ax1 + gap < bx0 or bx1 + gap < ax0 or ay1 + gap < by0 or by1 + gap < ay0)


def _merge_nearby_boxes(boxes: List[Box], *, gap: int) -> List[Box]:
    pending = [dict(b) for b in boxes]
    if len(pending) <= 1:
        return pending

    changed = True
    while changed:
        changed = False
        merged: List[Box] = []
        while pending:
            current = pending.pop()
            cx0, cy0, cx1, cy1 = _box_bounds(current)
            area = int(current.get("area", max(1, (cx1 - cx0) * (cy1 - cy0))))

            i = 0
            while i < len(pending):
                other = pending[i]
                if not _boxes_touch_or_near(current, other, gap=gap):
                    i += 1
                    continue
                ox0, oy0, ox1, oy1 = _box_bounds(other)
                cx0, cy0 = min(cx0, ox0), min(cy0, oy0)
                cx1, cy1 = max(cx1, ox1), max(cy1, oy1)
                area += int(other.get("area", max(1, (ox1 - ox0) * (oy1 - oy0))))
                current = _box_from_bounds(cx0, cy0, cx1, cy1, area=area)
                pending.pop(i)
                changed = True
            merged.append(current)
        pending = merged

    pending.sort(key=lambda b: b["area"], reverse=True)
    return pending


def _clip_mask_to_boxes(mask: np.ndarray, boxes: List[Box]) -> np.ndarray:
    if mask.size == 0 or not boxes:
        return np.zeros_like(mask, dtype=bool)
    clipped = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for box in boxes:
        x0 = max(0, int(box["x"]))
        y0 = max(0, int(box["y"]))
        x1 = min(w, x0 + max(0, int(box["w"])))
        y1 = min(h, y0 + max(0, int(box["h"])))
        if x1 > x0 and y1 > y0:
            clipped[y0:y1, x0:x1] = True
    return mask & clipped


def _soft_alpha_mask(mask: np.ndarray) -> np.ndarray:
    return np.clip(_mean_filter3(_mean_filter3(mask.astype(np.float32, copy=False))), 0.0, 1.0)


def _refine_dark_region_boxes(boxes: List[Box], tissue_mask: np.ndarray) -> List[Box]:
    refined = [trimmed for box in boxes if (trimmed := _trim_box_to_tissue(box, tissue_mask)) is not None]
    refined.sort(key=lambda b: b["area"], reverse=True)
    return refined


def _grow_mask_within(base_mask: np.ndarray, seed_mask: np.ndarray, steps: int) -> np.ndarray:
    grown = seed_mask.astype(bool, copy=True)
    allowed = base_mask.astype(bool, copy=False)
    if not np.any(grown) or not np.any(allowed):
        return np.zeros_like(allowed, dtype=bool)
    for _ in range(max(0, int(steps))):
        next_mask = allowed & (_mean_filter3(grown.astype(np.float32, copy=False)) > 0.0)
        if np.array_equal(next_mask, grown):
            break
        grown = next_mask
    return grown


def _select_dark_core_boxes(
    *,
    score: np.ndarray,
    tissue_mask: np.ndarray,
    threshold_pct: int,
    out_w: int,
    out_h: int,
    min_area: int,
    max_regions: int,
) -> Tuple[List[Box], np.ndarray]:
    empty_mask = np.zeros((out_h, out_w), dtype=bool)
    min_dim = min(out_w, out_h)
    score = np.clip(score.astype(np.float32, copy=False), 0.0, 1.0)
    tissue_float = tissue_mask.astype(np.float32, copy=False)
    coarse_context = _mean_filter3(_mean_filter3(score * tissue_float)).astype(np.float32, copy=False)

    core_threshold_pct = min(98.5, max(float(threshold_pct) + 6.0, 90.0))
    core_threshold = float(np.percentile(score[tissue_mask], core_threshold_pct))
    core_mask = tissue_mask & (score >= core_threshold)
    core_boxes = _find_connected_components(
        core_mask.reshape(-1).tolist(), out_w, out_h, min_area=max(24, int(min_area // 6))
    )

    if not core_boxes:
        base_threshold = float(np.percentile(score[tissue_mask], float(threshold_pct)))
        coarse_threshold = float(np.percentile(coarse_context[tissue_mask], float(max(threshold_pct - 10, 60))))
        base_mask = tissue_mask & ((score >= base_threshold) | (coarse_context >= coarse_threshold))
        base_boxes = _find_connected_components(base_mask.reshape(-1).tolist(), out_w, out_h, min_area=min_area)
        if not base_boxes:
            return [], empty_mask
        merged = _merge_nearby_boxes(
            _refine_dark_region_boxes(base_boxes[:max_regions], tissue_mask),
            gap=max(1, int(round(min_dim * 0.006))),
        )
        refined = _refine_dark_region_boxes(merged, tissue_mask)[:max_regions]
        return refined, _clip_mask_to_boxes(base_mask, refined)

    base_threshold = float(np.percentile(score[tissue_mask], float(max(threshold_pct - 6, 68))))
    coarse_threshold = float(np.percentile(coarse_context[tissue_mask], float(max(threshold_pct - 12, 60))))
    score_mid = float(np.percentile(score[tissue_mask], 50))
    score_hi = float(np.percentile(score[tissue_mask], 95))
    coarse_mid = float(np.percentile(coarse_context[tissue_mask], 50))
    coarse_hi = float(np.percentile(coarse_context[tissue_mask], 95))
    expansion_score_floor = max(base_threshold, score_mid + 0.35 * max(score_hi - score_mid, 0.08))
    expansion_coarse_floor = max(coarse_threshold, coarse_mid + 0.25 * max(coarse_hi - coarse_mid, 0.06))

    base_mask = tissue_mask & (
        (score >= expansion_score_floor)
        | ((score >= max(expansion_score_floor - 0.05, 0.0)) & (coarse_context >= expansion_coarse_floor))
    )
    region_mask = core_mask | (
        tissue_mask
        & (score >= max(expansion_score_floor - 0.03, 0.0))
        & (coarse_context >= max(expansion_coarse_floor - 0.02, 0.0))
    )

    growth_steps = max(4, int(round(min_dim * 0.007)))
    grown_mask = _grow_mask_within(base_mask, region_mask, steps=growth_steps)
    if np.any(grown_mask):
        region_mask = grown_mask

    region_boxes = _find_connected_components(
        region_mask.reshape(-1).tolist(), out_w, out_h, min_area=max(32, int(min_area // 4))
    )
    boxes = region_boxes if region_boxes else core_boxes
    pad = max(1, int(round(min_dim * 0.003)))
    expanded = [_expand_box(box, out_w, out_h, pad) for box in boxes]
    refined = _refine_dark_region_boxes(expanded, tissue_mask)
    merged = _merge_nearby_boxes(refined, gap=max(1, int(round(min_dim * 0.006))))
    refined = _refine_dark_region_boxes(merged, tissue_mask)[:max_regions]
    return refined, _clip_mask_to_boxes(region_mask, refined)


def _compute_cellularity_score(rgb: np.ndarray, gray: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ch_max = np.maximum(r, np.maximum(g, b))
    ch_min = np.minimum(r, np.minimum(g, b))
    chroma = ch_max - ch_min
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    edge_mag = np.pad(grad_x, ((0, 0), (0, 1)), mode="constant") + np.pad(grad_y, ((0, 1), (0, 0)), mode="constant")

    chroma_gate = np.clip(chroma / 32.0, 0.0, 1.0)
    texture_gate = np.clip(edge_mag / 12.0, 0.0, 1.0)
    mid_darkness = np.clip((205.0 - gray) / 90.0, 0.0, 1.0) * np.clip((gray - 55.0) / 45.0, 0.0, 1.0)

    blue_over_red = np.clip((b - r + 25.0) / 55.0, 0.0, 1.0)
    purple_blue = (blue_over_red * chroma_gate * mid_darkness * (0.35 + 0.65 * texture_gate)).astype(np.float32)
    pb_max = max(float(np.percentile(purple_blue[tissue_mask], 95)), 0.01)
    purple_blue_norm = np.clip(purple_blue / pb_max, 0.0, 1.0)

    chromatic_cellular = (
        chroma_gate * mid_darkness * texture_gate * np.clip((np.maximum(r, b) - g + 18.0) / 60.0, 0.0, 1.0)
    ).astype(np.float32)
    chromatic_max = max(float(np.percentile(chromatic_cellular[tissue_mask], 95)), 0.01)
    chromatic_cellular_norm = np.clip(chromatic_cellular / chromatic_max, 0.0, 1.0)

    red_over_blue = np.clip((r - b) / 60.0, 0.0, 1.0)
    red_smooth = (red_over_blue * chroma_gate * np.clip(1.0 - texture_gate, 0.0, 1.0)).astype(np.float32)
    red_smooth_max = max(float(np.percentile(red_smooth[tissue_mask], 95)), 0.01)
    red_smooth_norm = np.clip(red_smooth / red_smooth_max, 0.0, 1.0)

    density = _mean_filter3(tissue_mask.astype(np.float32, copy=False))
    artifact_dark = np.clip((60.0 - gray) / 60.0, 0.0, 1.0) * np.clip(1.0 - chroma / 22.0, 0.0, 1.0)
    gray_black_penalty = np.clip((95.0 - gray) / 55.0, 0.0, 1.0) * np.clip((18.0 - chroma) / 18.0, 0.0, 1.0)
    light_penalty = np.clip((gray - 178.0) / 42.0, 0.0, 1.0)

    return (
        0.56 * purple_blue_norm
        + 0.08 * chromatic_cellular_norm
        + 0.15 * density
        + 0.08 * texture_gate
        - 0.18 * red_smooth_norm
        - 0.14 * artifact_dark
        - 0.18 * gray_black_penalty
        - 0.12 * light_penalty
    ).astype(np.float32, copy=False)


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
        selected_mask = np.zeros((out_h, out_w), dtype=bool)

        if np.any(tissue_mask):
            score = _compute_cellularity_score(rgb, gray, tissue_mask)
            threshold = float(np.percentile(score[tissue_mask], float(threshold_pct)))
            boxes, selected_mask = _select_dark_core_boxes(
                score=score,
                tissue_mask=tissue_mask,
                threshold_pct=threshold_pct,
                out_w=out_w,
                out_h=out_h,
                min_area=min_area,
                max_regions=max_regions,
            )
            if not np.any(selected_mask):
                selected_mask = tissue_mask & (score >= threshold)
        else:
            hist = region.convert("L").histogram()
            threshold = float(_percentile_from_hist(hist, float(threshold_pct)))
            selected_mask = gray <= threshold
            boxes = _find_connected_components(selected_mask.reshape(-1).tolist(), out_w, out_h, min_area=min_area)
            boxes.sort(key=lambda b: b["area"], reverse=True)
            boxes = _refine_dark_region_boxes(boxes[:max_regions], tissue_mask)
            selected_mask = _clip_mask_to_boxes(selected_mask, boxes)

        base_w0, base_h0 = slide.level_dimensions[0]
        scale_x = base_w0 / float(out_w)
        scale_y = base_h0 / float(out_h)
        boxes_level0 = [
            {
                "x0": int(round(b["x"] * scale_x)),
                "y0": int(round(b["y"] * scale_y)),
                "w": int(round(b["w"] * scale_x)),
                "h": int(round(b["h"] * scale_y)),
                "area": int(b["area"]),
            }
            for b in boxes
        ]

        out_dir = os.path.join(DEBUG_ROOT_DIR, run_id, "dark")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "overview.jpg")
        region.save(out_path, format="JPEG", quality=90)

        mask_alpha = (_soft_alpha_mask(selected_mask) * 255.0).astype(np.uint8)
        mask_rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
        mask_rgba[..., 0:3] = 255
        mask_rgba[..., 3] = mask_alpha
        mask_path = os.path.join(out_dir, "mask.png")
        Image.fromarray(mask_rgba, mode="RGBA").save(mask_path, format="PNG")

        return {
            "image_path": out_path,
            "mask_path": mask_path,
            "image_dims": [out_w, out_h],
            "threshold": float(threshold),
            "threshold_pct": int(threshold_pct),
            "boxes": boxes,
            "boxes_level0": boxes_level0,
        }
    finally:
        try:
            slide.close()
        except Exception:
            pass
