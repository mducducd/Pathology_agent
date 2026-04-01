import os
from typing import Any, Dict, List

import numpy as np
import openslide

from .config import DEBUG_ROOT_DIR
from .slide_utils import _read_region_rgb, _resize_to_max_dim


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


def _find_connected_components(mask: List[bool], w: int, h: int, min_area: int) -> List[Dict[str, int]]:
    visited = bytearray(w * h)
    boxes: List[Dict[str, int]] = []

    for idx in range(w * h):
        if not mask[idx] or visited[idx]:
            continue
        stack = [idx]
        visited[idx] = 1
        area = 0
        minx = w
        miny = h
        maxx = 0
        maxy = 0

        while stack:
            i = stack.pop()
            x = i % w
            y = i // w
            area += 1
            if x < minx:
                minx = x
            if y < miny:
                miny = y
            if x > maxx:
                maxx = x
            if y > maxy:
                maxy = y

            if x > 0:
                n = i - 1
                if mask[n] and not visited[n]:
                    visited[n] = 1
                    stack.append(n)
            if x + 1 < w:
                n = i + 1
                if mask[n] and not visited[n]:
                    visited[n] = 1
                    stack.append(n)
            if y > 0:
                n = i - w
                if mask[n] and not visited[n]:
                    visited[n] = 1
                    stack.append(n)
            if y + 1 < h:
                n = i + w
                if mask[n] and not visited[n]:
                    visited[n] = 1
                    stack.append(n)

        if area >= min_area:
            boxes.append(
                {
                    "x": minx,
                    "y": miny,
                    "w": maxx - minx + 1,
                    "h": maxy - miny + 1,
                    "area": area,
                }
            )

    return boxes


def _expand_box(box: Dict[str, int], width: int, height: int, pad: int) -> Dict[str, int]:
    x0 = max(0, int(box["x"]) - int(pad))
    y0 = max(0, int(box["y"]) - int(pad))
    x1 = min(int(width), int(box["x"]) + int(box["w"]) + int(pad))
    y1 = min(int(height), int(box["y"]) + int(box["h"]) + int(pad))
    return {
        "x": x0,
        "y": y0,
        "w": max(1, x1 - x0),
        "h": max(1, y1 - y0),
        "area": int(max(1, box.get("area", 0))),
    }


def _box_bounds(box: Dict[str, int]) -> tuple[int, int, int, int]:
    x0 = int(box["x"])
    y0 = int(box["y"])
    x1 = x0 + int(box["w"])
    y1 = y0 + int(box["h"])
    return x0, y0, x1, y1


def _box_from_bounds(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    area: int,
) -> Dict[str, int]:
    return {
        "x": int(x0),
        "y": int(y0),
        "w": max(1, int(x1) - int(x0)),
        "h": max(1, int(y1) - int(y0)),
        "area": int(max(1, area)),
    }


def _trim_box_to_tissue(
    box: Dict[str, int],
    tissue_mask: np.ndarray,
    *,
    line_threshold: float = 0.72,
    min_side: int = 6,
) -> Dict[str, int] | None:
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


def _box_touches_tissue_edge(box: Dict[str, int], tissue_mask: np.ndarray) -> bool:
    x0, y0, x1, y1 = _box_bounds(box)
    box_mask = tissue_mask[y0:y1, x0:x1]
    if box_mask.size == 0:
        return True

    tissue_fraction = float(np.mean(box_mask))
    if tissue_fraction < 0.94:
        return True

    h, w = box_mask.shape
    band = max(1, int(round(min(h, w) * 0.12)))
    band = min(band, max(1, min(h, w) // 2))

    top = box_mask[:band, :]
    bottom = box_mask[h - band :, :]
    left = box_mask[:, :band]
    right = box_mask[:, w - band :]
    side_tissue_fraction = np.asarray(
        [np.mean(top), np.mean(bottom), np.mean(left), np.mean(right)],
        dtype=np.float32,
    )
    if np.any(side_tissue_fraction < 0.86):
        return True

    corners = (
        box_mask[:band, :band],
        box_mask[:band, w - band :],
        box_mask[h - band :, :band],
        box_mask[h - band :, w - band :],
    )
    corner_tissue_fraction = np.asarray([np.mean(corner) for corner in corners], dtype=np.float32)
    if np.any(corner_tissue_fraction < 0.82):
        return True

    border_mask = np.zeros_like(box_mask, dtype=bool)
    border_mask[:band, :] = True
    border_mask[h - band :, :] = True
    border_mask[:, :band] = True
    border_mask[:, w - band :] = True
    border_tissue_fraction = float(np.mean(box_mask[border_mask]))
    return border_tissue_fraction < 0.90


def _refine_dark_region_boxes(
    boxes: List[Dict[str, int]],
    tissue_mask: np.ndarray,
) -> List[Dict[str, int]]:
    refined: List[Dict[str, int]] = []
    for box in boxes:
        trimmed = _trim_box_to_tissue(box, tissue_mask)
        if trimmed is None:
            continue
        if _box_touches_tissue_edge(trimmed, tissue_mask):
            continue
        refined.append(trimmed)

    refined.sort(key=lambda b: b["area"], reverse=True)
    return refined


def _grow_mask_within(base_mask: np.ndarray, seed_mask: np.ndarray, steps: int) -> np.ndarray:
    grown = seed_mask.astype(bool, copy=True)
    allowed = base_mask.astype(bool, copy=False)
    if not np.any(grown) or not np.any(allowed):
        return np.zeros_like(allowed, dtype=bool)
    for _ in range(max(0, int(steps))):
        neighborhood = _mean_filter3(grown.astype(np.float32, copy=False)) > 0.0
        next_mask = allowed & neighborhood
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
) -> List[Dict[str, int]]:
    # Seed from dense purple cores, but keep the seed threshold broad enough that
    # lighter blue-purple basophilic regions are not missed entirely.
    core_threshold_pct = min(99.0, max(float(threshold_pct) + 8.0, 92.0))
    core_threshold = float(np.percentile(score[tissue_mask], core_threshold_pct))
    core_mask = tissue_mask & (score >= core_threshold)
    core_min_area = max(24, int(min_area // 6))  # Smaller minimum to catch small dense clusters
    core_boxes = _find_connected_components(
        core_mask.reshape(-1).tolist(),
        out_w,
        out_h,
        min_area=core_min_area,
    )

    if not core_boxes:
        # Fallback: use broader threshold if no dense cores found
        base_threshold = float(np.percentile(score[tissue_mask], float(threshold_pct)))
        base_mask = tissue_mask & (score >= base_threshold)
        base_boxes = _find_connected_components(base_mask.reshape(-1).tolist(), out_w, out_h, min_area=min_area)
        if not base_boxes:
            return []
        return base_boxes[:max_regions]

    # Expand cores into the surrounding basophilic region, not just the densest
    # nucleus core, while still avoiding pale background.
    base_threshold = float(np.percentile(score[tissue_mask], float(max(threshold_pct - 4, 72))))
    base_mask = tissue_mask & (score >= base_threshold)

    region_mask = core_mask
    if core_boxes:
        growth_steps = max(4, int(round(min(out_w, out_h) * 0.008)))
        grown_mask = _grow_mask_within(base_mask, core_mask, steps=growth_steps)
        if np.any(grown_mask):
            region_mask = grown_mask

    # Lower min_area to capture more small dense regions
    region_min_area = max(32, int(min_area // 4))
    region_boxes = _find_connected_components(
        region_mask.reshape(-1).tolist(),
        out_w,
        out_h,
        min_area=region_min_area,
    )
    boxes = region_boxes if region_boxes else core_boxes
    pad = max(2, int(round(min(out_w, out_h) * 0.005)))
    expanded = [_expand_box(box, out_w, out_h, pad) for box in boxes]
    refined = _refine_dark_region_boxes(expanded, tissue_mask)
    return refined[:max_regions]


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
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            ch_max = np.maximum(r, np.maximum(g, b))
            ch_min = np.minimum(r, np.minimum(g, b))
            chroma = ch_max - ch_min  # 0-255 range; high = colored tissue
            grad_x = np.abs(np.diff(gray, axis=1))
            grad_y = np.abs(np.diff(gray, axis=0))
            edge_mag = np.pad(grad_x, ((0, 0), (0, 1)), mode="constant") + np.pad(grad_y, ((0, 1), (0, 0)), mode="constant")

            # Prefer chromatic, textured cellular stain in a useful dark range.
            # This avoids collapsing onto low-chroma gray-black debris just because it
            # is darker than real cellular marrow.
            chroma_gate = np.clip(chroma / 32.0, 0.0, 1.0)
            texture_gate = np.clip(edge_mag / 12.0, 0.0, 1.0)
            mid_darkness = (
                np.clip((205.0 - gray) / 90.0, 0.0, 1.0) *
                np.clip((gray - 55.0) / 45.0, 0.0, 1.0)
            )

            # Purple-blue cellular signal remains important, but should be textured and
            # chromatic rather than simply dark.
            blue_over_red = np.clip((b - r + 25.0) / 55.0, 0.0, 1.0)
            purple_blue = (blue_over_red * chroma_gate * mid_darkness * (0.35 + 0.65 * texture_gate)).astype(np.float32)
            pb_max = max(float(np.percentile(purple_blue[tissue_mask], 95)), 0.01)
            purple_blue_norm = np.clip(purple_blue / pb_max, 0.0, 1.0)

            # Rare exception: some dark red-pink cellular marrow can be useful, but
            # deep blue-purple remains the primary target. Smooth red material is bad.
            chromatic_cellular = (
                chroma_gate *
                mid_darkness *
                texture_gate *
                np.clip((np.maximum(r, b) - g + 18.0) / 60.0, 0.0, 1.0)
            ).astype(np.float32)
            chromatic_max = max(float(np.percentile(chromatic_cellular[tissue_mask], 95)), 0.01)
            chromatic_cellular_norm = np.clip(chromatic_cellular / chromatic_max, 0.0, 1.0)

            red_over_blue = np.clip((r - b) / 60.0, 0.0, 1.0)
            red_smooth = (red_over_blue * chroma_gate * np.clip(1.0 - texture_gate, 0.0, 1.0)).astype(np.float32)
            red_smooth_max = max(float(np.percentile(red_smooth[tissue_mask], 95)), 0.01)
            red_smooth_norm = np.clip(red_smooth / red_smooth_max, 0.0, 1.0)

            # --- Density (cellularity) ---
            density = _mean_filter3(tissue_mask.astype(np.float32, copy=False))

            # Very dark + low chroma = artifact penalty (tissue folds, dark debris)
            artifact_dark = np.clip((60.0 - gray) / 60.0, 0.0, 1.0) * np.clip(1.0 - chroma / 22.0, 0.0, 1.0)
            gray_black_penalty = np.clip((95.0 - gray) / 55.0, 0.0, 1.0) * np.clip((18.0 - chroma) / 18.0, 0.0, 1.0)

            # Light area penalty: keep some penalty, but allow lighter blue-purple
            # cellular tissue to remain eligible.
            light_penalty = np.clip((gray - 178.0) / 42.0, 0.0, 1.0)

            # Score: favor textured chromatic cellular stain, not just darkness.
            score = (
                0.56 * purple_blue_norm
                + 0.08 * chromatic_cellular_norm
                + 0.15 * density
                + 0.08 * texture_gate
                - 0.18 * red_smooth_norm
                - 0.14 * artifact_dark
                - 0.18 * gray_black_penalty
                - 0.12 * light_penalty
            ).astype(np.float32, copy=False)
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
            mask_np = tissue_mask & (score >= threshold)
        else:
            hist = region.convert("L").histogram()
            threshold = float(_percentile_from_hist(hist, float(threshold_pct)))
            mask_np = gray <= threshold
            mask = mask_np.reshape(-1).tolist()
            boxes = _find_connected_components(mask, out_w, out_h, min_area=min_area)
            boxes.sort(key=lambda b: b["area"], reverse=True)
            boxes = _refine_dark_region_boxes(boxes[:max_regions], tissue_mask)

        base_w0, base_h0 = slide.level_dimensions[0]
        scale_x = base_w0 / float(out_w)
        scale_y = base_h0 / float(out_h)
        boxes_level0 = []
        for b in boxes:
            boxes_level0.append(
                {
                    "x0": int(round(b["x"] * scale_x)),
                    "y0": int(round(b["y"] * scale_y)),
                    "w": int(round(b["w"] * scale_x)),
                    "h": int(round(b["h"] * scale_y)),
                    "area": int(b["area"]),
                }
            )

        out_dir = os.path.join(DEBUG_ROOT_DIR, run_id, "dark")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "overview.jpg")
        region.save(out_path, format="JPEG", quality=90)

        return {
            "image_path": out_path,
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
