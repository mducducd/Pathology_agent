import os
from typing import Any, Dict, List

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


def detect_dark_regions(
    slide_path: str,
    run_id: str,
    max_dim: int = 1024,
    threshold_pct: int = 85,
    min_area: int = 800,
    max_regions: int = 30,
) -> Dict[str, Any]:
    slide = openslide.OpenSlide(slide_path)
    try:
        level = slide.level_count - 1
        level_w, level_h = slide.level_dimensions[level]

        region = _read_region_rgb(slide, 0, 0, level, (level_w, level_h))
        region, out_w, out_h = _resize_to_max_dim(region, max_dim=max_dim)

        gray = region.convert("L")
        hist = gray.histogram()
        threshold = _percentile_from_hist(hist, float(threshold_pct))

        pixels = list(gray.getdata())
        mask = [px <= threshold for px in pixels]

        boxes = _find_connected_components(mask, out_w, out_h, min_area=min_area)
        boxes.sort(key=lambda b: b["area"], reverse=True)
        boxes = boxes[:max_regions]

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
            "threshold": int(threshold),
            "threshold_pct": int(threshold_pct),
            "boxes": boxes,
            "boxes_level0": boxes_level0,
        }
    finally:
        try:
            slide.close()
        except Exception:
            pass
