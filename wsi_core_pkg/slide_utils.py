import json
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import openslide
from PIL import Image, ImageDraw

from . import state
from .config import MAX_IMG_DIM, MAX_NATIVE_VIEW_DIM


# ---------------------------------------------------------------------
# SLIDE LOADING / UTILS
# ---------------------------------------------------------------------


def _load_slide() -> openslide.OpenSlide:
    if state._slide is None:
        print(f"[WSI] Loading slide from: {state.SLIDE_PATH}")
        if not os.path.exists(state.SLIDE_PATH):
            raise FileNotFoundError(f"Slide not found at: {state.SLIDE_PATH}")
        state._slide = openslide.OpenSlide(state.SLIDE_PATH)
        print(
            f"[WSI] Slide loaded. "
            f"levels={state._slide.level_count}, "
            f"level_dimensions={state._slide.level_dimensions}, "
            f"level_downsamples={state._slide.level_downsamples}"
        )
    return state._slide


def _get_mpp_um(slide: openslide.OpenSlide) -> Optional[float]:
    props = slide.properties
    for key in ("openslide.mpp-x", "openslide.mpp-y", "aperio.MPP"):
        v = props.get(key)
        if v:
            try:
                return float(v)
            except Exception:
                pass

    obj = props.get("openslide.objective-power")
    if obj:
        try:
            obj = float(obj)
            return 10.0 / obj
        except Exception:
            pass

    return None


def _estimate_tissue_fraction(img: Image.Image) -> float:
    gray = img.convert("L")
    w, h = gray.size
    pixels = gray.load()

    total = 0
    tissue = 0

    step_x = max(1, w // 128)
    step_y = max(1, h // 128)

    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            total += 1
            if pixels[x, y] < 240:
                tissue += 1

    return float(tissue) / total if total > 0 else 0.0


def _next_debug_filename(tag: str) -> str:
    state._debug_img_counter += 1
    filename = f"{state._debug_img_counter:04d}_{tag}.jpg"
    return os.path.join(state.DEBUG_SAVE_DIR, filename)


def _save_debug_image(img: Image.Image, tag: str) -> str:
    path = _next_debug_filename(tag)
    img.save(path, format="JPEG", quality=90)
    print(f"[WSI][DEBUG] Saved image to: {path}")
    return path


def _safe(fn, **kwargs) -> str:
    try:
        out = fn(**kwargs)
        if isinstance(out, str):
            return out
        return json.dumps(out)
    except Exception as e:
        if isinstance(e, (FileNotFoundError, openslide.OpenSlideError)):
            state.HAS_FATAL_ERROR = True
            state.LAST_FATAL_ERROR = str(e)
        print("[WSI][ERROR]", e)
        print(traceback.format_exc())
        return json.dumps({"error": str(e), "trace": traceback.format_exc()[:4000]})


def _read_region_rgb(
    slide: openslide.OpenSlide, x: int, y: int, level: int, size: Tuple[int, int]
) -> Image.Image:
    region = slide.read_region((x, y), level, size)
    if region.mode == "RGBA":
        bg = Image.new("RGBA", region.size, (255, 255, 255, 255))
        region = Image.alpha_composite(bg, region)
    return region.convert("RGB")


def _resize_to_max_dim(region: Image.Image, max_dim: int) -> Tuple[Image.Image, int, int]:
    w, h = region.size
    if max(w, h) <= max_dim:
        return region, w, h
    scale = max_dim / float(max(w, h))
    out_w = int(round(w * scale))
    out_h = int(round(h * scale))
    region = region.resize((out_w, out_h), Image.BILINEAR)
    return region, out_w, out_h


def _choose_level_for_bbox(base_w: int, base_h: int, slide: openslide.OpenSlide) -> int:
    side0 = max(base_w, base_h)
    for level in range(slide.level_count):
        ds = float(slide.level_downsamples[level])
        side_lvl = side0 / ds
        if side_lvl <= MAX_NATIVE_VIEW_DIM:
            return level
    return slide.level_count - 1


def _add_view_to_history(info: Dict[str, Any], tag: str) -> None:
    entry = dict(info)
    entry["tag"] = tag
    state._view_history.append(entry)
    max_views = 8
    if len(state._view_history) > max_views:
        state._view_history = state._view_history[-max_views:]


def _make_overview_with_current_box(
    tag: str = "overview_with_current",
    draw_current_box: bool = False,
) -> Optional[str]:
    if not state._overview_cache or not state._current_view:
        return None

    slide = _load_slide()
    base_w0, base_h0 = slide.level_dimensions[0]

    ov_level = state._overview_cache["level"]
    lvl_w = state._overview_cache["level_w"]
    lvl_h = state._overview_cache["level_h"]

    region = _read_region_rgb(slide, 0, 0, ov_level, (lvl_w, lvl_h))
    region, out_w, out_h = _resize_to_max_dim(region, MAX_IMG_DIM)
    state._overview_cache["shown_w"] = out_w
    state._overview_cache["shown_h"] = out_h

    vx0 = state._current_view["x0"]
    vy0 = state._current_view["y0"]
    vw = state._current_view["w"]
    vh = state._current_view["h"]

    if draw_current_box:
        draw = ImageDraw.Draw(region)
        x0_px = int(round((vx0 / float(base_w0)) * out_w))
        y0_px = int(round((vy0 / float(base_h0)) * out_h))
        x1_px = int(round(((vx0 + vw) / float(base_w0)) * out_w))
        y1_px = int(round(((vy0 + vh) / float(base_h0)) * out_h))
        draw.rectangle([x0_px, y0_px, x1_px, y1_px], outline="orange", width=3)

    path = _save_debug_image(region, tag=tag)
    state._last_overview_with_box_path = path
    print(
        f"[WSI][OV_BOX] Saved overview image at "
        f"{path} for base bbox=({vx0},{vy0},{vw},{vh})"
    )
    return path


def _render_view_from_base_bbox(
    x0: int,
    y0: int,
    w: int,
    h: int,
    max_dim: int,
    tag: str,
    force_level: Optional[int] = None,
) -> Dict[str, Any]:
    slide = _load_slide()
    base_w0, base_h0 = slide.level_dimensions[0]

    x0 = max(0, min(x0, base_w0 - 1))
    y0 = max(0, min(y0, base_h0 - 1))
    w = max(1, min(w, base_w0 - x0))
    h = max(1, min(h, base_h0 - y0))

    if force_level is None:
        level = _choose_level_for_bbox(w, h, slide)
    else:
        level = max(0, min(force_level, slide.level_count - 1))

    ds = float(slide.level_downsamples[level])
    w_lvl = max(1, int(round(w / ds)))
    h_lvl = max(1, int(round(h / ds)))

    print(
        "[WSI][RENDER] base_bbox=(%d,%d,%d,%d), level=%d, ds=%.2f, level_bbox_dims=(%d,%d)"
        % (x0, y0, w, h, level, ds, w_lvl, h_lvl)
    )

    region = _read_region_rgb(slide, x0, y0, level, (w_lvl, h_lvl))
    region, out_w, out_h = _resize_to_max_dim(region, max_dim=min(max_dim, MAX_IMG_DIM))

    tissue_fraction = _estimate_tissue_fraction(region)

    debug_path = _save_debug_image(region, tag=tag)

    mpp = _get_mpp_um(slide)
    if mpp is not None:
        field_width_um = w * mpp
        field_height_um = h * mpp
    else:
        field_width_um = None
        field_height_um = None

    x_lvl = int(round(x0 / ds))
    y_lvl = int(round(y0 / ds))
    w_lvl_int = int(round(w / ds))
    h_lvl_int = int(round(h / ds))

    info = {
        "debug_path": debug_path,
        "view_level": level,
        "view_bbox_level": [x_lvl, y_lvl, w_lvl_int, h_lvl_int],
        "view_bbox_level0": [x0, y0, w, h],
        "view_image_dims": [out_w, out_h],
        "field_width_um": field_width_um,
        "field_height_um": field_height_um,
        "tissue_fraction": tissue_fraction,
    }

    state._current_view = {
        "x0": x0,
        "y0": y0,
        "w": w,
        "h": h,
        "level": level,
        "level_downsample": ds,
        "shown_w": out_w,
        "shown_h": out_h,
        "debug_path": debug_path,
        "field_width_um": field_width_um,
        "field_height_um": field_height_um,
        "tissue_fraction": tissue_fraction,
    }

    _add_view_to_history(info, tag)
    _make_overview_with_current_box(draw_current_box=False)

    return info


def _bbox_from_norm_with_aspect_controls(
    x0_999: int,
    y0_999: int,
    x1_999: int,
    y1_999: int,
    cv_x0: int,
    cv_y0: int,
    cv_w: int,
    cv_h: int,
    slide_w0: int,
    slide_h0: int,
    shrink_if_large: float = 0.35,
    max_aspect: float = 1.4,
) -> Tuple[int, int, int, int]:
    x0_999 = max(0, min(999, x0_999))
    y0_999 = max(0, min(999, y0_999))
    x1_999 = max(0, min(999, x1_999))
    y1_999 = max(0, min(999, y1_999))
    x0n, x1n = sorted([x0_999, x1_999])
    y0n, y1n = sorted([y0_999, y1_999])

    x0_rel = x0n / 999.0
    y0_rel = y0n / 999.0
    x1_rel = x1n / 999.0
    y1_rel = y1n / 999.0

    x0 = cv_x0 + int(round(x0_rel * cv_w))
    y0 = cv_y0 + int(round(y0_rel * cv_h))
    w = int(round((x1_rel - x0_rel) * cv_w))
    h = int(round((y1_rel - y0_rel) * cv_h))

    w = max(1, w)
    h = max(1, h)

    frac_w = w / float(cv_w)
    frac_h = h / float(cv_h)
    if frac_w > 0.7 or frac_h > 0.7:
        cx = x0 + w // 2
        cy = y0 + h // 2
        w = int(round(w * shrink_if_large))
        h = int(round(h * shrink_if_large))
        w = max(32, w)
        h = max(32, h)
        x0 = cx - w // 2
        y0 = cy - h // 2

    x0 = max(0, min(x0, slide_w0 - 1))
    y0 = max(0, min(y0, slide_h0 - 1))
    w = max(1, min(w, slide_w0 - x0))
    h = max(1, min(h, slide_h0 - y0))

    aspect = max(w / float(h), h / float(w))
    if aspect > max_aspect:
        if w > h:
            target_w = int(round(h * max_aspect))
            cx = x0 + w // 2
            w = max(1, target_w)
            x0 = max(0, min(cx - w // 2, slide_w0 - w))
        else:
            target_h = int(round(w * max_aspect))
            cy = y0 + h // 2
            h = max(1, target_h)
            y0 = max(0, min(cy - h // 2, slide_h0 - h))

    return x0, y0, w, h


def _log_step(tool_name: str, nav_reason: str, info: Dict[str, Any]) -> None:
    step_idx = len(state._step_log) + 1
    entry = {
        "step_index": step_idx,
        "tool": tool_name,
        "nav_reason": nav_reason.strip() if nav_reason else "(none provided)",
        "debug_path": info.get("debug_path"),
        "view_level": info.get("view_level"),
        "view_bbox_level0": info.get("view_bbox_level0"),
        "view_image_dims": info.get("view_image_dims"),
        "field_width_um": info.get("field_width_um"),
        "field_height_um": info.get("field_height_um"),
        "tissue_fraction": info.get("tissue_fraction"),
    }
    state._step_log.append(entry)
    print(f"[WSI][STEP_LOG] Step {step_idx}: {tool_name}, nav_reason='{nav_reason}'")


def _safe_filename(text: str, max_len: int = 64) -> str:
    cleaned = []
    for ch in text.strip():
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append("_")
    out = "".join(cleaned).strip("_")
    return out[:max_len] if out else "tile"
