import json
import os
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openslide
from PIL import Image, ImageDraw

from . import state
from .config import MAX_IMG_DIM, MAX_NATIVE_VIEW_DIM
from .tuning_config import tuning_value

DEFAULT_MPP_FALLBACK_UM = float(tuning_value("tools.slide", "DEFAULT_MPP_UM"))

_SAFE_FILENAME_DROP_RE = re.compile(r"[^\w\s-]")
_SAFE_FILENAME_SPACE_RE = re.compile(r"\s")

_LOG_STEP_KEYS = (
    "debug_path",
    "view_level",
    "view_bbox_level0",
    "view_image_dims",
    "field_width_um",
    "field_height_um",
    "tissue_fraction",
    "roi_candidate_count",
    "roi_candidate_source",
    "roi_candidate_warning",
    "roi_candidate_stage",
    "roi_candidate_pipeline",
    "roi_candidate_index_meta",
    "aml_reference_stats",
)

_VIEW_HISTORY_MAX = 8


# ---------------------------------------------------------------------
# SLIDE LOADING / UTILS
# ---------------------------------------------------------------------


def _load_slide() -> openslide.AbstractSlide:
    if state._slide is None:
        print(f"[WSI] Loading slide from: {state.SLIDE_PATH}")
        if not os.path.exists(state.SLIDE_PATH):
            raise FileNotFoundError(f"Slide not found at: {state.SLIDE_PATH}")
        state._slide = openslide.open_slide(state.SLIDE_PATH)
        print(
            f"[WSI] Slide loaded. "
            f"levels={state._slide.level_count}, "
        )
    return state._slide


def _get_mpp_um(slide: openslide.AbstractSlide) -> Optional[float]:
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
            return 10.0 / float(obj)
        except Exception:
            pass

    return None


def _estimate_tissue_fraction(img: Image.Image) -> float:
    gray = np.array(img.convert("L"))
    step = max(1, gray.shape[0] // 128), max(1, gray.shape[1] // 128)
    sampled = gray[::step[0], ::step[1]]
    return float(np.mean(sampled < 240))


def _next_debug_filename(tag: str) -> str:
    state._debug_img_counter += 1
    return os.path.join(state.DEBUG_SAVE_DIR, f"{state._debug_img_counter:04d}_{tag}.jpg")


def _save_debug_image(img: Image.Image, tag: str) -> str:
    path = _next_debug_filename(tag)
    img.save(path, format="JPEG", quality=90)
    return path


def _safe(fn, **kwargs) -> str:
    try:
        out = fn(**kwargs)
        return out if isinstance(out, str) else json.dumps(out)
    except Exception as e:
        if isinstance(e, (FileNotFoundError, openslide.OpenSlideError)):
            state.HAS_FATAL_ERROR = True
            state.LAST_FATAL_ERROR = str(e)
        print("[WSI][ERROR]", e)
        print(traceback.format_exc())
        return json.dumps({"error": str(e), "trace": traceback.format_exc()[:4000]})


def _read_region_rgb(
    slide: openslide.AbstractSlide, x: int, y: int, level: int, size: Tuple[int, int]
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
    out_w, out_h = int(round(w * scale)), int(round(h * scale))
    return region.resize((out_w, out_h), Image.BILINEAR), out_w, out_h


def _choose_level_for_bbox(base_w: int, base_h: int, slide: openslide.AbstractSlide) -> int:
    side0 = max(base_w, base_h)
    for level in range(slide.level_count):
        if side0 / float(slide.level_downsamples[level]) <= MAX_NATIVE_VIEW_DIM:
            return level
    return slide.level_count - 1


def _add_view_to_history(info: Dict[str, Any], tag: str) -> None:
    entry = {**info, "tag": tag}
    state._view_history.append(entry)
    if len(state._view_history) > _VIEW_HISTORY_MAX:
        state._view_history = state._view_history[-_VIEW_HISTORY_MAX:]


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
    return path


def _resolve_mpp(slide: openslide.AbstractSlide) -> float:
    override = getattr(state, "DEFAULT_MPP_UM_OVERRIDE", None)
    if override is not None:
        try:
            mpp = float(override)
            if mpp > 0:
                return mpp
        except Exception:
            pass
    mpp = _get_mpp_um(slide)
    return mpp if (mpp is not None and mpp > 0) else DEFAULT_MPP_FALLBACK_UM


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
    region, out_w, out_h = _resize_to_max_dim(region, max_dim=max(1, int(max_dim or MAX_IMG_DIM)))

    tissue_fraction = _estimate_tissue_fraction(region)
    debug_path = _save_debug_image(region, tag=tag)

    mpp = _resolve_mpp(slide)
    field_width_um = w * mpp
    field_height_um = h * mpp

    x_lvl = int(round(x0 / ds))
    y_lvl = int(round(y0 / ds))

    info = {
        "debug_path": debug_path,
        "view_tag": tag,
        "view_level": level,
        "view_bbox_level": [x_lvl, y_lvl, w_lvl, h_lvl],
        "view_bbox_level0": [x0, y0, w, h],
        "view_image_dims": [out_w, out_h],
        "field_width_um": field_width_um,
        "field_height_um": field_height_um,
        "tissue_fraction": tissue_fraction,
    }

    state._current_view = {
        "x0": x0, "y0": y0, "w": w, "h": h,
        "level": level,
        "level_downsample": ds,
        "shown_w": out_w,
        "shown_h": out_h,
        "debug_path": debug_path,
        "view_tag": tag,
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
    x0n, x1n = sorted([max(0, min(999, x0_999)), max(0, min(999, x1_999))])
    y0n, y1n = sorted([max(0, min(999, y0_999)), max(0, min(999, y1_999))])

    x0_rel, y0_rel = x0n / 999.0, y0n / 999.0
    x1_rel, y1_rel = x1n / 999.0, y1n / 999.0

    x0 = cv_x0 + int(round(x0_rel * cv_w))
    y0 = cv_y0 + int(round(y0_rel * cv_h))
    w = max(1, int(round((x1_rel - x0_rel) * cv_w)))
    h = max(1, int(round((y1_rel - y0_rel) * cv_h)))

    if w / float(cv_w) > 0.7 or h / float(cv_h) > 0.7:
        cx, cy = x0 + w // 2, y0 + h // 2
        w = max(32, int(round(w * shrink_if_large)))
        h = max(32, int(round(h * shrink_if_large)))
        x0, y0 = cx - w // 2, cy - h // 2

    x0 = max(0, min(x0, slide_w0 - 1))
    y0 = max(0, min(y0, slide_h0 - 1))
    w = max(1, min(w, slide_w0 - x0))
    h = max(1, min(h, slide_h0 - y0))

    if max(w, h) / float(min(w, h)) > max_aspect:
        if w > h:
            cx = x0 + w // 2
            w = max(1, int(round(h * max_aspect)))
            x0 = max(0, min(cx - w // 2, slide_w0 - w))
        else:
            cy = y0 + h // 2
            h = max(1, int(round(w * max_aspect)))
            y0 = max(0, min(cy - h // 2, slide_h0 - h))

    return x0, y0, w, h


def _log_step(tool_name: str, nav_reason: str, info: Dict[str, Any]) -> None:
    step_idx = len(state._step_log) + 1
    entry = {
        "step_index": step_idx,
        "tool": tool_name,
        "nav_reason": nav_reason.strip() if nav_reason else "(none provided)",
        **{k: info.get(k) for k in _LOG_STEP_KEYS},
    }
    state._step_log.append(entry)
    print(f"[WSI][STEP_LOG] Step {step_idx}: {tool_name}, nav_reason='{nav_reason}'")


def _safe_filename(text: str, max_len: int = 64) -> str:
    t = _SAFE_FILENAME_DROP_RE.sub("", text.strip())
    out = _SAFE_FILENAME_SPACE_RE.sub("_", t).strip("_")
    return out[:max_len] if out else "tile"
