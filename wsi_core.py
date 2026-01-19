import os
import json
import base64
import traceback
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import shutil

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from agents import (
    Agent,
    Runner,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
    function_tool,
    enable_verbose_stdout_logging,
)

import openslide
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
load_dotenv()

SLIDE_PATH = os.path.abspath("341476.svs")
MODEL_NAME = "Qwen3-VL-235B-A22B-Thinking-FP8"

client_async = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1"),
)
client_sync = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1"),
)

set_default_openai_client(client_async)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)
enable_verbose_stdout_logging()

# You can increase this for more detail (e.g. 1024 or 1200)
MAX_IMG_DIM = 1024         # max width/height for any image we send
MAX_NATIVE_VIEW_DIM = 4096 # size at slide level before downsampling for view
MAX_TURNS = 30

# Base-level side length in pixels for ROI fields (high power)
ROI_TARGET_SIDE_PX = 1500

DEBUG_ROOT_DIR = os.path.abspath("./wsi_debug")
os.makedirs(DEBUG_ROOT_DIR, exist_ok=True)
DEBUG_SAVE_DIR = DEBUG_ROOT_DIR
_debug_img_counter = 0

REPORT_ROOT_DIR = os.path.abspath("./wsi_reports")
os.makedirs(REPORT_ROOT_DIR, exist_ok=True)

RUN_ID: Optional[str] = None

# ---------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------

_slide: Optional[openslide.OpenSlide] = None

_current_view: Dict[str, Any] = {}
# keys: x0, y0, w, h (level-0),
#       level, level_downsample,
#       shown_w, shown_h,
#       debug_path,
#       field_width_um, field_height_um,
#       tissue_fraction

_overview_cache: Dict[str, Any] = {}
# keys: level, level_w, level_h, level_downsample, shown_w, shown_h

_last_overview_with_box_path: Optional[str] = None

_view_history: List[Dict[str, Any]] = []
# each: {debug_path, tag, view_level, view_bbox_level0,
#        view_image_dims, field_width_um, field_height_um, tissue_fraction}

_step_log: List[Dict[str, Any]] = []

_roi_marks: List[Dict[str, Any]] = []

# ---------------------------------------------------------------------
# SLIDE LOADING / UTILS
# ---------------------------------------------------------------------


def _load_slide() -> openslide.OpenSlide:
    global _slide
    if _slide is None:
        print(f"[WSI] Loading slide from: {SLIDE_PATH}")
        if not os.path.exists(SLIDE_PATH):
            raise FileNotFoundError(f"Slide not found at: {SLIDE_PATH}")
        _slide = openslide.OpenSlide(SLIDE_PATH)
        print(
            f"[WSI] Slide loaded. "
            f"levels={_slide.level_count}, "
            f"level_dimensions={_slide.level_dimensions}, "
            f"level_downsamples={_slide.level_downsamples}"
        )
    return _slide


def _get_mpp_um(slide: openslide.OpenSlide) -> Optional[float]:
    """
    Approximate microns-per-pixel at level 0.
    Prefer explicit metadata; fall back to an approximation from objective power.
    """
    props = slide.properties
    for key in ("openslide.mpp-x", "openslide.mpp-y", "aperio.MPP"):
        v = props.get(key)
        if v:
            try:
                return float(v)
            except Exception:
                pass

    # Fallback: approximate from objective power
    obj = props.get("openslide.objective-power")
    if obj:
        try:
            obj = float(obj)
            # crude heuristic: 40x -> ~0.25 µm/px → 10 / 40 = 0.25
            return 10.0 / obj
        except Exception:
            pass

    return None


def _estimate_tissue_fraction(img: Image.Image) -> float:
    """
    Rough estimate of how much of this image is tissue vs background.

    Implementation:
    - Convert to grayscale.
    - Sample pixels on a coarse grid.
    - Treat very bright pixels as background, darker as tissue.
    """
    gray = img.convert("L")
    w, h = gray.size
    pixels = gray.load()

    total = 0
    tissue = 0

    # Coarse sampling to keep it cheap
    step_x = max(1, w // 128)
    step_y = max(1, h // 128)

    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            total += 1
            # 255 = pure white; 240+ ~ background in H&E
            if pixels[x, y] < 240:
                tissue += 1

    return float(tissue) / total if total > 0 else 0.0


def _next_debug_filename(tag: str) -> str:
    global _debug_img_counter
    _debug_img_counter += 1
    filename = f"{_debug_img_counter:04d}_{tag}.jpg"
    return os.path.join(DEBUG_SAVE_DIR, filename)


def _save_debug_image(img: Image.Image, tag: str) -> str:
    path = _next_debug_filename(tag)
    img.save(path, format="JPEG", quality=90)
    print(f"[WSI][DEBUG] Saved image to: {path}")
    return path


def _safe(fn, **kwargs) -> str:
    """Safely execute a function tool and return its output as a JSON string."""
    try:
        out = fn(**kwargs)
        if isinstance(out, str):
            return out
        return json.dumps(out)
    except Exception as e:
        print("[WSI][ERROR]", e)
        print(traceback.format_exc())
        return json.dumps({"error": str(e), "trace": traceback.format_exc()[:4000]})


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
    """Choose the highest resolution level that fits the base_w x base_h region into MAX_NATIVE_VIEW_DIM."""
    side0 = max(base_w, base_h)
    for level in range(slide.level_count):
        ds = float(slide.level_downsamples[level])
        side_lvl = side0 / ds
        if side_lvl <= MAX_NATIVE_VIEW_DIM:
            return level
    return slide.level_count - 1


def _add_view_to_history(info: Dict[str, Any], tag: str) -> None:
    """
    Store a view in history (with metadata) so we can re-inject it into the context.
    """
    global _view_history
    entry = dict(info)
    entry["tag"] = tag
    _view_history.append(entry)
    # keep more recent views (tunable)
    max_views = 8
    if len(_view_history) > max_views:
        _view_history = _view_history[-max_views:]


def _make_overview_with_current_box(tag: str = "overview_with_current") -> Optional[str]:
    """
    Generate an overview image with the current view's bounding box drawn in red.
    """
    global _overview_cache, _current_view, _last_overview_with_box_path

    if not _overview_cache or not _current_view:
        return None

    slide = _load_slide()
    base_w0, base_h0 = slide.level_dimensions[0]

    ov_level = _overview_cache["level"]
    lvl_w = _overview_cache["level_w"]
    lvl_h = _overview_cache["level_h"]

    region = slide.read_region((0, 0), ov_level, (lvl_w, lvl_h)).convert("RGB")
    region, out_w, out_h = _resize_to_max_dim(region, MAX_IMG_DIM)
    _overview_cache["shown_w"] = out_w
    _overview_cache["shown_h"] = out_h

    vx0 = _current_view["x0"]
    vy0 = _current_view["y0"]
    vw = _current_view["w"]
    vh = _current_view["h"]

    draw = ImageDraw.Draw(region)
    x0_px = int(round((vx0 / float(base_w0)) * out_w))
    y0_px = int(round((vy0 / float(base_h0)) * out_h))
    x1_px = int(round(((vx0 + vw) / float(base_w0)) * out_w))
    y1_px = int(round(((vy0 + vh) / float(base_h0)) * out_h))
    draw.rectangle([x0_px, y0_px, x1_px, y1_px], outline="red", width=3)

    path = _save_debug_image(region, tag=tag)
    _last_overview_with_box_path = path
    print(
        f"[WSI][OV_BOX] Saved overview-with-current box at "
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
    """
    Read the region defined by the base-level (level 0) bounding box (x0, y0, w, h),
    optionally forcing a specific zoom level, and resize it for display.
    Updates _current_view and view history, and refreshes the overview-with-box.
    """
    global _current_view

    slide = _load_slide()
    base_w0, base_h0 = slide.level_dimensions[0]

    # Clamp region to slide boundaries
    x0 = max(0, min(x0, base_w0 - 1))
    y0 = max(0, min(y0, base_h0 - 1))
    w = max(1, min(w, base_w0 - x0))
    h = max(1, min(h, base_h0 - y0))

    # Determine appropriate level
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

    # Extract region at the chosen level
    region = slide.read_region((x0, y0), level, (w_lvl, h_lvl)).convert("RGB")
    region, out_w, out_h = _resize_to_max_dim(region, max_dim=min(max_dim, MAX_IMG_DIM))

    # Estimate tissue vs background
    tissue_fraction = _estimate_tissue_fraction(region)

    debug_path = _save_debug_image(region, tag=tag)

    # Approximate field size in micrometers at level 0
    mpp = _get_mpp_um(slide)
    if mpp is not None:
        field_width_um = w * mpp
        field_height_um = h * mpp
    else:
        field_width_um = None
        field_height_um = None

    # Base-level and level-coordinates for the view
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

    # Update current view
    _current_view = {
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
    _make_overview_with_current_box()

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
    """
    Convert a normalized (0–999) box relative to a view bbox into an absolute level-0 bbox,
    with optional shrinkage and aspect ratio control.
    """
    # Clamp and sort
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

    # Shrink if region is very large relative to the view
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

    # Clamp to slide
    x0 = max(0, min(x0, slide_w0 - 1))
    y0 = max(0, min(y0, slide_h0 - 1))
    w = max(1, min(w, slide_w0 - x0))
    h = max(1, min(h, slide_h0 - y0))

    # Enforce max aspect ratio
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
    """Log a navigation step with its tool name, reason, and view info for report."""
    step_idx = len(_step_log) + 1
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
    _step_log.append(entry)
    print(f"[WSI][STEP_LOG] Step {step_idx}: {tool_name}, nav_reason='{nav_reason}'")


# ---------------------------------------------------------------------
# MULTIMODAL INJECTION (CURRENT VIEW + OVERVIEW + HISTORY)
# ---------------------------------------------------------------------

_real_async_chat_create = client_async.chat.completions.create
_real_sync_chat_create = client_sync.chat.completions.create


def _encode_image_as_data_url(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        img_bytes = f.read()
    image_b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{image_b64}"


def _format_field_width_caption(field_width_um: Optional[float]) -> str:
    if field_width_um is None:
        return ""
    # round to nearest 50 µm for readability
    approx = int(round(field_width_um / 50.0) * 50)
    return f", field ~{approx} µm wide"


def _inject_wsi_images(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Insert WSI context into the conversation, always making it explicit which image
    is the CURRENT VIEW (i.e., the one whose coordinates the tools refer to).

    Priority:
    1) CURRENT VIEW (always).
       - If the last tool was wsi_mark_roi_norm and we have at least one ROI,
         the current view is the newly marked ROI and the caption reflects that.
    2) Whole-slide overview with the current bounding box.
    3) A few recent views (historical context).
    """
    last_tool_idx = None
    tool_name = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "tool":
            last_tool_idx = i
            tool_name = messages[i].get("name")
            break
    if last_tool_idx is None:
        return messages

    new_messages = list(messages)
    insert_pos = last_tool_idx + 1

    # 1) CURRENT VIEW
    curr_path = _current_view.get("debug_path") if _current_view else None
    curr_url = _encode_image_as_data_url(curr_path) if curr_path else None
    if curr_url:
        fw = _current_view.get("field_width_um")
        extra = _format_field_width_caption(fw)

        if tool_name == "wsi_mark_roi_norm" and _roi_marks:
            last_roi = _roi_marks[-1]
            roi_id = last_roi.get("roi_id")
            label = last_roi.get("label", "")
            text = (
                f"CURRENT VIEW = NEWLY MARKED ROI (ROI #{roi_id}: {label}{extra}). "
                "Carefully inspect this high-power field. If it is mostly background or "
                "not diagnostic, your very next action should be to call "
                "wsi_discard_last_roi. All coordinates for any subsequent tool call must "
                "be chosen relative to THIS image."
            )
        else:
            text = (
                f"CURRENT VIEW for navigation{extra}. "
                "All coordinates for your NEXT tool call must be chosen relative to THIS "
                "image. Do NOT select boxes centered on blank/white background; always "
                "place boxes tightly around tissue."
            )

        current_view_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": curr_url}},
            ],
        }
        new_messages.insert(insert_pos, current_view_msg)
        insert_pos += 1

    # 2) Overview with current box
    if _last_overview_with_box_path:
        url = _encode_image_as_data_url(_last_overview_with_box_path)
        if url:
            fw = _current_view.get("field_width_um") if _current_view else None
            extra = _format_field_width_caption(fw)
            overview_msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Whole-slide overview (red box = current view{extra}).",
                    },
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
            new_messages.insert(insert_pos, overview_msg)
            insert_pos += 1

    # 3) Last few views (historical context)
    for view in _view_history[-4:]:
        url = _encode_image_as_data_url(view["debug_path"])
        if not url:
            continue
        fw = view.get("field_width_um")
        extra = _format_field_width_caption(fw)
        tag = view.get("tag", "view")
        view_msg = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Previous view ({tag}{extra}).",
                },
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
        new_messages.append(view_msg)

    return new_messages


def _patched_async_chat_create(*args, **kwargs):
    msgs = kwargs.get("messages")
    if isinstance(msgs, list):
        kwargs["messages"] = _inject_wsi_images(msgs)
    return _real_async_chat_create(*args, **kwargs)


def _patched_sync_chat_create(*args, **kwargs):
    msgs = kwargs.get("messages")
    if isinstance(msgs, list):
        kwargs["messages"] = _inject_wsi_images(msgs)
    return _real_sync_chat_create(*args, **kwargs)


client_async.chat.completions.create = _patched_async_chat_create
client_sync.chat.completions.create = _patched_sync_chat_create

# ---------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------


@function_tool
def wsi_get_overview_view(
    nav_reason: str = "Initial overview of the whole slide",
    max_dim: int = MAX_IMG_DIM,
) -> str:
    """
    Get a low-resolution overview image of the whole slide and set the current view
    to the full slide. The approximate field width in micrometers is included.
    """
    def _inner(nav_reason: str, max_dim: int) -> Dict[str, Any]:
        global _overview_cache

        slide = _load_slide()
        base_w0, base_h0 = slide.level_dimensions[0]

        info = _render_view_from_base_bbox(
            x0=0,
            y0=0,
            w=base_w0,
            h=base_h0,
            max_dim=min(max_dim, MAX_IMG_DIM),
            tag="overview",
        )

        level = _current_view["level"]
        ds = _current_view["level_downsample"]
        lvl_w, lvl_h = slide.level_dimensions[level]
        _overview_cache = {
            "level": level,
            "level_w": lvl_w,
            "level_h": lvl_h,
            "level_downsample": ds,
            "shown_w": _current_view["shown_w"],
            "shown_h": _current_view["shown_h"],
        }

        _make_overview_with_current_box()
        _log_step("wsi_get_overview_view", nav_reason, info)
        return info

    return _safe(_inner, nav_reason=nav_reason, max_dim=min(max_dim, MAX_IMG_DIM))


@function_tool
def wsi_zoom_current_norm(
    x0_999: int,
    y0_999: int,
    x1_999: int,
    y1_999: int,
    nav_reason: str = "",
    max_dim: int = MAX_IMG_DIM,
) -> str:
    """
    Zoom into a rectangular region defined in normalized 0–999 coordinates
    relative to the CURRENT view.
    The tool output includes approximate field width/height in micrometers and
    a tissue_fraction estimate.
    """
    def _inner(
        x0_999: int,
        y0_999: int,
        x1_999: int,
        y1_999: int,
        nav_reason: str,
        max_dim: int,
    ) -> Dict[str, Any]:
        if not _current_view:
            raise RuntimeError("wsi_zoom_current_norm called before wsi_get_overview_view.")

        slide = _load_slide()
        cv_x0 = _current_view["x0"]
        cv_y0 = _current_view["y0"]
        cv_w = _current_view["w"]
        cv_h = _current_view["h"]
        slide_w0, slide_h0 = slide.level_dimensions[0]

        print(
            "[WSI][ZOOM_CUR] norm_box=(%d,%d,%d,%d), current_view_base_bbox=(%d,%d,%d,%d)"
            % (x0_999, y0_999, x1_999, y1_999, cv_x0, cv_y0, cv_w, cv_h)
        )

        x0_new, y0_new, w_new, h_new = _bbox_from_norm_with_aspect_controls(
            x0_999,
            y0_999,
            x1_999,
            y1_999,
            cv_x0,
            cv_y0,
            cv_w,
            cv_h,
            slide_w0,
            slide_h0,
            shrink_if_large=1.0,
            max_aspect=3.0,
        )

        info = _render_view_from_base_bbox(
            x0=x0_new,
            y0=y0_new,
            w=w_new,
            h=h_new,
            max_dim=min(max_dim, MAX_IMG_DIM),
            tag="zoom",
        )

        tf = info.get("tissue_fraction")
        if tf is not None and tf < 0.15:
            info["tissue_warning"] = (
                "This zoomed field is mostly background/empty glass (low tissue_fraction). "
                "You should NOT mark ROIs here. Instead, zoom or pan toward visible tissue "
                "in this CURRENT VIEW before proceeding."
            )

        _log_step("wsi_zoom_current_norm", nav_reason, info)
        return info

    return _safe(
        _inner,
        x0_999=x0_999,
        y0_999=y0_999,
        x1_999=x1_999,
        y1_999=y1_999,
        nav_reason=nav_reason,
        max_dim=max_dim,
    )


@function_tool
def wsi_zoom_full_norm(
    x0_999: int,
    y0_999: int,
    x1_999: int,
    y1_999: int,
    nav_reason: str = "",
    max_dim: int = MAX_IMG_DIM,
) -> str:
    """
    Zoom to a rectangular region specified in normalized 0–999 coordinates relative
    to the ENTIRE slide (level 0). Use this to jump directly to a region on the whole slide.
    Output includes approximate field width/height in micrometers and tissue_fraction.
    """
    def _inner(
        x0_999: int,
        y0_999: int,
        x1_999: int,
        y1_999: int,
        nav_reason: str,
        max_dim: int,
    ) -> Dict[str, Any]:
        slide = _load_slide()
        slide_w0, slide_h0 = slide.level_dimensions[0]
        print(f"[WSI][ZOOM_FULL] norm_box=({x0_999},{y0_999},{x1_999},{y1_999}) on full slide")

        x0_new, y0_new, w_new, h_new = _bbox_from_norm_with_aspect_controls(
            x0_999,
            y0_999,
            x1_999,
            y1_999,
            cv_x0=0,
            cv_y0=0,
            cv_w=slide_w0,
            cv_h=slide_h0,
            slide_w0=slide_w0,
            slide_h0=slide_h0,
            shrink_if_large=1.0,
            max_aspect=3.0,
        )

        info = _render_view_from_base_bbox(
            x0=x0_new,
            y0=y0_new,
            w=w_new,
            h=h_new,
            max_dim=min(max_dim, MAX_IMG_DIM),
            tag="zoom",
        )

        tf = info.get("tissue_fraction")
        if tf is not None and tf < 0.15:
            info["tissue_warning"] = (
                "Selected region is mostly background/empty glass (low tissue_fraction). "
                "You should pick coordinates over tissue areas in the overview and try again."
            )

        _log_step("wsi_zoom_full_norm", nav_reason, info)
        return info

    return _safe(
        _inner,
        x0_999=x0_999,
        y0_999=y0_999,
        x1_999=x1_999,
        y1_999=y1_999,
        nav_reason=nav_reason,
        max_dim=max_dim,
    )


@function_tool
def wsi_pan_current(
    dx_999: int,
    dy_999: int,
    nav_reason: str = "",
    max_dim: int = MAX_IMG_DIM,
) -> str:
    """
    Pan (move) the current view by a relative offset in normalized coordinates
    (dx_999, dy_999 in -999..999). Positive values move right/down.
    Output includes approximate field width/height in micrometers and tissue_fraction.
    """
    def _inner(
        dx_999: int,
        dy_999: int,
        nav_reason: str,
        max_dim: int,
    ) -> Dict[str, Any]:
        if not _current_view:
            raise RuntimeError("wsi_pan_current called before wsi_get_overview_view.")

        slide = _load_slide()
        dx_999 = max(-999, min(999, dx_999))
        dy_999 = max(-999, min(999, dy_999))

        cv_x0 = _current_view["x0"]
        cv_y0 = _current_view["y0"]
        cv_w = _current_view["w"]
        cv_h = _current_view["h"]

        dx_rel = dx_999 / 999.0
        dy_rel = dy_999 / 999.0

        dx_base = int(round(dx_rel * cv_w))
        dy_base = int(round(dy_rel * cv_h))

        slide_w0, slide_h0 = slide.level_dimensions[0]

        x0_new = max(0, min(cv_x0 + dx_base, slide_w0 - cv_w))
        y0_new = max(0, min(cv_y0 + dy_base, slide_h0 - cv_h))

        info = _render_view_from_base_bbox(
            x0=x0_new,
            y0=y0_new,
            w=cv_w,
            h=cv_h,
            max_dim=min(max_dim, MAX_IMG_DIM),
            tag="pan",
        )
        _log_step("wsi_pan_current", nav_reason, info)
        return info

    return _safe(
        _inner,
        dx_999=dx_999,
        dy_999=dy_999,
        nav_reason=nav_reason,
        max_dim=max_dim,
    )


@function_tool
def wsi_get_view_info(nav_reason: str = "Get current view info") -> str:
    """
    Return metadata about the current view, including:
    - bbox at level 0,
    - approximate field width/height in micrometers,
    - objective power and effective magnification (approximate),
    - tissue_fraction estimate.
    """
    def _inner(nav_reason: str) -> Dict[str, Any]:
        slide = _load_slide()
        if not _current_view:
            raise RuntimeError("No current view. Call wsi_get_overview_view first.")

        level = _current_view["level"]
        ds = float(_current_view["level_downsample"])
        objective = float(slide.properties.get("openslide.objective-power", 40.0))
        eff_mag = objective / ds if ds > 0 else None

        info = {
            "level": level,
            "bbox_level0": [
                _current_view["x0"],
                _current_view["y0"],
                _current_view["w"],
                _current_view["h"],
            ],
            "downsample": ds,
            "objective_power": objective,
            "effective_magnification": eff_mag,
            "field_width_um": _current_view.get("field_width_um"),
            "field_height_um": _current_view.get("field_height_um"),
            "tissue_fraction": _current_view.get("tissue_fraction"),
        }
        _log_step(
            "wsi_get_view_info",
            nav_reason,
            {
                "view_level": level,
                "view_bbox_level0": info["bbox_level0"],
                "field_width_um": info["field_width_um"],
                "field_height_um": info["field_height_um"],
                "tissue_fraction": info["tissue_fraction"],
            },
        )
        return info

    return _safe(_inner, nav_reason=nav_reason)


@function_tool
def wsi_mark_roi_norm(
    x0_999: int,
    y0_999: int,
    x1_999: int,
    y1_999: int,
    label: str,
    note: str = "",
    importance: int = 1,
    nav_reason: str = "Mark ROI in current view",
) -> str:
    """
    Mark a high-resolution Region of Interest (ROI) in the CURRENT view.

    IMPORTANT:
    - The normalized box is used primarily to choose the *center* of the ROI.
    - The ROI itself is always a relatively small, fixed-size high-power square at level 0
      (ROI_TARGET_SIDE_PX × ROI_TARGET_SIDE_PX), so individual cells and lymphocytes
      are clearly visible.
    - The tool output includes approximate field width/height in micrometers, tissue_fraction,
      and a hint about next steps (inspect & possibly discard).
    """
    def _inner(
        x0_999: int,
        y0_999: int,
        x1_999: int,
        y1_999: int,
        label: str,
        note: str,
        importance: int,
        nav_reason: str,
    ) -> Dict[str, Any]:
        global _roi_marks

        if not _current_view:
            raise RuntimeError("wsi_mark_roi_norm called before wsi_get_overview_view.")

        slide = _load_slide()
        cv_x0 = _current_view["x0"]
        cv_y0 = _current_view["y0"]
        cv_w = _current_view["w"]
        cv_h = _current_view["h"]
        slide_w0, slide_h0 = slide.level_dimensions[0]

        # Clamp normalized coords
        x0_999_cl = max(0, min(999, x0_999))
        x1_999_cl = max(0, min(999, x1_999))
        y0_999_cl = max(0, min(999, y0_999))
        y1_999_cl = max(0, min(999, y1_999))

        # Center of the user-selected box in normalized coords
        cx_999 = (x0_999_cl + x1_999_cl) / 2.0
        cy_999 = (y0_999_cl + y1_999_cl) / 2.0

        cx_rel = cx_999 / 999.0
        cy_rel = cy_999 / 999.0

        # Map center to base-level coordinates
        cx_base = cv_x0 + int(round(cx_rel * cv_w))
        cy_base = cv_y0 + int(round(cy_rel * cv_h))

        # Define fixed-size square ROI at level 0
        side = min(ROI_TARGET_SIDE_PX, slide_w0, slide_h0)
        w_new = side
        h_new = side

        x0_new = cx_base - w_new // 2
        y0_new = cy_base - h_new // 2

        # Clamp to slide boundaries
        x0_new = max(0, min(x0_new, slide_w0 - w_new))
        y0_new = max(0, min(y0_new, slide_h0 - h_new))

        print(
            "[WSI][ROI_NORM] center=(%.1f,%.1f) -> base_center=(%d,%d), "
            "ROI_bbox=(%d,%d,%d,%d)"
            % (cx_999, cy_999, cx_base, cy_base, x0_new, y0_new, w_new, h_new)
        )

        info = _render_view_from_base_bbox(
            x0=x0_new,
            y0=y0_new,
            w=w_new,
            h=h_new,
            max_dim=MAX_IMG_DIM,
            tag="roi",
            force_level=0,  # enforce highest resolution
        )

        tf = info.get("tissue_fraction")
        if tf is not None and tf < 0.15:
            info["tissue_warning"] = (
                "This high-power ROI is mostly background/empty glass (low tissue_fraction). "
                "You should immediately discard it using wsi_discard_last_roi and select an "
                "ROI centered on diagnostic tissue."
            )

        # Magnification metadata
        level = info["view_level"]
        ds = float(slide.level_downsamples[level])
        objective = float(slide.properties.get("openslide.objective-power", 40.0))
        eff_mag = objective / ds if ds > 0 else None

        roi_id = len(_roi_marks) + 1
        roi = {
            "roi_id": roi_id,
            "label": label,
            "note": note,
            "importance": int(importance),
            "view_level": level,
            "view_bbox_level0": info["view_bbox_level0"],
            "downsample": ds,
            "objective_power": objective,
            "effective_magnification": eff_mag,
            "debug_path": info["debug_path"],
            "field_width_um": info.get("field_width_um"),
            "field_height_um": info.get("field_height_um"),
            "tissue_fraction": info.get("tissue_fraction"),
        }

        # Hint for the model about what to do next
        roi["next_action_hint"] = (
            "You just requested a high-power ROI. "
            "Carefully inspect the newly shown ROI image in the conversation. "
            "If it is mostly background, out of focus, or not diagnostic, "
            "your very next step should be to call wsi_discard_last_roi with a brief nav_reason. "
            "If it is diagnostic, you may either mark additional ROIs or continue navigation."
        )

        _roi_marks.append(roi)

        print(f"[WSI][ROI_NORM] Marked ROI {roi_id}: {label} (importance={importance})")
        _log_step("wsi_mark_roi_norm", nav_reason, info)

        return roi

    return _safe(
        _inner,
        x0_999=x0_999,
        y0_999=y0_999,
        x1_999=x1_999,
        y1_999=y1_999,
        label=label,
        note=note,
        importance=importance,
        nav_reason=nav_reason,
    )


@function_tool
def wsi_discard_last_roi(
    nav_reason: str = "Discard last ROI if not useful",
) -> str:
    """
    Discard the most recently marked ROI (for example, if it was mostly background
    or not diagnostically useful). Use this immediately after marking an ROI if
    you decide it should not be kept.
    """
    def _inner(nav_reason: str) -> Dict[str, Any]:
        if not _roi_marks:
            return {"ok": False, "message": "No ROI to discard."}
        roi = _roi_marks.pop()
        print(f"[WSI][ROI] Discarded ROI {roi['roi_id']}: {roi['label']}")
        _log_step(
            "wsi_discard_last_roi",
            nav_reason,
            {"view_bbox_level0": roi.get("view_bbox_level0")},
        )
        return {"ok": True, "discarded_roi_id": roi["roi_id"], "label": roi["label"]}

    return _safe(_inner, nav_reason=nav_reason)

# ---------------------------------------------------------------------
# AGENTS
# ---------------------------------------------------------------------

WSIPathologyAgent = Agent(
    name="WSIPathologyAgent",
    model=MODEL_NAME,
    instructions=(
        "You are a whole-slide image (WSI) exploration agent, acting like an experienced pathologist "
        "using a digital slide viewer.\n"
        "\n"
        "GENERAL ROLE:\n"
        "- The user prompt defines your specific task (for example, general description, MSI screening, etc.).\n"
        "- Always follow the clinical / diagnostic task described in the user prompt while using the tools below.\n"
        "\n"
        "SLIDE AND STAIN:\n"
        "- Slides are H&E stained. Tissue appears in shades of pink/purple; background is white.\n"
        "- Always focus navigation on tissue, not blank background.\n"
        "\n"
        "COORDINATES:\n"
        "- All tool coordinates are integers 0–999 for x and y, always relative to the CURRENT image view.\n"
        "- (0,0) = top-left; (999,999) = bottom-right.\n"
        "- Rectangles are defined by two opposite corners (x0,y0) and (x1,y1).\n"
        "\n"
        "FIELD SIZE / MAGNIFICATION:\n"
        "- You do NOT need to reason about internal levels or downsample factors.\n"
        "- For each view/ROI, you will see an approximate field width in micrometers (µm) in the tool output / image captions.\n"
        "- Rough guide:\n"
        "  * Very low power / overview: field width in the tens of thousands of µm.\n"
        "  * Intermediate power: field width ~2000–4000 µm.\n"
        "  * High power (good for cellular detail and lymphocytes): field width ~300–800 µm.\n"
        "- If the field is still wider than ~2000 µm and you need cellular detail, zoom in further on tissue.\n"
        "- Tools also expose a tissue_fraction estimate; if tissue_fraction is low (<0.15), the view is mostly background and you should pan/zoom towards tissue.\n"
        "\n"
        "NAVIGATION STRATEGY (APPLIES TO ALL TASKS):\n"
        "1) Start with wsi_get_overview_view to see the entire slide.\n"
        "   - Identify where tissue fragments are and how they are distributed.\n"
        "   - If the task involves tumor assessment, roughly locate suspected tumor regions at low power.\n"
        "2) Systematically explore multiple regions:\n"
        "   - Use wsi_zoom_full_norm from the overview to zoom into major tissue fragments or distant parts of a large fragment.\n"
        "   - Use wsi_zoom_current_norm to step from overview → intermediate → high power on tissue areas.\n"
        "   - Use wsi_pan_current to move laterally at the same magnification along interfaces or lesions.\n"
        "3) Avoid getting stuck:\n"
        "   - After exploring one region, use wsi_get_overview_view or wsi_zoom_full_norm to deliberately move to a distinct region.\n"
        "   - Inspect at least a few distinct areas at high power before concluding.\n"
        "\n"
        "ROIs AND SELF-CHECK:\n"
        "- When you find diagnostically significant tissue (for ANY task), call wsi_mark_roi_norm on that area.\n"
        "- This will create a fixed-size high-power field (width reported in µm), centered on your selected region.\n"
        "- After each wsi_mark_roi_norm, a NEW ROI image is shown as the CURRENT VIEW. Carefully inspect it:\n"
        "  * If it is mostly background, out of focus, or uninformative, your very next step should be wsi_discard_last_roi.\n"
        "  * If it is useful, keep it and continue exploring or mark additional ROIs.\n"
        "- Keep only ROIs that truly help summarize the case (e.g., tumor, key inflammation, MSI-relevant areas, etc.).\n"
        "\n"
        "MSI-SPECIFIC GUIDANCE (USE ONLY IF THE PROMPT ASKS FOR MSI ASSESSMENT):\n"
        "- If the task in the prompt is MSI screening, pay particular attention to:\n"
        "  * Tumor architecture: poorly differentiated or solid/medullary areas, pushing borders, mucinous components, signet-ring cells.\n"
        "  * Cytology: marked nuclear pleomorphism, vesicular nuclei, prominent nucleoli in solid areas.\n"
        "  * Inflammation: tumor-infiltrating lymphocytes (TILs) within tumor nests, peritumoral lymphoid aggregates / Crohn-like reaction.\n"
        "- For MSI tasks, sample at least three distinct tumor regions at high power, mark representative ROIs, and give a qualitative assessment "
        "such as: 'strongly suggests MSI-H', 'compatible with MSI-H but not specific', or 'more in keeping with MSS'.\n"
        "- Always state that definitive MSI status requires immunohistochemistry (MLH1, PMS2, MSH2, MSH6) and/or molecular testing.\n"
        "\n"
        "WHEN TO STOP:\n"
        "- Continue using tools until you have:\n"
        "  * Viewed representative areas at high power (field width hundreds of µm, not thousands).\n"
        "  * Considered whether a lesion or abnormality is present, according to the prompt.\n"
        "  * Marked any useful ROIs relevant to the task.\n"
        "- Then stop calling tools and provide your final summary.\n"
        "\n"
        "FINAL REPORTING:\n"
        "- In the final response (after tools), summarize according to the user prompt. For example:\n"
        "  * Likely tissue/organ of origin.\n"
        "  * Overall histologic pattern and key structures.\n"
        "  * Any tumors or suspicious lesions.\n"
        "  * Other relevant findings (inflammation, necrosis, fibrosis, etc.).\n"
        "  * A brief description of each kept ROI and why it was chosen.\n"
        "- If no suspicious lesion is found after adequate exploration, clearly state that no obvious suspicious lesion was identified.\n"
    ),
    tools=[
        wsi_get_overview_view,
        wsi_zoom_current_norm,
        wsi_zoom_full_norm,
        wsi_pan_current,
        wsi_get_view_info,
        wsi_mark_roi_norm,
        wsi_discard_last_roi,
    ],
)

# ---------------------------------------------------------------------
# REPORT GENERATION
# ---------------------------------------------------------------------


def _copy_image_for_report(
    src_path: Optional[str],
    images_dir: str,
    run_dir: str,
    copied_map: Dict[str, str],
) -> Optional[str]:
    if not src_path or not os.path.exists(src_path):
        return None
    if src_path in copied_map:
        return copied_map[src_path]

    base = os.path.basename(src_path)
    dst = os.path.join(images_dir, base)
    i = 1
    name, ext = os.path.splitext(base)
    while os.path.exists(dst):
        dst = os.path.join(images_dir, f"{name}_{i}{ext}")
        i += 1

    shutil.copy2(src_path, dst)
    rel = os.path.relpath(dst, run_dir)
    copied_map[src_path] = rel
    print(f"[WSI][REPORT] Copied image {src_path} -> {dst}")
    return rel


def write_markdown_report(
    run_prompt: str,
    final_text: str,
    run_id: Optional[str] = None,
    reasoning_content: Optional[str] = None,
) -> str:
    if run_id is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        ts = run_id

    run_dir = os.path.join(REPORT_ROOT_DIR, ts)
    os.makedirs(run_dir, exist_ok=True)

    images_dir = os.path.join(run_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    report_path = os.path.join(run_dir, "report.md")

    copied_paths: Dict[str, str] = {}
    lines: List[str] = []

    lines.append(f"# WSI Agent Report ({ts})\n")

    lines.append("## Prompt\n")
    lines.append("```text")
    lines.append(run_prompt)
    lines.append("```")
    lines.append("")

    lines.append("## Final Report\n")
    lines.append(final_text)
    lines.append("")

    if reasoning_content:
        lines.append("## Model Reasoning\n")
        lines.append("```text")
        lines.append(reasoning_content)
        lines.append("```")
        lines.append("")

    # ROIs
    lines.append("## Regions of Interest (ROIs)\n")
    if not _roi_marks:
        lines.append("_No ROIs were kept in this run._\n")
    else:
        sorted_rois = sorted(
            _roi_marks,
            key=lambda r: (-int(r.get("importance", 1)), r["roi_id"]),
        )
        for roi in sorted_rois:
            rid = roi["roi_id"]
            label = roi["label"]
            note = roi.get("note", "")
            importance = roi.get("importance", 1)
            bbox0 = roi.get("view_bbox_level0")
            level = roi.get("view_level")
            eff_mag = roi.get("effective_magnification")
            field_w = roi.get("field_width_um")
            field_h = roi.get("field_height_um")
            tf = roi.get("tissue_fraction")
            debug_path = roi.get("debug_path")

            lines.append(f"### ROI {rid}: {label}\n")
            lines.append(f"- **Importance**: {importance}")
            if note:
                lines.append(f"- **Note**: {note}")
            if level is not None:
                lines.append(f"- **View level**: {level}")
            if bbox0 is not None:
                x0, y0, w, h = bbox0
                lines.append(f"- **BBox (level 0)**: x={x0}, y={y0}, w={w}, h={h}")
            if field_w is not None and field_h is not None:
                lines.append(
                    f"- **Approx field size**: ~{field_w:.0f} × {field_h:.0f} µm"
                )
            if tf is not None:
                lines.append(f"- **Tissue fraction**: {tf:.2f}")
            if eff_mag is not None:
                lines.append(f"- **Effective magnification (approx)**: ~{eff_mag:.1f}x")

            if debug_path:
                rel_img = _copy_image_for_report(
                    debug_path, images_dir, run_dir, copied_paths
                )
                if rel_img:
                    lines.append("")
                    lines.append(f"![ROI {rid}]({rel_img})")
            lines.append("")

    # Steps
    lines.append("## Navigation Steps\n")
    if not _step_log:
        lines.append("_No navigation steps recorded._\n")
    else:
        for step in _step_log:
            idx = step["step_index"]
            tool = step["tool"]
            nav_reason = step["nav_reason"] or "(no reason provided)"
            bbox = step.get("view_bbox_level0")
            view_level = step.get("view_level")
            dims = step.get("view_image_dims")
            debug_path = step.get("debug_path")
            field_w = step.get("field_width_um")
            field_h = step.get("field_height_um")
            tf = step.get("tissue_fraction")

            lines.append(f"### Step {idx}: `{tool}`\n")
            lines.append(f"- **Reason**: {nav_reason}")
            if view_level is not None:
                lines.append(f"- **View level**: {view_level}")
            if bbox is not None:
                x0, y0, w, h = bbox
                lines.append(f"- **BBox (level 0)**: x={x0}, y={y0}, w={w}, h={h}")
            if field_w is not None and field_h is not None:
                lines.append(
                    f"- **Approx field size**: ~{field_w:.0f} × {field_h:.0f} µm"
                )
            if tf is not None:
                lines.append(f"- **Tissue fraction**: {tf:.2f}")
            if dims is not None:
                lines.append(f"- **View image size**: {dims[0]}×{dims[1]} px")

            if debug_path:
                rel_img = _copy_image_for_report(
                    debug_path, images_dir, run_dir, copied_paths
                )
                if rel_img:
                    lines.append("")
                    lines.append(f"![Step {idx}]({rel_img})")
            lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[WSI][REPORT] Wrote Markdown report to: {report_path}")
    return report_path


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

# --- add this at the very end of wsi_core.py, after all your existing code ---

def reset_wsi_state(run_id: str) -> None:
    """
    Reset global state for a new run.
    """
    global RUN_ID, _debug_img_counter, DEBUG_SAVE_DIR
    global _step_log, _roi_marks, _view_history
    global _current_view, _overview_cache, _last_overview_with_box_path
    global _slide

    RUN_ID = run_id

    _debug_img_counter = 0
    DEBUG_SAVE_DIR = os.path.join(DEBUG_ROOT_DIR, RUN_ID)
    os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)

    _step_log = []
    _roi_marks = []
    _view_history = []
    _current_view = {}
    _overview_cache = {}
    _last_overview_with_box_path = None

    if _slide is not None:
        try:
            _slide.close()
        except Exception:
            pass
    _slide = None


def set_slide_path(path: str) -> None:
    """
    Set which slide file to use for this run.
    """
    global SLIDE_PATH
    SLIDE_PATH = os.path.abspath(path)


def get_public_state_snapshot() -> Dict[str, Any]:
    """
    Return a snapshot of the current global state that is safe to send to the frontend.
    """
    return {
        "run_id": RUN_ID,
        "current_view": dict(_current_view) if _current_view else None,
        "overview_cache": dict(_overview_cache) if _overview_cache else None,
        "step_log": list(_step_log),
        "roi_marks": list(_roi_marks),
        "last_overview_with_box_path": _last_overview_with_box_path,
    }


DEFAULT_MSI_PROMPT = (
    "Your only task is to assess this H&E whole-slide image for morphologic features "
    "suggestive of microsatellite instability (MSI-high).\n"
    "- First, identify the likely organ and tumor type.\n"
    "- Then, systematically explore multiple distinct tumor regions at high power, using the field width in micrometers and tissue_fraction "
    "to ensure you are truly at high magnification with tissue in view when assessing tumor-infiltrating lymphocytes and cytologic detail.\n"
    "- Use the WSI navigation tools as instructed and provide a nav_reason for every tool call.\n"
    "- Mark ROIs that best illustrate MSI-relevant features (or representative MSS-like morphology if MSI features are absent).\n"
    "- After each ROI, inspect the CURRENT VIEW ROI image; discard it if it is mostly background or not diagnostic.\n"
    "- At the end, give a qualitative assessment of MSI likelihood and explicitly recommend confirmatory immunohistochemistry or "
    "molecular testing.\n"
)

DEFAULT_WSI_PROMPT = (
    "Inspect the whole-slide image and describe the likely tissue of origin and any "
    "key findings (including tumors, inflammatory infiltrates, necrosis, etc.). "
    "Use the WSI tools to get an overview and then pan/zoom as needed, similar to a "
    "human pathologist using a digital slide viewer. Use the approximate field width in micrometers "
    "and tissue_fraction to ensure you reach true high-power views on tissue when you need cellular detail. "
    "Provide nav_reason for each tool call. Mark important regions of interest with wsi_mark_roi_norm so they "
    "can be highlighted in the final report. After each ROI, review the CURRENT VIEW ROI image and call "
    "wsi_discard_last_roi if the ROI is mostly background or not diagnostic. If you cannot find a suspicious lesion "
    "after exploring representative areas at adequate magnification, state that no obvious lesion was identified.\n"
)


def run_wsi_agent_for_web(
    slide_path: str,
    prompt: Optional[str],
    agent_type: str,
    run_id: str,
    max_turns: int = MAX_TURNS,
) -> Dict[str, Any]:
    """
    Entry point for the web app. Always uses the unified WSIPathologyAgent.
    The agent_type only controls which default prompt to use if prompt is empty.
    """
    set_slide_path(slide_path)
    reset_wsi_state(run_id)

    if not prompt:
        if agent_type.lower() == "msi":
            prompt = DEFAULT_MSI_PROMPT
        else:
            prompt = DEFAULT_WSI_PROMPT

    result = Runner.run_sync(WSIPathologyAgent, prompt, max_turns=max_turns)

    final_text = result.final_output
    reasoning = getattr(result, "reasoning_content", None)

    report_path = write_markdown_report(
        prompt,
        final_text,
        run_id=run_id,
        reasoning_content=reasoning,
    )

    state = get_public_state_snapshot()

    return {
        "final_output": final_text,
        "reasoning_content": reasoning,
        "report_path": report_path,
        "state": state,
    }
