import os
from datetime import datetime
from typing import Any, Dict

from agents import function_tool
from PIL import Image

from . import state
from .config import (
    DEFAULT_MPP_UM,
    MAX_BAD_TILES,
    MAX_GOOD_TILES,
    MAX_IMG_DIM,
    ROI_TARGET_SIDE_PX,
    SELECTED_TILES_ROOT,
    TILE_PX,
    TILE_SIZE_UM,
)
from .slide_utils import (
    _bbox_from_norm_with_aspect_controls,
    _get_mpp_um,
    _load_slide,
    _log_step,
    _make_overview_with_current_box,
    _render_view_from_base_bbox,
    _safe,
    _safe_filename,
)


@function_tool
def wsi_get_overview_view(
    nav_reason: str = "Initial overview of the whole slide",
    max_dim: int = MAX_IMG_DIM,
) -> str:
    def _inner(nav_reason: str, max_dim: int) -> Dict[str, Any]:
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
        state._last_overview_debug_path = info.get("debug_path")

        level = state._current_view["level"]
        ds = state._current_view["level_downsample"]
        lvl_w, lvl_h = slide.level_dimensions[level]
        state._overview_cache = {
            "level": level,
            "level_w": lvl_w,
            "level_h": lvl_h,
            "level_downsample": ds,
            "shown_w": state._current_view["shown_w"],
            "shown_h": state._current_view["shown_h"],
            "base_w0": base_w0,
            "base_h0": base_h0,
        }

        _make_overview_with_current_box(draw_current_box=False)
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
    def _inner(
        x0_999: int,
        y0_999: int,
        x1_999: int,
        y1_999: int,
        nav_reason: str,
        max_dim: int,
    ) -> Dict[str, Any]:
        if not state._current_view:
            raise RuntimeError("wsi_zoom_current_norm called before wsi_get_overview_view.")

        slide = _load_slide()
        cv_x0 = state._current_view["x0"]
        cv_y0 = state._current_view["y0"]
        cv_w = state._current_view["w"]
        cv_h = state._current_view["h"]
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
    def _inner(
        dx_999: int,
        dy_999: int,
        nav_reason: str,
        max_dim: int,
    ) -> Dict[str, Any]:
        if not state._current_view:
            raise RuntimeError("wsi_pan_current called before wsi_get_overview_view.")

        slide = _load_slide()
        dx_999 = max(-999, min(999, dx_999))
        dy_999 = max(-999, min(999, dy_999))

        cv_x0 = state._current_view["x0"]
        cv_y0 = state._current_view["y0"]
        cv_w = state._current_view["w"]
        cv_h = state._current_view["h"]

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
    def _inner(nav_reason: str) -> Dict[str, Any]:
        slide = _load_slide()
        if not state._current_view:
            raise RuntimeError("No current view. Call wsi_get_overview_view first.")

        level = state._current_view["level"]
        ds = float(state._current_view["level_downsample"])
        objective = float(slide.properties.get("openslide.objective-power", 40.0))
        eff_mag = objective / ds if ds > 0 else None

        info = {
            "level": level,
            "bbox_level0": [
                state._current_view["x0"],
                state._current_view["y0"],
                state._current_view["w"],
                state._current_view["h"],
            ],
            "downsample": ds,
            "objective_power": objective,
            "effective_magnification": eff_mag,
            "field_width_um": state._current_view.get("field_width_um"),
            "field_height_um": state._current_view.get("field_height_um"),
            "tissue_fraction": state._current_view.get("tissue_fraction"),
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
        if not state._current_view:
            raise RuntimeError("wsi_mark_roi_norm called before wsi_get_overview_view.")

        slide = _load_slide()
        cv_x0 = state._current_view["x0"]
        cv_y0 = state._current_view["y0"]
        cv_w = state._current_view["w"]
        cv_h = state._current_view["h"]
        slide_w0, slide_h0 = slide.level_dimensions[0]

        x0_999_cl = max(0, min(999, x0_999))
        x1_999_cl = max(0, min(999, x1_999))
        y0_999_cl = max(0, min(999, y0_999))
        y1_999_cl = max(0, min(999, y1_999))

        cx_999 = (x0_999_cl + x1_999_cl) / 2.0
        cy_999 = (y0_999_cl + y1_999_cl) / 2.0

        cx_rel = cx_999 / 999.0
        cy_rel = cy_999 / 999.0

        cx_base = cv_x0 + int(round(cx_rel * cv_w))
        cy_base = cv_y0 + int(round(cy_rel * cv_h))

        side = min(ROI_TARGET_SIDE_PX, slide_w0, slide_h0)
        w_new = side
        h_new = side

        x0_new = cx_base - w_new // 2
        y0_new = cy_base - h_new // 2

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
            force_level=0,
        )

        tf = info.get("tissue_fraction")
        if tf is not None and tf < 0.15:
            info["tissue_warning"] = (
                "This high-power ROI is mostly background/empty glass (low tissue_fraction). "
                "You should immediately discard it using wsi_discard_last_roi and select an "
                "ROI centered on diagnostic tissue."
            )

        level = info["view_level"]
        ds = float(slide.level_downsamples[level])
        objective = float(slide.properties.get("openslide.objective-power", 40.0))
        eff_mag = objective / ds if ds > 0 else None

        roi_id = len(state._roi_marks) + 1
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

        roi["next_action_hint"] = (
            "You just requested a high-power ROI. "
            "Carefully inspect the newly shown ROI image in the conversation. "
            "If it is mostly background, out of focus, or not diagnostic, "
            "your very next step should be to call wsi_discard_last_roi with a brief nav_reason. "
            "If it is diagnostic, you may either mark additional ROIs or continue navigation."
        )

        state._roi_marks.append(roi)

        _make_overview_with_current_box(draw_current_box=True)

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
def wsi_save_tile_norm(
    x0_999: int,
    y0_999: int,
    x1_999: int,
    y1_999: int,
    label: str,
    quality: str = "good",
    nav_reason: str = "Save a diagnostic tile",
) -> str:
    def _inner(
        x0_999: int,
        y0_999: int,
        x1_999: int,
        y1_999: int,
        label: str,
        quality: str,
        nav_reason: str,
    ) -> Dict[str, Any]:
        if not state._current_view:
            raise RuntimeError("wsi_save_tile_norm called before wsi_get_overview_view.")

        quality = (quality or "good").strip().lower()
        if quality not in {"good", "bad"}:
            raise ValueError("quality must be 'good' or 'bad'")

        if quality == "good" and len(state._saved_good_tiles) >= MAX_GOOD_TILES:
            return {"ok": False, "reason": "max_good_tiles_reached"}
        if quality == "bad" and len(state._saved_bad_tiles) >= MAX_BAD_TILES:
            return {"ok": False, "reason": "max_bad_tiles_reached"}

        slide = _load_slide()
        mpp = _get_mpp_um(slide) or DEFAULT_MPP_UM
        tile_px = int(round(TILE_SIZE_UM / mpp))
        tile_px = max(32, tile_px)

        x0_999_cl = max(0, min(999, x0_999))
        x1_999_cl = max(0, min(999, x1_999))
        y0_999_cl = max(0, min(999, y0_999))
        y1_999_cl = max(0, min(999, y1_999))

        cx_999 = (x0_999_cl + x1_999_cl) / 2.0
        cy_999 = (y0_999_cl + y1_999_cl) / 2.0

        cv_x0 = state._current_view["x0"]
        cv_y0 = state._current_view["y0"]
        cv_w = state._current_view["w"]
        cv_h = state._current_view["h"]
        slide_w0, slide_h0 = slide.level_dimensions[0]
        tile_px = min(tile_px, slide_w0, slide_h0)

        cx_base = cv_x0 + int(round((cx_999 / 999.0) * cv_w))
        cy_base = cv_y0 + int(round((cy_999 / 999.0) * cv_h))

        x0 = cx_base - tile_px // 2
        y0 = cy_base - tile_px // 2
        x0 = max(0, min(x0, slide_w0 - tile_px))
        y0 = max(0, min(y0, slide_h0 - tile_px))

        region = slide.read_region((x0, y0), 0, (tile_px, tile_px)).convert("RGB")
        region = region.resize((TILE_PX, TILE_PX), Image.BILINEAR)

        run_id = state.RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(SELECTED_TILES_ROOT, run_id, "Selected_Tiles", quality)
        os.makedirs(out_dir, exist_ok=True)

        idx = (len(state._saved_good_tiles) + 1) if quality == "good" else (len(state._saved_bad_tiles) + 1)
        label_safe = _safe_filename(label)
        x_um = x0 * mpp
        y_um = y0 * mpp
        out_path = os.path.join(
            out_dir,
            f"{quality}_{idx:04d}_tile_({x_um}, {y_um}).jpg",
        )
        region.save(out_path, format="JPEG", quality=95)

        record = {
            "quality": quality,
            "label": label,
            "path": out_path,
            "bbox_level0": [x0, y0, tile_px, tile_px],
            "tile_px": TILE_PX,
            "tile_um": TILE_SIZE_UM,
            "mpp_used": mpp,
        }
        _ = label_safe

        if quality == "good":
            state._saved_good_tiles.append(record)
        else:
            state._saved_bad_tiles.append(record)

        _log_step(
            "wsi_save_tile_norm",
            nav_reason,
            {
                "view_bbox_level0": record["bbox_level0"],
                "field_width_um": TILE_SIZE_UM,
                "field_height_um": TILE_SIZE_UM,
            },
        )

        return {
            "ok": True,
            "quality": quality,
            "path": out_path,
            "count_good": len(state._saved_good_tiles),
            "count_bad": len(state._saved_bad_tiles),
            "tile_px": TILE_PX,
            "tile_um": TILE_SIZE_UM,
            "mpp_used": mpp,
        }

    return _safe(
        _inner,
        x0_999=x0_999,
        y0_999=y0_999,
        x1_999=x1_999,
        y1_999=y1_999,
        label=label,
        quality=quality,
        nav_reason=nav_reason,
    )


@function_tool
def wsi_discard_last_roi(
    nav_reason: str = "Discard last ROI if not useful",
) -> str:
    def _inner(nav_reason: str) -> Dict[str, Any]:
        if not state._roi_marks:
            return {"ok": False, "message": "No ROI to discard."}
        roi = state._roi_marks.pop()
        print(f"[WSI][ROI] Discarded ROI {roi['roi_id']}: {roi['label']}")
        _log_step(
            "wsi_discard_last_roi",
            nav_reason,
            {"view_bbox_level0": roi.get("view_bbox_level0")},
        )
        return {"ok": True, "discarded_roi_id": roi["roi_id"], "label": roi["label"]}

    return _safe(_inner, nav_reason=nav_reason)
