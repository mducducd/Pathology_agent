import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agents import function_tool
from PIL import Image, ImageDraw

from . import state
from .config import (
    DEFAULT_MPP_UM,
    EXAMPLE_TILES_ROOT,
    MAX_BAD_TILES,
    MAX_GOOD_TILES,
    MAX_IMG_DIM,
    OUTPUTS_ROOT_DIR,
    ROI_TARGET_SIDE_PX,
    SELECTED_TILES_ROOT,
    TILE_PX,
    TILE_SIZE_UM,
)
from .embeddings.roi_ranker import (
    build_unsupervised_roi_index,
    select_topk_candidates_for_view,
)
from .slide_utils import (
    _bbox_from_norm_with_aspect_controls,
    _get_mpp_um,
    _load_slide,
    _log_step,
    _make_overview_with_current_box,
    _render_view_from_base_bbox,
    _save_debug_image,
    _safe,
    _safe_filename,
)

ROI_CANDIDATE_TOP_K = int(os.getenv("ROI_CANDIDATE_TOP_K", "12"))
ROI_CANDIDATE_MIN_SEPARATION_PX = int(os.getenv("ROI_CANDIDATE_MIN_SEPARATION_PX", "512"))
ROI_MARK_CANDIDATE_TOLERANCE_NORM = int(os.getenv("ROI_MARK_CANDIDATE_TOLERANCE_NORM", "170"))
ROI_CANDIDATE_ALLOW_FALLBACK = os.getenv("ROI_CANDIDATE_ALLOW_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "y"}
ROI_RANKER_BATCH_SIZE = int(os.getenv("ROI_RANKER_BATCH_SIZE", "32"))
ROI_RANKER_MAX_WORKERS = int(os.getenv("ROI_RANKER_MAX_WORKERS", "4"))
ROI_TILE_CACHE_DIR = os.getenv("ROI_TILE_CACHE_DIR", "").strip()


def _set_roi_candidate_prep(
    *,
    phase: str,
    status: str,
    message: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    now = datetime.utcnow().isoformat() + "Z"
    prev = state._roi_candidate_prep if isinstance(state._roi_candidate_prep, dict) else {}
    started_at = prev.get("started_at") if prev else None
    if not started_at or status in {"starting"}:
        started_at = now
    payload: Dict[str, Any] = {
        "phase": phase,
        "status": status,
        "message": message,
        "started_at": started_at,
        "updated_at": now,
        "active": status not in {"done", "failed", "idle"},
    }
    if extra:
        payload.update(extra)
    state._roi_candidate_prep = payload


def _ensure_unsupervised_roi_index():
    cached = state._roi_ranker_index
    meta = state._roi_ranker_meta or {}
    if cached is not None and meta.get("slide_path") == state.SLIDE_PATH:
        _set_roi_candidate_prep(
            phase="ready",
            status="done",
            message="ROI candidates already prepared for this run.",
            extra={"source": "cache", "slide_path": state.SLIDE_PATH},
        )
        return cached

    aml_mode = str(getattr(state, "AGENT_TYPE", "") or "").lower() == "aml"
    pipeline_desc = (
        "UNI2 tile embeddings -> kNN novelty + bad-vs-good reference scoring -> top-K per view"
        if aml_mode
        else "UNI2 tile embeddings -> kNN novelty ranking -> top-K per view"
    )
    cache_dir = Path(ROI_TILE_CACHE_DIR) if ROI_TILE_CACHE_DIR else Path(OUTPUTS_ROOT_DIR) / "_tile_cache"
    _set_roi_candidate_prep(
        phase="starting",
        status="starting",
        message="Preparing ROI candidates from slide tiles...",
        extra={"source": "uni2_knn", "slide_path": state.SLIDE_PATH},
    )
    _log_step(
        "wsi_prepare_roi_candidates",
        "Start ROI candidate preparation: UNI2 embedding extraction + kNN build.",
        {
            "roi_candidate_stage": "index_running",
            "roi_candidate_pipeline": pipeline_desc,
            "roi_candidate_count": 0,
            "roi_candidate_warning": "Preparation in progress...",
        },
    )

    try:
        slide = _load_slide()
        default_mpp = _get_mpp_um(slide) or DEFAULT_MPP_UM

        def _on_progress(evt: Dict[str, Any]) -> None:
            phase = str(evt.get("phase") or "working")
            status = str(evt.get("status") or "running")
            if phase == "load_extractor":
                msg = "Loading UNI2 foundation model..."
            elif phase == "extract_embeddings":
                pt = evt.get("processed_tiles")
                pb = evt.get("processed_batches")
                msg = "Extracting UNI2 tile embeddings..."
                if pt is not None:
                    msg += f" tiles={pt}"
                if pb is not None:
                    msg += f", batches={pb}"
            elif phase == "build_knn":
                nt = evt.get("num_tiles")
                msg = "Building kNN index over tile embeddings..."
                if nt is not None:
                    msg += f" N={nt}"
            elif phase == "embed_reference_tiles":
                rt = evt.get("reference_tiles_total")
                rg = evt.get("reference_tiles_good")
                rb = evt.get("reference_tiles_bad")
                msg = "Embedding AML reference tiles (good/bad)..."
                if rt is not None:
                    msg += f" total={rt}"
                if rg is not None and rb is not None:
                    msg += f", good={rg}, bad={rb}"
            elif phase == "rank_candidates":
                msg = "Ranking top-K ROI candidates for current views..."
            else:
                msg = "Preparing ROI candidates..."
            _set_roi_candidate_prep(
                phase=phase,
                status=status,
                message=msg,
                extra=evt,
            )

        index = build_unsupervised_roi_index(
            slide_path=state.SLIDE_PATH,
            tile_size_um=TILE_SIZE_UM,
            tile_size_px=TILE_PX,
            batch_size=ROI_RANKER_BATCH_SIZE,
            cache_dir=cache_dir,
            max_workers=ROI_RANKER_MAX_WORKERS,
            brightness_cutoff=240,
            canny_cutoff=0.02,
            default_slide_mpp=float(default_mpp),
            k_neighbors=20,
            use_reference_labels=aml_mode,
            reference_tiles_root=EXAMPLE_TILES_ROOT if aml_mode else None,
            progress_cb=_on_progress,
        )
        state._roi_ranker_index = index
        state._roi_ranker_meta = {
            "slide_path": state.SLIDE_PATH,
            "num_tiles": index.num_tiles,
            "feature_dim": index.feature_dim,
            "extractor_id": index.extractor_id,
            "tile_size_px": index.tile_size_px,
            "tile_size_um": index.tile_size_um,
            "agent_type": getattr(state, "AGENT_TYPE", None),
            "reference_mode": getattr(index, "reference_mode", "none"),
            "reference_stats": dict(getattr(index, "reference_stats", {}) or {}),
        }
        _set_roi_candidate_prep(
            phase="ready",
            status="done",
            message=(
                "ROI candidates ready."
                f" extractor={index.extractor_id}, tiles={index.num_tiles}, dim={index.feature_dim}"
            ),
            extra={
                "source": "uni2_knn",
                "extractor_id": index.extractor_id,
                "num_tiles": index.num_tiles,
                "feature_dim": index.feature_dim,
                "reference_mode": getattr(index, "reference_mode", "none"),
                "reference_stats": dict(getattr(index, "reference_stats", {}) or {}),
            },
        )
        _log_step(
            "wsi_prepare_roi_candidates",
            "Extract UNI2 tile embeddings, build kNN index, then rank top-K ROI candidates per view.",
            {
                "roi_candidate_stage": "index_built",
                "roi_candidate_pipeline": pipeline_desc,
                "roi_candidate_index_meta": dict(state._roi_ranker_meta),
                "roi_candidate_count": 0,
            },
        )
        return index
    except Exception as exc:
        prev_meta = state._roi_ranker_meta if isinstance(state._roi_ranker_meta, dict) else {}
        err_text = f"{type(exc).__name__}: {exc}"
        _set_roi_candidate_prep(
            phase="failed",
            status="failed",
            message=f"ROI candidate preparation failed: {err_text}",
            extra={"source": "uni2_knn", "error": err_text},
        )
        state._roi_ranker_index = None
        state._roi_ranker_meta = {
            "slide_path": state.SLIDE_PATH,
            "error": err_text,
        }
        # Record failure once per distinct error so it appears in navigation steps.
        if prev_meta.get("slide_path") != state.SLIDE_PATH or prev_meta.get("error") != err_text:
            _log_step(
                "wsi_prepare_roi_candidates",
                "Extract UNI2 tile embeddings, build kNN index, then rank top-K ROI candidates per view.",
                {
                    "roi_candidate_stage": "index_failed",
                    "roi_candidate_pipeline": pipeline_desc,
                    "roi_candidate_index_meta": dict(state._roi_ranker_meta),
                    "roi_candidate_warning": f"Candidate index build failed: {err_text}",
                    "roi_candidate_count": 0,
                },
            )
        print(f"[WSI][ROI_CAND] Failed to build ROI index: {type(exc).__name__}: {exc}")
        return None


def _fallback_candidates_from_current_view(top_k: int) -> List[Dict[str, Any]]:
    if not state._current_view:
        return []
    debug_path = state._current_view.get("debug_path")
    if not debug_path or not os.path.exists(debug_path):
        return []

    with Image.open(debug_path) as im:
        img = im.convert("RGB")
        w, h = img.size
        cols = 7
        rows = 7
        patch_w = max(16, w // cols)
        patch_h = max(16, h // rows)

        raw: List[Dict[str, Any]] = []
        cv_x0 = int(state._current_view["x0"])
        cv_y0 = int(state._current_view["y0"])
        cv_w = int(state._current_view["w"])
        cv_h = int(state._current_view["h"])

        for gy in range(rows):
            for gx in range(cols):
                x0 = gx * patch_w
                y0 = gy * patch_h
                x1 = min(w, x0 + patch_w)
                y1 = min(h, y0 + patch_h)
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                patch = img.crop((x0, y0, x1, y1))
                gray = np.asarray(patch.convert("L"), dtype=np.float32) / 255.0
                if gray.size == 0:
                    continue
                tissue = float(np.mean(gray < 0.92))
                gx_edge = float(np.abs(np.diff(gray, axis=1)).mean()) if gray.shape[1] > 1 else 0.0
                gy_edge = float(np.abs(np.diff(gray, axis=0)).mean()) if gray.shape[0] > 1 else 0.0
                edge = (gx_edge + gy_edge) / 2.0
                score = (0.70 * tissue) + (0.30 * edge)
                if tissue < 0.08:
                    continue

                cx = int(round(((x0 + x1) / 2.0) / max(1, w - 1) * 999.0))
                cy = int(round(((y0 + y1) / 2.0) / max(1, h - 1) * 999.0))
                bx0 = int(round(x0 / max(1, w - 1) * 999.0))
                by0 = int(round(y0 / max(1, h - 1) * 999.0))
                bx1 = int(round(x1 / max(1, w - 1) * 999.0))
                by1 = int(round(y1 / max(1, h - 1) * 999.0))

                cx_level0 = cv_x0 + int(round((cx / 999.0) * cv_w))
                cy_level0 = cv_y0 + int(round((cy / 999.0) * cv_h))

                raw.append(
                    {
                        "score": float(score),
                        "center_norm": [max(0, min(999, cx)), max(0, min(999, cy))],
                        "bbox_norm": [
                            max(0, min(999, bx0)),
                            max(0, min(999, by0)),
                            max(0, min(999, bx1)),
                            max(0, min(999, by1)),
                        ],
                        "center_level0": [cx_level0, cy_level0],
                        "tile_bbox_level0": [cv_x0, cv_y0, cv_x0 + cv_w, cv_y0 + cv_h],
                    }
                )

    if not raw:
        return []

    raw.sort(key=lambda r: float(r["score"]), reverse=True)
    selected: List[Dict[str, Any]] = []
    min_dist_sq = float(120 * 120)
    for item in raw:
        cxi, cyi = item["center_norm"]
        keep = True
        for prev in selected:
            px, py = prev["center_norm"]
            dx = float(cxi - px)
            dy = float(cyi - py)
            if dx * dx + dy * dy < min_dist_sq:
                keep = False
                break
        if keep:
            selected.append(item)
        if len(selected) >= top_k:
            break

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(selected, start=1):
        rec = dict(item)
        rec["rank"] = i
        rec["tile_index"] = -1
        out.append(rec)
    return out


def _build_roi_candidate_overlay(candidates: List[Dict[str, Any]]) -> Optional[str]:
    if not candidates or not state._current_view:
        return None
    debug_path = state._current_view.get("debug_path")
    if not debug_path or not os.path.exists(debug_path):
        return None

    with Image.open(debug_path) as im:
        img = im.convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    sx = (w - 1) / 999.0 if w > 1 else 1.0
    sy = (h - 1) / 999.0 if h > 1 else 1.0
    colors = ["#00c853", "#ff6d00", "#00b0ff", "#ff1744", "#651fff", "#ffd600"]

    for idx, cand in enumerate(candidates):
        bx0, by0, bx1, by1 = cand.get("bbox_norm", [0, 0, 0, 0])
        cx, cy = cand.get("center_norm", [0, 0])
        x0 = int(round(float(bx0) * sx))
        y0 = int(round(float(by0) * sy))
        x1 = int(round(float(bx1) * sx))
        y1 = int(round(float(by1) * sy))
        px = int(round(float(cx) * sx))
        py = int(round(float(cy) * sy))
        color = colors[idx % len(colors)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color)
        draw.text((x0 + 4, y0 + 2), f"#{cand.get('rank', idx + 1)}", fill=color)

    return _save_debug_image(img, tag="roi_candidates")


def _refresh_roi_candidates_for_current_view(top_k: int = ROI_CANDIDATE_TOP_K) -> List[Dict[str, Any]]:
    if not state._current_view:
        state._last_roi_candidates = []
        state._last_roi_candidate_source = None
        state._last_roi_candidate_overlay_path = None
        return []

    top_k = max(1, int(top_k))
    candidates: List[Dict[str, Any]] = []
    source: Optional[str] = None

    index = _ensure_unsupervised_roi_index()
    if index is not None and getattr(index, "num_tiles", 0) > 0:
        view_bbox = (
            int(state._current_view["x0"]),
            int(state._current_view["y0"]),
            int(state._current_view["w"]),
            int(state._current_view["h"]),
        )
        candidates = select_topk_candidates_for_view(
            index=index,
            view_bbox_level0=view_bbox,
            top_k=top_k,
            min_center_separation_px=max(64, ROI_CANDIDATE_MIN_SEPARATION_PX),
        )
        source = "uni2_knn"

    if not candidates and ROI_CANDIDATE_ALLOW_FALLBACK:
        candidates = _fallback_candidates_from_current_view(top_k)
        if candidates:
            source = "fallback_heuristic"

    state._last_roi_candidates = candidates
    state._last_roi_candidate_source = source
    state._last_roi_candidate_overlay_path = _build_roi_candidate_overlay(candidates)
    return candidates


def _attach_roi_candidates(info: Dict[str, Any], top_k: int = ROI_CANDIDATE_TOP_K) -> Dict[str, Any]:
    candidates = _refresh_roi_candidates_for_current_view(top_k=top_k)
    aml_mode = str(getattr(state, "AGENT_TYPE", "") or "").lower() == "aml"
    info["roi_candidates"] = candidates
    info["roi_candidate_count"] = len(candidates)
    info["roi_candidate_source"] = state._last_roi_candidate_source
    info["roi_candidate_prep"] = dict(state._roi_candidate_prep) if state._roi_candidate_prep else None
    info["roi_candidate_overlay_path"] = state._last_roi_candidate_overlay_path
    if aml_mode:
        info["roi_candidate_pipeline"] = (
            "UNI2 tile embeddings -> kNN novelty + bad-vs-good reference scoring -> top-K per current view"
        )
    else:
        info["roi_candidate_pipeline"] = "UNI2 tile embeddings -> kNN novelty ranking -> top-K per current view"
    if state._roi_ranker_meta:
        info["roi_candidate_index_meta"] = dict(state._roi_ranker_meta)
        ref_stats = state._roi_ranker_meta.get("reference_stats")
        if aml_mode and isinstance(ref_stats, dict):
            info["aml_reference_stats"] = dict(ref_stats)
    if state._last_roi_candidate_source != "uni2_knn":
        info["roi_candidate_warning"] = (
            "UNI2/kNN candidate source unavailable for this view."
            + (" Using fallback heuristic." if ROI_CANDIDATE_ALLOW_FALLBACK else " Fallback disabled.")
        )
    if candidates:
        if aml_mode:
            info["roi_candidate_guidance"] = (
                "For AML, prioritize candidates with quality_hint='bad_like' and higher bad_likelihood. "
                "Use one of the top-K candidate centers/bboxes for wsi_mark_roi_norm; "
                "arbitrary ROI coordinates are rejected."
            )
        else:
            info["roi_candidate_guidance"] = (
                "Use one of the top-K candidate centers/bboxes for wsi_mark_roi_norm. "
                "Arbitrary ROI coordinates are rejected."
            )
    return info


def _closest_candidate(
    cx_999: float,
    cy_999: float,
) -> tuple[Optional[Dict[str, Any]], float]:
    if not state._last_roi_candidates:
        return None, float("inf")
    best: Optional[Dict[str, Any]] = None
    best_dist = float("inf")
    for cand in state._last_roi_candidates:
        cc = cand.get("center_norm") or [0, 0]
        dx = float(cx_999) - float(cc[0])
        dy = float(cy_999) - float(cc[1])
        dist = float((dx * dx + dy * dy) ** 0.5)
        if dist < best_dist:
            best = cand
            best_dist = dist
    return best, best_dist


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
        info = _attach_roi_candidates(info)
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

        info = _attach_roi_candidates(info)
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

        info = _attach_roi_candidates(info)
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
        info = _attach_roi_candidates(info)
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
        info = _attach_roi_candidates(info)
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

        # Always refresh candidate ranking on the active view before marking ROI.
        candidates = _refresh_roi_candidates_for_current_view(top_k=ROI_CANDIDATE_TOP_K)
        if not candidates:
            return {
                "ok": False,
                "reason": "no_roi_candidates",
                "message": "No ROI candidates available in current view. Navigate to tissue and try again.",
            }

        x0_999_cl = max(0, min(999, x0_999))
        x1_999_cl = max(0, min(999, x1_999))
        y0_999_cl = max(0, min(999, y0_999))
        y1_999_cl = max(0, min(999, y1_999))

        requested_cx_999 = (x0_999_cl + x1_999_cl) / 2.0
        requested_cy_999 = (y0_999_cl + y1_999_cl) / 2.0
        chosen, dist = _closest_candidate(requested_cx_999, requested_cy_999)
        if chosen is None or dist > float(ROI_MARK_CANDIDATE_TOLERANCE_NORM):
            return {
                "ok": False,
                "reason": "roi_outside_topk_candidates",
                "message": (
                    "ROI center is outside allowed candidate set. "
                    "Choose one of roi_candidates[*].center_norm from the latest navigation output."
                ),
                "distance_to_nearest_candidate": float(dist),
                "tolerance_norm": ROI_MARK_CANDIDATE_TOLERANCE_NORM,
                "roi_candidates": candidates,
            }

        cx_999 = float(chosen["center_norm"][0])
        cy_999 = float(chosen["center_norm"][1])

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
            "[WSI][ROI_NORM] requested_center=(%.1f,%.1f), snapped_center=(%.1f,%.1f), "
            "candidate_rank=%s, base_center=(%d,%d), ROI_bbox=(%d,%d,%d,%d)"
            % (
                requested_cx_999,
                requested_cy_999,
                cx_999,
                cy_999,
                str(chosen.get("rank")),
                cx_base,
                cy_base,
                x0_new,
                y0_new,
                w_new,
                h_new,
            )
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
            "candidate_rank": chosen.get("rank"),
            "candidate_score": chosen.get("score"),
            "candidate_center_norm": chosen.get("center_norm"),
            "requested_center_norm": [
                int(round(requested_cx_999)),
                int(round(requested_cy_999)),
            ],
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
