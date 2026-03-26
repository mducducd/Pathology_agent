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

ROI_CANDIDATE_TOP_K = int(os.getenv("ROI_CANDIDATE_TOP_K", "24"))
ROI_CANDIDATE_MIN_SEPARATION_PX = int(os.getenv("ROI_CANDIDATE_MIN_SEPARATION_PX", "192"))
ROI_MARK_CANDIDATE_TOLERANCE_NORM = int(os.getenv("ROI_MARK_CANDIDATE_TOLERANCE_NORM", "170"))
ROI_CANDIDATE_ALLOW_FALLBACK = os.getenv("ROI_CANDIDATE_ALLOW_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "y"}
# Hard cap on how many candidates the VLM sees in AML mode after raw retrieval ranking.
ROI_CANDIDATE_TOP_K_AML = int(os.getenv("ROI_CANDIDATE_TOP_K_AML", "15"))
ROI_RANKER_BATCH_SIZE = int(os.getenv("ROI_RANKER_BATCH_SIZE", "32"))
ROI_RANKER_MAX_WORKERS = int(os.getenv("ROI_RANKER_MAX_WORKERS", "4"))
ROI_COARSE_PREFILTER_TRIGGER_SUPERTILES = int(os.getenv("ROI_COARSE_PREFILTER_TRIGGER_SUPERTILES", "128"))
ROI_COARSE_PREFILTER_KEEP_RATIO = float(os.getenv("ROI_COARSE_PREFILTER_KEEP_RATIO", "0.40"))
ROI_COARSE_PREFILTER_MIN_KEEP_SUPERTILES = int(os.getenv("ROI_COARSE_PREFILTER_MIN_KEEP_SUPERTILES", "48"))
ROI_COARSE_PREFILTER_MAX_KEEP_SUPERTILES = int(os.getenv("ROI_COARSE_PREFILTER_MAX_KEEP_SUPERTILES", "96"))
ROI_QUALITY_PREFILTER_KEEP_RATIO = float(os.getenv("ROI_QUALITY_PREFILTER_KEEP_RATIO", "0.35"))
ROI_QUALITY_PREFILTER_MIN_KEEP_TILES = int(os.getenv("ROI_QUALITY_PREFILTER_MIN_KEEP_TILES", "4"))
ROI_QUALITY_PREFILTER_TRIGGER_TILES = int(os.getenv("ROI_QUALITY_PREFILTER_TRIGGER_TILES", "12"))
ROI_QUALITY_PREFILTER_RANDOM_RESERVE_RATIO = float(os.getenv("ROI_QUALITY_PREFILTER_RANDOM_RESERVE_RATIO", "0.08"))
ROI_TILE_CACHE_DIR = os.getenv("ROI_TILE_CACHE_DIR", "").strip()


def _selected_batch_size() -> int:
    try:
        return max(1, int(getattr(state, "BATCH_SIZE", ROI_RANKER_BATCH_SIZE) or ROI_RANKER_BATCH_SIZE))
    except Exception:
        return int(ROI_RANKER_BATCH_SIZE)


def _selected_extractor_label() -> str:
    return getattr(state, "EXTRACTOR_NAME", "uni2").replace("_onnx", " (ONNX)").title()


def _selected_tile_prefilter_method() -> str:
    raw = str(getattr(state, "TILE_PREFILTER_METHOD", "quality") or "quality").strip().lower()
    return raw if raw in {"none", "coarse", "quality", "hybrid"} else "quality"


def _use_coarse_prefilter(method: str | None = None) -> bool:
    return (method or _selected_tile_prefilter_method()) in {"coarse", "hybrid"}


def _use_quality_prefilter(method: str | None = None) -> bool:
    return (method or _selected_tile_prefilter_method()) in {"quality", "hybrid"}


def _tile_prefilter_label(method: str | None = None) -> str:
    value = method or _selected_tile_prefilter_method()
    return {
        "none": "No extra prefilter",
        "coarse": "Thumbnail coarse-to-fine",
        "quality": "Raw-tile quality score",
        "hybrid": "Hybrid coarse + quality",
    }.get(value, "Raw-tile quality score")


def _selected_candidate_source(aml_mode: bool) -> str:
    extractor_name = str(getattr(state, "EXTRACTOR_NAME", "uni2") or "uni2").strip().lower()
    return f"{extractor_name}_exact_retrieval" if aml_mode else f"{extractor_name}_knn"


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
        cached_source = _selected_candidate_source(
            str(getattr(state, "AGENT_TYPE", "") or "").lower() == "aml"
        )
        _set_roi_candidate_prep(
            phase="ready",
            status="done",
            message="ROI candidates already prepared for this run.",
            extra={"source": "cache", "candidate_source": cached_source, "slide_path": state.SLIDE_PATH},
        )
        return cached

    aml_mode = str(getattr(state, "AGENT_TYPE", "") or "").lower() == "aml"
    candidate_source = _selected_candidate_source(aml_mode)
    extractor_label = _selected_extractor_label()
    tile_prefilter_method = _selected_tile_prefilter_method()
    use_coarse_prefilter = _use_coarse_prefilter(tile_prefilter_method)
    use_quality_prefilter = _use_quality_prefilter(tile_prefilter_method)
    if aml_mode:
        if tile_prefilter_method == "hybrid":
            pipeline_desc = (
                f"Thumbnail coarse region filter -> raw-tile quality score filter -> {extractor_label} tile embeddings -> exact good/bad exemplar retrieval -> rank by raw nearest bad similarity -> top-K per view"
            )
        elif tile_prefilter_method == "quality":
            pipeline_desc = (
                f"Raw-tile quality score filter -> {extractor_label} tile embeddings -> exact good/bad exemplar retrieval -> rank by raw nearest bad similarity -> top-K per view"
            )
        elif tile_prefilter_method == "coarse":
            pipeline_desc = (
                f"Thumbnail coarse region filter -> {extractor_label} tile embeddings -> exact good/bad exemplar retrieval -> rank by raw nearest bad similarity -> top-K per view"
            )
        else:
            pipeline_desc = (
                f"{extractor_label} tile embeddings -> exact good/bad exemplar retrieval -> rank by raw nearest bad similarity -> top-K per view"
            )
    else:
        if tile_prefilter_method == "hybrid":
            pipeline_desc = f"Thumbnail coarse region filter -> raw-tile quality score filter -> {extractor_label} tile embeddings -> kNN novelty ranking -> top-K per view"
        elif tile_prefilter_method == "quality":
            pipeline_desc = f"Raw-tile quality score filter -> {extractor_label} tile embeddings -> kNN novelty ranking -> top-K per view"
        elif tile_prefilter_method == "coarse":
            pipeline_desc = f"Thumbnail coarse region filter -> {extractor_label} tile embeddings -> kNN novelty ranking -> top-K per view"
        else:
            pipeline_desc = f"{extractor_label} tile embeddings -> kNN novelty ranking -> top-K per view"
    cache_dir = Path(ROI_TILE_CACHE_DIR) if ROI_TILE_CACHE_DIR else Path(OUTPUTS_ROOT_DIR) / "_tile_cache"
    _set_roi_candidate_prep(
        phase="starting",
        status="starting",
        message="Preparing ROI candidates from slide tiles...",
        extra={
            "source": candidate_source,
            "slide_path": state.SLIDE_PATH,
            "tile_prefilter_method": tile_prefilter_method,
        },
    )
    _log_step(
        "wsi_prepare_roi_candidates",
        f"Start ROI candidate preparation: {extractor_label} embedding extraction + kNN build.",
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
            if phase == "coarse_prefilter":
                total = evt.get("coarse_total_supertile_count")
                kept = evt.get("coarse_selected_supertile_count")
                used = bool(evt.get("coarse_prefilter_used"))
                if not use_coarse_prefilter:
                    if total is not None:
                        msg = f"Scanning all {total} foreground slide regions..."
                    else:
                        msg = "Scanning foreground slide regions..."
                elif used and total is not None and kept is not None:
                    msg = f"Coarse pass kept {kept}/{total} slide regions for fine embedding..."
                elif total is not None:
                    msg = f"Coarse pass kept all {total} slide regions..."
                else:
                    msg = "Running thumbnail coarse pass..."
            elif phase == "quality_prefilter":
                total = evt.get("quality_total_tiles")
                kept = evt.get("quality_kept_tiles")
                hard_rejected = evt.get("quality_hard_rejected_tiles")
                if total is not None and kept is not None:
                    msg = f"Quality prefilter kept {kept}/{total} tiles for embedding..."
                    if hard_rejected is not None:
                        msg += f" hard_reject={hard_rejected}"
                else:
                    msg = "Scoring raw tiles for focus, stain, and texture..."
            elif phase == "load_extractor":
                msg = f"Loading {_selected_extractor_label()} foundation model..."
            elif phase == "extract_embeddings":
                pt = evt.get("processed_tiles")
                pb = evt.get("processed_batches")
                msg = f"Extracting {_selected_extractor_label()} tile embeddings..."
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
                msg = "Embedding AML reference tiles and running exact exemplar retrieval..."
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
            extractor_name=state.EXTRACTOR_NAME,
            tile_size_um=state.TILE_SIZE_UM,
            tile_size_px=state.TILE_SIZE_PX,
            batch_size=_selected_batch_size(),
            cache_dir=cache_dir,
            max_workers=ROI_RANKER_MAX_WORKERS,
            brightness_cutoff=240,
            canny_cutoff=0.02,
            default_slide_mpp=float(default_mpp),
            tile_prefilter_method=tile_prefilter_method,
            coarse_trigger_supertile_count=ROI_COARSE_PREFILTER_TRIGGER_SUPERTILES if use_coarse_prefilter else None,
            coarse_keep_ratio=ROI_COARSE_PREFILTER_KEEP_RATIO if use_coarse_prefilter else None,
            coarse_min_keep_supertile_count=ROI_COARSE_PREFILTER_MIN_KEEP_SUPERTILES if use_coarse_prefilter else 0,
            coarse_max_keep_supertile_count=ROI_COARSE_PREFILTER_MAX_KEEP_SUPERTILES if use_coarse_prefilter else None,
            quality_keep_ratio=ROI_QUALITY_PREFILTER_KEEP_RATIO if use_quality_prefilter else None,
            quality_min_keep_tile_count=ROI_QUALITY_PREFILTER_MIN_KEEP_TILES if use_quality_prefilter else 0,
            quality_trigger_tile_count=ROI_QUALITY_PREFILTER_TRIGGER_TILES if use_quality_prefilter else None,
            quality_random_reserve_ratio=ROI_QUALITY_PREFILTER_RANDOM_RESERVE_RATIO if use_quality_prefilter else None,
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
            "tile_prefilter_method": tile_prefilter_method,
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
                "source": candidate_source,
                "extractor_id": index.extractor_id,
                "num_tiles": index.num_tiles,
                "feature_dim": index.feature_dim,
                "tile_prefilter_method": tile_prefilter_method,
                "reference_mode": getattr(index, "reference_mode", "none"),
                "reference_stats": dict(getattr(index, "reference_stats", {}) or {}),
            },
        )
        _log_step(
            "wsi_prepare_roi_candidates",
            f"Extract {_selected_extractor_label()} tile embeddings, build kNN index, then rank top-K ROI candidates per view.",
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
            extra={"source": candidate_source, "error": err_text},
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
                f"Extract {_selected_extractor_label()} tile embeddings, build kNN index, then rank top-K ROI candidates per view.",
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


def _current_view_cache_key() -> Optional[tuple[Any, ...]]:
    if not state._current_view:
        return None
    cv = state._current_view
    return (
        state.SLIDE_PATH,
        str(getattr(state, "AGENT_TYPE", "") or "").lower(),
        int(cv["x0"]),
        int(cv["y0"]),
        int(cv["w"]),
        int(cv["h"]),
    )


def _refresh_roi_candidates_for_current_view(top_k: int = ROI_CANDIDATE_TOP_K) -> List[Dict[str, Any]]:
    if not state._current_view:
        state._last_roi_candidates = []
        state._last_roi_candidate_source = None
        state._last_roi_candidate_overlay_path = None
        state._last_roi_candidate_view_key = None
        state._last_roi_candidate_top_k = None
        return []

    top_k = max(1, int(top_k))
    view_key = _current_view_cache_key()
    if (
        view_key is not None
        and state._last_roi_candidate_view_key == view_key
        and state._last_roi_candidate_top_k == top_k
    ):
        return list(state._last_roi_candidates)

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
        source = _selected_candidate_source(str(getattr(state, "AGENT_TYPE", "") or "").lower() == "aml")

    if not candidates and ROI_CANDIDATE_ALLOW_FALLBACK:
        candidates = _fallback_candidates_from_current_view(top_k)
        if candidates:
            source = "fallback_heuristic"

    state._last_roi_candidates = candidates
    state._last_roi_candidate_source = source
    state._last_roi_candidate_overlay_path = _build_roi_candidate_overlay(candidates)
    state._last_roi_candidate_view_key = view_key
    state._last_roi_candidate_top_k = top_k
    return candidates


def _attach_roi_candidates(info: Dict[str, Any], top_k: int = ROI_CANDIDATE_TOP_K) -> Dict[str, Any]:
    candidates = _refresh_roi_candidates_for_current_view(top_k=top_k)
    aml_mode = str(getattr(state, "AGENT_TYPE", "") or "").lower() == "aml"

    # Strip candidates that overlap an already-marked ROI so the VLM is not tempted
    # to re-navigate to or re-mark the same location.
    if candidates and state._roi_marks:
        marked_centers = []
        for roi in state._roi_marks:
            bbox = roi.get("view_bbox_level0")
            if bbox:
                marked_centers.append((bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2))
        if marked_centers:
            min_sep_sq = (ROI_TARGET_SIDE_PX * 0.5) ** 2
            def _not_marked(c: Dict[str, Any]) -> bool:
                cl = c.get("center_level0")
                if not cl:
                    return True
                cx, cy = cl[0], cl[1]
                return all((cx - mx) ** 2 + (cy - my) ** 2 >= min_sep_sq for mx, my in marked_centers)
            candidates = [c for c in candidates if _not_marked(c)]

    if aml_mode and candidates:
        # In AML mode, keep suspicious candidates prominent but still expose a small
        # amount of contrast so the model does not see only "bad_like" regions.
        # The full list stays in state._last_roi_candidates so wsi_mark_roi_norm
        # validation still accepts any of these coordinates.
        bad_like = [c for c in candidates if c.get("quality_hint") == "bad_like"]
        contrast = [c for c in candidates if c.get("quality_hint") != "bad_like"]
        if bad_like:
            bad_like = sorted(
                bad_like,
                key=lambda c: (
                    float(c.get("retrieval_score", c.get("score", float("-inf")))),
                    float(c.get("bad_margin", 0.0)),
                ),
                reverse=True,
            )
            if contrast:
                contrast = sorted(
                    contrast,
                    key=lambda c: (
                        0 if c.get("quality_hint") == "good_like" else 1,
                        -float(c.get("good_top1_similarity", c.get("score", 0.0))),
                        float(c.get("bad_margin", 0.0)),
                    ),
                )
                contrast_slots = min(2, len(contrast), max(0, ROI_CANDIDATE_TOP_K_AML - 1))
                chosen = bad_like[: max(1, ROI_CANDIDATE_TOP_K_AML - contrast_slots)]
                remaining = ROI_CANDIDATE_TOP_K_AML - len(chosen)
                if remaining > 0:
                    chosen.extend(contrast[:remaining])
                candidates = chosen
            else:
                candidates = bad_like[:ROI_CANDIDATE_TOP_K_AML]
        else:
            # Nothing is bad_like in this view — keep the raw retrieval-ranked list so
            # the VLM can see that bad exemplars are not winning here.
            candidates = candidates[:ROI_CANDIDATE_TOP_K_AML]
    info["roi_candidates"] = candidates
    info["roi_candidate_count"] = len(candidates)
    info["marked_roi_count"] = len(state._roi_marks)
    info["marked_roi_labels"] = [r.get("label", "") for r in state._roi_marks]
    if aml_mode:
        kept_roi_count = len(state._roi_marks)
        if kept_roi_count >= 2:
            info["aml_stop_hint"] = (
                "If the evidence you already have is enough for a stable final AML decision "
                "(Normal marrow / Acute leukemia / Call for more diagnostics), stop now and give the final answer. "
                f"You already have {kept_roi_count} kept ROI(s); do not explore another ROI unless it could materially change the decision. "
                "Final class must follow morphology and blast percentage, not retrieval labels alone."
            )
        else:
            info["aml_stop_hint"] = (
                "Stop as soon as the current evidence is enough for a stable final AML decision. "
                "Do not keep exploring for extra confirmation once another ROI is unlikely to change the final category. "
                "Final class must follow morphology and blast percentage, not retrieval labels alone."
            )

    # Detect how many consecutive recent steps have stayed in the same slide region.
    # Uses level-0 view centers from the step log; if the last N centers all cluster
    # within 1.5× the current view width of each other, the agent is stuck.
    same_region_steps = 0
    if state._step_log and state._current_view:
        cur_cx = state._current_view["x0"] + state._current_view["w"] / 2.0
        cur_cy = state._current_view["y0"] + state._current_view["h"] / 2.0
        radius_sq = (state._current_view["w"] * 1.5) ** 2
        for entry in reversed(state._step_log[-8:]):
            bbox = entry.get("view_bbox_level0")
            if not bbox:
                break
            ecx = bbox[0] + bbox[2] / 2.0
            ecy = bbox[1] + bbox[3] / 2.0
            if (ecx - cur_cx) ** 2 + (ecy - cur_cy) ** 2 <= radius_sq:
                same_region_steps += 1
            else:
                break
    info["same_region_steps"] = same_region_steps
    if same_region_steps >= 3:
        info["region_loop_warning"] = (
            f"You have taken {same_region_steps} consecutive steps in the same slide region. "
            "Call wsi_get_overview_view or wsi_zoom_full_norm NOW to move to a completely different area."
        )

    # Count consecutive recent steps with very low tissue content (white/background views).
    # If the agent has been stuck in empty/background territory for 2+ steps, force an escape.
    _LOW_TISSUE_THRESHOLD = 0.10
    low_tissue_steps = 0
    for entry in reversed(state._step_log[-6:]):
        tf = entry.get("tissue_fraction")
        if tf is not None and float(tf) < _LOW_TISSUE_THRESHOLD:
            low_tissue_steps += 1
        else:
            break
    info["low_tissue_steps"] = low_tissue_steps
    if low_tissue_steps >= 2:
        info["low_tissue_loop_warning"] = (
            f"ALERT: {low_tissue_steps} consecutive views have been mostly empty background "
            "(tissue_fraction < 0.10). You are zoomed into empty glass. "
            "You MUST call wsi_get_overview_view RIGHT NOW to reset to the full slide, "
            "then navigate to a region with visible tissue (pink/purple staining)."
        )

    info["roi_candidate_source"] = state._last_roi_candidate_source
    info["roi_candidate_prep"] = dict(state._roi_candidate_prep) if state._roi_candidate_prep else None
    info["roi_candidate_overlay_path"] = state._last_roi_candidate_overlay_path
    extractor_label = _selected_extractor_label()
    if aml_mode:
        info["roi_candidate_pipeline"] = (
            f"{extractor_label} tile embeddings -> exact good/bad exemplar retrieval -> rank by raw nearest bad similarity -> top-K per current view"
        )
    else:
        info["roi_candidate_pipeline"] = f"{extractor_label} tile embeddings -> kNN novelty ranking -> top-K per current view"
    if state._roi_ranker_meta:
        info["roi_candidate_index_meta"] = dict(state._roi_ranker_meta)
        ref_stats = state._roi_ranker_meta.get("reference_stats")
        if aml_mode and isinstance(ref_stats, dict):
            info["aml_reference_stats"] = dict(ref_stats)
    expected_source = _selected_candidate_source(aml_mode)
    if state._last_roi_candidate_source != expected_source:
        info["roi_candidate_warning"] = (
            "Primary candidate source unavailable for this view."
            + (" Using fallback heuristic." if ROI_CANDIDATE_ALLOW_FALLBACK else " Fallback disabled.")
        )
    if candidates:
        if aml_mode:
            info["roi_candidate_guidance"] = (
                "For AML, use bad_like candidates to find cellular high-yield fields, but treat retrieval evidence as advisory only. "
                "Do NOT diagnose AML from bad_like / closer-to-bad alone. "
                "If morphology shows normal maturation with blasts <5%, classify Normal marrow even if retrieval looked suspicious. "
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


def _aml_candidate_reference_evidence(candidate: Dict[str, Any]) -> Dict[str, Any]:
    bad_top1 = candidate.get("bad_top1_similarity")
    good_top1 = candidate.get("good_top1_similarity")
    bad_margin = candidate.get("bad_margin")
    retrieval_score = candidate.get("retrieval_score", candidate.get("score"))
    quality_hint = str(candidate.get("quality_hint") or "uncertain")
    bad_refs = candidate.get("retrieved_bad_refs") if isinstance(candidate.get("retrieved_bad_refs"), list) else []
    good_refs = candidate.get("retrieved_good_refs") if isinstance(candidate.get("retrieved_good_refs"), list) else []
    top_bad = bad_refs[0] if bad_refs else None
    top_good = good_refs[0] if good_refs else None

    match_label = "uncertain"
    if quality_hint == "bad_like":
        match_label = "closer_to_bad"
    elif quality_hint == "good_like":
        match_label = "closer_to_good"

    parts: List[str] = [match_label.replace("_", " ")]
    if isinstance(bad_top1, (int, float)):
        parts.append(f"bad_top1={float(bad_top1):.3f}")
    if isinstance(good_top1, (int, float)):
        parts.append(f"good_top1={float(good_top1):.3f}")
    if isinstance(bad_margin, (int, float)):
        parts.append(f"margin={float(bad_margin):.3f}")
    if isinstance(retrieval_score, (int, float)):
        parts.append(f"retrieval={float(retrieval_score):.3f}")
    if isinstance(top_bad, dict) and top_bad.get("name"):
        parts.append(f"nearest_bad={top_bad['name']}")
    if isinstance(top_good, dict) and top_good.get("name"):
        parts.append(f"nearest_good={top_good['name']}")

    return {
        "match_label": match_label,
        "quality_hint": quality_hint,
        "retrieval_score": float(retrieval_score) if isinstance(retrieval_score, (int, float)) else None,
        "bad_likelihood": float(candidate["bad_likelihood"]) if isinstance(candidate.get("bad_likelihood"), (int, float)) else None,
        "bad_margin": float(bad_margin) if isinstance(bad_margin, (int, float)) else None,
        "bad_top1_similarity": float(bad_top1) if isinstance(bad_top1, (int, float)) else None,
        "good_top1_similarity": float(good_top1) if isinstance(good_top1, (int, float)) else None,
        "nearest_bad_ref": dict(top_bad) if isinstance(top_bad, dict) else None,
        "nearest_good_ref": dict(top_good) if isinstance(top_good, dict) else None,
        "retrieved_bad_refs": [dict(x) for x in bad_refs[:3] if isinstance(x, dict)],
        "retrieved_good_refs": [dict(x) for x in good_refs[:3] if isinstance(x, dict)],
        "summary": ", ".join(parts),
    }


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

        # Duplicate guard: reject if an existing ROI center is within half a tile-width.
        new_cx = x0_new + w_new // 2
        new_cy = y0_new + h_new // 2
        min_sep = w_new * 0.5  # half the ROI side in level-0 pixels
        for existing in state._roi_marks:
            ex_bbox = existing.get("view_bbox_level0")
            if ex_bbox:
                ex_cx = ex_bbox[0] + ex_bbox[2] // 2
                ex_cy = ex_bbox[1] + ex_bbox[3] // 2
                dist_sq = (new_cx - ex_cx) ** 2 + (new_cy - ex_cy) ** 2
                if dist_sq < min_sep ** 2:
                    return {
                        "ok": False,
                        "reason": "duplicate_roi",
                        "message": (
                            f"This location is too close to an already-marked ROI (roi_id={existing['roi_id']}, "
                            f"label='{existing['label']}'). Choose a different candidate or navigate to a new region."
                        ),
                        "existing_roi_id": existing["roi_id"],
                    }

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
        aml_reference_evidence = (
            _aml_candidate_reference_evidence(chosen)
            if str(getattr(state, "AGENT_TYPE", "") or "").lower() == "aml"
            else {}
        )
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
            "candidate_quality_hint": chosen.get("quality_hint"),
            "candidate_retrieval_score": chosen.get("retrieval_score", chosen.get("score")),
            "candidate_bad_likelihood": chosen.get("bad_likelihood"),
            "candidate_bad_margin": chosen.get("bad_margin"),
            "candidate_bad_top1_similarity": chosen.get("bad_top1_similarity"),
            "candidate_good_top1_similarity": chosen.get("good_top1_similarity"),
            "candidate_retrieved_bad_refs": list(chosen.get("retrieved_bad_refs") or []),
            "candidate_retrieved_good_refs": list(chosen.get("retrieved_good_refs") or []),
            "aml_reference_evidence": aml_reference_evidence,
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
