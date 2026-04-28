import hashlib
import json
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
    ROI_TARGET_HEIGHT_UM,
    ROI_TARGET_SIDE_PX,
    ROI_TARGET_WIDTH_UM,
    SELECTED_TILES_ROOT,
    TILE_PX,
    TILE_SIZE_UM,
)
from .embeddings import embedding_extractor_display_name, get_embedding_extractor
from .embeddings.roi_ranker import (
    build_unsupervised_roi_index,
    select_topk_candidates_for_view,
)
from .embeddings.tile_prefilter import tile_touches_tissue_edge
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
from .tuning_config import tuning_value


def _tools_int(key: str, section: str = "tools.candidates", default: int = 0) -> int:
    env = os.getenv(key)
    if env is not None:
        try:
            return int(env)
        except Exception:
            pass
    try:
        return int(tuning_value(section, key))
    except Exception:
        return default


def _tools_float(key: str, section: str = "tools.candidates", default: float = 0.0) -> float:
    env = os.getenv(key)
    if env is not None:
        try:
            return float(env)
        except Exception:
            pass
    try:
        return float(tuning_value(section, key))
    except Exception:
        return default


def _tools_bool(key: str, section: str = "tools.candidates", default: bool = True) -> bool:
    env = os.getenv(key)
    if env is not None:
        text = env.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    try:
        return bool(tuning_value(section, key))
    except Exception:
        return default


def _tools_str(key: str, section: str = "tools.cache", default: str = "") -> str:
    env = os.getenv(key)
    if env is not None:
        return env.strip()
    try:
        return str(tuning_value(section, key))
    except Exception:
        return default


ROI_CANDIDATE_TOP_K = _tools_int("ROI_CANDIDATE_TOP_K", "tools.candidates", 72)
ROI_MARK_CANDIDATE_TOLERANCE_NORM = int(os.getenv("ROI_MARK_CANDIDATE_TOLERANCE_NORM", "170"))
ROI_CANDIDATE_ALLOW_FALLBACK = os.getenv("ROI_CANDIDATE_ALLOW_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "y"}
# Hard cap on how many candidates the VLM sees in AML mode after raw retrieval ranking.
ROI_CANDIDATE_TOP_K_AML = int(os.getenv("ROI_CANDIDATE_TOP_K_AML", "30"))
ROI_RANKER_BATCH_SIZE = int(os.getenv("ROI_RANKER_BATCH_SIZE", "32"))
ROI_RANKER_MAX_WORKERS = int(os.getenv("ROI_RANKER_MAX_WORKERS", "4"))
ROI_COARSE_PREFILTER_TRIGGER_SUPERTILES = int(os.getenv("ROI_COARSE_PREFILTER_TRIGGER_SUPERTILES", "128"))
ROI_COARSE_PREFILTER_KEEP_RATIO = float(os.getenv("ROI_COARSE_PREFILTER_KEEP_RATIO", "0.22"))
ROI_COARSE_PREFILTER_MIN_KEEP_SUPERTILES = int(os.getenv("ROI_COARSE_PREFILTER_MIN_KEEP_SUPERTILES", "36"))
ROI_COARSE_PREFILTER_MAX_KEEP_SUPERTILES = int(os.getenv("ROI_COARSE_PREFILTER_MAX_KEEP_SUPERTILES", "56"))
ROI_QUALITY_PREFILTER_KEEP_RATIO = float(os.getenv("ROI_QUALITY_PREFILTER_KEEP_RATIO", "0.25"))
ROI_QUALITY_PREFILTER_MIN_KEEP_TILES = int(os.getenv("ROI_QUALITY_PREFILTER_MIN_KEEP_TILES", "4"))
ROI_QUALITY_PREFILTER_TRIGGER_TILES = int(os.getenv("ROI_QUALITY_PREFILTER_TRIGGER_TILES", "12"))
ROI_QUALITY_PREFILTER_RANDOM_RESERVE_RATIO = float(os.getenv("ROI_QUALITY_PREFILTER_RANDOM_RESERVE_RATIO", "0.05"))
CACHE_ROOT_DIR = os.getenv("CACHE_ROOT_DIR", "").strip()
if not CACHE_ROOT_DIR:
    CACHE_ROOT_DIR = os.path.abspath("./outputs/_cache")

ROI_TILE_CACHE_DIR = os.getenv("ROI_TILE_CACHE_DIR", "").strip()
if not ROI_TILE_CACHE_DIR:
    ROI_TILE_CACHE_DIR = os.path.join(CACHE_ROOT_DIR, "feature_cache")

ROI_FEATURE_CACHE_DIR = os.getenv("ROI_FEATURE_CACHE_DIR", "").strip()
if not ROI_FEATURE_CACHE_DIR:
    ROI_FEATURE_CACHE_DIR = os.path.join(CACHE_ROOT_DIR, "feature_cache")

os.environ["CACHE_ROOT_DIR"] = CACHE_ROOT_DIR
os.environ["AML_REFERENCE_CACHE_DIR"] = os.path.join(CACHE_ROOT_DIR, "reference_hnsw")

ROI_INDEX_PIPELINE_VERSION = 3
DEFAULT_MPP_FALLBACK_UM = float(tuning_value("tools.slide", "DEFAULT_MPP_UM"))
DEFAULT_MAX_ACCEPTED_ROIS = int(tuning_value("tools.navigation", "MAX_ACCEPTED_ROIS"))
DEFAULT_TARGET_ACCEPTED_ROIS = int(tuning_value("tools.navigation", "TARGET_ACCEPTED_ROIS"))
DEFAULT_CANDIDATE_NAV_FIELD_UM = float(tuning_value("tools.navigation", "CANDIDATE_NAV_FIELD_UM"))


def _selected_batch_size() -> int:
    try:
        return max(1, int(getattr(state, "BATCH_SIZE", ROI_RANKER_BATCH_SIZE) or ROI_RANKER_BATCH_SIZE))
    except Exception:
        return int(ROI_RANKER_BATCH_SIZE)


def _selected_extractor_label() -> str:
    return embedding_extractor_display_name(getattr(state, "EXTRACTOR_NAME", "uni2"))


def _selected_roi_output_size_px() -> int:
    try:
        return max(640, int(getattr(state, "ROI_OUTPUT_SIZE_PX", 1024) or 1024))
    except Exception:
        return 1280


def _selected_max_accepted_rois() -> int:
    try:
        return max(1, int(getattr(state, "MAX_ACCEPTED_ROIS", DEFAULT_MAX_ACCEPTED_ROIS) or DEFAULT_MAX_ACCEPTED_ROIS))
    except Exception:
        return int(DEFAULT_MAX_ACCEPTED_ROIS)


def _selected_target_accepted_rois() -> int:
    try:
        raw_target = int(getattr(state, "TARGET_ACCEPTED_ROIS", DEFAULT_TARGET_ACCEPTED_ROIS) or DEFAULT_TARGET_ACCEPTED_ROIS)
    except Exception:
        raw_target = int(DEFAULT_TARGET_ACCEPTED_ROIS)
    return min(_selected_max_accepted_rois(), max(1, raw_target))


def _allow_discard_last_roi(roi: Dict[str, Any]) -> tuple[bool, str]:
    """Near the AML target ROI count, allow discard only for clearly bad ROIs."""
    if not _agent_is_aml():
        return True, ""

    kept_roi_count = len(getattr(state, "_roi_marks", []) or [])
    target_accepted_rois = _selected_target_accepted_rois()
    tissue_fraction = roi.get("tissue_fraction")
    clearly_empty = isinstance(tissue_fraction, (int, float)) and float(tissue_fraction) < 0.15

    if kept_roi_count >= max(1, target_accepted_rois - 1) and not clearly_empty:
        return (
            False,
            (
                f"Discard blocked: you already have {kept_roi_count} kept ROI(s) and target is "
                f"{target_accepted_rois}. Near the target, keep borderline/interpretable AML ROIs. "
                "Reserve discard for clearly bad ROIs such as mostly background or empty views."
            ),
        )

    return True, ""


def _save_tile_from_current_view_center(
    *,
    label: str,
    quality: str = "good",
    nav_reason: str = "Auto-save diagnostic tile from kept ROI",
) -> Dict[str, Any]:
    if not state._current_view:
        raise RuntimeError("Tile save requested before any current view exists.")

    quality = (quality or "good").strip().lower()
    if quality not in {"good", "bad"}:
        raise ValueError("quality must be 'good' or 'bad'")

    if quality == "good" and len(state._saved_good_tiles) >= MAX_GOOD_TILES:
        return {"ok": False, "reason": "max_good_tiles_reached"}
    if quality == "bad" and len(state._saved_bad_tiles) >= MAX_BAD_TILES:
        return {"ok": False, "reason": "max_bad_tiles_reached"}

    slide = _load_slide()
    mpp = _effective_slide_mpp_um(slide)
    tile_px = int(round(TILE_SIZE_UM / mpp))
    tile_px = max(32, tile_px)

    cv_x0 = int(state._current_view["x0"])
    cv_y0 = int(state._current_view["y0"])
    cv_w = int(state._current_view["w"])
    cv_h = int(state._current_view["h"])
    slide_w0, slide_h0 = slide.level_dimensions[0]
    tile_px = min(tile_px, slide_w0, slide_h0)

    cx_base = cv_x0 + (cv_w // 2)
    cy_base = cv_y0 + (cv_h // 2)
    x0 = cx_base - (tile_px // 2)
    y0 = cy_base - (tile_px // 2)
    x0 = max(0, min(x0, slide_w0 - tile_px))
    y0 = max(0, min(y0, slide_h0 - tile_px))

    region = slide.read_region((x0, y0), 0, (tile_px, tile_px)).convert("RGB")
    region = region.resize((TILE_PX, TILE_PX), Image.BILINEAR)

    run_id = state.RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SELECTED_TILES_ROOT, run_id, "Selected_Tiles", quality)
    os.makedirs(out_dir, exist_ok=True)

    idx = (len(state._saved_good_tiles) + 1) if quality == "good" else (len(state._saved_bad_tiles) + 1)
    x_um = x0 * mpp
    y_um = y0 * mpp
    out_path = os.path.join(out_dir, f"{quality}_{idx:04d}_tile_({x_um}, {y_um}).jpg")
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
    if quality == "good":
        state._saved_good_tiles.append(record)
    else:
        state._saved_bad_tiles.append(record)

    try:
        from wsi_core_pkg.embeddings.roi_ranker import _clear_reference_hnsw_cache
        _clear_reference_hnsw_cache()
    except Exception:
        pass

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


def _selected_candidate_nav_field_um() -> float:
    try:
        override = getattr(state, "CANDIDATE_NAV_FIELD_UM_OVERRIDE", None)
        if override is not None:
            return max(100.0, float(override))
        return max(100.0, float(getattr(state, "CANDIDATE_NAV_FIELD_UM", DEFAULT_CANDIDATE_NAV_FIELD_UM) or DEFAULT_CANDIDATE_NAV_FIELD_UM))
    except Exception:
        return float(DEFAULT_CANDIDATE_NAV_FIELD_UM)


def _is_free_local_search_view() -> bool:
    try:
        return str(state._current_view.get("candidate_navigation_mode") or "") == "free_local_search"
    except Exception:
        return False


def _strip_candidate_payload_for_free_local_search(info: Dict[str, Any]) -> Dict[str, Any]:
    if not (_agent_is_aml() and _is_free_local_search_view()):
        return info
    for key in (
        "roi_candidates",
        "roi_candidate_count",
        "roi_candidate_source",
        "roi_candidate_overlay_path",
        "roi_candidate_warning",
        "roi_candidate_guidance",
        "recommended_action",
        "local_roi_budget",
        "selected_candidate_rank",
        "selected_candidate_center_norm",
        "selected_candidate_bbox_norm",
    ):
        info.pop(key, None)
    return info


def _selected_default_mpp_um() -> float:
    try:
        value = getattr(state, "DEFAULT_MPP_UM_OVERRIDE", DEFAULT_MPP_FALLBACK_UM)
        return max(1e-6, float(value or DEFAULT_MPP_FALLBACK_UM))
    except Exception:
        return float(DEFAULT_MPP_UM)


def _agent_is_aml() -> bool:
    return str(getattr(state, "AGENT_TYPE", "") or "").strip().lower() == "aml"


def _effective_slide_mpp_um(slide) -> float:
    state_override = getattr(state, "DEFAULT_MPP_UM_OVERRIDE", None)
    if state_override is not None:
        try:
            override_val = float(state_override)
        except Exception:
            override_val = None
        if override_val is not None and override_val > 0:
            return override_val

    mpp = _get_mpp_um(slide)
    if mpp is not None and float(mpp) > 0:
        return float(mpp)
    return _selected_default_mpp_um()


def _roi_cap_reached() -> bool:
    return len(getattr(state, "_roi_marks", []) or []) >= _selected_max_accepted_rois()


def _roi_cap_response() -> Dict[str, Any]:
    kept = len(getattr(state, "_roi_marks", []) or [])
    cap = _selected_max_accepted_rois()
    return {
        "ok": False,
        "reason": "max_accepted_rois_reached",
        "message": (
            f"You already have {kept} kept ROI(s), which reaches the configured cap of {cap}. "
            "Stop searching immediately and give the final answer from the ROIs already kept."
        ),
        "marked_roi_count": kept,
        "max_accepted_rois": cap,
    }


def _target_roi_size_level0_px(slide=None) -> tuple[int, int]:
    slide_obj = slide or _load_slide()
    slide_w0, slide_h0 = slide_obj.level_dimensions[0]
    width_px = max(1, int(_selected_roi_output_size_px() or 1024))
    height_px = width_px
    width_px = min(width_px, slide_w0)
    height_px = min(height_px, slide_h0)
    return width_px, height_px


def _enforce_min_roi_view_size(
    *,
    x0: int,
    y0: int,
    w: int,
    h: int,
    slide,
) -> tuple[int, int, int, int, bool]:
    slide_w0, slide_h0 = slide.level_dimensions[0]
    min_w, min_h = _target_roi_size_level0_px(slide)
    clamped_w = max(int(w), int(min_w))
    clamped_h = max(int(h), int(min_h))
    if clamped_w == int(w) and clamped_h == int(h):
        return int(x0), int(y0), int(w), int(h), False

    cx = int(x0) + int(w) // 2
    cy = int(y0) + int(h) // 2
    new_x0 = cx - clamped_w // 2
    new_y0 = cy - clamped_h // 2
    new_x0 = max(0, min(new_x0, slide_w0 - clamped_w))
    new_y0 = max(0, min(new_y0, slide_h0 - clamped_h))
    return int(new_x0), int(new_y0), int(clamped_w), int(clamped_h), True


def _sanitize_cache_component(value: str | None, default: str = "item") -> str:
    raw = str(value or "").strip()
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
    safe = safe.strip("._-")
    return safe or default


def _selected_tile_prefilter_method() -> str:
    raw = str(getattr(state, "TILE_PREFILTER_METHOD", "quality") or "quality").strip().lower()
    return raw if raw in {"none", "coarse", "quality", "hybrid"} else "quality"


def _selected_feature_cache_dir() -> Path | None:
    raw = os.getenv("ROI_FEATURE_CACHE_DIR", "").strip()
    if raw:
        feature_cache_dir = Path(raw)
    else:
        extractor_name = _sanitize_cache_component(getattr(state, "EXTRACTOR_NAME", "uni2"), default="uni2")
        tile_prefilter_method = _sanitize_cache_component(_selected_tile_prefilter_method(), default="quality")
        feature_cache_dir = (
            Path(CACHE_ROOT_DIR)
            / "feature_cache"
            / extractor_name
            / tile_prefilter_method
        )
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    return feature_cache_dir


def _selected_reference_cache_dir() -> Path:
    raw = os.getenv("AML_REFERENCE_CACHE_DIR", "").strip()
    if raw:
        reference_cache_dir = Path(raw)
    else:
        extractor_name = _sanitize_cache_component(getattr(state, "EXTRACTOR_NAME", "uni2"), default="uni2")
        reference_cache_dir = (
            Path(CACHE_ROOT_DIR)
            / "reference_hnsw"
            / extractor_name
        )
    reference_cache_dir.mkdir(parents=True, exist_ok=True)
    return reference_cache_dir


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


def _quality_hint_bonus(candidate: Dict[str, Any]) -> float:
    hint = str(candidate.get("quality_hint") or "good_like")
    if hint == "good_like":
        return 1.0
    if hint == "bad_like":
        return 0.0
    return 0.5


def _aml_candidate_rank_key(candidate: Dict[str, Any]) -> tuple[float, ...]:
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    base_score = candidate.get("combined_rank_score", candidate.get("score"))
    retrieval_score = candidate.get("retrieval_score", candidate.get("score"))
    return (
        _quality_hint_bonus(candidate),
        _as_float(candidate.get("good_top1_similarity"), 0.0),
        _as_float(retrieval_score, 0.0),
        _as_float(base_score, float("-inf")),
    )


def _record_attempted_roi_bbox(bbox_level0: Any) -> None:
    try:
        if not bbox_level0 or len(bbox_level0) < 4:
            return
        x0, y0, w, h = [int(v) for v in bbox_level0[:4]]
        if w <= 0 or h <= 0:
            return
        bbox = [x0, y0, w, h]
        attempted = getattr(state, "_attempted_roi_bboxes_level0", None)
        if attempted is None:
            state._attempted_roi_bboxes_level0 = [bbox]
            return
        if bbox not in attempted:
            attempted.append(bbox)
    except Exception:
        return


def _next_unattempted_candidate_rank() -> Optional[int]:
    candidate_bank = list(getattr(state, "_overview_roi_candidates", None) or [])
    if not candidate_bank:
        candidate_bank = list(state._last_roi_candidates or [])
    if not candidate_bank:
        return None

    attempted_bboxes = list(getattr(state, "_attempted_roi_bboxes_level0", []) or [])
    blocked_centers: list[tuple[float, float, float]] = []
    for bbox in attempted_bboxes:
        try:
            ex_w = max(1, int(bbox[2]))
            ex_h = max(1, int(bbox[3]))
            ex_cx = int(bbox[0]) + ex_w // 2
            ex_cy = int(bbox[1]) + ex_h // 2
            min_sep = 0.3 * float(max(ex_w, ex_h))
            blocked_centers.append((float(ex_cx), float(ex_cy), float(min_sep * min_sep)))
        except Exception:
            continue

    for cand in sorted(candidate_bank, key=lambda item: int(item.get("rank", 10**9))):
        try:
            rank = int(cand.get("rank"))
        except Exception:
            continue
        center = cand.get("center_level0")
        if not center or len(center) < 2:
            return rank
        cx, cy = float(center[0]), float(center[1])
        if all((cx - mx) ** 2 + (cy - my) ** 2 >= min_sep_sq for mx, my, min_sep_sq in blocked_centers):
            return rank
    return None


def _postprocess_roi_candidates_for_view(
    candidates: List[Dict[str, Any]],
    *,
    top_k: int,
    aml_mode: bool,
    source: Optional[str],
) -> tuple[List[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    processed = _filter_edge_touching_candidates_for_current_view(list(candidates))
    meta: Dict[str, Any] = {
        "bad_like_only_view": False,
        "bad_like_hidden_count": 0,
        "forced_min_candidate": False,
    }

    if not processed and ROI_CANDIDATE_ALLOW_FALLBACK:
        fallback = _fallback_candidates_from_current_view(top_k)
        if fallback:
            processed = _filter_edge_touching_candidates_for_current_view(fallback)
            source = "fallback_heuristic"

    if aml_mode and processed:
        aml_cap = max(ROI_CANDIDATE_TOP_K_AML, 14)
        ranked = sorted(processed, key=_aml_candidate_rank_key, reverse=True)
        non_bad_like = [
            candidate
            for candidate in ranked
            if str(candidate.get("quality_hint") or "uncertain") != "bad_like"
        ]
        bad_like = [candidate for candidate in ranked if candidate not in non_bad_like]

        if non_bad_like:
            meta["bad_like_hidden_count"] = len(bad_like)
            processed = non_bad_like[:aml_cap]
        else:
            meta["bad_like_only_view"] = True
            processed = ranked[:1]
            if processed:
                meta["forced_min_candidate"] = True

    return processed, source, meta


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


def _try_load_cached_roi_index(
    slide_path: str,
    extractor_name: str,
    tile_prefilter_method: str,
    tile_size_px: int,
    aml_mode: bool,
) -> Optional[Any]:
    """Try to load cached ROI index from disk if foundation model + tile filter + tile size match."""
    import pickle
    from pathlib import Path

    cache_root = CACHE_ROOT_DIR or os.path.abspath("./outputs/_cache")
    cache_path = Path(cache_root) / "roi_index"

    if not cache_path.exists():
        return None

    # Build fingerprint: foundation model (extractor) + tile filter + tile size
    # Tile size is normally 224, but kept in fingerprint for flexibility
    settings_str = json.dumps({
        "slide_path": slide_path,
        "extractor_name": extractor_name,
        "tile_prefilter_method": tile_prefilter_method,
        "tile_size_px": tile_size_px,
        "aml_mode": aml_mode,
    }, sort_keys=True)
    settings_hash = hashlib.sha256(settings_str.encode()).hexdigest()

    index_cache_file = cache_path / f"roi_index_{settings_hash}.pkl"

    if index_cache_file.exists():
        try:
            with open(index_cache_file, "rb") as f:
                cached_index = pickle.load(f)
            print(f"[WSI][CACHE] Loaded cached ROI index from {index_cache_file}")
            return cached_index
        except Exception as e:
            print(f"[WSI][CACHE] Failed to load cached index: {e}")
            return None

    return None


def _ensure_unsupervised_roi_index():
    cached = state._roi_ranker_index
    meta = state._roi_ranker_meta or {}

    state.CURRENT_AGENT_ACTION = "Preparing ROI candidates from slide..."

    aml_mode = str(getattr(state, "AGENT_TYPE", "") or "").lower() == "aml"
    candidate_source = _selected_candidate_source(aml_mode)
    extractor_label = _selected_extractor_label()
    extractor_name = str(getattr(state, "EXTRACTOR_NAME", "uni2") or "uni2")
    tile_prefilter_method = _selected_tile_prefilter_method()
    use_coarse_prefilter = _use_coarse_prefilter(tile_prefilter_method)
    use_quality_prefilter = _use_quality_prefilter(tile_prefilter_method)

    # Try to load cached index FIRST (before any expensive operations)
    cached_index = _try_load_cached_roi_index(
        slide_path=state.SLIDE_PATH,
        extractor_name=state.EXTRACTOR_NAME,
        tile_prefilter_method=tile_prefilter_method,
        tile_size_px=state.TILE_SIZE_PX,
        aml_mode=aml_mode,
    )
    if cached_index is not None:
        _set_roi_candidate_prep(
            phase="ready",
            status="done",
            message="ROI index loaded from disk cache (skipping tile filtering)",
            extra={"source": "disk_cache", "candidate_source": candidate_source},
        )
        state._roi_ranker_index = cached_index
        state._roi_ranker_meta = {
            "slide_path": state.SLIDE_PATH,
            "cache_source": "disk",
        }
        return cached_index

    if (
        cached is not None
        and meta.get("slide_path") == state.SLIDE_PATH
        and meta.get("pipeline_version") == ROI_INDEX_PIPELINE_VERSION
        and meta.get("extractor_name") == extractor_name
        and meta.get("tile_prefilter_method") == tile_prefilter_method
        and meta.get("agent_type") == getattr(state, "AGENT_TYPE", None)
    ):
        _set_roi_candidate_prep(
            phase="ready",
            status="done",
            message="ROI candidates already prepared for this run.",
            extra={"source": "cache", "candidate_source": candidate_source, "slide_path": state.SLIDE_PATH},
        )
        return cached
    if aml_mode:
        if tile_prefilter_method == "hybrid":
            pipeline_desc = (
                f"Detect deep blue-purple basophilic regions -> dark-guided supertile coarse filter -> raw-tile hybrid quality filter -> {extractor_label} tile embeddings -> ROI-quality exemplar retrieval + nuclei/dark-region priors -> top-K candidate blast-suspected ROIs per view"
            )
        elif tile_prefilter_method == "quality":
            pipeline_desc = (
                f"Detect deep blue-purple basophilic regions -> dark-guided supertile coarse filter -> raw-tile quality filter -> {extractor_label} tile embeddings -> ROI-quality exemplar retrieval + nuclei/dark-region priors -> top-K candidate blast-suspected ROIs per view"
            )
        elif tile_prefilter_method == "coarse":
            pipeline_desc = (
                f"Detect deep blue-purple basophilic regions -> dark-guided supertile coarse filter -> {extractor_label} tile embeddings -> ROI-quality exemplar retrieval + nuclei/dark-region priors -> top-K candidate blast-suspected ROIs per view"
            )
        else:
            pipeline_desc = (
                f"No tile prefilter -> {extractor_label} tile embeddings -> ROI-quality exemplar retrieval + nuclei/dark-region heuristics -> top-K candidate blast-suspected ROIs per view"
            )
    else:
        # Non-AML mode: no dark region gating
        if tile_prefilter_method == "hybrid":
            pipeline_desc = f"Thumbnail coarse region filter -> raw-tile quality score filter -> {extractor_label} tile embeddings -> kNN novelty ranking -> top-K per view"
        elif tile_prefilter_method == "quality":
            pipeline_desc = f"Raw-tile quality score filter -> {extractor_label} tile embeddings -> kNN novelty ranking -> top-K per view"
        elif tile_prefilter_method == "coarse":
            pipeline_desc = f"Thumbnail coarse region filter -> {extractor_label} tile embeddings -> kNN novelty ranking -> top-K per view"
        else:
            pipeline_desc = f"{extractor_label} tile embeddings -> kNN novelty ranking -> top-K per view"
    feature_cache_dir = _selected_feature_cache_dir()
    reference_cache_dir = _selected_reference_cache_dir()
    os.environ["AML_REFERENCE_CACHE_DIR"] = str(reference_cache_dir)
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
        default_mpp = _effective_slide_mpp_um(slide)

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
                if status == "cached":
                    msg = f"Loaded cached {_selected_extractor_label()} feature embeddings..."
                else:
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
                if status == "cached":
                    msg = "Loaded cached AML reference embeddings and retrieval index..."
                else:
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

        # Build the index (cache was already checked at function start)
        index = build_unsupervised_roi_index(
            slide_path=state.SLIDE_PATH,
            extractor_name=state.EXTRACTOR_NAME,
            tile_size_um=state.TILE_SIZE_UM,
            tile_size_px=state.TILE_SIZE_PX,
            batch_size=_selected_batch_size(),
            cache_dir=None,
            feature_cache_dir=feature_cache_dir,
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
            dark_region_boxes_level0=None,
            k_neighbors=20,
            use_reference_labels=aml_mode,
            reference_tiles_root=EXAMPLE_TILES_ROOT if aml_mode else None,
            quality_method="embedding",
            progress_cb=_on_progress,
        )
        state._roi_ranker_index = index
        state._roi_ranker_meta = {
            "slide_path": state.SLIDE_PATH,
            "pipeline_version": ROI_INDEX_PIPELINE_VERSION,
            "num_tiles": index.num_tiles,
            "feature_dim": index.feature_dim,
            "extractor_id": index.extractor_id,
            "extractor_name": extractor_name,
            "tile_size_px": index.tile_size_px,
            "tile_size_um": index.tile_size_um,
            "tile_prefilter_method": tile_prefilter_method,
            "agent_type": getattr(state, "AGENT_TYPE", None),
            "feature_cache_dir": str(feature_cache_dir) if feature_cache_dir is not None else None,
            "reference_cache_dir": str(reference_cache_dir),
            "reference_mode": getattr(index, "reference_mode", "none"),
            "reference_stats": dict(getattr(index, "reference_stats", {}) or {}),
        }

        # Save index to CACHE_ROOT_DIR for next time (fingerprint: foundation model + tile filter)
        try:
            import pickle
            from pathlib import Path

            cache_root = CACHE_ROOT_DIR or os.path.abspath("./outputs/_cache")
            cache_path = Path(cache_root) / "roi_index"
            cache_path.mkdir(parents=True, exist_ok=True)

            # Fingerprint: foundation model (extractor) + tile filter + tile size
            # Tile size is normally 224, but kept for flexibility
            settings_str = json.dumps({
                "slide_path": state.SLIDE_PATH,
                "extractor_name": extractor_name,
                "tile_prefilter_method": tile_prefilter_method,
                "tile_size_px": state.TILE_SIZE_PX,
                "aml_mode": aml_mode,
            }, sort_keys=True)
            settings_hash = hashlib.sha256(settings_str.encode()).hexdigest()
            index_cache_file = cache_path / f"roi_index_{settings_hash}.pkl"

            with open(index_cache_file, "wb") as f:
                pickle.dump(index, f)
            print(f"[WSI][CACHE] Saved ROI index to {index_cache_file}")
        except Exception as e:
            print(f"[WSI][CACHE] Failed to save index to CACHE_ROOT_DIR: {e}")
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
                if tile_touches_tissue_edge(patch):
                    continue
                tissue = float(np.mean(gray < 0.92))
                gx_edge = float(np.abs(np.diff(gray, axis=1)).mean()) if gray.shape[1] > 1 else 0.0
                gy_edge = float(np.abs(np.diff(gray, axis=0)).mean()) if gray.shape[0] > 1 else 0.0
                edge = (gx_edge + gy_edge) / 2.0
                score = (0.70 * tissue) + (0.30 * edge)
                if tissue < 0.05:
                    continue

                cx = int(round(((x0 + x1) / 2.0) / max(1, w - 1) * 999.0))
                cy = int(round(((y0 + y1) / 2.0) / max(1, h - 1) * 999.0))
                bx0 = int(round(x0 / max(1, w - 1) * 999.0))
                by0 = int(round(y0 / max(1, h - 1) * 999.0))
                bx1 = int(round(x1 / max(1, w - 1) * 999.0))
                by1 = int(round(y1 / max(1, h - 1) * 999.0))

                cx_level0 = cv_x0 + int(round((cx / 999.0) * cv_w))
                cy_level0 = cv_y0 + int(round((cy / 999.0) * cv_h))
                bx0_level0 = cv_x0 + int(round((x0 / max(1, w)) * cv_w))
                by0_level0 = cv_y0 + int(round((y0 / max(1, h)) * cv_h))
                bx1_level0 = cv_x0 + int(round((x1 / max(1, w)) * cv_w))
                by1_level0 = cv_y0 + int(round((y1 / max(1, h)) * cv_h))

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
                        "tile_bbox_level0": [bx0_level0, by0_level0, bx1_level0, by1_level0],
                    }
                )

    if not raw:
        return []

    raw.sort(key=lambda r: float(r["score"]), reverse=True)
    selected: List[Dict[str, Any]] = []
    min_dist_sq = float(96 * 96)
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


def _candidate_patch_from_current_view(
    image: Image.Image,
    candidate: Dict[str, Any],
) -> Optional[Image.Image]:
    bbox = candidate.get("bbox_norm")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    w, h = image.size
    if w <= 1 or h <= 1:
        return None

    try:
        bx0, by0, bx1, by1 = [int(v) for v in bbox]
    except Exception:
        return None

    sx = (w - 1) / 999.0
    sy = (h - 1) / 999.0
    x0 = max(0, min(w - 1, int(round(bx0 * sx))))
    y0 = max(0, min(h - 1, int(round(by0 * sy))))
    x1 = max(x0 + 1, min(w, int(round(bx1 * sx)) + 1))
    y1 = max(y0 + 1, min(h, int(round(by1 * sy)) + 1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return image.crop((x0, y0, x1, y1))


def _candidate_patch_from_level0(candidate: Dict[str, Any]) -> Optional[Image.Image]:
    bbox = candidate.get("tile_bbox_level0")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x0, y0, x1, y1 = [int(v) for v in bbox]
    except Exception:
        return None

    w = max(0, x1 - x0)
    h = max(0, y1 - y0)
    if w < 8 or h < 8:
        return None

    slide = _load_slide()
    slide_w0, slide_h0 = slide.level_dimensions[0]
    x0 = max(0, min(x0, slide_w0 - 1))
    y0 = max(0, min(y0, slide_h0 - 1))
    w = max(1, min(w, slide_w0 - x0))
    h = max(1, min(h, slide_h0 - y0))

    patch = slide.read_region((x0, y0), 0, (w, h)).convert("RGB")
    if patch.size != (TILE_PX, TILE_PX):
        patch = patch.resize((TILE_PX, TILE_PX), Image.BILINEAR)
    return patch


def _filter_edge_touching_candidates_for_current_view(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not candidates or not state._current_view:
        return candidates

    debug_path = state._current_view.get("debug_path")
    if not debug_path or not os.path.exists(debug_path):
        return candidates

    with Image.open(debug_path) as im:
        img = im.convert("RGB")
        filtered: List[Dict[str, Any]] = []
        for candidate in candidates:
            patch = _candidate_patch_from_level0(candidate)
            if patch is None:
                patch = _candidate_patch_from_current_view(img, candidate)
            if patch is not None and tile_touches_tissue_edge(patch):
                continue
            filtered.append(candidate)
    return filtered


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


def _clear_roi_candidate_cache() -> None:
    state._last_roi_candidates = []
    state._last_roi_candidate_meta = {}
    state._last_roi_candidate_source = None
    state._last_roi_candidate_overlay_path = None
    state._last_roi_candidate_view_key = None
    state._last_roi_candidate_top_k = None


def _remember_overview_candidate_bank() -> None:
    state._overview_roi_candidates = [dict(c) for c in (state._last_roi_candidates or [])]


def _refresh_roi_candidates_for_current_view(top_k: int = ROI_CANDIDATE_TOP_K, skip_refresh: bool = False) -> List[Dict[str, Any]]:
    if not state._current_view:
        _clear_roi_candidate_cache()
        return []

    top_k = max(1, int(top_k))
    view_key = _current_view_cache_key()
    if (
        view_key is not None
        and state._last_roi_candidate_view_key == view_key
        and state._last_roi_candidate_top_k == top_k
    ):
        return list(state._last_roi_candidates)

    if skip_refresh and state._last_roi_candidates:
        return list(state._last_roi_candidates)

    candidates: List[Dict[str, Any]] = []
    source: Optional[str] = None
    aml_mode = str(getattr(state, "AGENT_TYPE", "") or "").lower() == "aml"
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
            min_center_separation_px=0,
            focus_boxes_level0=None,
        )
        source = _selected_candidate_source(aml_mode)

    if not candidates and ROI_CANDIDATE_ALLOW_FALLBACK:
        candidates = _fallback_candidates_from_current_view(top_k)
        if candidates:
            source = "fallback_heuristic"

    if not skip_refresh:
        candidates, source, candidate_meta = _postprocess_roi_candidates_for_view(
            candidates,
            top_k=top_k,
            aml_mode=aml_mode,
            source=source,
        )
        if not candidates and ROI_CANDIDATE_ALLOW_FALLBACK and source != "fallback_heuristic":
            fallback_candidates = _fallback_candidates_from_current_view(top_k)
            if fallback_candidates:
                candidates, source, candidate_meta = _postprocess_roi_candidates_for_view(
                    fallback_candidates,
                    top_k=top_k,
                    aml_mode=aml_mode,
                    source="fallback_heuristic",
                )
    else:
        candidate_meta = {}

    state._last_roi_candidates = candidates
    state._last_roi_candidate_meta = candidate_meta
    state._last_roi_candidate_source = source
    if not skip_refresh:
        state._last_roi_candidate_overlay_path = _build_roi_candidate_overlay(candidates)
    state._last_roi_candidate_view_key = view_key
    state._last_roi_candidate_top_k = top_k
    return candidates


_CANDIDATE_REFERENCE_FIELDS = {
    "bad_likelihood",
    "bad_margin",
    "bad_top1_similarity",
    "good_top1_similarity",
    "blast_top1_similarity",
    "blast_similarity_score",
    "retrieved_bad_refs",
    "retrieved_good_refs",
    "retrieved_blast_refs",
    "reference_mode",
    "aml_reference_evidence",
}


def _candidate_for_vlm(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of candidate with internal reference/ranking fields removed."""
    return {k: v for k, v in candidate.items() if k not in _CANDIDATE_REFERENCE_FIELDS}


def _candidate_navigation_bbox_norm(
    candidate: Dict[str, Any],
    *,
    target_field_um: Optional[float] = None,
) -> Optional[List[int]]:
    if not state._current_view:
        return None
    center = candidate.get("center_level0")
    if not center or len(center) < 2:
        return None
    try:
        slide = _load_slide()
        mpp = _effective_slide_mpp_um(slide)
        if not mpp or mpp <= 0:
            return None
        cv_x0 = int(state._current_view["x0"])
        cv_y0 = int(state._current_view["y0"])
        cv_w = max(1, int(state._current_view["w"]))
        cv_h = max(1, int(state._current_view["h"]))
        cxi = float(center[0])
        cyi = float(center[1])
        target_um = float(target_field_um or _selected_candidate_nav_field_um())
        target_px = max(1, int(round(target_um / mpp)))
        half_w_norm = max(10, int(round((target_px / cv_w) * 999.0 / 2.0)))
        half_h_norm = max(10, int(round((target_px / cv_h) * 999.0 / 2.0)))
        cx_norm = int(round(((cxi - cv_x0) / cv_w) * 999.0))
        cy_norm = int(round(((cyi - cv_y0) / cv_h) * 999.0))
        cx_norm = max(0, min(999, cx_norm))
        cy_norm = max(0, min(999, cy_norm))
        return [
            max(0, min(999, cx_norm - half_w_norm)),
            max(0, min(999, cy_norm - half_h_norm)),
            max(0, min(999, cx_norm + half_w_norm)),
            max(0, min(999, cy_norm + half_h_norm)),
        ]
    except Exception:
        return None


def _attach_roi_candidates(info: Dict[str, Any], top_k: int = ROI_CANDIDATE_TOP_K, skip_refresh: bool = False) -> Dict[str, Any]:
    kept_roi_count = len(getattr(state, "_roi_marks", []) or [])
    target_accepted_rois = _selected_target_accepted_rois()
    if kept_roi_count >= target_accepted_rois:
        return info

    candidates = _refresh_roi_candidates_for_current_view(top_k=top_k, skip_refresh=skip_refresh)
    aml_mode = str(getattr(state, "AGENT_TYPE", "") or "").lower() == "aml"
    nav_field_um = _selected_candidate_nav_field_um()
    for candidate in candidates:
        nav_bbox = _candidate_navigation_bbox_norm(candidate, target_field_um=nav_field_um)
        if nav_bbox:
            candidate["navigation_bbox_norm"] = nav_bbox
            candidate["navigation_field_width_um"] = nav_field_um
    candidate_meta = getattr(state, "_last_roi_candidate_meta", None)
    if not isinstance(candidate_meta, dict):
        candidate_meta = {}
    bad_like_only_view = bool(candidate_meta.get("bad_like_only_view"))
    bad_like_hidden_count = int(candidate_meta.get("bad_like_hidden_count") or 0)
    forced_min_candidate = bool(candidate_meta.get("forced_min_candidate"))
    info["roi_candidates"] = [_candidate_for_vlm(c) for c in candidates]
    info["roi_candidate_count"] = len(candidates)
    info["local_roi_budget"] = len(candidates)
    info["marked_roi_count"] = len(state._roi_marks)
    info["max_accepted_rois"] = _selected_max_accepted_rois()
    info["target_accepted_rois"] = _selected_target_accepted_rois()
    info["roi_cap_reached"] = _roi_cap_reached()
    info["marked_roi_labels"] = [r.get("label", "") for r in state._roi_marks]

    # Option E: Auto-recommend action + exploration budget (force faster VLM behavior)
    if aml_mode and candidates and not _roi_cap_reached():
        top = candidates[0]
        current_field_width_um = state._current_view.get("field_width_um") if state._current_view else None
        if (
            isinstance(current_field_width_um, (int, float))
            and current_field_width_um > (nav_field_um * 1.25)
        ):
            info["recommended_action"] = {
                "tool": "wsi_open_candidate",
                "urgency": "HIGH",
                "reason": (
                    f"Candidate #1 is pre-ranked by AML relevance. Jump first to an approximately "
                    f"{int(round(nav_field_um))} um field around the candidate, then mark or skip."
                ),
                "params": {
                    "rank": int(top["rank"]),
                },
            }

    # Exploration budget counter - shows VLM how many steps it has used
    total_steps = len(state._step_log) if state._step_log else 0
    navigation_steps = sum(1 for e in (state._step_log[-12:] if state._step_log else [])
                          if e.get("tool") in ("wsi_get_overview_view", "wsi_zoom_current_norm", "wsi_zoom_full_norm", "wsi_pan_current"))
    roi_steps = len(state._roi_marks)
    info["exploration_budget"] = {
        "total_steps": total_steps,
        "navigation_steps": navigation_steps,
        "rois_marked": roi_steps,
        "recommended_max_steps": 12,
        "status": "OVER_BUDGET" if navigation_steps > 4 else ("WARNING" if navigation_steps >= 3 else "OK"),
        "message": (
            f"Used {total_steps} steps ({navigation_steps} navigation, {roi_steps} ROIs). "
            f"Recommended max ~12 steps total. "
            "If navigation_steps > 2, stop wandering between views: search within a strong opened region and mark once you find a representative interpretable high-power ROI."
        ),
    }

    if aml_mode:
        kept_roi_count = len(state._roi_marks)
        max_accepted_rois = _selected_max_accepted_rois()
        target_accepted_rois = _selected_target_accepted_rois()
        if kept_roi_count >= max_accepted_rois:
            info["aml_stop_hint"] = (
                f"Accepted ROI cap reached ({kept_roi_count}/{max_accepted_rois}). "
                "Finalize now — do not call any more tools."
            )
        elif target_accepted_rois == max_accepted_rois and kept_roi_count >= target_accepted_rois:
            info["aml_stop_hint"] = (
                f"ROI target reached ({kept_roi_count}/{target_accepted_rois}). "
                "Finalize now — do not call any more tools."
            )
        elif kept_roi_count >= target_accepted_rois:
            info["aml_stop_hint"] = (
                f"Soft ROI target reached ({kept_roi_count}/{target_accepted_rois}; hard cap {max_accepted_rois}). "
                "Finalize unless an additional ROI would materially change the decision — do not call wsi_get_view_info to confirm."
            )
        elif kept_roi_count == max(1, target_accepted_rois - 1):
            info["aml_stop_hint"] = (
                f"You already have {kept_roi_count} kept ROI(s). "
                f"Soft target is {target_accepted_rois} kept AML ROIs, so inspect one more distinct informative ROI unless none can be found after reasonable search."
            )
        elif kept_roi_count == 1:
            info["aml_stop_hint"] = (
                f"One ROI is screening evidence only. Soft target is {target_accepted_rois} kept AML ROIs from distinct slide regions if feasible."
            )
        else:
            info["aml_stop_hint"] = (
                f"You have {kept_roi_count} kept ROI(s). Continue toward the soft target of {target_accepted_rois} representative AML ROIs across distinct slide regions unless additional informative ROIs cannot be found."
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
        next_rank_hint = _next_unattempted_candidate_rank() if aml_mode else None
        info["region_loop_warning"] = (
            f"You have taken {same_region_steps} consecutive steps in the same slide region. "
            + (
                f"Do NOT keep searching here. Call wsi_open_candidate(rank={next_rank_hint}) now to move to a different candidate region."
                if next_rank_hint is not None
                else
                "Call wsi_get_overview_view or wsi_zoom_full_norm NOW to move to a completely different area."
            )
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
        next_rank_hint = _next_unattempted_candidate_rank() if aml_mode else None
        low_tissue_action = (
            f"You MUST call wsi_open_candidate(rank={next_rank_hint}) RIGHT NOW to jump back to a candidate region with visible tissue."
            if next_rank_hint is not None
            else
            "You MUST call wsi_get_overview_view RIGHT NOW to reset to the full slide, "
            "then navigate to a region with visible tissue (pink/purple staining)."
        )
        info["low_tissue_loop_warning"] = (
            f"ALERT: {low_tissue_steps} consecutive views have been mostly empty background "
            "(tissue_fraction < 0.10). You are zoomed into empty glass. "
            + low_tissue_action
        )

    # CRITICAL: Exploration budget exceeded warning
    if aml_mode and navigation_steps >= 5 and roi_steps == 0:
        next_rank_hint = _next_unattempted_candidate_rank()
        exploration_action = (
            f"Open candidate region #{next_rank_hint} next, then search inside that field with zoom/pan until you find a representative interpretable high-power ROI with adequate nucleated cells, readable morphology, and acceptable focus; mark only after that search, or skip the region. "
            if next_rank_hint is not None
            else
            "Search within the current strong region with zoom/pan until you find a representative local interpretable ROI with adequate nucleated cells and readable morphology, or call wsi_get_overview_view and move to a different region. "
        )
        info["exploration_over_budget_warning"] = (
            f"CRITICAL: You have navigated {navigation_steps} times without marking any ROI. "
            "This is excessive exploration. "
            + exploration_action
            + "Do not mark the candidate center by default."
        )
    elif aml_mode and navigation_steps >= 3 and roi_steps == 0:
        info["exploration_warning"] = (
            f"WARNING: {navigation_steps} navigation steps with 0 ROIs marked. "
            "You are over-exploring. Use wsi_open_candidate(rank) to jump into a strong region, then search within that field by zooming to high power before you mark. "
            "Do not treat the candidate center tile as the ROI."
        )

    info["roi_candidate_source"] = state._last_roi_candidate_source
    info["roi_candidate_prep"] = dict(state._roi_candidate_prep) if state._roi_candidate_prep else None
    info["roi_candidate_overlay_path"] = state._last_roi_candidate_overlay_path
    extractor_label = _selected_extractor_label()
    tile_prefilter_method = _selected_tile_prefilter_method()

    if aml_mode:
        info["roi_candidate_pipeline"] = (
            f"No tile prefilter -> {extractor_label} tile embeddings -> ROI-quality exemplar retrieval + nuclei/dark-region heuristics -> top-K candidate blast-suspected ROIs per current view"
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
    if aml_mode and bad_like_hidden_count > 0:
        msg = (
            f"Hid {bad_like_hidden_count} bad_like ROI candidate(s). "
            "Do not mark bad_like candidates when good_like or uncertain alternatives exist."
        )
        if info.get("roi_candidate_warning"):
            info["roi_candidate_warning"] = f"{info['roi_candidate_warning']} {msg}"
        else:
            info["roi_candidate_warning"] = msg
    if aml_mode and forced_min_candidate:
        msg = (
            "Only weak ROI support was available in this view, so one best-effort fallback candidate was kept "
            "to avoid an empty candidate list. Treat it as low-confidence and navigate to a better cellular region if feasible."
        )
        if info.get("roi_candidate_warning"):
            info["roi_candidate_warning"] = f"{info['roi_candidate_warning']} {msg}"
        else:
            info["roi_candidate_warning"] = msg
    if aml_mode and bad_like_only_view:
        if forced_min_candidate:
            msg = (
                "Current view only produced bad_like ROI candidates. One best-effort fallback candidate is shown, "
                "but prefer moving to a different region or zoom level for better candidates if feasible."
            )
        else:
            msg = (
                "Current view only produced bad_like ROI candidates. "
                "Do NOT mark them. Navigate to a different region or zoom level and look for good_like/uncertain candidates."
            )
        if info.get("roi_candidate_warning"):
            info["roi_candidate_warning"] = f"{info['roi_candidate_warning']} {msg}"
        else:
            info["roi_candidate_warning"] = msg
    if candidates:
        if aml_mode:
            guidance_intro = (
                "Treat roi_candidates as candidate blast-suspected ROIs selected from tissue, nucleated-cell, focus, RBC, and artifact heuristics across the current view. Prioritize deep dark blue-purple cellular fields; dark red-pink is only a rare fallback when clearly cellular, and gray-black low-chroma junk should be rejected. "
            )
            info["roi_candidate_guidance"] = (
                guidance_intro +
                "Follow a standard practical hierarchy: tissue first, then nucleated-cell-rich interpretable marrow over RBC-rich/empty areas, then blast-suspected morphology. Prefer ROIs with adequate nucleated cells, readable single-cell detail, acceptable focus, and limited artifact. Moderate cellularity is acceptable if morphology is still assessable; do not reject a usable ROI only because it is not the single densest field in the region. "
                f"A single ROI is screening evidence only. For AML, soft target is {_selected_target_accepted_rois()} kept ROIs from representative distinct slide regions when feasible; hard cap is {_selected_max_accepted_rois()}. "
                "Use roi_candidates only to jump into a promising region quickly. After opening a candidate region, search within that field by zooming/panning until you find a representative high-power ROI. Choose a nearby area with better readability or less artifact when available, but do not over-search indefinitely for a marginally denser patch. Then use wsi_mark_roi_norm."
            )
        else:
            info["roi_candidate_guidance"] = (
                "Use roi_candidates as region-level guidance, then search locally within the opened field before choosing a final ROI with wsi_mark_roi_norm."
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


def _candidate_by_rank(rank: int, *, top_k: int = ROI_CANDIDATE_TOP_K) -> Optional[Dict[str, Any]]:
    candidates = list(state._last_roi_candidates or [])
    if not candidates and state._current_view:
        candidates = _refresh_roi_candidates_for_current_view(top_k=top_k)
    for cand in candidates:
        if int(cand.get("rank", -1)) == int(rank):
            return cand
    for cand in (getattr(state, "_overview_roi_candidates", None) or []):
        if int(cand.get("rank", -1)) == int(rank):
            return cand
    return None


def _invalid_candidate_rank_response(rank: int) -> Dict[str, Any]:
    return {
        "ok": False,
        "reason": "invalid_candidate_rank",
        "message": (
            f"Candidate rank #{rank} is not available in the latest current-view candidate list. "
            "Use one of the available roi_candidates ranks from the latest tool output."
        ),
        "available_candidate_ranks": sorted(
            {
                int(c.get("rank", -1))
                for c in list(state._last_roi_candidates or []) + list(getattr(state, "_overview_roi_candidates", None) or [])
                if c.get("rank") is not None
            }
        ),
    }


def _dark_blue_flood_fraction(image: Image.Image) -> float:
    """Return fraction of pixels that are saturated dark-blue flood (no cell outlines)."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    ch_max = rgb.max(axis=2)
    ch_min = rgb.min(axis=2)
    chroma = ch_max - ch_min
    padded = np.pad(gray, 1, mode="edge")
    lap = (
        padded[1:-1, :-2] + padded[1:-1, 2:] +
        padded[:-2, 1:-1] + padded[2:, 1:-1] -
        4.0 * padded[1:-1, 1:-1]
    )
    edge_mag = np.abs(lap)
    mask = (
        (gray < 0.22) &
        (chroma > 0.18) &
        (rgb[:, :, 2] > rgb[:, :, 0] + 0.15) &
        (rgb[:, :, 2] > rgb[:, :, 1] + 0.05) &
        (edge_mag < 0.05)
    )
    return float(np.mean(mask))


def _mark_roi_from_candidate(
    *,
    chosen: Dict[str, Any],
    label: str,
    note: str,
    importance: int,
    nav_reason: str,
    requested_center_norm: Optional[List[int]] = None,
) -> Dict[str, Any]:
    if not state._current_view:
        raise RuntimeError("ROI marking called before wsi_get_overview_view.")

    slide = _load_slide()
    slide_w0, slide_h0 = slide.level_dimensions[0]
    center_level0 = chosen.get("center_level0")
    if center_level0 and len(center_level0) >= 2:
        cx_base = int(round(float(center_level0[0])))
        cy_base = int(round(float(center_level0[1])))
    else:
        cv_x0 = state._current_view["x0"]
        cv_y0 = state._current_view["y0"]
        cv_w = state._current_view["w"]
        cv_h = state._current_view["h"]
        cx_999 = float(chosen["center_norm"][0])
        cy_999 = float(chosen["center_norm"][1])
        cx_rel = cx_999 / 999.0
        cy_rel = cy_999 / 999.0
        cx_base = cv_x0 + int(round(cx_rel * cv_w))
        cy_base = cv_y0 + int(round(cy_rel * cv_h))
    cx_999 = float((chosen.get("center_norm") or [0, 0])[0])
    cy_999 = float((chosen.get("center_norm") or [0, 0])[1])

    w_new, h_new = _target_roi_size_level0_px(slide)

    x0_new = cx_base - w_new // 2
    y0_new = cy_base - h_new // 2

    x0_new = max(0, min(x0_new, slide_w0 - w_new))
    y0_new = max(0, min(y0_new, slide_h0 - h_new))

    new_cx = x0_new + w_new // 2
    new_cy = y0_new + h_new // 2
    min_sep = max(w_new, h_new) * 0.5
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

    requested_cx = requested_center_norm[0] if requested_center_norm else cx_999
    requested_cy = requested_center_norm[1] if requested_center_norm else cy_999
    print(
        "[WSI][ROI_NORM] requested_center=(%.1f,%.1f), snapped_center=(%.1f,%.1f), "
        "candidate_rank=%s, base_center=(%d,%d), ROI_bbox=(%d,%d,%d,%d)"
        % (
            requested_cx,
            requested_cy,
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
        max_dim=_selected_roi_output_size_px(),
        tag="roi",
        force_level=0,
    )

    tf = info.get("tissue_fraction")
    if tf is not None and tf < 0.15:
        info["tissue_warning"] = (
            "This high-power ROI is mostly background/empty glass (low tissue_fraction). "
            "This is one of the few cases where immediate discard is appropriate. "
            "Use wsi_discard_last_roi and select an ROI centered on diagnostic tissue."
        )

    if _agent_is_aml():
        roi_img_path = info.get("debug_path", "")
        try:
            flood_frac = _dark_blue_flood_fraction(Image.open(roi_img_path)) if roi_img_path else 0.0
        except Exception:
            flood_frac = 0.0
        if flood_frac > 0.35:
            return {
                "ok": False,
                "reason": "dark_blue_flood_rejected",
                "flood_fraction": round(flood_frac, 2),
                "message": (
                    f"ROI auto-rejected: {round(flood_frac * 100)}% is saturated dark-blue flood "
                    f"(stain pool, no cell outlines). Move the box away from the flood region and try again."
                ),
            }

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
        "roi_output_size_px": _selected_roi_output_size_px(),
        "candidate_rank": chosen.get("rank"),
        "candidate_score": chosen.get("score"),
        "candidate_center_norm": chosen.get("center_norm"),
        "requested_center_norm": [
            int(round(requested_cx)),
            int(round(requested_cy)),
        ],
    }

    state._roi_marks.append(roi)
    state.CURRENT_AGENT_ACTION = f"Marked ROI #{roi_id}: {label}"
    _record_attempted_roi_bbox(roi.get("view_bbox_level0"))

    if _agent_is_aml() and len(state._saved_good_tiles) < 5:
        auto_tile = _save_tile_from_current_view_center(
            label=label or f"roi_{roi_id}",
            quality="good",
            nav_reason=f"Auto-save one good tile for kept AML ROI #{roi_id}",
        )
        roi["auto_saved_tile"] = dict(auto_tile)
        if auto_tile.get("ok"):
            roi["auto_saved_tile_path"] = auto_tile.get("path")

    next_rank_hint = _next_unattempted_candidate_rank()
    roi["next_candidate_rank_hint"] = next_rank_hint
    roi["next_action_hint"] = (
        "You just requested a high-power ROI. "
        + (
            "In AML mode, discard only if this ROI is clearly bad on review, such as mostly background/empty glass. "
            "If it is borderline but interpretable, keep it and continue searching for additional ROIs. "
            if _agent_is_aml()
            else
            "Carefully inspect the newly shown ROI image in the conversation. "
            "If it is mostly background, out of focus, or not diagnostic, "
            "very next step should be to call wsi_discard_last_roi with a brief nav_reason. "
        )
        + (
            f"If you still need more evidence, use candidate #{next_rank_hint} next instead of continuing local search inside this ROI. "
            if next_rank_hint is not None
            else
            "If you still need more evidence, jump directly to the next unvisited candidate instead of continuing local search inside this ROI. "
        )
    )

    print(f"[WSI][ROI_NORM] Marked ROI {roi_id}: {label} (importance={importance})")
    _log_step("wsi_mark_roi_norm", nav_reason, info)

    # Do NOT auto-advance to next candidate. Allow agent to mark multiple ROIs within current subregion if strong.
    # Agent should explicitly call wsi_open_candidate(rank) to move to next candidate when current one is exhausted.
    roi["guidance_for_next_action"] = (
        f"ROI #{roi_id} has been marked and saved. You can now:\n"
        f"1) Continue searching WITHIN THIS CANDIDATE REGION for additional strong ROIs by using wsi_zoom_current_norm to explore other sub-areas.\n"
        f"2) When this candidate region is exhausted (no more strong tissue), call wsi_open_candidate(rank={next_rank_hint}) to move to the next candidate.\n"
        f"If you just discarded an ROI, you can search for a different position in this region OR jump to the next candidate."
    )

    _make_overview_with_current_box(draw_current_box=True)
    # Do not auto clear candidate cache - allow continued exploration within current region

    return roi


def _open_candidate_by_rank(
    *,
    rank: int,
    nav_reason: str,
    max_dim: int,
) -> Dict[str, Any]:
    if _roi_cap_reached():
        return _roi_cap_response()
    if not state._current_view and not getattr(state, "_overview_roi_candidates", None):
        raise RuntimeError("wsi_open_candidate called before wsi_get_overview_view.")

    chosen = _candidate_by_rank(rank)
    if chosen is None:
        return _invalid_candidate_rank_response(rank)

    center_level0 = chosen.get("center_level0")
    if not center_level0 or len(center_level0) < 2:
        return {
            "ok": False,
            "reason": "candidate_center_missing",
            "message": f"Candidate #{rank} does not have a usable level-0 center.",
        }

    slide = _load_slide()
    slide_w0, slide_h0 = slide.level_dimensions[0]
    mpp = _effective_slide_mpp_um(slide)
    target_field_um = float(chosen.get("navigation_field_width_um") or _selected_candidate_nav_field_um())
    target_px = max(1, int(round(target_field_um / max(1e-6, mpp))))
    cx_base = int(round(float(center_level0[0])))
    cy_base = int(round(float(center_level0[1])))
    x0_new = cx_base - target_px // 2
    y0_new = cy_base - target_px // 2
    w_new = target_px
    h_new = target_px
    x0_new = max(0, min(x0_new, slide_w0 - w_new))
    y0_new = max(0, min(y0_new, slide_h0 - h_new))
    x0_new, y0_new, w_new, h_new, min_zoom_floor_applied = _enforce_min_roi_view_size(
        x0=x0_new,
        y0=y0_new,
        w=w_new,
        h=h_new,
        slide=slide,
    )

    info = _render_view_from_base_bbox(
        x0=x0_new,
        y0=y0_new,
        w=w_new,
        h=h_new,
        max_dim=min(max_dim, MAX_IMG_DIM),
        tag="zoom",
    )
    if min_zoom_floor_applied:
        info["zoom_floor_applied"] = True
        info["zoom_floor_warning"] = (
            "Requested zoom was smaller than the minimum ROI inspection size, so the system expanded it "
            "back to ROI scale. Do not zoom smaller than the ROI crop; if the candidate is centered, use "
            "wsi_mark_candidate or wsi_mark_roi_norm instead."
        )
    info["candidate_navigation_mode"] = "free_local_search"
    info["candidate_search_guidance"] = (
        "CRITICAL: This is a candidate-search region centered on a promising area. Systematically zoom into different "
        "sub-areas (corners, edges, quadrants) using wsi_zoom_current_norm to find ALL strong cellular, high-density subregions. "
        "Mark ROIs at EACH strong position you discover. If this candidate region contains multiple distinct high-quality areas, "
        "mark multiple ROIs here before moving to the next candidate. After marking each ROI, continue zooming to different areas "
        "of this region to search for additional strong ROIs. Do NOT mark ROI at this view's center - mark it where you find the "
        "best tissue via systematic zooming. Only call wsi_open_candidate(rank) to move to the next candidate when this region is exhausted."
    )
    if isinstance(state._current_view, dict):
        state._current_view["candidate_navigation_mode"] = "free_local_search"
    info = _attach_roi_candidates(info)
    info = _strip_candidate_payload_for_free_local_search(info)
    state.CURRENT_AGENT_ACTION = f"Opened candidate #{rank}: {nav_reason}" if nav_reason else f"Opened candidate #{rank}"
    _log_step("wsi_open_candidate", nav_reason, info)
    return info




@function_tool
def wsi_get_overview_view(
    nav_reason: str = "Initial overview of the whole slide",
    max_dim: int = MAX_IMG_DIM,
) -> str:
    def _inner(nav_reason: str, max_dim: int) -> Dict[str, Any]:
        if _roi_cap_reached():
            return _roi_cap_response()
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
        _remember_overview_candidate_bank()
        state.CURRENT_AGENT_ACTION = f"Overview: {nav_reason}" if nav_reason else "Viewing overview"
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
        if _roi_cap_reached():
            return _roi_cap_response()
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
        x0_new, y0_new, w_new, h_new, min_zoom_floor_applied = _enforce_min_roi_view_size(
            x0=x0_new,
            y0=y0_new,
            w=w_new,
            h=h_new,
            slide=slide,
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
        if min_zoom_floor_applied:
            info["zoom_floor_applied"] = True
            info["zoom_floor_warning"] = (
                "Requested zoom was smaller than the minimum ROI inspection size, so the system expanded it "
                "back to ROI scale. Do not zoom smaller than the ROI crop; if the candidate is centered, use "
                "wsi_mark_roi_norm instead."
            )

        info = _attach_roi_candidates(info, skip_refresh=True)
        info = _strip_candidate_payload_for_free_local_search(info)
        state.CURRENT_AGENT_ACTION = f"Zooming in: {nav_reason}" if nav_reason else "Zooming in"
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
        if _roi_cap_reached():
            return _roi_cap_response()
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
        x0_new, y0_new, w_new, h_new, min_zoom_floor_applied = _enforce_min_roi_view_size(
            x0=x0_new,
            y0=y0_new,
            w=w_new,
            h=h_new,
            slide=slide,
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
        if min_zoom_floor_applied:
            info["zoom_floor_applied"] = True
            info["zoom_floor_warning"] = (
                "Requested zoom was smaller than the minimum ROI inspection size, so the system expanded it "
                "back to ROI scale. Do not zoom smaller than the ROI crop; if the candidate is centered, use "
                "wsi_mark_roi_norm instead."
            )

        info = _attach_roi_candidates(info, skip_refresh=True)
        info = _strip_candidate_payload_for_free_local_search(info)
        state.CURRENT_AGENT_ACTION = f"Zooming overview: {nav_reason}" if nav_reason else "Zooming overview"
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
        if _roi_cap_reached():
            return _roi_cap_response()
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
        info = _attach_roi_candidates(info, skip_refresh=True)
        info = _strip_candidate_payload_for_free_local_search(info)
        state.CURRENT_AGENT_ACTION = f"Panning: {nav_reason}" if nav_reason else "Panning"
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
        info = _strip_candidate_payload_for_free_local_search(info)
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
        if _roi_cap_reached():
            return _roi_cap_response()
        if not state._current_view:
            raise RuntimeError("wsi_mark_roi_norm called before wsi_get_overview_view.")

        # Always refresh candidate ranking on the active view before marking ROI.
        _refresh_roi_candidates_for_current_view(top_k=ROI_CANDIDATE_TOP_K)

        x0_999_cl = max(0, min(999, x0_999))
        x1_999_cl = max(0, min(999, x1_999))
        y0_999_cl = max(0, min(999, y0_999))
        y1_999_cl = max(0, min(999, y1_999))

        requested_cx_999 = (x0_999_cl + x1_999_cl) / 2.0
        requested_cy_999 = (y0_999_cl + y1_999_cl) / 2.0
        cv_x0 = state._current_view["x0"]
        cv_y0 = state._current_view["y0"]
        cv_w = state._current_view["w"]
        cv_h = state._current_view["h"]
        cx_base = cv_x0 + int(round((requested_cx_999 / 999.0) * cv_w))
        cy_base = cv_y0 + int(round((requested_cy_999 / 999.0) * cv_h))
        chosen_for_mark = {
            "rank": None,
            "score": None,
            "center_norm": [int(round(requested_cx_999)), int(round(requested_cy_999))],
            "center_level0": [cx_base, cy_base],
        }
        return _mark_roi_from_candidate(
            chosen=chosen_for_mark,
            label=label,
            note=note,
            importance=importance,
            nav_reason=nav_reason,
            requested_center_norm=[
                int(round(requested_cx_999)),
                int(round(requested_cy_999)),
            ],
        )

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
def wsi_open_candidate(
    rank: int,
    nav_reason: str = "Open ROI candidate by rank",
    max_dim: int = MAX_IMG_DIM,
) -> str:
    def _inner(rank: int, nav_reason: str, max_dim: int) -> Dict[str, Any]:
        return _open_candidate_by_rank(
            rank=int(rank),
            nav_reason=nav_reason,
            max_dim=max_dim,
        )

    return _safe(
        _inner,
        rank=rank,
        nav_reason=nav_reason,
        max_dim=max_dim,
    )


@function_tool
def wsi_mark_candidate(
    rank: int,
    label: str,
    note: str = "",
    importance: int = 1,
    nav_reason: str = "Mark ROI candidate by rank",
) -> str:
    def _inner(
        rank: int,
        label: str,
        note: str,
        importance: int,
        nav_reason: str,
    ) -> Dict[str, Any]:
        if _roi_cap_reached():
            return _roi_cap_response()
        if not state._current_view:
            raise RuntimeError("wsi_mark_candidate called before wsi_get_overview_view.")
        chosen = _candidate_by_rank(int(rank))
        if chosen is None:
            return _invalid_candidate_rank_response(int(rank))
        center_norm = chosen.get("center_norm") or [0, 0]
        return _mark_roi_from_candidate(
            chosen=chosen,
            label=label,
            note=note,
            importance=importance,
            nav_reason=nav_reason,
            requested_center_norm=[
                int(center_norm[0]),
                int(center_norm[1]),
            ],
        )

    return _safe(
        _inner,
        rank=rank,
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
        mpp = _effective_slide_mpp_um(slide)
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

        # Invalidate HNSW cache so new tiles are included in future retrievals
        try:
            from wsi_core_pkg.embeddings.roi_ranker import _clear_reference_hnsw_cache
            _clear_reference_hnsw_cache()
        except Exception:
            pass  # Non-critical - cache will rebuild naturally

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
        roi = state._roi_marks[-1]
        allow_discard, block_message = _allow_discard_last_roi(roi)
        if not allow_discard:
            return {
                "ok": False,
                "reason": "discard_blocked_near_target",
                "message": block_message,
                "kept_roi_count": len(state._roi_marks),
                "target_accepted_rois": _selected_target_accepted_rois(),
            }
        roi = state._roi_marks.pop()
        auto_tile_path = str(roi.get("auto_saved_tile_path") or "")
        if auto_tile_path:
            state._saved_good_tiles = [
                record for record in state._saved_good_tiles
                if str(record.get("path") or "") != auto_tile_path
            ]
            try:
                if os.path.exists(auto_tile_path):
                    os.remove(auto_tile_path)
            except Exception:
                pass
        next_rank_hint = _next_unattempted_candidate_rank()
        state.CURRENT_AGENT_ACTION = f"Discarded ROI #{roi['roi_id']}: {roi['label']}"
        print(f"[WSI][ROI] Discarded ROI {roi['roi_id']}: {roi['label']}")
        _log_step(
            "wsi_discard_last_roi",
            nav_reason,
            {"view_bbox_level0": roi.get("view_bbox_level0")},
        )
        response = {
            "ok": True,
            "discarded_roi_id": roi["roi_id"],
            "label": roi["label"],
            "message": f"ROI discarded.",
        }
        if next_rank_hint is not None:
            response["next_candidate_rank_hint"] = next_rank_hint
            response["message"] += f" Next candidate available: #{next_rank_hint}."
        return response

    return _safe(_inner, nav_reason=nav_reason)


@function_tool
def wsi_rebuild_reference_index(
    include_saved_tiles: bool = True,
    progress_message: str = "Rebuilding reference index with curated tiles",
) -> Dict[str, Any]:
    """Rebuild the HNSW reference index including newly saved tiles.

    This function incorporates tiles saved via wsi_save_tile_norm into the
    reference prototype bank, making them immediately available for retrieval.

    Args:
        include_saved_tiles: If True, include tiles saved in this session.
                            If False, only clear cache (will rebuild on next use).
        progress_message: Optional message for progress tracking.

    Returns:
        Dictionary with status and statistics about the rebuild operation.
    """
    from wsi_core_pkg.embeddings.roi_ranker import (
        _clear_reference_embeddings_cache,
        _clear_reference_hnsw_cache,
        _save_reference_hnsw_cache,
        _save_reference_embeddings_to_cache,
        _embed_reference_tiles,
    )
    import torch

    def _inner(include_saved_tiles: bool, progress_message: str) -> Dict[str, Any]:
        # Clear existing cache
        _clear_reference_hnsw_cache()
        _clear_reference_embeddings_cache()

        if not include_saved_tiles:
            return {
                "ok": True,
                "action": "cache_cleared",
                "message": "Reference cache cleared. Will rebuild on next retrieval.",
            }

        # Collect saved tiles from current session
        saved_records = []
        for record in state._saved_good_tiles:
            saved_records.append((Path(record["path"]), "good"))
        for record in state._saved_bad_tiles:
            saved_records.append((Path(record["path"]), "bad"))

        if not saved_records:
            return {
                "ok": True,
                "action": "no_saved_tiles",
                "message": "No saved tiles in current session. Cache cleared, will use EXAMPLE_TILES_ROOT on next retrieval.",
                "saved_good_count": len(state._saved_good_tiles),
                "saved_bad_count": len(state._saved_bad_tiles),
            }

        # Embed saved tiles
        try:
            extractor = get_embedding_extractor(getattr(state, "EXTRACTOR_NAME", "uni2"))
            run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            extractor.model = extractor.model.to(run_device)
            extractor.model.eval()

            ref_feat_l2, ref_labels, ref_paths = _embed_reference_tiles(
                records=saved_records,
                extractor=extractor,
                device=run_device,
                batch_size=32,
                extractor_id=extractor.identifier,
            )

            # Save canonical reference embedding cache and HNSW cache
            embedding_cache_dir = _save_reference_embeddings_to_cache(
                ref_feat_l2, ref_labels, ref_paths, extractor.identifier
            )
            hnsw_cache_dir = _save_reference_hnsw_cache(
                ref_feat_l2, ref_labels, ref_paths, extractor.identifier
            )

            return {
                "ok": True,
                "action": "rebuilt_with_saved_tiles",
                "message": f"Reference index rebuilt with {len(saved_records)} saved tiles.",
                "saved_good_count": len(state._saved_good_tiles),
                "saved_bad_count": len(state._saved_bad_tiles),
                "embedding_cache_dir": str(embedding_cache_dir) if embedding_cache_dir else None,
                "hnsw_cache_dir": str(hnsw_cache_dir) if hnsw_cache_dir else None,
            }

        except Exception as e:
            return {
                "ok": False,
                "action": "rebuild_failed",
                "error": str(e),
                "message": "Failed to rebuild index. Will use fallback exact search.",
                "saved_good_count": len(state._saved_good_tiles),
                "saved_bad_count": len(state._saved_bad_tiles),
            }

    return _safe(_inner, include_saved_tiles=include_saved_tiles, progress_message=progress_message)
