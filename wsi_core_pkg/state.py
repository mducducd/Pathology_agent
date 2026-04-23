import os
from typing import Any, Dict, List, Optional

import openslide

from .config import DEBUG_ROOT_DIR, DEFAULT_MPP_UM, DEFAULT_SLIDE_PATH, MAX_IMG_DIM, OUTPUTS_ROOT_DIR

# ---------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------

SLIDE_PATH = DEFAULT_SLIDE_PATH
RUN_ID: Optional[str] = None
AGENT_TYPE: str = "aml"
EXTRACTOR_NAME: str = "reddino_base"
TILE_SIZE_UM: float = 256.0
TILE_SIZE_PX: int = 224
BATCH_SIZE: int = 128
TILE_PREFILTER_METHOD: str = "quality"
ROI_OUTPUT_SIZE_PX: int = 1024
MAX_ACCEPTED_ROIS: int = 10
TARGET_ACCEPTED_ROIS: int = 5
DEFAULT_MPP_UM_OVERRIDE: float | None = None
QUALITY_METHOD: str = "embedding"

_slide: Optional[openslide.AbstractSlide] = None

_current_view: Dict[str, Any] = {}
_overview_cache: Dict[str, Any] = {}

_last_overview_with_box_path: Optional[str] = None
_last_overview_debug_path: Optional[str] = None

_view_history: List[Dict[str, Any]] = []
_step_log: List[Dict[str, Any]] = []
_roi_marks: List[Dict[str, Any]] = []
_attempted_roi_bboxes_level0: List[List[int]] = []

_saved_good_tiles: List[Dict[str, Any]] = []
_saved_bad_tiles: List[Dict[str, Any]] = []

_example_tiles_injected = False
_example_rois_injected = False

_roi_ranker_index: Optional[Any] = None
_roi_ranker_meta: Dict[str, Any] = {}
_roi_candidate_prep: Dict[str, Any] = {}
_last_roi_candidates: List[Dict[str, Any]] = []
_last_roi_candidate_meta: Dict[str, Any] = {}
_last_roi_candidate_source: Optional[str] = None
_last_roi_candidate_overlay_path: Optional[str] = None
_last_roi_candidate_view_key: Optional[Any] = None
_last_roi_candidate_top_k: Optional[int] = None
_overview_roi_candidates: List[Dict[str, Any]] = []
_dark_region_boxes_level0: List[Dict[str, Any]] = []
_dark_region_slide_path: Optional[str] = None
_dark_region_cache_signature: Optional[Any] = None

TRACE_DIR: Optional[str] = None
TRACE_FILE_PATH: Optional[str] = None

DEBUG_SAVE_DIR = DEBUG_ROOT_DIR
_debug_img_counter = 0

HAS_FATAL_ERROR = False
LAST_FATAL_ERROR: Optional[str] = None


def _loaded_slide_mpp_um() -> float | None:
    if _slide is None:
        return None
    props = getattr(_slide, "properties", {}) or {}
    for key in ("openslide.mpp-x", "openslide.mpp-y", "aperio.MPP"):
        value = props.get(key)
        if value:
            try:
                parsed = float(value)
            except Exception:
                continue
            if parsed > 0:
                return parsed
    obj = props.get("openslide.objective-power")
    if obj:
        try:
            parsed_obj = float(obj)
        except Exception:
            parsed_obj = None
        if parsed_obj and parsed_obj > 0:
            return 10.0 / parsed_obj
    return None


def reset_wsi_state(
    run_id: str,
    extractor_name: str = "reddino_base",
    tile_size_um: float = 256.0,
    tile_size_px: int = 224,
    batch_size: int = 128,
    tile_prefilter_method: str = "quality",
    roi_output_size_px: int = 1024,
    max_accepted_rois: int = 10,
    target_accepted_rois: int = 5,
    default_mpp_um: float | None = None,
    quality_method: str = "embedding",
) -> None:
    global RUN_ID, AGENT_TYPE, EXTRACTOR_NAME, TILE_SIZE_UM, TILE_SIZE_PX, BATCH_SIZE, TILE_PREFILTER_METHOD, ROI_OUTPUT_SIZE_PX, MAX_ACCEPTED_ROIS, TARGET_ACCEPTED_ROIS, DEFAULT_MPP_UM_OVERRIDE, QUALITY_METHOD, _debug_img_counter, DEBUG_SAVE_DIR
    global _step_log, _roi_marks, _attempted_roi_bboxes_level0, _view_history
    global _current_view, _overview_cache, _last_overview_with_box_path, _last_overview_debug_path
    global _slide, _saved_good_tiles, _saved_bad_tiles, _example_tiles_injected, _example_rois_injected
    global _roi_ranker_index, _roi_ranker_meta, _roi_candidate_prep
    global _last_roi_candidates, _last_roi_candidate_meta, _last_roi_candidate_source, _last_roi_candidate_overlay_path
    global _last_roi_candidate_view_key, _last_roi_candidate_top_k, _overview_roi_candidates
    global _dark_region_boxes_level0, _dark_region_slide_path, _dark_region_cache_signature
    global TRACE_DIR, TRACE_FILE_PATH
    global HAS_FATAL_ERROR, LAST_FATAL_ERROR

    RUN_ID = run_id
    AGENT_TYPE = "aml"
    EXTRACTOR_NAME = extractor_name
    TILE_SIZE_UM = tile_size_um
    TILE_SIZE_PX = tile_size_px
    BATCH_SIZE = int(batch_size)
    TILE_PREFILTER_METHOD = str(tile_prefilter_method or "quality").strip().lower()
    ROI_OUTPUT_SIZE_PX = max(640, int(roi_output_size_px or 1024))
    MAX_ACCEPTED_ROIS = max(1, int(max_accepted_rois or 10))
    TARGET_ACCEPTED_ROIS = min(MAX_ACCEPTED_ROIS, max(1, int(target_accepted_rois or 5)))
    DEFAULT_MPP_UM_OVERRIDE = max(1e-6, float(default_mpp_um)) if default_mpp_um is not None else None
    QUALITY_METHOD = str(quality_method or "embedding").strip().lower()

    _debug_img_counter = 0
    DEBUG_SAVE_DIR = os.path.join(DEBUG_ROOT_DIR, RUN_ID, "wsi_debug")
    os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)

    TRACE_DIR = os.path.join(OUTPUTS_ROOT_DIR, RUN_ID, "traces")
    os.makedirs(TRACE_DIR, exist_ok=True)
    TRACE_FILE_PATH = os.path.join(TRACE_DIR, "trace.jsonl")

    _step_log = []
    _roi_marks = []
    _attempted_roi_bboxes_level0 = []
    _view_history = []
    _current_view = {}
    _overview_cache = {}
    _last_overview_with_box_path = None
    _last_overview_debug_path = None
    _saved_good_tiles = []
    _saved_bad_tiles = []
    _example_tiles_injected = False
    _example_rois_injected = False
    _roi_ranker_index = None
    _roi_ranker_meta = {}
    _roi_candidate_prep = {}
    _last_roi_candidates = []
    _last_roi_candidate_meta = {}
    _last_roi_candidate_source = None
    _last_roi_candidate_overlay_path = None
    _last_roi_candidate_view_key = None
    _last_roi_candidate_top_k = None
    _overview_roi_candidates = []
    _dark_region_boxes_level0 = []
    _dark_region_slide_path = None
    _dark_region_cache_signature = None
    HAS_FATAL_ERROR = False
    LAST_FATAL_ERROR = None

    if _slide is not None:
        try:
            _slide.close()
        except Exception:
            pass
    _slide = None


def set_slide_path(path: str) -> None:
    global SLIDE_PATH
    SLIDE_PATH = os.path.abspath(path)


def clear_wsi_outputs_state() -> None:
    """
    Clear generated WSI run outputs shown in UI without changing run id/slide path.
    """
    global AGENT_TYPE, BATCH_SIZE, TILE_PREFILTER_METHOD, ROI_OUTPUT_SIZE_PX, MAX_ACCEPTED_ROIS, TARGET_ACCEPTED_ROIS, DEFAULT_MPP_UM_OVERRIDE, QUALITY_METHOD, _step_log, _roi_marks, _attempted_roi_bboxes_level0, _view_history
    global _current_view, _overview_cache, _last_overview_with_box_path, _last_overview_debug_path
    global _saved_good_tiles, _saved_bad_tiles, _example_tiles_injected, _example_rois_injected
    global _roi_ranker_index, _roi_ranker_meta, _roi_candidate_prep
    global _last_roi_candidates, _last_roi_candidate_meta, _last_roi_candidate_source, _last_roi_candidate_overlay_path
    global _last_roi_candidate_view_key, _last_roi_candidate_top_k, _overview_roi_candidates
    global _dark_region_boxes_level0, _dark_region_slide_path, _dark_region_cache_signature
    global HAS_FATAL_ERROR, LAST_FATAL_ERROR

    _step_log = []
    AGENT_TYPE = "wsi"
    BATCH_SIZE = 128
    TILE_PREFILTER_METHOD = "quality"
    ROI_OUTPUT_SIZE_PX = 1024
    MAX_ACCEPTED_ROIS = 10
    TARGET_ACCEPTED_ROIS = 5
    DEFAULT_MPP_UM_OVERRIDE = None
    QUALITY_METHOD = "embedding"
    _roi_marks = []
    _attempted_roi_bboxes_level0 = []
    _view_history = []
    _current_view = {}
    _overview_cache = {}
    _last_overview_with_box_path = None
    _last_overview_debug_path = None
    _saved_good_tiles = []
    _saved_bad_tiles = []
    _example_tiles_injected = False
    _example_rois_injected = False
    _roi_ranker_index = None
    _roi_ranker_meta = {}
    _roi_candidate_prep = {}
    _last_roi_candidates = []
    _last_roi_candidate_meta = {}
    _last_roi_candidate_source = None
    _last_roi_candidate_overlay_path = None
    _last_roi_candidate_view_key = None
    _last_roi_candidate_top_k = None
    _overview_roi_candidates = []
    _dark_region_boxes_level0 = []
    _dark_region_slide_path = None
    _dark_region_cache_signature = None
    HAS_FATAL_ERROR = False
    LAST_FATAL_ERROR = None


def get_public_state_snapshot() -> Dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "agent_type": AGENT_TYPE,
        "current_view": dict(_current_view) if _current_view else None,
        "batch_size": BATCH_SIZE,
        "tile_prefilter_method": TILE_PREFILTER_METHOD,
        "roi_output_size_px": ROI_OUTPUT_SIZE_PX,
        "max_accepted_rois": MAX_ACCEPTED_ROIS,
        "target_accepted_rois": TARGET_ACCEPTED_ROIS,
        "default_mpp_um": DEFAULT_MPP_UM_OVERRIDE,
        "slide_mpp_um": _loaded_slide_mpp_um(),
        "overview_cache": dict(_overview_cache) if _overview_cache else None,
        "step_log": list(_step_log),
        "roi_marks": list(_roi_marks),
        "attempted_roi_bboxes_level0": list(_attempted_roi_bboxes_level0),
        "last_overview_with_box_path": _last_overview_with_box_path,
        "last_overview_debug_path": _last_overview_debug_path,
        "last_roi_candidates": list(_last_roi_candidates),
        "overview_roi_candidates": list(_overview_roi_candidates),
        "last_roi_candidate_meta": dict(_last_roi_candidate_meta) if _last_roi_candidate_meta else None,
        "last_roi_candidate_source": _last_roi_candidate_source,
        "roi_candidate_prep": dict(_roi_candidate_prep) if _roi_candidate_prep else None,
        "last_roi_candidate_overlay_path": _last_roi_candidate_overlay_path,
        "has_fatal_error": HAS_FATAL_ERROR,
        "last_fatal_error": LAST_FATAL_ERROR,
    }
