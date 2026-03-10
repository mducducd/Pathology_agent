import os
from typing import Any, Dict, List, Optional

import openslide

from .config import DEBUG_ROOT_DIR, DEFAULT_SLIDE_PATH, OUTPUTS_ROOT_DIR

# ---------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------

SLIDE_PATH = DEFAULT_SLIDE_PATH
RUN_ID: Optional[str] = None
AGENT_TYPE: str = "wsi"

_slide: Optional[openslide.AbstractSlide] = None

_current_view: Dict[str, Any] = {}
_overview_cache: Dict[str, Any] = {}

_last_overview_with_box_path: Optional[str] = None
_last_overview_debug_path: Optional[str] = None

_view_history: List[Dict[str, Any]] = []
_step_log: List[Dict[str, Any]] = []
_roi_marks: List[Dict[str, Any]] = []

_saved_good_tiles: List[Dict[str, Any]] = []
_saved_bad_tiles: List[Dict[str, Any]] = []

_example_tiles_injected = False
_example_rois_injected = False

_roi_ranker_index: Optional[Any] = None
_roi_ranker_meta: Dict[str, Any] = {}
_roi_candidate_prep: Dict[str, Any] = {}
_last_roi_candidates: List[Dict[str, Any]] = []
_last_roi_candidate_source: Optional[str] = None
_last_roi_candidate_overlay_path: Optional[str] = None

TRACE_DIR: Optional[str] = None
TRACE_FILE_PATH: Optional[str] = None

DEBUG_SAVE_DIR = DEBUG_ROOT_DIR
_debug_img_counter = 0

HAS_FATAL_ERROR = False
LAST_FATAL_ERROR: Optional[str] = None


def reset_wsi_state(run_id: str) -> None:
    global RUN_ID, AGENT_TYPE, _debug_img_counter, DEBUG_SAVE_DIR
    global _step_log, _roi_marks, _view_history
    global _current_view, _overview_cache, _last_overview_with_box_path, _last_overview_debug_path
    global _slide, _saved_good_tiles, _saved_bad_tiles, _example_tiles_injected, _example_rois_injected
    global _roi_ranker_index, _roi_ranker_meta, _roi_candidate_prep
    global _last_roi_candidates, _last_roi_candidate_source, _last_roi_candidate_overlay_path
    global TRACE_DIR, TRACE_FILE_PATH
    global HAS_FATAL_ERROR, LAST_FATAL_ERROR

    RUN_ID = run_id
    AGENT_TYPE = "wsi"

    _debug_img_counter = 0
    DEBUG_SAVE_DIR = os.path.join(DEBUG_ROOT_DIR, RUN_ID, "wsi_debug")
    os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)

    TRACE_DIR = os.path.join(OUTPUTS_ROOT_DIR, RUN_ID, "traces")
    os.makedirs(TRACE_DIR, exist_ok=True)
    TRACE_FILE_PATH = os.path.join(TRACE_DIR, "trace.jsonl")

    _step_log = []
    _roi_marks = []
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
    _last_roi_candidate_source = None
    _last_roi_candidate_overlay_path = None
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
    global AGENT_TYPE, _step_log, _roi_marks, _view_history
    global _current_view, _overview_cache, _last_overview_with_box_path, _last_overview_debug_path
    global _saved_good_tiles, _saved_bad_tiles, _example_tiles_injected, _example_rois_injected
    global _roi_ranker_index, _roi_ranker_meta, _roi_candidate_prep
    global _last_roi_candidates, _last_roi_candidate_source, _last_roi_candidate_overlay_path
    global HAS_FATAL_ERROR, LAST_FATAL_ERROR

    _step_log = []
    AGENT_TYPE = "wsi"
    _roi_marks = []
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
    _last_roi_candidate_source = None
    _last_roi_candidate_overlay_path = None
    HAS_FATAL_ERROR = False
    LAST_FATAL_ERROR = None


def get_public_state_snapshot() -> Dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "agent_type": AGENT_TYPE,
        "current_view": dict(_current_view) if _current_view else None,
        "overview_cache": dict(_overview_cache) if _overview_cache else None,
        "step_log": list(_step_log),
        "roi_marks": list(_roi_marks),
        "last_overview_with_box_path": _last_overview_with_box_path,
        "last_overview_debug_path": _last_overview_debug_path,
        "last_roi_candidates": list(_last_roi_candidates),
        "last_roi_candidate_source": _last_roi_candidate_source,
        "roi_candidate_prep": dict(_roi_candidate_prep) if _roi_candidate_prep else None,
        "last_roi_candidate_overlay_path": _last_roi_candidate_overlay_path,
        "has_fatal_error": HAS_FATAL_ERROR,
        "last_fatal_error": LAST_FATAL_ERROR,
    }
