"""Compatibility layer for the refactored WSI core package.

The implementation now lives in ``wsi_core_pkg``.
"""

from wsi_core_pkg import (
    DEBUG_ROOT_DIR,
    MODEL_NAME,
    OUTPUTS_ROOT_DIR,
    REPORT_ROOT_DIR,
    clear_wsi_outputs_state,
    detect_dark_regions,
    get_public_state_snapshot,
    run_wsi_agent_for_web,
)

__all__ = [
    "run_wsi_agent_for_web",
    "get_public_state_snapshot",
    "OUTPUTS_ROOT_DIR",
    "DEBUG_ROOT_DIR",
    "REPORT_ROOT_DIR",
    "detect_dark_regions",
    "MODEL_NAME",
    "clear_wsi_outputs_state",
]
