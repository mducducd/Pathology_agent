from .config import DEBUG_ROOT_DIR, MODEL_NAME, OUTPUTS_ROOT_DIR, REPORT_ROOT_DIR
from .context_injection import install_chat_patches
from .dark_regions import detect_dark_regions
from .runtime import run_wsi_agent_for_web
from .state import clear_wsi_outputs_state, get_public_state_snapshot, reset_wsi_state, set_slide_path

install_chat_patches()

__all__ = [
    "run_wsi_agent_for_web",
    "get_public_state_snapshot",
    "detect_dark_regions",
    "reset_wsi_state",
    "clear_wsi_outputs_state",
    "set_slide_path",
    "MODEL_NAME",
    "OUTPUTS_ROOT_DIR",
    "DEBUG_ROOT_DIR",
    "REPORT_ROOT_DIR",
]
