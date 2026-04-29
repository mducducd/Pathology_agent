import os

from dotenv import load_dotenv
from .tuning_config import tuning_value
from openai import AsyncOpenAI, OpenAI

from agents import (
    enable_verbose_stdout_logging,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

load_dotenv()

DEFAULT_SLIDE_PATH = os.path.abspath(os.getenv("DEFAULT_SLIDE_PATH") or str(tuning_value("agent", "DEFAULT_SLIDE_PATH") or "341476.svs"))
MODEL_NAME = os.getenv("MODEL_NAME") or str(tuning_value("agent", "MODEL_NAME") or "GPT-OSS-120B")

_api_key = os.getenv("OPENAI_API_KEY") or str(tuning_value("agent", "OPENAI_API_KEY") or "local")
_api_base = os.getenv("OPENAI_API_BASE") or str(tuning_value("agent", "OPENAI_API_BASE") or "http://pluto/v1")

client_async = AsyncOpenAI(api_key=_api_key, base_url=_api_base)
client_sync = OpenAI(api_key=_api_key, base_url=_api_base)

set_default_openai_client(client_async)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

MAX_IMG_DIM = int(os.getenv("MAX_IMG_DIM", "") or tuning_value("agent", "MAX_IMG_DIM") or 1024)
MAX_NATIVE_VIEW_DIM = int(os.getenv("MAX_NATIVE_VIEW_DIM", "") or tuning_value("agent", "MAX_NATIVE_VIEW_DIM") or 4096)
MAX_TURNS = int(os.getenv("MAX_TURNS", "") or tuning_value("agent", "MAX_TURNS") or 140)
def _tv_bool(section: str, key: str, default: bool) -> bool:
    try:
        v = tuning_value(section, key)
        if isinstance(v, bool):
            return v
        return str(v).lower() not in ("0", "false", "no", "")
    except Exception:
        return default

WSI_AGENT_TEMPERATURE = float(os.getenv("WSI_AGENT_TEMPERATURE") or tuning_value("agent", "WSI_AGENT_TEMPERATURE") or 0.9)
ENABLE_THINKING = _tv_bool("agent", "ENABLE_THINKING", False)

TILE_SIZE_UM = float(os.getenv("TILE_SIZE_UM", "") or tuning_value("agent", "TILE_SIZE_UM") or 256.0)
TILE_PX = int(os.getenv("TILE_PX", "") or tuning_value("agent", "TILE_PX") or 224)
MAX_GOOD_TILES = int(os.getenv("MAX_GOOD_TILES", "") or tuning_value("agent", "MAX_GOOD_TILES") or 200)
MAX_BAD_TILES = int(os.getenv("MAX_BAD_TILES", "") or tuning_value("agent", "MAX_BAD_TILES") or 50)
DEFAULT_MPP_UM = float(tuning_value("tools.slide", "DEFAULT_MPP_UM"))

OUTPUTS_ROOT_DIR = os.path.abspath(os.getenv("OUTPUTS_ROOT_DIR") or str(tuning_value("agent", "OUTPUTS_ROOT_DIR") or "./outputs"))
os.makedirs(OUTPUTS_ROOT_DIR, exist_ok=True)

DEBUG_ROOT_DIR = OUTPUTS_ROOT_DIR
os.makedirs(DEBUG_ROOT_DIR, exist_ok=True)

REPORT_ROOT_DIR = OUTPUTS_ROOT_DIR
os.makedirs(REPORT_ROOT_DIR, exist_ok=True)

SELECTED_TILES_ROOT = OUTPUTS_ROOT_DIR
os.makedirs(SELECTED_TILES_ROOT, exist_ok=True)

EXAMPLE_TILES_ROOT = os.path.abspath(os.getenv("EXAMPLE_TILES_ROOT") or str(tuning_value("agent", "EXAMPLE_TILES_ROOT") or "./Selected_Tiles"))
EXAMPLE_TILES_GOOD_DIR = os.path.join(EXAMPLE_TILES_ROOT, "Good_Tiles")
EXAMPLE_TILES_BAD_DIR = os.path.join(EXAMPLE_TILES_ROOT, "Bad_Tiles")
EXAMPLE_TILES_MAX_PER_CLASS = int(os.getenv("EXAMPLE_TILES_MAX_PER_CLASS") or tuning_value("agent", "EXAMPLE_TILES_MAX_PER_CLASS") or 2)

_example_rois_root = os.path.abspath(os.getenv("EXAMPLE_ROIS_ROOT") or str(tuning_value("agent", "EXAMPLE_ROIS_ROOT") or "./Example_ROIs"))
EXAMPLE_ROIS_POS_DIR = os.path.join(_example_rois_root, "ROI")
EXAMPLE_ROIS_NEG_DIR = os.path.join(_example_rois_root, "Non_ROI")
EXAMPLE_ROIS_MAX_PER_CLASS = int(os.getenv("EXAMPLE_ROIS_MAX_PER_CLASS") or tuning_value("agent", "EXAMPLE_ROIS_MAX_PER_CLASS") or 2)

def _tv_int(section: str, key: str, default: int) -> int:
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

CONTEXT_PREVIOUS_VIEWS_MAX = _tv_int("context_injection.candidates", "CONTEXT_PREVIOUS_VIEWS_MAX", 0)
CONTEXT_ROI_CANDIDATE_LINES_MAX = _tv_int("context_injection.candidates", "CONTEXT_ROI_CANDIDATE_LINES_MAX", 8)
