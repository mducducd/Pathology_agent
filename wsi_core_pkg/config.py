import os
from typing import Callable, TypeVar

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from agents import (
    enable_verbose_stdout_logging,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

from .tuning_config import tuning_value

load_dotenv()

# ---------- helpers ----------

T = TypeVar("T")


def _cfg(key: str, section: str, default: T, cast: Callable[[object], T] = str) -> T:  # type: ignore[assignment]
    env = os.getenv(key)
    if env is not None and env != "":
        return cast(env)
    tv = tuning_value(section, key)
    if tv is not None and tv != "":
        return cast(tv)
    return cast(default)


def _cfg_bool(section: str, key: str, default: bool) -> bool:
    try:
        v = tuning_value(section, key)
        if isinstance(v, bool):
            return v
        return str(v).lower() not in ("0", "false", "no", "")
    except Exception:
        return default


def _path_cfg(key: str, section: str, default: str) -> str:
    return os.path.abspath(_cfg(key, section, default))


# ---------- model / API ----------

DEFAULT_SLIDE_PATH = _path_cfg("DEFAULT_SLIDE_PATH", "agent", "341476.svs")
MODEL_NAME = _cfg("MODEL_NAME", "agent", "GLM-4.6V-FP8")

_api_key = _cfg("OPENAI_API_KEY", "agent", "local")
_api_base = _cfg("OPENAI_API_BASE", "agent", "http://pluto/v1")

client_async = AsyncOpenAI(api_key=_api_key, base_url=_api_base, max_retries=6, timeout=120.0)
client_sync = OpenAI(api_key=_api_key, base_url=_api_base, max_retries=6, timeout=120.0)

set_default_openai_client(client_async)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# ---------- agent tuning ----------

MAX_IMG_DIM = _cfg("MAX_IMG_DIM", "agent", 1024, int)
MAX_NATIVE_VIEW_DIM = _cfg("MAX_NATIVE_VIEW_DIM", "agent", 4096, int)
MAX_TURNS = _cfg("MAX_TURNS", "agent", 140, int)
WSI_AGENT_TEMPERATURE = _cfg("WSI_AGENT_TEMPERATURE", "agent", 0.9, float)
ENABLE_THINKING = _cfg_bool("agent", "ENABLE_THINKING", False)

# ---------- tile / embedding ----------

TILE_SIZE_UM = _cfg("TILE_SIZE_UM", "agent", 256.0, float)
TILE_PX = _cfg("TILE_PX", "agent", 224, int)
MAX_GOOD_TILES = _cfg("MAX_GOOD_TILES", "agent", 200, int)
MAX_BAD_TILES = _cfg("MAX_BAD_TILES", "agent", 50, int)
DEFAULT_MPP_UM = float(tuning_value("tools.slide", "DEFAULT_MPP_UM"))

# ---------- output directories ----------

OUTPUTS_ROOT_DIR = _path_cfg("OUTPUTS_ROOT_DIR", "agent", "./outputs")
os.makedirs(OUTPUTS_ROOT_DIR, exist_ok=True)

DEBUG_ROOT_DIR = OUTPUTS_ROOT_DIR
REPORT_ROOT_DIR = OUTPUTS_ROOT_DIR
SELECTED_TILES_ROOT = OUTPUTS_ROOT_DIR

# ---------- example tiles / ROIs ----------

EXAMPLE_TILES_ROOT = _path_cfg("EXAMPLE_TILES_ROOT", "agent", "./Selected_Tiles")
EXAMPLE_TILES_GOOD_DIR = os.path.join(EXAMPLE_TILES_ROOT, "Good_Tiles")
EXAMPLE_TILES_BAD_DIR = os.path.join(EXAMPLE_TILES_ROOT, "Bad_Tiles")
EXAMPLE_TILES_MAX_PER_CLASS = _cfg("EXAMPLE_TILES_MAX_PER_CLASS", "agent", 2, int)

_example_rois_root = _path_cfg("EXAMPLE_ROIS_ROOT", "agent", "./Example_ROIs")
EXAMPLE_ROIS_POS_DIR = os.path.join(_example_rois_root, "ROI")
EXAMPLE_ROIS_NEG_DIR = os.path.join(_example_rois_root, "Non_ROI")
EXAMPLE_ROIS_MAX_PER_CLASS = _cfg("EXAMPLE_ROIS_MAX_PER_CLASS", "agent", 2, int)

# ---------- context injection ----------

CONTEXT_PREVIOUS_VIEWS_MAX = _cfg("CONTEXT_PREVIOUS_VIEWS_MAX", "context_injection.candidates", 0, int)
CONTEXT_ROI_CANDIDATE_LINES_MAX = _cfg("CONTEXT_ROI_CANDIDATE_LINES_MAX", "context_injection.candidates", 8, int)
