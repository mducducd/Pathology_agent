import os

from dotenv import load_dotenv
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

DEFAULT_SLIDE_PATH = os.path.abspath("341476.svs")
MODEL_NAME = "GPT-OSS-120B"

client_async = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "http://pluto/v1"),
)
client_sync = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "http://pluto/v1"),
)

set_default_openai_client(client_async)
set_default_openai_api("chat_completions")
set_tracing_disabled(False)
enable_verbose_stdout_logging()

MAX_IMG_DIM = int(os.getenv("MAX_IMG_DIM", "1024"))
MAX_NATIVE_VIEW_DIM = 4096
MAX_TURNS = int(os.getenv("MAX_TURNS", "100"))

ROI_TARGET_SIDE_PX = 1500

TILE_SIZE_UM = 256.0
TILE_PX = 224
MAX_GOOD_TILES = 200
MAX_BAD_TILES = 50
DEFAULT_MPP_UM = 0.25

OUTPUTS_ROOT_DIR = os.path.abspath("./outputs")
os.makedirs(OUTPUTS_ROOT_DIR, exist_ok=True)

DEBUG_ROOT_DIR = OUTPUTS_ROOT_DIR
os.makedirs(DEBUG_ROOT_DIR, exist_ok=True)

REPORT_ROOT_DIR = OUTPUTS_ROOT_DIR
os.makedirs(REPORT_ROOT_DIR, exist_ok=True)

SELECTED_TILES_ROOT = OUTPUTS_ROOT_DIR
os.makedirs(SELECTED_TILES_ROOT, exist_ok=True)

EXAMPLE_TILES_ROOT = os.path.abspath(os.getenv("EXAMPLE_TILES_ROOT", "./Selected_Tiles"))
EXAMPLE_TILES_GOOD_DIR = os.path.join(EXAMPLE_TILES_ROOT, "Good_Tiles")
EXAMPLE_TILES_BAD_DIR = os.path.join(EXAMPLE_TILES_ROOT, "Bad_Tiles")
EXAMPLE_TILES_MAX_PER_CLASS = int(os.getenv("EXAMPLE_TILES_MAX_PER_CLASS", "2"))

EXAMPLE_ROIS_ROOT = os.path.abspath(os.getenv("EXAMPLE_ROIS_ROOT", "./Example_ROIs"))
EXAMPLE_ROIS_POS_DIR = os.path.join(EXAMPLE_ROIS_ROOT, "ROI")
EXAMPLE_ROIS_NEG_DIR = os.path.join(EXAMPLE_ROIS_ROOT, "Non_ROI")
EXAMPLE_ROIS_MAX_PER_CLASS = int(os.getenv("EXAMPLE_ROIS_MAX_PER_CLASS", "2"))
CONTEXT_PREVIOUS_VIEWS_MAX = int(os.getenv("CONTEXT_PREVIOUS_VIEWS_MAX", "0"))
CONTEXT_ROI_CANDIDATE_LINES_MAX = int(os.getenv("CONTEXT_ROI_CANDIDATE_LINES_MAX", "4"))
