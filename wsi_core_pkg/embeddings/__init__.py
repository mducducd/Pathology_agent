from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from typing import Any, Generic, TypeVar

from PIL import Image

ExtractorModel = TypeVar("ExtractorModel")


@dataclass(frozen=True)
class Extractor(Generic[ExtractorModel]):
    _: KW_ONLY
    model: ExtractorModel
    transform: Callable[[Image.Image], Any]
    identifier: str
    """Uniquely identifies a model + transform pair.

    In production, this should include the exact model/version/weights
    identifier so cached embeddings can be invalidated safely.
    """


from .extractors.dinobloom import (
    dino_bloom,
    dinobloom,
    dinobloom_base,
    dinobloom_giant,
    dinobloom_large,
)
from .extractors.h_optimus_1 import h_optimus_1
from .extractors.reddino import red_dino, reddino, reddino_base, reddino_large
from .extractors.uni2 import uni2
from .extractors.virchow2 import virchow2
from .index_tiles_hnsw import embed_tiles_to_hnsw
from .tiling import TileFeatureMatrix, extract_wsi_features_by_tiles, save_tile_features_npz

DEFAULT_EMBEDDING_EXTRACTOR = "uni2"
_EMBEDDING_EXTRACTOR_BUILDERS = {
    "uni2": uni2,
    "dinobloom": dinobloom,
    "dinobloom_base": dinobloom_base,
    "dinobloom_large": dinobloom_large,
    "dinobloom_giant": dinobloom_giant,
    "virchow2": virchow2,
    "h_optimus_1": h_optimus_1,
    "reddino": reddino,
    "reddino_base": reddino_base,
    "reddino_large": reddino_large,
}
_EMBEDDING_EXTRACTOR_DISPLAY_NAMES = {
    "uni2": "UNI2-h",
    "dinobloom": "DinoBloom-S",
    "dinobloom_base": "DinoBloom-B",
    "dinobloom_large": "DinoBloom-L",
    "dinobloom_giant": "DinoBloom-G",
    "virchow2": "Virchow2",
    "h_optimus_1": "H-optimus-1",
    "reddino": "RedDino-Small",
    "reddino_base": "RedDino-base",
    "reddino_large": "RedDino-large",
    "uni2_onnx": "UNI2-h (ONNX)",
    "dinobloom_onnx": "DinoBloom-S (ONNX)",
    "reddino_onnx": "RedDino-Small (ONNX)",
}
_EMBEDDING_EXTRACTOR_DEFAULT_IDENTIFIERS = {
    "uni2": "UNI2-h",
    "dinobloom": "DinoBloom-S",
    "dinobloom_base": "DinoBloom-B",
    "dinobloom_large": "DinoBloom-L",
    "dinobloom_giant": "DinoBloom-G",
    "virchow2": "Virchow2",
    "h_optimus_1": "H-optimus-1",
    "reddino": "RedDino-Small",
    "reddino_base": "RedDino-base",
    "reddino_large": "RedDino-large",
    "uni2_onnx": "ONNX-uni2",
    "dinobloom_onnx": "ONNX-dinobloom",
    "reddino_onnx": "ONNX-reddino",
}

# ONNX Runtime extractors (lazy loaded to avoid dependency if not used)
def _get_onnx_extractors() -> dict[str, Callable[[], Any]]:
    """Lazy load ONNX extractors if onnxruntime is available."""
    try:
        from .extractors.onnx_runtime import dinobloom_onnx, reddino_onnx, uni2_onnx
        return {
            "uni2_onnx": uni2_onnx,
            "dinobloom_onnx": dinobloom_onnx,
            "reddino_onnx": reddino_onnx,
        }
    except ImportError:
        return {}


def available_embedding_extractors() -> tuple[str, ...]:
    """Return list of available extractor names including ONNX variants."""
    base_extractors = tuple(_EMBEDDING_EXTRACTOR_BUILDERS.keys())
    onnx_extractors = tuple(_get_onnx_extractors().keys())
    return base_extractors + onnx_extractors


def normalize_embedding_extractor_name(name: str | None) -> str:
    key = (name or DEFAULT_EMBEDDING_EXTRACTOR).strip().lower()
    if key.endswith("_onnx"):
        if key in _get_onnx_extractors():
            return key
    if key not in _EMBEDDING_EXTRACTOR_BUILDERS:
        raise ValueError(
            "extractor_name must be one of: " + ", ".join(sorted(available_embedding_extractors()))
        )
    return key


def get_embedding_extractor(name: str | None) -> Any:
    key = normalize_embedding_extractor_name(name)

    # Handle ONNX variants
    if key.endswith("_onnx"):
        onnx_extractors = _get_onnx_extractors()
        if key not in onnx_extractors:
            raise RuntimeError(
                f"ONNX extractor '{key}' requested but onnxruntime not available. "
                "Install with: pip install onnxruntime-gpu"
            )
        return onnx_extractors[key]()

    return _EMBEDDING_EXTRACTOR_BUILDERS[key]()


def embedding_extractor_display_name(name: str | None) -> str:
    key = normalize_embedding_extractor_name(name)
    return _EMBEDDING_EXTRACTOR_DISPLAY_NAMES.get(key, key)


def embedding_extractor_identifier(name: str | None) -> str:
    key = normalize_embedding_extractor_name(name)
    return _EMBEDDING_EXTRACTOR_DEFAULT_IDENTIFIERS.get(
        key,
        _EMBEDDING_EXTRACTOR_DISPLAY_NAMES.get(key, key),
    )


__all__ = [
    "ExtractorModel",
    "Extractor",
    "DEFAULT_EMBEDDING_EXTRACTOR",
    "available_embedding_extractors",
    "normalize_embedding_extractor_name",
    "get_embedding_extractor",
    "embedding_extractor_display_name",
    "embedding_extractor_identifier",
    # PyTorch extractors
    "dino_bloom",
    "dinobloom",
    "dinobloom_base",
    "dinobloom_large",
    "dinobloom_giant",
    "virchow2",
    "h_optimus_1",
    "reddino",
    "reddino_base",
    "reddino_large",
    "red_dino",
    "uni2",
    # Core functions
    "embed_tiles_to_hnsw",
    "TileFeatureMatrix",
    "extract_wsi_features_by_tiles",
    "save_tile_features_npz",
]
