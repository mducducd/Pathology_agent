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


from .extractors.dinobloom import dinobloom
from .extractors.reddino import red_dino, reddino
from .extractors.uni2 import uni2
from .index_tiles_hnsw import embed_tiles_to_hnsw
from .tiling import TileFeatureMatrix, extract_wsi_features_by_tiles, save_tile_features_npz

DEFAULT_EMBEDDING_EXTRACTOR = "uni2"
_EMBEDDING_EXTRACTOR_BUILDERS = {
    "uni2": uni2,
    "dinobloom": dinobloom,
    "reddino": reddino,
}
_EMBEDDING_EXTRACTOR_DISPLAY_NAMES = {
    "uni2": "UNI2-h",
    "dinobloom": "DinoBloom-S",
    "reddino": "RedDino-Small",
    "uni2_onnx": "UNI2-h (ONNX)",
    "dinobloom_onnx": "DinoBloom-S (ONNX)",
    "reddino_onnx": "RedDino-Small (ONNX)",
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
    # Allow ONNX variants
    if key.endswith("_onnx"):
        base_key = key[:-5]  # Remove "_onnx" suffix
        if base_key in _EMBEDDING_EXTRACTOR_BUILDERS:
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
        base_key = key[:-5]  # Remove "_onnx" suffix
        try:
            from .extractors.onnx_runtime import (
                dinobloom_onnx,
                reddino_onnx,
                uni2_onnx,
            )
            onnx_extractors = {
                "uni2_onnx": uni2_onnx,
                "dinobloom_onnx": dinobloom_onnx,
                "reddino_onnx": reddino_onnx,
            }
            return onnx_extractors[key]()
        except ImportError:
            raise RuntimeError(
                f"ONNX extractor '{key}' requested but onnxruntime not available. "
                "Install with: pip install onnxruntime-gpu"
            )

    return _EMBEDDING_EXTRACTOR_BUILDERS[key]()


def embedding_extractor_display_name(name: str | None) -> str:
    key = normalize_embedding_extractor_name(name)
    return _EMBEDDING_EXTRACTOR_DISPLAY_NAMES.get(key, key)


__all__ = [
    "ExtractorModel",
    "Extractor",
    "DEFAULT_EMBEDDING_EXTRACTOR",
    "available_embedding_extractors",
    "normalize_embedding_extractor_name",
    "get_embedding_extractor",
    "embedding_extractor_display_name",
    # PyTorch extractors
    "dinobloom",
    "reddino",
    "red_dino",
    "uni2",
    # Core functions
    "embed_tiles_to_hnsw",
    "TileFeatureMatrix",
    "extract_wsi_features_by_tiles",
    "save_tile_features_npz",
]
