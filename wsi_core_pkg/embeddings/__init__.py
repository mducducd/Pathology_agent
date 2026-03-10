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


from .extractors.uni2 import uni2
from .index_tiles_hnsw import embed_tiles_to_hnsw
from .tiling import TileFeatureMatrix, extract_wsi_features_by_tiles, save_tile_features_npz

__all__ = [
    "ExtractorModel",
    "Extractor",
    "uni2",
    "embed_tiles_to_hnsw",
    "TileFeatureMatrix",
    "extract_wsi_features_by_tiles",
    "save_tile_features_npz",
]
