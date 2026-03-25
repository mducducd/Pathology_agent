"""
DinoBloom Extractor for WSI Embeddings

Ported from https://github.com/KatherLab/dinobloom
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .. import Extractor

_DEFAULT_IDENTIFIER = "DinoBloom-S"


@lru_cache(maxsize=1)
def _load_dinobloom_assets() -> tuple[Any, Any]:
    """Load DinoBloom model + transform once per process."""
    try:
        import timm
        import torch
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "DinoBloom dependencies not installed. "
            "Please install with `pip install torch timm`."
        ) from e

    model = timm.create_model("hf-hub:KatherLab/DinoBloom-S", pretrained=True)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval()
    return model, transform


def dinobloom(identifier: str = _DEFAULT_IDENTIFIER) -> Extractor[Any]:
    """Build a DinoBloom extractor."""
    model, transform = _load_dinobloom_assets()
    return Extractor(
        model=model,
        transform=transform,
        identifier=identifier,
    )
