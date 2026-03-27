"""
SSIM-based tile quality scoring (no foundation model required).

This extractor uses SSIM (Structural Similarity Index) to score tiles
based on their similarity to reference good/bad tiles.

Unlike foundation model extractors (UNI2, RedDino, DinoBloom), SSIM:
- Does NOT produce high-dimensional embeddings
- Directly computes scalar similarity scores
- Is much faster (CPU-only, no GPU needed)
- Works purely on visual/structural similarity
"""

from __future__ import annotations

import torch
from functools import lru_cache
from typing import Any

from PIL import Image
from torchvision import transforms

from .. import Extractor

__license__ = "MIT"

_DEFAULT_IDENTIFIER = "SSIM"


class SSIMPlaceholder:
    """
    Placeholder model for SSIM mode.

    SSIM doesn't use a neural network - it directly computes
    structural similarity between images. This placeholder exists
    only to satisfy the Extractor interface.
    """

    def __init__(self) -> None:
        pass

    def to(self, device: Any) -> "SSIMPlaceholder":
        return self

    def eval(self) -> "SSIMPlaceholder":
        return self

    def __call__(self, batch: Any) -> Any:
        # SSIM doesn't use model inference
        # Returns a dummy tensor - actual scoring happens in roi_ranker.py
        if isinstance(batch, torch.Tensor):
            return torch.zeros(batch.shape[0], 1)
        return batch


@lru_cache(maxsize=1)
def _load_ssim_assets() -> tuple[Any, Any]:
    # SSIM doesn't actually use these - they're just placeholders
    # The actual SSIM computation happens in roi_ranker.py using skimage
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return SSIMPlaceholder(), transform


def ssim_extractor(identifier: str = _DEFAULT_IDENTIFIER) -> Extractor[Any]:
    """
    Create an SSIM-based extractor.

    This extractor doesn't use a foundation model. Instead, it:
    1. Extracts tiles from the WSI (standard tiling)
    2. Computes SSIM similarity to reference good/bad tiles
    3. Ranks tiles by SSIM score (no embeddings needed)

    Args:
        identifier: Extractor identifier (default: "SSIM")

    Returns:
        Extractor object (SSIM uses placeholder model)
    """
    model, transform = _load_ssim_assets()
    return Extractor(
        model=model,
        transform=transform,
        identifier=identifier,
    )
