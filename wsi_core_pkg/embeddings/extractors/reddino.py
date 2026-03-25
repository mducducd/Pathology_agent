"""
Port from https://github.com/Snarci/RedDino
RedDino: A Foundation Model for Red Blood Cell Analysis
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast

from .. import Extractor

__license__ = "MIT"

_DEFAULT_IDENTIFIER = "RedDino-Small"
_DEFAULT_HF_MODEL = "hf-hub:Snarcy/RedDino-small"


class RedDinoClsOnly:
    def __init__(self, model: Any) -> None:
        self.model = model

    def to(self, device: Any) -> "RedDinoClsOnly":
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)
        return self

    def eval(self) -> "RedDinoClsOnly":
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self

    def __call__(self, batch: Any) -> Any:
        out = self.model(batch)
        if isinstance(out, tuple):
            out = out[0]
        # If the model returns token embeddings [B, T, D], keep the class token.
        # Do not slice already-pooled [B, D] outputs, or the feature width becomes batch-size dependent.
        if getattr(out, "ndim", 0) == 3 and out.shape[1] > 1:
            return out[:, 0]
        return out


@lru_cache(maxsize=1)
def _load_reddino_assets() -> tuple[Any, Any]:
    try:
        import timm
        import torch
        from PIL import Image
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RedDino dependencies not installed. Please install with `pip install torch torchvision timm`."
        ) from exc

    model = timm.create_model(
        _DEFAULT_HF_MODEL,
        pretrained=True,
        num_classes=0,
        pretrained_strict=False,
    )
    model.eval()

    transform = cast(
        Callable[[Image.Image], torch.Tensor],
        transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    )
    return RedDinoClsOnly(model), transform


def reddino(identifier: str = _DEFAULT_IDENTIFIER) -> Extractor[Any]:
    model, transform = _load_reddino_assets()
    return Extractor(
        model=model,
        transform=transform,
        identifier=identifier,
    )


def red_dino(identifier: str = _DEFAULT_IDENTIFIER) -> Extractor[Any]:
    return reddino(identifier=identifier)
