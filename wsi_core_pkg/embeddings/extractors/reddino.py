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
_BASE_IDENTIFIER = "RedDino-base"
_BASE_HF_MODEL = "hf-hub:Snarcy/RedDino-base"
_LARGE_IDENTIFIER = "RedDino-large"
_LARGE_HF_MODEL = "hf-hub:Snarcy/RedDino-large"


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


@lru_cache(maxsize=3)
def _load_reddino_assets(hf_model: str) -> tuple[Any, Any]:
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
        hf_model,
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


def _build_reddino_extractor(identifier: str, hf_model: str) -> Extractor[Any]:
    model, transform = _load_reddino_assets(hf_model)
    return Extractor(
        model=model,
        transform=transform,
        identifier=identifier,
    )


def reddino(identifier: str = _DEFAULT_IDENTIFIER) -> Extractor[Any]:
    return _build_reddino_extractor(identifier=identifier, hf_model=_DEFAULT_HF_MODEL)


def reddino_base(identifier: str = _BASE_IDENTIFIER) -> Extractor[Any]:
    return _build_reddino_extractor(identifier=identifier, hf_model=_BASE_HF_MODEL)


def reddino_large(identifier: str = _LARGE_IDENTIFIER) -> Extractor[Any]:
    return _build_reddino_extractor(identifier=identifier, hf_model=_LARGE_HF_MODEL)


def red_dino(identifier: str = _DEFAULT_IDENTIFIER) -> Extractor[Any]:
    return reddino(identifier=identifier)
