from __future__ import annotations

from functools import lru_cache
from typing import Any

from .. import Extractor

_DEFAULT_IDENTIFIER = "Virchow2"


class Virchow2ClsOnly:
    def __init__(self, model: Any) -> None:
        self.model = model

    def to(self, device: Any) -> "Virchow2ClsOnly":
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)
        return self

    def eval(self) -> "Virchow2ClsOnly":
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self

    def __call__(self, batch: Any) -> Any:
        out = self.model(batch)
        if isinstance(out, tuple):
            out = out[0]
        if getattr(out, "ndim", 0) == 3 and out.shape[1] > 1:
            return out[:, 0]
        return out


@lru_cache(maxsize=1)
def _load_virchow2_assets() -> tuple[Any, Any]:
    try:
        import timm
        import torch
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers.mlp import SwiGLUPacked
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Virchow2 dependencies not installed. "
            "Please install with `pip install torch torchvision timm`."
        ) from exc

    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model.eval()

    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return Virchow2ClsOnly(model), transform


def virchow2(identifier: str = _DEFAULT_IDENTIFIER) -> Extractor[Any]:
    model, transform = _load_virchow2_assets()
    return Extractor(
        model=model,
        transform=transform,
        identifier=identifier,
    )
