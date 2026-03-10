from __future__ import annotations

from functools import lru_cache
from typing import Any

from .. import Extractor


@lru_cache(maxsize=1)
def _load_uni2_assets() -> tuple[Any, Any]:
    """Load UNI2 model + transform once per process."""
    try:
        import timm
        import torch
        from timm.data import resolve_data_config  # type: ignore
        from timm.data.transforms_factory import create_transform
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "UNI2 dependencies not installed. "
            "Please install them with `pip install torch timm`."
        ) from e

    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }

    # pretrained=True is required so timm resolves UNI2-h pretrained weights.
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval()
    return model, transform


def uni2(identifier: str = "UNI2-h") -> Extractor[Any]:
    """Build a UNI2 extractor (MahmoodLab/UNI2-h via timm hf-hub)."""
    model, transform = _load_uni2_assets()
    return Extractor(
        model=model,
        transform=transform,
        identifier=identifier,
    )
