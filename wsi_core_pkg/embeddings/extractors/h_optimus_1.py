from __future__ import annotations

from functools import lru_cache
from typing import Any

from .. import Extractor

_DEFAULT_IDENTIFIER = "H-optimus-1"


@lru_cache(maxsize=1)
def _load_h_optimus_1_assets() -> tuple[Any, Any]:
    try:
        import timm
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "H-optimus-1 dependencies not installed. "
            "Please install with `pip install torch torchvision timm`."
        ) from exc

    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-1",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617),
                std=(0.211883, 0.230117, 0.177517),
            ),
        ]
    )
    return model, transform


def h_optimus_1(identifier: str = _DEFAULT_IDENTIFIER) -> Extractor[Any]:
    model, transform = _load_h_optimus_1_assets()
    return Extractor(
        model=model,
        transform=transform,
        identifier=identifier,
    )
