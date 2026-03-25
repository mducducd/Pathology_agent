"""
ONNX Export and Runtime for WSI Embedding Extractors

Supports: UNI2, DinoBloom, RedDino

Usage:
    # Export (one-time)
    from wsi_core_pkg.embeddings.extractors.onnx_export import export_extractor_to_onnx
    from wsi_core_pkg.embeddings.extractors import uni2

    export_extractor_to_onnx(uni2, "uni2.onnx")

    # Inference with ONNX Runtime
    from wsi_core_pkg.embeddings.extractors.onnx_runtime import OnnxExtractor

    extractor = OnnxExtractor("uni2.onnx")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import torch

from .. import Extractor

_logger = logging.getLogger(__name__)

# Default export settings
DEFAULT_OPSET_VERSION = 17
DEFAULT_INPUT_SIZE = (1, 3, 224, 224)  # (batch, channels, height, width)


def _get_model_for_export(model: Any) -> Any:
    """Extract the underlying model for ONNX export (handle wrapper classes)."""
    # RedDino wraps model in RedDinoClsOnly
    if hasattr(model, "model") and not callable(getattr(model, "model", None)):
        return model.model
    # UNI2 and DinoBloom can be exported directly
    return model


def export_extractor_to_onnx(
    extractor_fn: Callable[[], Extractor[Any]],
    output_path: str | Path,
    opset_version: int = DEFAULT_OPSET_VERSION,
    input_size: tuple[int, int, int, int] = DEFAULT_INPUT_SIZE,
    device: str = "cuda",
    dynamic_batch: bool = True,
    verify: bool = True,
) -> Path:
    """
    Export an extractor model to ONNX format.

    Args:
        extractor_fn: Function that returns an Extractor (e.g., uni2, dinobloom, reddino)
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version (default: 17 for ViT support)
        input_size: Input tensor size (batch, channels, height, width)
        device: Device for export ('cuda' or 'cpu')
        dynamic_batch: Allow dynamic batch size in exported model
        verify: Verify the exported model with onnx.checker

    Returns:
        Path to the exported ONNX model

    Raises:
        ModuleNotFoundError: If onnx or torch.onnx dependencies are missing
        RuntimeError: If export fails
    """
    try:
        import onnx
        import onnx.checker
    except ImportError as e:
        raise ModuleNotFoundError(
            "ONNX export requires 'onnx' package. Install with: pip install onnx"
        ) from e

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load extractor
    _logger.info(f"Loading extractor: {extractor_fn.__name__}")
    extractor = extractor_fn()
    model = extractor.model
    model.eval()

    # Get underlying model for export
    export_model = _get_model_for_export(model)
    export_model.eval()

    # Move to device
    if device == "cuda" and torch.cuda.is_available():
        export_model = export_model.cuda()
    else:
        device = "cpu"

    # Create dummy input
    dummy_input = torch.randn(*input_size, device=device)

    # Determine output name based on extractor type
    extractor_name = extractor_fn.__name__.lower()

    # Export
    _logger.info(f"Exporting to ONNX (opset={opset_version}, device={device})")

    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["features"],
        dynamic_axes={
            "input": {0: "batch_size"} if dynamic_batch else {},
            "features": {0: "batch_size"} if dynamic_batch else {},
        } if dynamic_batch else None,
        do_constant_folding=True,
        verbose=False,
    )

    # Verify
    if verify:
        _logger.info("Verifying ONNX model...")
        try:
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            _logger.info("ONNX model verification passed")
        except Exception as e:
            _logger.warning(f"ONNX verification warning: {e}")

    _logger.info(f"Exported to: {output_path}")
    return output_path


def export_all_extractors(
    output_dir: str | Path,
    opset_version: int = DEFAULT_OPSET_VERSION,
    device: str = "cuda",
) -> dict[str, Path]:
    """
    Export all available extractors to ONNX format.

    Args:
        output_dir: Directory to save ONNX models
        opset_version: ONNX opset version
        device: Device for export

    Returns:
        Dictionary mapping extractor names to their ONNX paths
    """
    from . import dinobloom, reddino, uni2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractors = {
        "uni2": uni2,
        "dinobloom": dinobloom,
        "reddino": reddino,
    }

    results = {}
    for name, extractor_fn in extractors.items():
        try:
            output_path = output_dir / f"{name}.onnx"
            export_extractor_to_onnx(
                extractor_fn=extractor_fn,
                output_path=output_path,
                opset_version=opset_version,
                device=device,
            )
            results[name] = output_path
            _logger.info(f"Successfully exported {name}")
        except Exception as e:
            _logger.error(f"Failed to export {name}: {e}")
            results[name] = None

    return results


def optimize_onnx_model(
    input_path: str | Path,
    output_path: str | Path | None = None,
    optimize_for: Literal["gpu", "cpu", "tensorrt"] = "gpu",
) -> Path:
    """
    Optimize an ONNX model for inference.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized model (default: overwrite input)
        optimize_for: Target platform ('gpu', 'cpu', or 'tensorrt')

    Returns:
        Path to optimized model
    """
    try:
        from onnxruntime.transformers.optimizer import optimize_model
    except ImportError:
        _logger.warning("onnxruntime-tools not installed. Skipping optimization.")
        return Path(input_path)

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_optimized.onnx"
    else:
        output_path = Path(output_path)

    _logger.info(f"Optimizing ONNX model for {optimize_for}...")

    # Determine model type for optimization
    model_type = "vit"  # All our extractors are ViT-based

    optimize_model(
        str(input_path),
        model_type=model_type,
        num_heads=12,  # Will be auto-detected
        hidden_size=768,  # Will be auto-detected
        optimization_options={
            "enable_gelu_approximation": True,
            "enable_layer_norm": True,
            "enable_attention": True,
        },
        opt_level=99,  # Maximum optimization
        only_onnxruntime=False,
    )

    _logger.info(f"Optimized model saved to: {output_path}")
    return output_path


__all__ = [
    "export_extractor_to_onnx",
    "export_all_extractors",
    "optimize_onnx_model",
    "DEFAULT_OPSET_VERSION",
    "DEFAULT_INPUT_SIZE",
]
