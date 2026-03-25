"""
ONNX Runtime Wrapper for WSI Embedding Extractors

Provides drop-in replacement for PyTorch extractors using ONNX Runtime.

Usage:
    from wsi_core_pkg.embeddings.extractors.onnx_runtime import OnnxExtractor
    from wsi_core_pkg.embeddings.extractors import uni2

    # Get the transform from original extractor
    pytorch_extractor = uni2()
    transform = pytorch_extractor.transform

    # Create ONNX extractor
    onnx_extractor = OnnxExtractor("uni2.onnx", transform=transform)

    # Use same as PyTorch extractor
    features = onnx_extractor.model(tensor)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .. import Extractor

_logger = logging.getLogger(__name__)


class OnnxRuntimeModel:
    """
    ONNX Runtime wrapper that mimics PyTorch model interface.

    Provides .to(), .eval(), and __call__() methods for compatibility
    with the existing extraction pipeline.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        provider: str = "cuda",
        intra_op_num_threads: int = 4,
        inter_op_num_threads: int = 4,
    ) -> None:
        """
        Initialize ONNX Runtime session.

        Args:
            onnx_path: Path to ONNX model file
            provider: Execution provider ('cuda' or 'cpu')
            intra_op_num_threads: Threads within operators
            inter_op_num_threads: Threads across operators
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ModuleNotFoundError(
                "ONNX Runtime not installed. Install with: pip install onnxruntime-gpu "
                "(or 'onnxruntime' for CPU-only)"
            ) from e

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Configure providers
        if provider == "cuda":
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kSameAsRequested",
                        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4GB limit
                        "cudnn_conv_algo_search": "HEURISTIC",
                    },
                ),
                "CPUExecutionProvider",
            ]
            _logger.info("Using CUDAExecutionProvider for ONNX Runtime")
        else:
            providers = ["CPUExecutionProvider"]
            _logger.info("Using CPUExecutionProvider for ONNX Runtime")

        # Configure session options
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = intra_op_num_threads
        session_options.inter_op_num_threads = inter_op_num_threads
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Optional: Enable memory pattern optimization
        session_options.enable_mem_pattern = True

        # Create session
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=session_options,
            providers=providers,
        )

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Track device for .to() compatibility
        self._device = provider

        _logger.info(f"Loaded ONNX model: {onnx_path}")
        _logger.info(f"  Input: {self.session.get_inputs()[0]}")
        _logger.info(f"  Output: {self.session.get_outputs()[0]}")

    def to(self, device: Any) -> "OnnxRuntimeModel":
        """Compatibility method - ONNX Runtime handles device internally."""
        # ONNX Runtime doesn't support device switching after initialization
        # This is a no-op for compatibility with existing code
        if isinstance(device, str):
            if "cuda" in device.lower() and "cuda" not in str(self.session.get_providers()):
                _logger.warning("Cannot switch ONNX model to CUDA after initialization")
        return self

    def eval(self) -> "OnnxRuntimeModel":
        """Compatibility method - ONNX models are always in eval mode."""
        return self

    def __call__(self, input_tensor: Any) -> np.ndarray:
        """
        Run inference on input tensor.

        Args:
            input_tensor: Input tensor (torch.Tensor or numpy array)
                         Shape: [batch, channels, height, width]

        Returns:
            Feature embeddings, Shape: [batch, feature_dim]
        """
        # Convert to numpy if torch tensor
        if hasattr(input_tensor, "cpu"):
            input_tensor = input_tensor.cpu().numpy()
        elif not isinstance(input_tensor, np.ndarray):
            input_tensor = np.array(input_tensor)

        # Ensure correct dtype (float32)
        if input_tensor.dtype != np.float32:
            input_tensor = input_tensor.astype(np.float32)

        # Run inference
        result = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

        return result

    @property
    def device(self) -> str:
        """Return current device."""
        return self._device


def create_onnx_extractor(
    onnx_path: str | Path,
    transform: Any,
    provider: str = "cuda",
    identifier: Optional[str] = None,
) -> Extractor[Any]:
    """
    Create an Extractor using ONNX Runtime for inference.

    Args:
        onnx_path: Path to ONNX model file
        transform: Image transformation (from original PyTorch extractor)
        provider: Execution provider ('cuda' or 'cpu')
        identifier: Optional identifier for the extractor

    Returns:
        Extractor compatible with existing pipeline
    """
    onnx_path = Path(onnx_path)
    if identifier is None:
        identifier = f"ONNX-{onnx_path.stem}"

    model = OnnxRuntimeModel(onnx_path, provider=provider)

    return Extractor(
        model=model,
        transform=transform,
        identifier=identifier,
    )


def get_onnx_model_path(
    extractor_name: str,
    cache_dir: Optional[str | Path] = None,
) -> Path:
    """
    Get cached ONNX model path or return None if not exists.

    Args:
        extractor_name: Name of extractor ('uni2', 'dinobloom', 'reddino')
        cache_dir: Directory to search for ONNX models

    Returns:
        Path to ONNX model if exists, None otherwise
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/slide-agent/onnx")

    cache_dir = Path(cache_dir)
    model_path = cache_dir / f"{extractor_name.lower()}.onnx"

    if model_path.exists():
        return model_path
    return None


def export_and_load(
    extractor_fn: Callable[[], Extractor[Any]],
    extractor_name: str,
    cache_dir: Optional[str | Path] = None,
    provider: str = "cuda",
) -> Extractor[Any]:
    """
    Export extractor to ONNX if not cached, then load with ONNX Runtime.

    Args:
        extractor_fn: Function that returns PyTorch Extractor
        extractor_name: Name for the extractor ('uni2', 'dinobloom', 'reddino')
        cache_dir: Directory to cache ONNX models
        provider: Execution provider ('cuda' or 'cpu')

    Returns:
        ONNX-based Extractor
    """
    from .onnx_export import export_extractor_to_onnx

    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/slide-agent/onnx")

    cache_dir = Path(cache_dir)
    onnx_path = cache_dir / f"{extractor_name.lower()}.onnx"

    # Export if not cached
    if not onnx_path.exists():
        _logger.info(f"Exporting {extractor_name} to ONNX...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        export_extractor_to_onnx(extractor_fn, onnx_path)
    else:
        _logger.info(f"Using cached ONNX model: {onnx_path}")

    # Load original extractor for transform
    pytorch_extractor = extractor_fn()
    transform = pytorch_extractor.transform

    # Create ONNX extractor
    return create_on_extractor(onnx_path, transform, provider=provider)


# Convenience functions for each extractor
def uni2_onnx(
    cache_dir: Optional[str | Path] = None,
    provider: str = "cuda",
) -> Extractor[Any]:
    """Load UNI2 with ONNX Runtime."""
    from .uni2 import uni2 as pytorch_uni2

    return export_and_load(pytorch_uni2, "uni2", cache_dir, provider)


def dinobloom_onnx(
    cache_dir: Optional[str | Path] = None,
    provider: str = "cuda",
) -> Extractor[Any]:
    """Load DinoBloom with ONNX Runtime."""
    from .dinobloom import dinobloom as pytorch_dinobloom

    return export_and_load(pytorch_dinobloom, "dinobloom", cache_dir, provider)


def reddino_onnx(
    cache_dir: Optional[str | Path] = None,
    provider: str = "cuda",
) -> Extractor[Any]:
    """Load RedDino with ONNX Runtime."""
    from .reddino import reddino as pytorch_reddino

    return export_and_load(pytorch_reddino, "reddino", cache_dir, provider)


__all__ = [
    "OnnxRuntimeModel",
    "create_onnx_extractor",
    "get_onnx_model_path",
    "export_and_load",
    "uni2_onnx",
    "dinobloom_onnx",
    "reddino_onnx",
]
