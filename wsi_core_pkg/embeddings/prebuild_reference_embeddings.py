#!/usr/bin/env python3
"""Pre-build reference embeddings from curated Good_Tiles and Bad_Tiles.

This script extracts embeddings from curated tile images and saves them
to disk for fast loading during inference. This avoids re-extracting
embeddings every time the AML agent starts.

Usage:
    python -m wsi_core_pkg.embeddings.prebuild_reference_embeddings \\
        --tiles-root ./Selected_Tiles \\
        --output-dir ./outputs/cache/reference_hnsw

The script produces:
- {extractor_id}_prebuilt_embeddings.npy: L2-normalized embeddings
- {extractor_id}_prebuilt_meta.json: Metadata with paths and labels
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _discover_tile_records(tiles_root: Path) -> list[dict[str, Any]]:
    """Discover tile images and their labels from directory structure."""
    class_dirs = [
        ("good", tiles_root / "Good_Tiles"),
        ("bad", tiles_root / "Bad_Tiles"),
        ("good", tiles_root / "Good"),
        ("bad", tiles_root / "Bad"),
    ]

    records: list[dict[str, Any]] = []
    seen: set[Path] = set()

    for label, class_dir in class_dirs:
        if not class_dir.exists() or not class_dir.is_dir():
            continue
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            abs_path = path.resolve()
            if abs_path in seen:
                continue
            seen.add(abs_path)
            records.append({
                "path": str(abs_path),
                "label": label,
            })

    return records


def _prepare_input_tensor(image: Image.Image, transform: Any) -> torch.Tensor:
    transformed = transform(image)
    if isinstance(transformed, np.ndarray):
        transformed = torch.from_numpy(transformed)
    if not isinstance(transformed, torch.Tensor):
        raise TypeError(f"Transform returned unsupported type: {type(transformed)!r}")

    if transformed.ndim == 4 and transformed.shape[0] == 1:
        transformed = transformed[0]
    if transformed.ndim != 3:
        raise ValueError(f"Expected transformed tensor shape [C,H,W], got {tuple(transformed.shape)}")
    if not torch.is_floating_point(transformed):
        transformed = transformed.float()
    return transformed


def _normalize_feature_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        tensor = output
    elif isinstance(output, (list, tuple)) and output:
        tensor = output[0]
    elif isinstance(output, dict):
        for key in ("features", "embeddings", "x", "logits"):
            if key in output and isinstance(output[key], torch.Tensor):
                tensor = output[key]
                break
        else:
            raise TypeError("Model output dict does not contain a tensor under known keys.")
    else:
        raise TypeError(f"Unsupported model output type: {type(output)!r}")

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor[:, 0, :]
    elif tensor.ndim > 3:
        tensor = tensor.reshape(tensor.shape[0], -1)

    if tensor.ndim != 2:
        raise ValueError(f"Expected feature tensor shape [B,D], got {tuple(tensor.shape)}")
    return tensor


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def prebuild_embeddings(
    tiles_root: str | Path,
    output_dir: str | Path,
    batch_size: int = 32,
    device: str | None = None,
    extractor_name: str = "reddino_base",
) -> dict[str, Any]:
    """Extract and save embeddings from curated tile images.

    Args:
        tiles_root: Root directory containing Good_Tiles and Bad_Tiles folders
        output_dir: Directory to save embeddings and metadata
        batch_size: Batch size for embedding extraction
        device: torch device (e.g., "cuda", "cuda:0", "cpu")
        extractor_name: Name of the extractor to use

    Returns:
        Dictionary with paths to saved files and statistics
    """
    tiles_root = Path(tiles_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover tiles
    print(f"Discovering tiles in {tiles_root}...")
    records = _discover_tile_records(tiles_root)
    if not records:
        raise ValueError(f"No image tiles found under: {tiles_root}")

    print(f"Found {len(records)} tiles:")
    good_count = sum(1 for r in records if r["label"] == "good")
    bad_count = sum(1 for r in records if r["label"] == "bad")
    print(f"  - Good tiles: {good_count}")
    print(f"  - Bad tiles: {bad_count}")

    # Load extractor
    print(f"Loading {extractor_name} extractor...")
    from . import get_embedding_extractor

    extractor = get_embedding_extractor(extractor_name)
    model = extractor.model

    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {run_device}")
    model = model.to(run_device)
    model.eval()

    # Extract embeddings
    print("Extracting embeddings...")
    batch_tensors: list[torch.Tensor] = []
    feature_chunks: list[torch.Tensor] = []
    labels: list[str] = []
    paths: list[str] = []

    for record in records:
        try:
            with Image.open(record["path"]) as im:
                rgb = im.convert("RGB")
                batch_tensors.append(_prepare_input_tensor(rgb, extractor.transform))
                labels.append(record["label"])
                paths.append(record["path"])
        except Exception as e:
            print(f"Warning: Failed to load {record['path']}: {e}")
            continue

        if len(batch_tensors) >= batch_size:
            x = torch.stack(batch_tensors, dim=0).to(run_device, non_blocking=True)
            with torch.no_grad():
                y = model(x)
            feature_chunks.append(_normalize_feature_output(y).detach().cpu())
            batch_tensors.clear()

    if batch_tensors:
        x = torch.stack(batch_tensors, dim=0).to(run_device, non_blocking=True)
        with torch.no_grad():
            y = model(x)
        feature_chunks.append(_normalize_feature_output(y).detach().cpu())

    # Concatenate and normalize
    features = torch.cat(feature_chunks, dim=0).numpy().astype(np.float32, copy=False)
    features = _l2_normalize_rows(features)

    print(f"Extracted embeddings shape: {features.shape}")

    # Compute fingerprint
    paths_tuple = tuple(paths)
    fingerprint = hashlib.sha256(str(paths_tuple).encode()).hexdigest()[:16]
    base_name = f"{extractor.identifier}_{fingerprint}"

    # Save embeddings
    embeddings_path = output_dir / f"{base_name}_embeddings.npy"
    labels_path = output_dir / f"{base_name}_labels.npy"
    meta_path = output_dir / f"{base_name}_meta.json"

    print(f"Saving embeddings to {embeddings_path}...")
    np.save(str(embeddings_path), features, allow_pickle=True)

    print(f"Saving labels to {labels_path}...")
    np.save(str(labels_path), np.array(labels, dtype=np.str_), allow_pickle=True)

    print(f"Saving metadata to {meta_path}...")
    meta = {
        "extractor_id": extractor.identifier,
        "count": len(paths),
        "dim": int(features.shape[1]),
        "fingerprint": fingerprint,
        "paths": paths,
        "labels": labels,
        "tiles_root": str(tiles_root),
        "good_count": good_count,
        "bad_count": bad_count,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    # Also save with fixed names for easy loading
    fixed_embeddings_path = output_dir / f"{extractor.identifier}_prebuilt_embeddings.npy"
    fixed_meta_path = output_dir / f"{extractor.identifier}_prebuilt_meta.json"

    np.save(str(fixed_embeddings_path), features, allow_pickle=True)
    fixed_meta_path.write_text(json.dumps(meta, indent=2))

    return {
        "ok": True,
        "count": len(paths),
        "dim": int(features.shape[1]),
        "extractor_id": extractor.identifier,
        "embeddings_path": str(embeddings_path),
        "labels_path": str(labels_path),
        "meta_path": str(meta_path),
        "fixed_embeddings_path": str(fixed_embeddings_path),
        "fixed_meta_path": str(fixed_meta_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-build reference embeddings from curated tile images."
    )
    parser.add_argument(
        "--tiles-root",
        type=Path,
        default=Path("Selected_Tiles"),
        help="Root dir containing Good_Tiles and Bad_Tiles folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cache/reference_hnsw"),
        help="Directory to save embeddings and metadata.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument(
        "--extractor",
        type=str,
        default="reddino_base",
        help="Name of the extractor to use (default: reddino_base, options: reddino_base, reddino_large, reddino, dinobloom, uni2)",
    )

    args = parser.parse_args()

    result = prebuild_embeddings(
        tiles_root=args.tiles_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        extractor_name=args.extractor,
    )

    print("\nDone!")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
