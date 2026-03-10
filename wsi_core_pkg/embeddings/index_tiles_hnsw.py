from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

try:
    from .extractors.uni2 import uni2
except ImportError:
    # Allows direct execution: `python wsi_core_pkg/embeddings/index_tiles_hnsw.py ...`
    from wsi_core_pkg.embeddings.extractors.uni2 import uni2

try:
    import hnswlib
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "hnswlib is not installed. Please install dependencies first (e.g. `uv add hnswlib`)."
    ) from e


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class TileRecord:
    tile_id: int
    label: str
    path: Path


def _discover_tile_records(tiles_root: Path) -> list[TileRecord]:
    class_dirs = [
        ("good", tiles_root / "Good_Tiles"),
        ("bad", tiles_root / "Bad_Tiles"),
        ("good", tiles_root / "Good"),
        ("bad", tiles_root / "Bad"),
    ]

    records: list[TileRecord] = []
    next_id = 0
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
            records.append(TileRecord(tile_id=next_id, label=label, path=abs_path))
            next_id += 1

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


def embed_tiles_to_hnsw(
    *,
    tiles_root: str | Path,
    output_dir: str | Path,
    batch_size: int = 32,
    device: str | None = None,
    space: str = "cosine",
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 100,
) -> dict[str, Any]:
    tiles_root = Path(tiles_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    records = _discover_tile_records(tiles_root)
    if not records:
        raise ValueError(f"No image tiles found under: {tiles_root}")

    extractor = uni2()
    model = extractor.model

    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(run_device)
    model.eval()

    batch_tensors: list[torch.Tensor] = []
    feature_chunks: list[torch.Tensor] = []

    for record in records:
        with Image.open(record.path) as im:
            rgb = im.convert("RGB")
            batch_tensors.append(_prepare_input_tensor(rgb, extractor.transform))

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

    features = torch.cat(feature_chunks, dim=0).numpy().astype(np.float32, copy=False)
    if features.shape[0] != len(records):
        raise RuntimeError(
            f"Embedding count mismatch: features={features.shape[0]} records={len(records)}"
        )

    if space == "cosine":
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
        features = features / norms

    dim = int(features.shape[1])
    ids = np.arange(len(records), dtype=np.int64)

    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=len(records), ef_construction=ef_construction, M=m)
    index.add_items(features, ids)
    index.set_ef(ef_search)

    index_path = output_dir / "tiles_hnsw.bin"
    meta_json_path = output_dir / "tiles_meta.json"
    labels_np_path = output_dir / "tile_labels.npy"

    index.save_index(str(index_path))

    meta = {
        "space": space,
        "dim": dim,
        "count": len(records),
        "extractor_id": extractor.identifier,
        "tiles_root": str(tiles_root),
        "index_path": str(index_path),
        "labels_path": str(labels_np_path),
        "items": [
            {
                "id": r.tile_id,
                "label": r.label,
                "path": str(r.path),
            }
            for r in records
        ],
    }
    meta_json_path.write_text(json.dumps(meta, indent=2))

    label_map = np.array([r.label for r in records], dtype=object)
    np.save(labels_np_path, label_map, allow_pickle=True)

    return {
        "ok": True,
        "count": len(records),
        "dim": dim,
        "extractor_id": extractor.identifier,
        "index_path": str(index_path),
        "meta_path": str(meta_json_path),
        "labels_path": str(labels_np_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed Good/Bad tile folders with UNI2 and build HNSW index.")
    parser.add_argument(
        "--tiles-root",
        type=Path,
        default=Path("Selected_Tiles"),
        help="Root dir containing Good_Tiles and Bad_Tiles folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tile_hnsw"),
        help="Directory to store HNSW index and metadata.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--space", type=str, default="cosine", choices=["cosine", "l2", "ip"])
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = embed_tiles_to_hnsw(
        tiles_root=args.tiles_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        space=args.space,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
