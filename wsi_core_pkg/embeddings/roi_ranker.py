from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from .extractors.uni2 import uni2
from .tiling import SlideMPP, extract_wsi_features_by_tiles, get_slide_mpp_

try:
    import hnswlib
except Exception:  # pragma: no cover - optional runtime acceleration
    hnswlib = None

ROI_KNN_RANDOM_SEED = int(os.getenv("ROI_KNN_RANDOM_SEED", "42"))
ROI_CANDIDATE_MAX_IOU = float(os.getenv("ROI_CANDIDATE_MAX_IOU", "0.20"))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Scoring weights for generic WSI mode: score = w_nov*z(novelty) + w_cen*z(centroid_dist)
WSI_W_NOVELTY = float(os.getenv("WSI_W_NOVELTY", "0.65"))
WSI_W_CENTROID = float(os.getenv("WSI_W_CENTROID", "0.35"))

AML_REFERENCE_TOP_K = int(os.getenv("AML_REFERENCE_TOP_K", "5"))
AML_REFERENCE_QUERY_BLOCK_ROWS = int(os.getenv("AML_REFERENCE_QUERY_BLOCK_ROWS", "1024"))
AML_REFERENCE_LOGIT_SCALE = float(os.getenv("AML_REFERENCE_LOGIT_SCALE", "4.0"))
AML_REFERENCE_EVIDENCE_PER_CLASS = int(os.getenv("AML_REFERENCE_EVIDENCE_PER_CLASS", "3"))

# Novelty outlier clipping: tiles above this percentile of novelty are likely artifacts
# (tissue folds, pen marks, torn edges). Clipped to this ceiling before z-scoring so
# they don't dominate the ranking. Set to 100 to disable.
ROI_NOVELTY_CLIP_PERCENTILE = float(os.getenv("ROI_NOVELTY_CLIP_PERCENTILE", "97.0"))


@dataclass(frozen=True)
class UnsupervisedROIIndex:
    slide_path: str
    extractor_id: str
    tile_size_um: float
    tile_size_px: int
    tile_size_level0_px: int
    coordinates_level0_xy: npt.NDArray[np.float32]
    scores: npt.NDArray[np.float32]
    num_tiles: int
    feature_dim: int
    bad_margin: npt.NDArray[np.float32] = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    bad_likelihood: npt.NDArray[np.float32] = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    reference_mode: str = "none"
    reference_stats: dict[str, Any] = field(default_factory=dict)
    bad_neighbor_indices: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty((0, 0), dtype=np.int32))
    bad_neighbor_sims: npt.NDArray[np.float32] = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    good_neighbor_indices: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty((0, 0), dtype=np.int32))
    good_neighbor_sims: npt.NDArray[np.float32] = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    reference_tile_paths: tuple[str, ...] = field(default_factory=tuple)
    reference_tile_labels: tuple[str, ...] = field(default_factory=tuple)
    reference_neighbor_k: int = 0


@dataclass(frozen=True)
class ReferenceKNNScoring:
    margin: npt.NDArray[np.float32]
    bad_likelihood: npt.NDArray[np.float32]
    reference_mode: str
    rank_scores: npt.NDArray[np.float32] = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    bad_top1_similarity: npt.NDArray[np.float32] = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    good_top1_similarity: npt.NDArray[np.float32] = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    bad_neighbor_indices: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty((0, 0), dtype=np.int32))
    bad_neighbor_sims: npt.NDArray[np.float32] = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    good_neighbor_indices: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty((0, 0), dtype=np.int32))
    good_neighbor_sims: npt.NDArray[np.float32] = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))


def _l2_normalize_rows(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def _zscore(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    mu = float(np.mean(x)) if x.size else 0.0
    sigma = float(np.std(x)) if x.size else 0.0
    if sigma < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mu) / sigma).astype(np.float32, copy=False)


def _sigmoid(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))).astype(np.float32, copy=False)


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


def _discover_reference_tiles(reference_root: Path) -> list[tuple[Path, str]]:
    class_dirs = [
        ("good", reference_root / "Good_Tiles"),
        ("bad", reference_root / "Bad_Tiles"),
        ("good", reference_root / "Good"),
        ("bad", reference_root / "Bad"),
    ]

    out: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for label, class_dir in class_dirs:
        if not class_dir.exists() or not class_dir.is_dir():
            continue
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            p = path.resolve()
            if p in seen:
                continue
            seen.add(p)
            out.append((p, label))
    return out


def _embed_reference_tiles(
    *,
    records: list[tuple[Path, str]],
    extractor: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.str_], tuple[str, ...]]:
    if not records:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.str_), ()

    model = extractor.model.to(device)
    model.eval()

    label_buf: list[str] = []
    path_buf: list[str] = []
    batch_tensors: list[torch.Tensor] = []
    chunks: list[torch.Tensor] = []

    for path, label in records:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            batch_tensors.append(_prepare_input_tensor(rgb, extractor.transform))
            label_buf.append(label)
            path_buf.append(str(path))

        if len(batch_tensors) >= batch_size:
            x = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
            with torch.no_grad():
                y = model(x)
            chunks.append(_normalize_feature_output(y).detach().cpu())
            batch_tensors.clear()

    if batch_tensors:
        x = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            y = model(x)
        chunks.append(_normalize_feature_output(y).detach().cpu())

    feat = torch.cat(chunks, dim=0).numpy().astype(np.float32, copy=False)
    feat_l2 = _l2_normalize_rows(feat)
    labels = np.asarray(label_buf, dtype=np.str_)
    return feat_l2, labels, tuple(path_buf)


def _topk_from_similarity_matrix(
    *,
    similarities: npt.NDArray[np.float32],
    k: int,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    rows = int(similarities.shape[0]) if similarities.ndim == 2 else 0
    cols = int(similarities.shape[1]) if similarities.ndim == 2 else 0
    if rows == 0 or cols == 0 or k <= 0:
        return np.empty((rows, 0), dtype=np.int32), np.empty((rows, 0), dtype=np.float32)

    k_eff = max(1, min(int(k), cols))
    kth = max(0, cols - k_eff)
    part = np.argpartition(similarities, kth=kth, axis=1)[:, -k_eff:]
    part_sims = np.take_along_axis(similarities, part, axis=1)
    order = np.argsort(part_sims, axis=1)[:, ::-1]
    top_idx = np.take_along_axis(part, order, axis=1).astype(np.int32, copy=False)
    top_sims = np.take_along_axis(part_sims, order, axis=1).astype(np.float32, copy=False)
    return top_idx, top_sims


def _exact_topk_reference_matches(
    *,
    features_l2: npt.NDArray[np.float32],
    ref_features_l2: npt.NDArray[np.float32],
    k: int,
    row_block_size: int,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    rows = int(features_l2.shape[0]) if features_l2.ndim == 2 else 0
    cols = int(ref_features_l2.shape[0]) if ref_features_l2.ndim == 2 else 0
    if rows == 0 or cols == 0 or k <= 0:
        return np.empty((rows, 0), dtype=np.int32), np.empty((rows, 0), dtype=np.float32)

    k_eff = max(1, min(int(k), cols))
    block_rows = max(1, int(row_block_size))
    top_idx = np.empty((rows, k_eff), dtype=np.int32)
    top_sims = np.empty((rows, k_eff), dtype=np.float32)

    for start in range(0, rows, block_rows):
        stop = min(rows, start + block_rows)
        sims = (features_l2[start:stop] @ ref_features_l2.T).astype(np.float32, copy=False)
        block_idx, block_sims = _topk_from_similarity_matrix(similarities=sims, k=k_eff)
        top_idx[start:stop] = block_idx
        top_sims[start:stop] = block_sims

    return top_idx, top_sims


def _top1_sims(
    sims: npt.NDArray[np.float32],
    *,
    rows: int,
) -> npt.NDArray[np.float32]:
    if sims.ndim != 2 or sims.shape[1] == 0:
        return np.zeros((rows,), dtype=np.float32)
    return sims[:, 0].astype(np.float32, copy=False)


def _compute_reference_knn_scores(
    *,
    features_l2: npt.NDArray[np.float32],
    ref_features_l2: npt.NDArray[np.float32],
    ref_labels: npt.NDArray[np.str_],
    top_k: int,
    row_block_size: int,
) -> ReferenceKNNScoring:
    rows = int(features_l2.shape[0]) if features_l2.ndim == 2 else 0
    empty_margin = np.zeros((rows,), dtype=np.float32)
    empty_like = np.full((rows,), 0.5, dtype=np.float32)
    empty_idx = np.empty((rows, 0), dtype=np.int32)
    empty_sims = np.empty((rows, 0), dtype=np.float32)

    if features_l2.size == 0 or ref_features_l2.size == 0 or ref_labels.size == 0:
        return ReferenceKNNScoring(
            margin=empty_margin,
            bad_likelihood=empty_like,
            reference_mode="none",
            rank_scores=empty_margin,
            bad_top1_similarity=empty_margin,
            good_top1_similarity=empty_margin,
            bad_neighbor_indices=empty_idx,
            bad_neighbor_sims=empty_sims,
            good_neighbor_indices=empty_idx,
            good_neighbor_sims=empty_sims,
        )

    bad_ref_ids = np.flatnonzero(ref_labels == "bad").astype(np.int32, copy=False)
    good_ref_ids = np.flatnonzero(ref_labels == "good").astype(np.int32, copy=False)

    bad_neighbor_indices = empty_idx
    bad_neighbor_sims = empty_sims
    good_neighbor_indices = empty_idx
    good_neighbor_sims = empty_sims

    if bad_ref_ids.size:
        bad_local_idx, bad_neighbor_sims = _exact_topk_reference_matches(
            features_l2=features_l2,
            ref_features_l2=ref_features_l2[bad_ref_ids],
            k=top_k,
            row_block_size=row_block_size,
        )
        bad_neighbor_indices = bad_ref_ids[bad_local_idx] if bad_local_idx.size else empty_idx

    if good_ref_ids.size:
        good_local_idx, good_neighbor_sims = _exact_topk_reference_matches(
            features_l2=features_l2,
            ref_features_l2=ref_features_l2[good_ref_ids],
            k=top_k,
            row_block_size=row_block_size,
        )
        good_neighbor_indices = good_ref_ids[good_local_idx] if good_local_idx.size else empty_idx

    bad_top1 = _top1_sims(bad_neighbor_sims, rows=rows)
    good_top1 = _top1_sims(good_neighbor_sims, rows=rows)

    if bad_ref_ids.size and good_ref_ids.size:
        margin = (bad_top1 - good_top1).astype(np.float32, copy=False)
        bad_like = _sigmoid((AML_REFERENCE_LOGIT_SCALE * margin).astype(np.float32, copy=False))
        rank_scores = bad_top1.astype(np.float32, copy=False)
        mode = "good_bad_exact_knn"
    elif bad_ref_ids.size:
        margin = bad_top1.astype(np.float32, copy=False)
        bad_like = _sigmoid((AML_REFERENCE_LOGIT_SCALE * margin).astype(np.float32, copy=False))
        rank_scores = bad_top1.astype(np.float32, copy=False)
        mode = "bad_only_exact_knn"
    elif good_ref_ids.size:
        margin = (-good_top1).astype(np.float32, copy=False)
        bad_like = _sigmoid((AML_REFERENCE_LOGIT_SCALE * margin).astype(np.float32, copy=False))
        rank_scores = (-good_top1).astype(np.float32, copy=False)
        mode = "good_only_exact_knn"
    else:
        margin = empty_margin
        bad_like = empty_like
        rank_scores = empty_margin
        mode = "none"

    return ReferenceKNNScoring(
        margin=margin,
        bad_likelihood=bad_like,
        reference_mode=mode,
        rank_scores=rank_scores,
        bad_top1_similarity=bad_top1,
        good_top1_similarity=good_top1,
        bad_neighbor_indices=bad_neighbor_indices,
        bad_neighbor_sims=bad_neighbor_sims,
        good_neighbor_indices=good_neighbor_indices,
        good_neighbor_sims=good_neighbor_sims,
    )


def _suppress_artifact_outliers(
    novelty: npt.NDArray[np.float32],
    *,
    percentile: float,
) -> npt.NDArray[np.float32]:
    """Cap extreme novelty scores at `percentile` to prevent artifact tiles
    (folds, pen marks, torn edges) from dominating the ranking after z-scoring.

    Tiles above the ceiling are clipped to the ceiling value — they remain
    present and their relative ordering is preserved up to that cap, but they
    cannot pull the z-score distribution so far that all normal tissue tiles
    collapse to near-zero score.

    Set ROI_NOVELTY_CLIP_PERCENTILE=100 to disable.
    """
    if novelty.size == 0 or percentile >= 100.0:
        return novelty
    ceiling = float(np.percentile(novelty, percentile))
    return np.minimum(novelty, ceiling).astype(np.float32, copy=False)


def _novelty_scores_from_knn(
    features_l2: npt.NDArray[np.float32],
    *,
    k_neighbors: int,
) -> npt.NDArray[np.float32]:
    n = int(features_l2.shape[0])
    if n == 0:
        return np.empty((0,), dtype=np.float32)
    if n == 1:
        return np.zeros((1,), dtype=np.float32)

    k_eff = max(2, min(k_neighbors + 1, n))

    if hnswlib is not None:
        index = hnswlib.Index(space="cosine", dim=int(features_l2.shape[1]))
        index.init_index(
            max_elements=n,
            ef_construction=200,
            M=32,
            random_seed=ROI_KNN_RANDOM_SEED,
        )
        ids = np.arange(n, dtype=np.int64)
        index.add_items(features_l2, ids, num_threads=1)
        index.set_ef(max(64, k_eff))
        _, distances = index.knn_query(features_l2, k=k_eff)
        # For cosine space in hnswlib, distance = 1 - cosine_similarity.
        return np.mean(distances[:, 1:], axis=1).astype(np.float32, copy=False)

    # Fallback if hnswlib is unavailable.
    k_no_self = max(1, min(k_neighbors, n - 1))
    sims = features_l2 @ features_l2.T
    np.fill_diagonal(sims, -1.0)
    nearest = np.partition(sims, kth=n - k_no_self, axis=1)[:, -k_no_self:]
    dists = 1.0 - nearest
    return np.mean(dists, axis=1).astype(np.float32, copy=False)


def build_unsupervised_roi_index(
    *,
    slide_path: str | Path,
    extractor_name: str = "uni2",
    tile_size_um: float = 256.0,
    tile_size_px: int = 224,
    batch_size: int = 32,
    device: str | None = None,
    cache_dir: Path | None = None,
    max_supertile_size_slide_px: int = 4096,
    max_workers: int = 4,
    brightness_cutoff: int | None = 240,
    canny_cutoff: float | None = 0.02,
    default_slide_mpp: float | None = None,
    tile_prefilter_method: str = "none",
    coarse_trigger_supertile_count: int | None = None,
    coarse_keep_ratio: float | None = None,
    coarse_min_keep_supertile_count: int = 0,
    coarse_max_keep_supertile_count: int | None = None,
    quality_keep_ratio: float | None = None,
    quality_min_keep_tile_count: int = 0,
    quality_trigger_tile_count: int | None = None,
    quality_random_reserve_ratio: float | None = None,
    k_neighbors: int = 20,
    use_reference_labels: bool = False,
    reference_tiles_root: str | Path | None = None,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> UnsupervisedROIIndex:
    slide_path = Path(slide_path).resolve()
    if progress_cb is not None:
        progress_cb({"phase": "load_extractor", "status": "running"})

    # Load the specified extractor
    from wsi_core_pkg.embeddings import get_embedding_extractor
    extractor = get_embedding_extractor(extractor_name)
    if progress_cb is not None:
        progress_cb(
            {
                "phase": "load_extractor",
                "status": "done",
                "extractor_id": extractor.identifier,
            }
        )

    if progress_cb is not None:
        progress_cb({"phase": "extract_embeddings", "status": "running"})
    result = extract_wsi_features_by_tiles(
        slide_path=slide_path,
        extractor=extractor,
        tile_size_um=tile_size_um,
        tile_size_px=tile_size_px,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
        max_supertile_size_slide_px=max_supertile_size_slide_px,
        max_workers=max_workers,
        brightness_cutoff=brightness_cutoff,
        canny_cutoff=canny_cutoff,
        default_slide_mpp=default_slide_mpp,
        tile_prefilter_method=tile_prefilter_method,
        coarse_trigger_supertile_count=coarse_trigger_supertile_count,
        coarse_keep_ratio=coarse_keep_ratio,
        coarse_min_keep_supertile_count=coarse_min_keep_supertile_count,
        coarse_max_keep_supertile_count=coarse_max_keep_supertile_count,
        quality_keep_ratio=quality_keep_ratio,
        quality_min_keep_tile_count=quality_min_keep_tile_count,
        quality_trigger_tile_count=quality_trigger_tile_count,
        quality_random_reserve_ratio=quality_random_reserve_ratio,
        progress_cb=progress_cb,
        use_amp=True,
    )

    features = result.features.numpy().astype(np.float32, copy=False)
    num_tiles = int(features.shape[0]) if features.ndim == 2 else 0
    feature_dim = int(features.shape[1]) if features.ndim == 2 and features.size else 0

    if num_tiles == 0 or feature_dim == 0:
        return UnsupervisedROIIndex(
            slide_path=str(slide_path),
            extractor_id=result.extractor_id,
            tile_size_um=float(tile_size_um),
            tile_size_px=int(tile_size_px),
            tile_size_level0_px=int(tile_size_px),
            coordinates_level0_xy=np.empty((0, 2), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            num_tiles=0,
            feature_dim=feature_dim,
            bad_margin=np.empty((0,), dtype=np.float32),
            bad_likelihood=np.empty((0,), dtype=np.float32),
            reference_mode="none",
            reference_stats={},
            bad_neighbor_indices=np.empty((0, 0), dtype=np.int32),
            bad_neighbor_sims=np.empty((0, 0), dtype=np.float32),
            good_neighbor_indices=np.empty((0, 0), dtype=np.int32),
            good_neighbor_sims=np.empty((0, 0), dtype=np.float32),
            reference_tile_paths=(),
            reference_tile_labels=(),
            reference_neighbor_k=0,
        )

    features_l2 = _l2_normalize_rows(features)
    scores = np.zeros((num_tiles,), dtype=np.float32)
    bad_margin = np.zeros((num_tiles,), dtype=np.float32)
    bad_likelihood = np.full((num_tiles,), 0.5, dtype=np.float32)
    reference_mode = "none"
    reference_stats: dict[str, Any] = {}
    bad_neighbor_indices = np.empty((num_tiles, 0), dtype=np.int32)
    bad_neighbor_sims = np.empty((num_tiles, 0), dtype=np.float32)
    good_neighbor_indices = np.empty((num_tiles, 0), dtype=np.int32)
    good_neighbor_sims = np.empty((num_tiles, 0), dtype=np.float32)
    reference_tile_paths: tuple[str, ...] = ()
    reference_tile_labels: tuple[str, ...] = ()
    reference_neighbor_k = 0
    ref_root = Path(reference_tiles_root).resolve() if (use_reference_labels and reference_tiles_root) else None
    ref_records = _discover_reference_tiles(ref_root) if ref_root else []
    use_retrieval_ranking = False

    if use_reference_labels:
        good_n = sum(1 for _, lbl in ref_records if lbl == "good")
        bad_n = sum(1 for _, lbl in ref_records if lbl == "bad")
        reference_stats = {
            "reference_tiles_root": str(ref_root) if ref_root else None,
            "reference_tiles_total": len(ref_records),
            "reference_tiles_good": int(good_n),
            "reference_tiles_bad": int(bad_n),
            "reference_neighbor_k": int(AML_REFERENCE_TOP_K),
            "reference_similarity": "cosine_exact",
            "ranking_strategy": "raw_exact_retrieval",
        }

        if ref_records:
            if progress_cb is not None:
                progress_cb(
                    {
                        "phase": "embed_reference_tiles",
                        "status": "running",
                        "reference_tiles_total": len(ref_records),
                        "reference_tiles_good": int(good_n),
                        "reference_tiles_bad": int(bad_n),
                    }
                )

            run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            ref_feat_l2, ref_labels, ref_paths = _embed_reference_tiles(
                records=ref_records,
                extractor=extractor,
                device=run_device,
                batch_size=max(1, min(64, int(batch_size))),
            )
            retrieval = _compute_reference_knn_scores(
                features_l2=features_l2,
                ref_features_l2=ref_feat_l2,
                ref_labels=ref_labels,
                top_k=AML_REFERENCE_TOP_K,
                row_block_size=AML_REFERENCE_QUERY_BLOCK_ROWS,
            )
            bad_margin = retrieval.margin
            bad_likelihood = retrieval.bad_likelihood
            reference_mode = retrieval.reference_mode
            bad_neighbor_indices = retrieval.bad_neighbor_indices
            bad_neighbor_sims = retrieval.bad_neighbor_sims
            good_neighbor_indices = retrieval.good_neighbor_indices
            good_neighbor_sims = retrieval.good_neighbor_sims
            reference_tile_paths = ref_paths
            reference_tile_labels = tuple(str(label) for label in ref_labels.tolist())
            reference_neighbor_k = int(AML_REFERENCE_TOP_K)

            if reference_mode != "none":
                scores = retrieval.rank_scores.astype(np.float32, copy=False)
                use_retrieval_ranking = True

            reference_stats.update(
                {
                    "reference_mode": reference_mode,
                    "wsi_bad_like_fraction": float(np.mean(bad_likelihood >= 0.5)),
                    "wsi_bad_like_strong_fraction": float(np.mean(bad_likelihood >= 0.65)),
                    "reference_query_block_rows": int(AML_REFERENCE_QUERY_BLOCK_ROWS),
                }
            )
            if progress_cb is not None:
                progress_cb(
                    {
                        "phase": "embed_reference_tiles",
                        "status": "done",
                        "reference_mode": reference_mode,
                        "wsi_bad_like_fraction": reference_stats["wsi_bad_like_fraction"],
                    }
                )
        else:
            reference_mode = "no_reference_tiles"
            reference_stats["reference_mode"] = reference_mode

    if not use_retrieval_ranking:
        if progress_cb is not None:
            progress_cb(
                {
                    "phase": "build_knn",
                    "status": "running",
                    "num_tiles": num_tiles,
                    "feature_dim": feature_dim,
                }
            )
        novelty = _novelty_scores_from_knn(features_l2, k_neighbors=k_neighbors)
        novelty = _suppress_artifact_outliers(novelty, percentile=ROI_NOVELTY_CLIP_PERCENTILE)

        centroid = np.mean(features_l2, axis=0, keepdims=True).astype(np.float32, copy=False)
        centroid = _l2_normalize_rows(centroid)[0]
        centroid_dist = (1.0 - (features_l2 @ centroid)).astype(np.float32, copy=False)

        scores = (WSI_W_NOVELTY * _zscore(novelty) + WSI_W_CENTROID * _zscore(centroid_dist)).astype(np.float32, copy=False)
        if progress_cb is not None:
            progress_cb(
                {
                    "phase": "build_knn",
                    "status": "done",
                    "num_tiles": num_tiles,
                    "feature_dim": feature_dim,
                }
            )
    default_mpp_obj = SlideMPP(default_slide_mpp) if default_slide_mpp is not None else None
    slide_mpp = get_slide_mpp_(slide_path, default_mpp=default_mpp_obj)
    if slide_mpp is None:
        raise RuntimeError("Could not infer slide MPP for ROI index.")
    slide_mpp_f = float(slide_mpp)

    tile_size_level0_px = int(np.ceil(float(tile_size_um) / slide_mpp_f))
    xy_um = result.coordinates_um.astype(np.float32, copy=False)
    coordinates_level0_xy = (xy_um / slide_mpp_f).astype(np.float32, copy=False)

    if progress_cb is not None:
        progress_cb(
            {
                "phase": "rank_candidates",
                "status": "done",
                "num_tiles": num_tiles,
                "feature_dim": feature_dim,
            }
        )

    return UnsupervisedROIIndex(
        slide_path=str(slide_path),
        extractor_id=result.extractor_id,
        tile_size_um=float(tile_size_um),
        tile_size_px=int(tile_size_px),
        tile_size_level0_px=max(1, tile_size_level0_px),
        coordinates_level0_xy=coordinates_level0_xy,
        scores=scores,
        num_tiles=num_tiles,
        feature_dim=feature_dim,
        bad_margin=bad_margin,
        bad_likelihood=bad_likelihood,
        reference_mode=reference_mode,
        reference_stats=reference_stats,
        bad_neighbor_indices=bad_neighbor_indices,
        bad_neighbor_sims=bad_neighbor_sims,
        good_neighbor_indices=good_neighbor_indices,
        good_neighbor_sims=good_neighbor_sims,
        reference_tile_paths=reference_tile_paths,
        reference_tile_labels=reference_tile_labels,
        reference_neighbor_k=reference_neighbor_k,
    )


def _reference_matches_for_tile(
    *,
    index: UnsupervisedROIIndex,
    tile_idx: int,
    max_items: int = AML_REFERENCE_EVIDENCE_PER_CLASS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, float]:
    def _collect(
        neighbor_indices: npt.NDArray[np.int32],
        neighbor_sims: npt.NDArray[np.float32],
    ) -> list[dict[str, Any]]:
        if neighbor_indices.ndim != 1 or neighbor_sims.ndim != 1:
            return []

        out: list[dict[str, Any]] = []
        limit = max(0, min(int(max_items), int(neighbor_indices.shape[0]), int(neighbor_sims.shape[0])))
        for ref_idx, sim in zip(neighbor_indices[:limit], neighbor_sims[:limit]):
            ref_i = int(ref_idx)
            if ref_i < 0 or ref_i >= len(index.reference_tile_paths):
                continue
            out.append(
                {
                    "path": index.reference_tile_paths[ref_i],
                    "name": Path(index.reference_tile_paths[ref_i]).name,
                    "label": index.reference_tile_labels[ref_i] if ref_i < len(index.reference_tile_labels) else "",
                    "similarity": float(sim),
                }
            )
        return out

    bad_refs: list[dict[str, Any]] = []
    good_refs: list[dict[str, Any]] = []
    bad_top1 = 0.0
    good_top1 = 0.0

    if index.bad_neighbor_sims.ndim == 2 and index.bad_neighbor_sims.shape[0] > tile_idx:
        bad_top1 = float(index.bad_neighbor_sims[tile_idx, 0]) if index.bad_neighbor_sims.shape[1] else 0.0
        bad_refs = _collect(index.bad_neighbor_indices[tile_idx], index.bad_neighbor_sims[tile_idx])
    if index.good_neighbor_sims.ndim == 2 and index.good_neighbor_sims.shape[0] > tile_idx:
        good_top1 = float(index.good_neighbor_sims[tile_idx, 0]) if index.good_neighbor_sims.shape[1] else 0.0
        good_refs = _collect(index.good_neighbor_indices[tile_idx], index.good_neighbor_sims[tile_idx])

    return bad_refs, good_refs, bad_top1, good_top1


def select_topk_candidates_for_view(
    *,
    index: UnsupervisedROIIndex,
    view_bbox_level0: tuple[int, int, int, int],
    top_k: int = 12,
    min_center_separation_px: int = 256,
) -> list[dict[str, Any]]:
    if top_k <= 0 or index.num_tiles == 0:
        return []

    vx0, vy0, vw, vh = view_bbox_level0
    vx1 = vx0 + vw
    vy1 = vy0 + vh

    half = index.tile_size_level0_px / 2.0
    cx = index.coordinates_level0_xy[:, 0] + half
    cy = index.coordinates_level0_xy[:, 1] + half

    in_view = (cx >= vx0) & (cx <= vx1) & (cy >= vy0) & (cy <= vy1)
    idxs = np.nonzero(in_view)[0]
    if idxs.size == 0:
        return []

    order = idxs[np.argsort(index.scores[idxs])[::-1]]
    # Keep candidates spatially distinct: require near-tile-sized center spacing.
    # This reduces heavy overlap even when tile_size_level0_px is larger than the
    # external min_center_separation_px setting.
    adaptive_min_sep_px = max(int(max(1, min_center_separation_px)), int(round(index.tile_size_level0_px * 0.55)))
    min_sep_sq = float(adaptive_min_sep_px ** 2)

    def _bbox_iou(
        ax0: float,
        ay0: float,
        ax1: float,
        ay1: float,
        bx0: float,
        by0: float,
        bx1: float,
        by1: float,
    ) -> float:
        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)
        iw = max(0.0, inter_x1 - inter_x0)
        ih = max(0.0, inter_y1 - inter_y0)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, (ax1 - ax0)) * max(0.0, (ay1 - ay0))
        area_b = max(0.0, (bx1 - bx0)) * max(0.0, (by1 - by0))
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return float(inter / union)

    selected: list[int] = []
    selected_bboxes: list[tuple[float, float, float, float]] = []
    for tile_idx in order:
        cxi = float(cx[tile_idx])
        cyi = float(cy[tile_idx])
        half_tile = float(index.tile_size_level0_px) / 2.0
        tx0 = cxi - half_tile
        ty0 = cyi - half_tile
        tx1 = cxi + half_tile
        ty1 = cyi + half_tile
        if selected:
            too_close = False
            too_overlapped = False
            for prev_idx, prev_bbox in zip(selected, selected_bboxes):
                dx = cxi - float(cx[prev_idx])
                dy = cyi - float(cy[prev_idx])
                if (dx * dx + dy * dy) < min_sep_sq:
                    too_close = True
                    break
                iou = _bbox_iou(tx0, ty0, tx1, ty1, prev_bbox[0], prev_bbox[1], prev_bbox[2], prev_bbox[3])
                if iou > ROI_CANDIDATE_MAX_IOU:
                    too_overlapped = True
                    break
            if too_close:
                continue
            if too_overlapped:
                continue
        selected.append(int(tile_idx))
        selected_bboxes.append((tx0, ty0, tx1, ty1))
        if len(selected) >= top_k:
            break

    out: list[dict[str, Any]] = []
    for rank, tile_idx in enumerate(selected, start=1):
        cxi = int(round(float(cx[tile_idx])))
        cyi = int(round(float(cy[tile_idx])))
        tile_x0 = int(round(float(index.coordinates_level0_xy[tile_idx, 0])))
        tile_y0 = int(round(float(index.coordinates_level0_xy[tile_idx, 1])))
        tile_x1 = tile_x0 + index.tile_size_level0_px
        tile_y1 = tile_y0 + index.tile_size_level0_px

        cx_norm = int(round(((cxi - vx0) / max(1, vw)) * 999.0))
        cy_norm = int(round(((cyi - vy0) / max(1, vh)) * 999.0))
        cx_norm = max(0, min(999, cx_norm))
        cy_norm = max(0, min(999, cy_norm))

        half_x_norm = max(10, int(round((index.tile_size_level0_px / max(1, vw)) * 999.0 / 2.0)))
        half_y_norm = max(10, int(round((index.tile_size_level0_px / max(1, vh)) * 999.0 / 2.0)))
        bx0n = max(0, min(999, cx_norm - half_x_norm))
        by0n = max(0, min(999, cy_norm - half_y_norm))
        bx1n = max(0, min(999, cx_norm + half_x_norm))
        by1n = max(0, min(999, cy_norm + half_y_norm))
        bad_like = (
            float(index.bad_likelihood[tile_idx])
            if index.bad_likelihood.size > tile_idx
            else 0.5
        )
        bad_margin = (
            float(index.bad_margin[tile_idx])
            if index.bad_margin.size > tile_idx
            else 0.0
        )
        retrieval_score = float(index.scores[tile_idx])
        if index.reference_mode.startswith("good_bad_exact_knn"):
            if bad_margin > 0.0:
                quality_hint = "bad_like"
            elif bad_margin < 0.0:
                quality_hint = "good_like"
            else:
                quality_hint = "uncertain"
        elif bad_like >= 0.60:
            quality_hint = "bad_like"
        elif bad_like <= 0.40:
            quality_hint = "good_like"
        else:
            quality_hint = "uncertain"
        bad_refs, good_refs, bad_top1, good_top1 = _reference_matches_for_tile(index=index, tile_idx=int(tile_idx))

        out.append(
            {
                "rank": rank,
                "tile_index": int(tile_idx),
                "score": float(index.scores[tile_idx]),
                "retrieval_score": retrieval_score,
                "bad_likelihood": bad_like,
                "bad_margin": bad_margin,
                "quality_hint": quality_hint,
                "reference_mode": index.reference_mode,
                "reference_neighbor_k": index.reference_neighbor_k,
                "bad_top1_similarity": bad_top1,
                "good_top1_similarity": good_top1,
                "retrieved_bad_refs": bad_refs,
                "retrieved_good_refs": good_refs,
                "center_norm": [cx_norm, cy_norm],
                "bbox_norm": [bx0n, by0n, bx1n, by1n],
                "center_level0": [cxi, cyi],
                "tile_bbox_level0": [tile_x0, tile_y0, tile_x1, tile_y1],
            }
        )

    return out


__all__ = [
    "UnsupervisedROIIndex",
    "build_unsupervised_roi_index",
    "select_topk_candidates_for_view",
]
