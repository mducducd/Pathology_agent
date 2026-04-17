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
from ..tuning_config import tuning_section

try:
    import hnswlib
except Exception:  # pragma: no cover - optional runtime acceleration
    hnswlib = None

SHARED_AML_SUPPORT_CFG = tuning_section("shared.aml_quality_support")
ROI_RANKER_CORE_CFG = tuning_section("roi_ranker.core")
ROI_RANKER_REFERENCE_HNSW_CFG = tuning_section("roi_ranker.reference_hnsw")
ROI_RANKER_REFERENCE_SCORING_CFG = tuning_section("roi_ranker.reference_scoring")
ROI_RANKER_BLAST_CFG = tuning_section("roi_ranker.blast_reference")
ROI_RANKER_VLLM_CFG = tuning_section("roi_ranker.vllm")
ROI_RANKER_KNN_GRAPH_CFG = tuning_section("roi_ranker.knn_graph")
ROI_RANKER_QUALITY_CFG = tuning_section("roi_ranker.quality_scoring")

# HNSW parameters for reference tile indexing
AML_REFERENCE_HNSW_M = int(ROI_RANKER_REFERENCE_HNSW_CFG["AML_REFERENCE_HNSW_M"])
AML_REFERENCE_HNSW_EF_CONSTRUCTION = int(ROI_RANKER_REFERENCE_HNSW_CFG["AML_REFERENCE_HNSW_EF_CONSTRUCTION"])
AML_REFERENCE_HNSW_EF_SEARCH = int(ROI_RANKER_REFERENCE_HNSW_CFG["AML_REFERENCE_HNSW_EF_SEARCH"])
AML_REFERENCE_USE_HNSW = bool(ROI_RANKER_REFERENCE_HNSW_CFG["AML_REFERENCE_USE_HNSW"])
AML_REFERENCE_EMBEDDING_CACHE = bool(ROI_RANKER_REFERENCE_HNSW_CFG["AML_REFERENCE_EMBEDDING_CACHE"])
AML_REFERENCE_CACHE_DIR = str(ROI_RANKER_REFERENCE_HNSW_CFG["AML_REFERENCE_CACHE_DIR"])

# Module-level cache for HNSW reference index
_reference_hnsw_index: hnswlib.Index | None = None
_reference_features: npt.NDArray[np.float32] | None = None
_reference_labels: npt.NDArray[np.str_] | None = None
_reference_paths: tuple[str, ...] | None = None

ROI_KNN_RANDOM_SEED = int(ROI_RANKER_CORE_CFG["ROI_KNN_RANDOM_SEED"])
ROI_CANDIDATE_MAX_IOU = float(ROI_RANKER_CORE_CFG["ROI_CANDIDATE_MAX_IOU"])
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Scoring weights for generic WSI mode: score = w_nov*z(novelty) + w_cen*z(centroid_dist)
# Reduced novelty weight to less aggressive outlier-seeking (reduces false positives)
WSI_W_NOVELTY = float(ROI_RANKER_CORE_CFG["WSI_W_NOVELTY"])
WSI_W_CENTROID = float(ROI_RANKER_CORE_CFG["WSI_W_CENTROID"])
ROI_NOVELTY_CLIP_PERCENTILE = float(ROI_RANKER_CORE_CFG["ROI_NOVELTY_CLIP_PERCENTILE"])

AML_REFERENCE_TOP_K = int(ROI_RANKER_REFERENCE_SCORING_CFG["AML_REFERENCE_TOP_K"])
AML_REFERENCE_QUERY_BLOCK_ROWS = int(ROI_RANKER_REFERENCE_SCORING_CFG["AML_REFERENCE_QUERY_BLOCK_ROWS"])
AML_REFERENCE_LOGIT_SCALE = float(ROI_RANKER_REFERENCE_SCORING_CFG["AML_REFERENCE_LOGIT_SCALE"])
AML_REFERENCE_EVIDENCE_PER_CLASS = int(ROI_RANKER_REFERENCE_SCORING_CFG["AML_REFERENCE_EVIDENCE_PER_CLASS"])
AML_DISABLE_BAD_REFERENCES = bool(ROI_RANKER_REFERENCE_SCORING_CFG["AML_DISABLE_BAD_REFERENCES"])
AML_BAD_TOP1_REJECT_THRESHOLD = float(ROI_RANKER_REFERENCE_SCORING_CFG["AML_BAD_TOP1_REJECT_THRESHOLD"])
AML_BAD_TOP1_AMBIGUOUS_THRESHOLD = float(ROI_RANKER_REFERENCE_SCORING_CFG["AML_BAD_TOP1_AMBIGUOUS_THRESHOLD"])
AML_GOOD_BAD_TOP1_MIN_GAP = float(ROI_RANKER_REFERENCE_SCORING_CFG["AML_GOOD_BAD_TOP1_MIN_GAP"])
AML_BAD_LIKE_REJECT_THRESHOLD = float(ROI_RANKER_REFERENCE_SCORING_CFG["AML_BAD_LIKE_REJECT_THRESHOLD"])
AML_BAD_MARGIN_REJECT_THRESHOLD = float(ROI_RANKER_REFERENCE_SCORING_CFG["AML_BAD_MARGIN_REJECT_THRESHOLD"])
AML_GOOD_LIKE_MARGIN_THRESHOLD = float(ROI_RANKER_REFERENCE_SCORING_CFG["AML_GOOD_LIKE_MARGIN_THRESHOLD"])
AML_GOOD_LIKE_BAD_LIKELIHOOD_MAX = float(ROI_RANKER_REFERENCE_SCORING_CFG["AML_GOOD_LIKE_BAD_LIKELIHOOD_MAX"])
AML_GOOD_LIKE_TOP1_GAP = float(ROI_RANKER_REFERENCE_SCORING_CFG["AML_GOOD_LIKE_TOP1_GAP"])
AML_BORDERLINE_BAD_LIKELIHOOD = float(ROI_RANKER_REFERENCE_SCORING_CFG["AML_BORDERLINE_BAD_LIKELIHOOD"])
AML_BORDERLINE_BAD_MARGIN_MAX = float(ROI_RANKER_REFERENCE_SCORING_CFG["AML_BORDERLINE_BAD_MARGIN_MAX"])
AML_BORDERLINE_BAD_TOP1_GAP = float(ROI_RANKER_REFERENCE_SCORING_CFG["AML_BORDERLINE_BAD_TOP1_GAP"])
AML_DARK_VIEW_PERCENTILE = float(ROI_RANKER_REFERENCE_SCORING_CFG["AML_DARK_VIEW_PERCENTILE"])
AML_DARK_VIEW_MIN_TILES = int(ROI_RANKER_REFERENCE_SCORING_CFG["AML_DARK_VIEW_MIN_TILES"])
AML_REFERENCE_AGGREGATION = str(ROI_RANKER_REFERENCE_SCORING_CFG["AML_REFERENCE_AGGREGATION"])
AML_MIN_DARK_SCORE_PERCENTILE = float(ROI_RANKER_REFERENCE_SCORING_CFG["AML_MIN_DARK_SCORE_PERCENTILE"])
AML_QUALITY_REJECT_MARGIN = float(ROI_RANKER_REFERENCE_SCORING_CFG["AML_QUALITY_REJECT_MARGIN"])

# Blast cell reference configuration
AML_BLAST_CELLS_ROOT = str(ROI_RANKER_BLAST_CFG["AML_BLAST_CELLS_ROOT"])
AML_BLAST_SIMILARITY_WEIGHT = float(ROI_RANKER_BLAST_CFG["AML_BLAST_SIMILARITY_WEIGHT"])
AML_BLAST_TOP_K = int(ROI_RANKER_BLAST_CFG["AML_BLAST_TOP_K"])
AML_ENABLE_BLAST_REFERENCES = bool(ROI_RANKER_BLAST_CFG["AML_ENABLE_BLAST_REFERENCES"])

# Cache for blast cell embeddings
_blast_features: npt.NDArray[np.float32] | None = None
_blast_paths: tuple[str, ...] | None = None
_blast_extractor_id: str | None = None

# VLLM optimization: pre-filter candidates before sending to VLM
# These filters remove low-quality, out-of-domain candidates EARLY to reduce VLM token count
VLLM_PREFILTER_ENABLED = bool(ROI_RANKER_VLLM_CFG["VLLM_PREFILTER_ENABLED"])
VLLM_MAX_CANDIDATES = int(ROI_RANKER_VLLM_CFG["VLLM_MAX_CANDIDATES"])
VLLM_MIN_DARK_SCORE = float(ROI_RANKER_VLLM_CFG["VLLM_MIN_DARK_SCORE"])
VLLM_BAD_LIKE_REJECT = bool(ROI_RANKER_VLLM_CFG["VLLM_BAD_LIKE_REJECT"])
AML_VLLM_GOOD_SUPPORT_EXEMPTION_TOP1_MIN = float(ROI_RANKER_VLLM_CFG["AML_VLLM_GOOD_SUPPORT_EXEMPTION_TOP1_MIN"])
AML_SUPPORT_GOOD_TOP1_FLOOR = float(SHARED_AML_SUPPORT_CFG["AML_SUPPORT_GOOD_TOP1_FLOOR"])
AML_SUPPORT_BAD_TOP1_CEILING = float(SHARED_AML_SUPPORT_CFG["AML_SUPPORT_BAD_TOP1_CEILING"])
AML_SUPPORT_BAD_LIKE_CEILING = float(SHARED_AML_SUPPORT_CFG["AML_SUPPORT_BAD_LIKE_CEILING"])
ROI_KNN_HNSW_EF_CONSTRUCTION = int(ROI_RANKER_KNN_GRAPH_CFG["ROI_KNN_HNSW_EF_CONSTRUCTION"])
ROI_KNN_HNSW_M = int(ROI_RANKER_KNN_GRAPH_CFG["ROI_KNN_HNSW_M"])
ROI_KNN_HNSW_EF_SEARCH_MIN = int(ROI_RANKER_KNN_GRAPH_CFG["ROI_KNN_HNSW_EF_SEARCH_MIN"])
AML_QUALITY_PRIOR_BAD_MARGIN_WEIGHT = float(ROI_RANKER_QUALITY_CFG["AML_QUALITY_PRIOR_BAD_MARGIN_WEIGHT"])
AML_QUALITY_PRIOR_SIMILARITY_GAP_WEIGHT = float(ROI_RANKER_QUALITY_CFG["AML_QUALITY_PRIOR_SIMILARITY_GAP_WEIGHT"])
AML_BAD_LIKE_SOFT_PENALTY_BASELINE = float(ROI_RANKER_QUALITY_CFG["AML_BAD_LIKE_SOFT_PENALTY_BASELINE"])
AML_BAD_MATCH_SOFT_PENALTY_OFFSET = float(ROI_RANKER_QUALITY_CFG["AML_BAD_MATCH_SOFT_PENALTY_OFFSET"])
AML_BAD_MATCH_SOFT_PENALTY_SCALE = float(ROI_RANKER_QUALITY_CFG["AML_BAD_MATCH_SOFT_PENALTY_SCALE"])
AML_QUALITY_PENALTY_BAD_LIKE_WEIGHT = float(ROI_RANKER_QUALITY_CFG["AML_QUALITY_PENALTY_BAD_LIKE_WEIGHT"])
AML_QUALITY_PENALTY_BAD_MATCH_WEIGHT = float(ROI_RANKER_QUALITY_CFG["AML_QUALITY_PENALTY_BAD_MATCH_WEIGHT"])
AML_ABSOLUTE_MIN_DARK_SCORE = float(ROI_RANKER_QUALITY_CFG["AML_ABSOLUTE_MIN_DARK_SCORE"])
AML_COMBINED_RANK_DARK_WEIGHT = float(ROI_RANKER_QUALITY_CFG["AML_COMBINED_RANK_DARK_WEIGHT"])
AML_COMBINED_RANK_BASE_SCORE_WEIGHT = float(ROI_RANKER_QUALITY_CFG["AML_COMBINED_RANK_BASE_SCORE_WEIGHT"])
AML_COMBINED_RANK_QUALITY_PRIOR_WEIGHT = float(ROI_RANKER_QUALITY_CFG["AML_COMBINED_RANK_QUALITY_PRIOR_WEIGHT"])
AML_COMBINED_RANK_QUALITY_PENALTY_WEIGHT = float(ROI_RANKER_QUALITY_CFG["AML_COMBINED_RANK_QUALITY_PENALTY_WEIGHT"])
AML_COMBINED_RANK_GOOD_SUPPORT_BONUS = float(ROI_RANKER_QUALITY_CFG["AML_COMBINED_RANK_GOOD_SUPPORT_BONUS"])
ROI_ADAPTIVE_MIN_SEPARATION_TILE_RATIO = float(ROI_RANKER_QUALITY_CFG["ROI_ADAPTIVE_MIN_SEPARATION_TILE_RATIO"])
AML_DARK_REGION_BOX_PRIOR_WEIGHT = 0.08


def _load_blast_cell_embeddings(
    blast_cells_root: str | Path,
    extractor: Any,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[npt.NDArray[np.float32], tuple[str, ...]] | None:
    """Load and embed blast cell images for similarity scoring.

    Args:
        blast_cells_root: Path to directory containing blast cell images
        extractor: Feature extractor model
        device: Torch device
        batch_size: Batch size for embedding

    Returns:
        Tuple of (features_l2, paths) or None if no blast cells found
    """
    global _blast_features, _blast_paths, _blast_extractor_id

    # Return cached embeddings if available
    extractor_id = getattr(extractor, "identifier", None)
    if _blast_features is not None and _blast_extractor_id == extractor_id:
        return _blast_features, _blast_paths

    blast_path = Path(blast_cells_root)
    if not blast_path.exists():
        return None

    # Discover blast cell images
    blast_images: list[Path] = []
    seen: set[Path] = set()
    for ext in IMAGE_EXTS:
        for path in blast_path.rglob(f"*{ext}"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            blast_images.append(resolved)

    if not blast_images:
        return None

    # Embed blast cells
    model = extractor.model.to(device)
    model.eval()

    batch_tensors: list[torch.Tensor] = []
    feature_chunks: list[torch.Tensor] = []
    paths: list[str] = []

    for img_path in blast_images:
        try:
            with Image.open(img_path) as im:
                rgb = im.convert("RGB")
                batch_tensors.append(_prepare_input_tensor(rgb, extractor.transform))
                paths.append(str(img_path))
        except Exception:
            continue

        if len(batch_tensors) >= batch_size:
            x = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
            with torch.no_grad():
                y = model(x)
            feature_chunks.append(_normalize_feature_output(y).detach().cpu())
            batch_tensors.clear()

    if batch_tensors:
        x = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            y = model(x)
        feature_chunks.append(_normalize_feature_output(y).detach().cpu())

    if not feature_chunks:
        return None

    features = torch.cat(feature_chunks, dim=0).numpy().astype(np.float32, copy=False)
    features_l2 = _l2_normalize_rows(features)

    # Cache for future use
    _blast_features = features_l2
    _blast_paths = tuple(paths)
    _blast_extractor_id = extractor_id

    return features_l2, tuple(paths)

@dataclass(frozen=True)
class UnsupervisedROIIndex:
    slide_path: str
    extractor_id: str
    tile_size_um: float
    tile_size_px: int
    tile_size_level0_px: int
    coordinates_level0_xy: npt.NDArray[np.float32]
    scores: npt.NDArray[np.float32]
    dark_roi_scores: npt.NDArray[np.float32]
    num_tiles: int
    feature_dim: int
    blast_scores: npt.NDArray[np.float32] = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
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
    blast_neighbor_indices: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty((0, 0), dtype=np.int32))
    blast_neighbor_sims: npt.NDArray[np.float32] = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    blast_reference_paths: tuple[str, ...] = field(default_factory=tuple)
    blast_neighbor_k: int = 0
    quality_method: str = "embedding"  # "embedding" or "hybrid"


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


def _bad_reference_reject_mask(
    *,
    bad_top1: npt.NDArray[np.float32],
    good_top1: npt.NDArray[np.float32] | None = None,
    bad_like: npt.NDArray[np.float32] | None = None,
    bad_margin: npt.NDArray[np.float32] | None = None,
) -> npt.NDArray[np.bool_]:
    rows = int(bad_top1.shape[0]) if bad_top1.ndim == 1 else 0
    reject = np.zeros((rows,), dtype=np.bool_)
    if rows == 0:
        return reject

    reject |= bad_top1 >= AML_BAD_TOP1_REJECT_THRESHOLD
    if good_top1 is not None and good_top1.ndim == 1 and good_top1.shape == bad_top1.shape:
        reject |= (
            (bad_top1 >= AML_BAD_TOP1_AMBIGUOUS_THRESHOLD)
            & ((good_top1 - bad_top1) <= AML_GOOD_BAD_TOP1_MIN_GAP)
        )
    if bad_like is not None and bad_like.ndim == 1 and bad_like.shape == bad_top1.shape:
        reject |= bad_like >= AML_BAD_LIKE_REJECT_THRESHOLD
    if bad_margin is not None and bad_margin.ndim == 1 and bad_margin.shape == bad_top1.shape:
        reject |= bad_margin <= AML_BAD_MARGIN_REJECT_THRESHOLD
    return reject


def _bad_reference_is_rejected(
    *,
    bad_top1: float | None,
    good_top1: float | None,
    bad_like: float | None,
    bad_margin: float | None,
) -> bool:
    bad_top1_val = float(bad_top1) if isinstance(bad_top1, (int, float)) else 0.0
    strong_bad = bad_top1_val >= AML_BAD_TOP1_REJECT_THRESHOLD
    ambiguous_bad = False
    if isinstance(good_top1, (int, float)):
        ambiguous_bad = (
            bad_top1_val >= AML_BAD_TOP1_AMBIGUOUS_THRESHOLD
            and (float(good_top1) - bad_top1_val) <= AML_GOOD_BAD_TOP1_MIN_GAP
        )
    bad_like_reject = isinstance(bad_like, (int, float)) and float(bad_like) >= AML_BAD_LIKE_REJECT_THRESHOLD
    bad_margin_reject = isinstance(bad_margin, (int, float)) and float(bad_margin) <= AML_BAD_MARGIN_REJECT_THRESHOLD
    return bool(strong_bad or ambiguous_bad or bad_like_reject or bad_margin_reject)


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


def _active_reference_records(records: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    if not AML_DISABLE_BAD_REFERENCES:
        return list(records)
    return [(path, label) for path, label in records if label == "good"]


def _embed_reference_tiles(
    *,
    records: list[tuple[Path, str]],
    extractor: Any,
    device: torch.device,
    batch_size: int,
    extractor_id: str,
    use_cache: bool = True,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.str_], tuple[str, ...]]:
    if not records:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.str_), ()

    ref_paths = tuple(str(path) for path, _ in records)
    ref_labels_list = [label for _, label in records]

    # Respect environment variable for embedding cache
    use_cache = use_cache and AML_REFERENCE_EMBEDDING_CACHE

    # Try to load from cache first
    if use_cache:
        cached_result = _load_reference_embeddings_from_cache(ref_paths, extractor_id)
        if cached_result is not None:
            return cached_result

    # Cache miss - need to embed
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

    # Save to cache
    if use_cache:
        _save_reference_embeddings_to_cache(feat_l2, labels, ref_paths, extractor_id)

    return feat_l2, labels, tuple(path_buf)


def _build_hnsw_index(
    features: npt.NDArray[np.float32],
    space: str = "cosine",
    m: int | None = None,
    ef_construction: int | None = None,
    ef_search: int | None = None,
) -> hnswlib.Index:
    """Build a HNSW index from feature vectors.

    Args:
        features: L2-normalized feature array of shape (n_samples, dim)
        space: Distance space ("cosine", "l2", "ip")
        m: HNSW M parameter (number of connections per node)
        ef_construction: Size of dynamic candidate list during construction
        ef_search: Size of dynamic candidate list during search

    Returns:
        hnswlib.Index object with features added
    """
    if hnswlib is None:
        raise RuntimeError("hnswlib is not installed. Install with: pip install hnswlib")

    dim = int(features.shape[1])
    n = int(features.shape[0])
    m = m or AML_REFERENCE_HNSW_M
    ef_construction = ef_construction or AML_REFERENCE_HNSW_EF_CONSTRUCTION
    ef_search = ef_search or AML_REFERENCE_HNSW_EF_SEARCH

    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=m)
    index.set_ef(ef_search)

    ids = np.arange(n, dtype=np.int64)
    index.add_items(features, ids)

    return index


def _hnsw_knn_query(
    index: hnswlib.Index,
    query_features: npt.NDArray[np.float32],
    k: int,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Query a HNSW index for k-nearest neighbors.

    Args:
        index: Pre-built hnswlib.Index
        query_features: Query feature array of shape (n_queries, dim)
        k: Number of neighbors to return

    Returns:
        Tuple of (indices, distances) arrays, each of shape (n_queries, k)
    """
    if hnswlib is None:
        raise RuntimeError("hnswlib is not installed")

    k_eff = min(k, int(index.get_current_count()))
    labels, distances = index.knn_query(query_features, k=k_eff)

    # Convert distances to similarities for cosine space
    # In hnswlib cosine space: distance = 1 - cosine_similarity
    if index.space == "cosine":
        similarities = (1.0 - distances).astype(np.float32, copy=False)
    else:
        # For L2 or IP, keep distances as-is
        similarities = distances.astype(np.float32, copy=False)

    return labels.astype(np.int32, copy=False), similarities


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


def _aggregate_knn_similarities(
    sims: npt.NDArray[np.float32],
    *,
    method: str = "mean",
) -> npt.NDArray[np.float32]:
    """Aggregate K-NN similarities into a single score per query tile.

    Args:
        sims: Array of shape (n_query_tiles, k_neighbors) with similarity scores
        method: Aggregation method - "mean", "max", or "weighted"
            - "mean": Average of all K neighbors (robust, reduces noise)
            - "max": Maximum similarity (equivalent to top-1, kept for backward compat)
            - "weighted": Exponentially weighted mean, higher weight to closer neighbors

    Returns:
        Array of shape (n_query_tiles,) with aggregated scores
    """
    if sims.ndim != 2 or sims.shape[1] == 0:
        return np.zeros((sims.shape[0] if sims.ndim == 2 else 0,), dtype=np.float32)

    if method == "max":
        return np.max(sims, axis=1).astype(np.float32, copy=False)

    if method == "weighted":
        # Exponential decay weights: w_i = exp(-i/3) for i in 0..k-1
        k = sims.shape[1]
        weights = np.exp(-np.arange(k) / 3.0).astype(np.float32)
        weights = weights / weights.sum()
        return np.sum(sims * weights, axis=1).astype(np.float32, copy=False)

    # Default: mean aggregation
    return np.mean(sims, axis=1).astype(np.float32, copy=False)


def _ensure_reference_hnsw_index(
    ref_features_l2: npt.NDArray[np.float32],
    ref_labels: npt.NDArray[np.str_],
    ref_paths: tuple[str, ...],
    extractor_id: str = "uni2",
) -> tuple[hnswlib.Index | None, npt.NDArray[np.float32], npt.NDArray[np.str_], tuple[str, ...]]:
    """Build or retrieve cached HNSW index for reference tiles.

    Uses a two-level caching strategy:
    1. In-memory cache (module-level) for reuse across multiple slides in same session
    2. Persistent cache (disk) for reuse across sessions

    Args:
        ref_features_l2: L2-normalized reference features
        ref_labels: Reference labels ("good" or "bad")
        ref_paths: Reference tile paths
        extractor_id: Extractor identifier for cache key

    Returns:
        Tuple of (hnsw_index or None, features, labels, paths)
    """
    global _reference_hnsw_index, _reference_features, _reference_labels, _reference_paths

    # Check in-memory cache first
    if _reference_hnsw_index is not None:
        if (_reference_features is not None and
            _reference_features.shape == ref_features_l2.shape and
            _reference_features.shape[0] > 0 and
            np.allclose(_reference_features[0], ref_features_l2[0], atol=1e-6) and
            np.allclose(_reference_features[-1], ref_features_l2[-1], atol=1e-6)):
            return _reference_hnsw_index, _reference_features, _reference_labels, _reference_paths

    # Try persistent cache
    if hnswlib is not None:
        cached_index = _load_reference_hnsw_cache(ref_paths, ref_features_l2, extractor_id)
        if cached_index is not None:
            # Populate in-memory cache
            _reference_hnsw_index = cached_index
            _reference_features = ref_features_l2
            _reference_labels = ref_labels
            _reference_paths = ref_paths
            return cached_index, ref_features_l2, ref_labels, ref_paths

    # Build new index
    if hnswlib is None:
        return None, ref_features_l2, ref_labels, ref_paths

    try:
        _reference_hnsw_index = _build_hnsw_index(
            ref_features_l2,
            space="cosine",
            m=AML_REFERENCE_HNSW_M,
            ef_construction=AML_REFERENCE_HNSW_EF_CONSTRUCTION,
            ef_search=AML_REFERENCE_HNSW_EF_SEARCH,
        )
        _reference_features = ref_features_l2
        _reference_labels = ref_labels
        _reference_paths = ref_paths

        # Save to persistent cache
        _save_reference_hnsw_cache(ref_features_l2, ref_labels, ref_paths, extractor_id)

    except Exception:
        return None, ref_features_l2, ref_labels, ref_paths

    return _reference_hnsw_index, _reference_features, _reference_labels, _reference_paths


def _clear_reference_hnsw_cache() -> None:
    """Clear the cached HNSW reference index. Call when reference tiles change."""
    global _reference_hnsw_index, _reference_features, _reference_labels, _reference_paths
    _reference_hnsw_index = None
    _reference_features = None
    _reference_labels = None
    _reference_paths = None


def _clear_reference_embeddings_cache() -> None:
    """Clear the in-memory reference embeddings cache.

    Call when reference tiles change or when you need to force re-embedding.
    Note: This does NOT clear the persistent disk cache.
    """
    global _reference_embeddings_cache
    _reference_embeddings_cache = None


def clear_all_reference_caches() -> None:
    """Clear all reference-related caches (both in-memory and persistent disk cache).

    Call this when reference tiles have been modified, added, or removed.
    """
    global _persistent_cache_dir, _reference_embeddings_cache

    # Clear in-memory caches
    _clear_reference_hnsw_cache()
    _clear_reference_embeddings_cache()

    # Clear persistent disk cache
    if _persistent_cache_dir is not None and _persistent_cache_dir.exists():
        import shutil
        shutil.rmtree(str(_persistent_cache_dir))
    _persistent_cache_dir = None


# Persistent cache paths
_persistent_cache_dir: Path | None = None

# Module-level cache for precomputed reference embeddings
_reference_embeddings_cache: dict[str, tuple[npt.NDArray[np.float32], npt.NDArray[np.str_], tuple[str, ...]]] | None = None

def _get_persistent_cache_dir() -> Path:
    """Get or create the persistent cache directory for HNSW indices and embeddings."""
    global _persistent_cache_dir
    env_override = os.getenv("AML_REFERENCE_CACHE_DIR", "").strip()
    cache_root = Path(env_override) if env_override else Path(AML_REFERENCE_CACHE_DIR)
    if _persistent_cache_dir is None or _persistent_cache_dir != cache_root:
        cache_dir = cache_root
        cache_dir.mkdir(parents=True, exist_ok=True)
        _persistent_cache_dir = cache_dir
    return _persistent_cache_dir


def _compute_reference_fingerprint(
    ref_paths: tuple[str, ...],
    ref_features_l2: npt.NDArray[np.float32] | None = None,
) -> str:
    """Compute a fingerprint for cache invalidation.

    Uses file paths and modification times to detect changes.
    Optionally includes feature hash for additional verification.
    """
    import hashlib
    import os

    # Hash the sorted paths with their modification times
    path_mtime_pairs = []
    for p in sorted(ref_paths):
        try:
            mtime = os.path.getmtime(p)
            path_mtime_pairs.append(f"{p}:{mtime}")
        except OSError:
            path_mtime_pairs.append(f"{p}:0")
    paths_str = "|".join(path_mtime_pairs)

    # Optionally include feature hash if provided
    if ref_features_l2 is not None and ref_features_l2.size > 0:
        n_samples = min(10, len(ref_features_l2))
        indices = np.linspace(0, len(ref_features_l2) - 1, n_samples, dtype=int)
        feature_sample = ref_features_l2[indices].tobytes()
        combined = f"{paths_str}:{feature_sample.hex()}"
    else:
        combined = paths_str

    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _find_matching_reference_embedding_cache(
    cache_dir: Path,
    *,
    ref_paths: tuple[str, ...],
    extractor_id: str,
) -> tuple[Path, Path, Path] | None:
    """Locate a compatible on-disk embedding cache entry by metadata.

    This provides backward compatibility for cache files written before the
    cache-name fingerprint was stabilized for embedding reuse.
    """
    expected_paths = list(ref_paths)
    for meta_path in sorted(cache_dir.glob(f"{extractor_id}_embed_*_meta.json")):
        try:
            import json

            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if str(meta.get("extractor_id") or "") != str(extractor_id):
            continue
        if list(meta.get("paths") or []) != expected_paths:
            continue

        stem = meta_path.name.removesuffix("_meta.json")
        features_path = cache_dir / f"{stem}_features.npy"
        labels_path = cache_dir / f"{stem}_labels.npy"
        if features_path.exists() and labels_path.exists():
            return features_path, labels_path, meta_path
    return None


def _load_reference_embeddings_from_cache(
    ref_paths: tuple[str, ...],
    extractor_id: str,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.str_], tuple[str, ...]] | None:
    """Load precomputed reference embeddings from persistent cache.

    Returns (features_l2, labels, paths) if cache hit, None if cache miss or invalid.
    """
    global _reference_embeddings_cache

    # Check in-memory cache first
    if _reference_embeddings_cache is not None:
        cache_key = f"{extractor_id}_{len(ref_paths)}"
        if cache_key in _reference_embeddings_cache:
            cached_feats, cached_labels, cached_paths = _reference_embeddings_cache[cache_key]
            if cached_paths == ref_paths:
                return cached_feats, cached_labels, cached_paths

    # Try persistent cache
    try:
        cache_dir = _get_persistent_cache_dir()
        fingerprint = _compute_reference_fingerprint(ref_paths)
        cache_name = f"{extractor_id}_embed_{fingerprint}"

        features_path = cache_dir / f"{cache_name}_features.npy"
        labels_path = cache_dir / f"{cache_name}_labels.npy"
        meta_path = cache_dir / f"{cache_name}_meta.json"

        if not (features_path.exists() and labels_path.exists() and meta_path.exists()):
            matched = _find_matching_reference_embedding_cache(
                cache_dir,
                ref_paths=ref_paths,
                extractor_id=extractor_id,
            )
            if matched is None:
                return None
            features_path, labels_path, meta_path = matched

        # Verify cache metadata
        import json
        meta = json.loads(meta_path.read_text())
        meta_paths = tuple(str(path) for path in meta.get("paths") or ())
        if meta_paths != ref_paths:
            return None
        if str(meta.get("extractor_id") or "") != str(extractor_id):
            return None

        # Load cached data
        features_l2 = np.load(str(features_path), allow_pickle=False)
        ref_labels = np.load(str(labels_path), allow_pickle=True)

        # Validate shapes
        if features_l2.ndim != 2 or len(ref_labels) != len(ref_paths):
            return None

        # Populate in-memory cache
        cache_key = f"{extractor_id}_{len(ref_paths)}"
        _reference_embeddings_cache = _reference_embeddings_cache or {}
        _reference_embeddings_cache[cache_key] = (features_l2, ref_labels, ref_paths)

        return features_l2, ref_labels, ref_paths

    except Exception:
        return None


def _save_reference_embeddings_to_cache(
    ref_features_l2: npt.NDArray[np.float32],
    ref_labels: npt.NDArray[np.str_],
    ref_paths: tuple[str, ...],
    extractor_id: str,
) -> Path | None:
    """Save reference embeddings to persistent cache.

    Returns the cache directory path if successful, None otherwise.
    """
    global _reference_embeddings_cache

    try:
        cache_dir = _get_persistent_cache_dir()
        # Use only the reference paths/mtimes for the persistent embedding cache
        # key so future runs can locate the cache before re-embedding.
        fingerprint = _compute_reference_fingerprint(ref_paths)
        cache_name = f"{extractor_id}_embed_{fingerprint}"

        features_path = cache_dir / f"{cache_name}_features.npy"
        labels_path = cache_dir / f"{cache_name}_labels.npy"
        meta_path = cache_dir / f"{cache_name}_meta.json"

        # Save features and labels
        np.save(str(features_path), ref_features_l2, allow_pickle=False)
        np.save(str(labels_path), ref_labels, allow_pickle=True)

        # Save metadata
        import json
        meta = {
            "extractor_id": extractor_id,
            "count": len(ref_paths),
            "dim": int(ref_features_l2.shape[1]),
            "fingerprint": fingerprint,
            "paths": ref_paths,
            "created_at": __import__("datetime").datetime.now().isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        # Populate in-memory cache
        cache_key = f"{extractor_id}_{len(ref_paths)}"
        _reference_embeddings_cache = _reference_embeddings_cache or {}
        _reference_embeddings_cache[cache_key] = (ref_features_l2, ref_labels, ref_paths)

        return cache_dir
    except Exception:
        return None


def _save_reference_hnsw_cache(
    ref_features_l2: npt.NDArray[np.float32],
    ref_labels: npt.NDArray[np.str_],
    ref_paths: tuple[str, ...],
    extractor_id: str,
) -> Path | None:
    """Save reference index to persistent cache.

    Returns the cache directory path if successful, None otherwise.
    """
    if hnswlib is None:
        return None

    try:
        cache_dir = _get_persistent_cache_dir()
        fingerprint = _compute_reference_fingerprint(ref_paths, ref_features_l2)
        cache_name = f"{extractor_id}_{fingerprint}"

        index_path = cache_dir / f"{cache_name}.bin"
        meta_path = cache_dir / f"{cache_name}.json"
        labels_path = cache_dir / f"{cache_name}_labels.npy"

        # Build index
        index = _build_hnsw_index(ref_features_l2)
        index.save_index(str(index_path))

        # Save metadata
        import json
        meta = {
            "extractor_id": extractor_id,
            "count": len(ref_paths),
            "dim": int(ref_features_l2.shape[1]),
            "fingerprint": fingerprint,
            "paths": ref_paths,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        # Save labels
        np.save(str(labels_path), ref_labels, allow_pickle=True)

        return cache_dir
    except Exception:
        return None


def _load_reference_hnsw_cache(
    ref_paths: tuple[str, ...],
    ref_features_l2: npt.NDArray[np.float32],
    extractor_id: str,
) -> hnswlib.Index | None:
    """Load reference index from persistent cache.

    Returns the index if a valid cache exists, None otherwise.
    """
    if hnswlib is None:
        return None

    try:
        cache_dir = _get_persistent_cache_dir()
        fingerprint = _compute_reference_fingerprint(ref_paths, ref_features_l2)
        cache_name = f"{extractor_id}_{fingerprint}"

        index_path = cache_dir / f"{cache_name}.bin"
        meta_path = cache_dir / f"{cache_name}.json"
        labels_path = cache_dir / f"{cache_name}_labels.npy"

        if not (index_path.exists() and meta_path.exists() and labels_path.exists()):
            return None

        # Verify cache metadata
        import json
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") != fingerprint:
            return None

        # Load index
        dim = int(meta.get("dim", 0))
        count = int(meta.get("count", 0))
        if dim == 0 or count == 0:
            return None

        index = hnswlib.Index(space="cosine", dim=dim)
        index.load_index(str(index_path), max_elements=count)

        return index
    except Exception:
        return None


def _compute_reference_knn_scores(
    *,
    features_l2: npt.NDArray[np.float32],
    ref_features_l2: npt.NDArray[np.float32],
    ref_labels: npt.NDArray[np.str_],
    ref_paths: tuple[str, ...] = (),
    top_k: int,
    row_block_size: int,
    use_hnsw: bool | None = None,
    extractor_id: str = "uni2",
) -> ReferenceKNNScoring:
    """Compute reference-based KNN scoring for WSI tiles.

    Scoring philosophy for AML detection:
    - "good" tiles = HIGH QUALITY tiles (clear nuclei, high cellularity, low artifact)
    - "bad" tiles = LOW QUALITY tiles (blurry, folded, RBC-heavy, overstained, background)

    Good tiles provide reliable morphological evidence for AML assessment.
    Bad tiles are non-diagnostic and may introduce misleading signals.

    Scoring strategy:
    - Prioritize tiles similar to "good" reference tiles (high quality, interpretable)
    - Deprioritize tiles similar to "bad" reference tiles (artifact, noise, uninformative)
    - Uses aggregate similarity over K neighbors (not just top-1) for robustness

    Args:
        features_l2: WSI tile features (L2-normalized)
        ref_features_l2: Reference tile features (L2-normalized)
        ref_labels: Reference tile labels ("good" or "bad")
        top_k: Number of neighbors to retrieve from each class
        row_block_size: Block size for exact search (ignored if HNSW enabled)
        use_hnsw: Whether to use HNSW indexing. None = auto-detect based on ref size
    """
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
    if AML_DISABLE_BAD_REFERENCES:
        bad_ref_ids = np.empty((0,), dtype=np.int32)
    good_ref_ids = np.flatnonzero(ref_labels == "good").astype(np.int32, copy=False)

    bad_neighbor_indices = empty_idx
    bad_neighbor_sims = empty_sims
    good_neighbor_indices = empty_idx
    good_neighbor_sims = empty_sims

    # Determine whether to use HNSW
    if use_hnsw is None:
        # Auto-detect: use HNSW if more than 100 reference tiles and hnswlib is available
        use_hnsw = hnswlib is not None and len(ref_labels) > 100

    hnsw_idx = None
    if use_hnsw and hnswlib is not None:
        # Build or retrieve cached HNSW index
        hnsw_idx, _, _, _ = _ensure_reference_hnsw_index(ref_features_l2, ref_labels, ref_paths, extractor_id=extractor_id)

    if use_hnsw and hnsw_idx is not None:
        # HNSW mode: query the index for all neighbors at once
        all_indices, all_sims = _hnsw_knn_query(hnsw_idx, features_l2, k=top_k * 2)

        # Separate by label and build results
        bad_indices_list = []
        bad_sims_list = []
        good_indices_list = []
        good_sims_list = []

        for i in range(rows):
            bad_idx = []
            bad_sim = []
            good_idx = []
            good_sim = []

            for j in range(all_indices.shape[1]):
                ref_idx = all_indices[i, j]
                sim = all_sims[i, j]
                if ref_labels[ref_idx] == "bad":
                    bad_idx.append(ref_idx)
                    bad_sim.append(sim)
                else:
                    good_idx.append(ref_idx)
                    good_sim.append(sim)

            # Pad or truncate to top_k
            bad_idx = (bad_idx + [-1] * top_k)[:top_k]
            bad_sim = (bad_sim + [0.0] * top_k)[:top_k]
            good_idx = (good_idx + [-1] * top_k)[:top_k]
            good_sim = (good_sim + [0.0] * top_k)[:top_k]

            bad_indices_list.append(bad_idx)
            bad_sims_list.append(bad_sim)
            good_indices_list.append(good_idx)
            good_sims_list.append(good_sim)

        bad_neighbor_indices = np.array(bad_indices_list, dtype=np.int32)
        bad_neighbor_sims = np.array(bad_sims_list, dtype=np.float32)
        good_neighbor_indices = np.array(good_indices_list, dtype=np.int32)
        good_neighbor_sims = np.array(good_sims_list, dtype=np.float32)
        mode_prefix = "hnsw"
    else:
        # Exact search mode
        mode_prefix = "exact"
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

    # Compute aggregate scores over K neighbors (more robust than top-1 alone)
    bad_top1 = _top1_sims(bad_neighbor_sims, rows=rows)
    good_top1 = _top1_sims(good_neighbor_sims, rows=rows)
    bad_score_agg = _aggregate_knn_similarities(bad_neighbor_sims, method=AML_REFERENCE_AGGREGATION)
    good_score_agg = _aggregate_knn_similarities(good_neighbor_sims, method=AML_REFERENCE_AGGREGATION)

    # AML scoring: higher = more AML-like (similar to good tiles, dissimilar to bad)
    # margin = good_similarity - bad_similarity (positive = more AML-like)
    if bad_ref_ids.size and good_ref_ids.size:
        # Use aggregate scores for robustness
        margin = (good_score_agg - bad_score_agg).astype(np.float32, copy=False)
        bad_like = _sigmoid((AML_REFERENCE_LOGIT_SCALE * (-margin)).astype(np.float32, copy=False))
        # Rank by AML-likeness: how similar to good (AML) tiles, adjusted by bad dissimilarity
        rank_scores = good_score_agg.astype(np.float32, copy=False)
        mode = f"good_bad_{mode_prefix}_knn"
    elif bad_ref_ids.size:
        # Only bad references: rank by dissimilarity to bad (normal marrow)
        margin = (-bad_score_agg).astype(np.float32, copy=False)
        bad_like = _sigmoid((AML_REFERENCE_LOGIT_SCALE * (-margin)).astype(np.float32, copy=False))
        rank_scores = (-bad_score_agg).astype(np.float32, copy=False)
        mode = f"bad_only_{mode_prefix}_knn"
    elif good_ref_ids.size:
        # Only good references: rank by similarity to good (AML) tiles
        margin = good_score_agg.astype(np.float32, copy=False)
        bad_like = _sigmoid((AML_REFERENCE_LOGIT_SCALE * (-margin)).astype(np.float32, copy=False))
        rank_scores = good_score_agg.astype(np.float32, copy=False)
        mode = f"good_only_{mode_prefix}_knn"
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


def _compute_blast_reference_scores(
    *,
    features_l2: npt.NDArray[np.float32],
    blast_features_l2: npt.NDArray[np.float32],
    top_k: int,
    row_block_size: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    rows = int(features_l2.shape[0]) if features_l2.ndim == 2 else 0
    empty_scores = np.zeros((rows,), dtype=np.float32)
    empty_idx = np.empty((rows, 0), dtype=np.int32)
    empty_sims = np.empty((rows, 0), dtype=np.float32)
    if features_l2.size == 0 or blast_features_l2.size == 0:
        return empty_scores, empty_idx, empty_sims, empty_scores

    local_idx, neighbor_sims = _exact_topk_reference_matches(
        features_l2=features_l2,
        ref_features_l2=blast_features_l2,
        k=max(1, int(top_k)),
        row_block_size=max(1, int(row_block_size)),
    )
    neighbor_indices = local_idx.astype(np.int32, copy=False) if local_idx.size else empty_idx
    scores = _aggregate_knn_similarities(neighbor_sims, method=AML_REFERENCE_AGGREGATION).astype(np.float32, copy=False)
    top1 = _top1_sims(neighbor_sims, rows=rows)
    return scores, neighbor_indices, neighbor_sims, top1


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

    Set `ROI_NOVELTY_CLIP_PERCENTILE = 100` in `configs/config.yaml`
    (or via env override) to disable.
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
            ef_construction=ROI_KNN_HNSW_EF_CONSTRUCTION,
            M=ROI_KNN_HNSW_M,
            random_seed=ROI_KNN_RANDOM_SEED,
        )
        ids = np.arange(n, dtype=np.int64)
        index.add_items(features_l2, ids, num_threads=1)
        index.set_ef(max(ROI_KNN_HNSW_EF_SEARCH_MIN, k_eff))
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
    dark_region_boxes_level0: list[dict[str, int]] | None = None,
    k_neighbors: int = 20,
    use_reference_labels: bool = False,
    reference_tiles_root: str | Path | None = None,
    quality_method: str = "embedding",  # "embedding" or "hybrid"
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
        dark_region_boxes_level0=dark_region_boxes_level0,
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
            dark_roi_scores=np.empty((0,), dtype=np.float32),
            blast_scores=np.empty((0,), dtype=np.float32),
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
            blast_neighbor_indices=np.empty((0, 0), dtype=np.int32),
            blast_neighbor_sims=np.empty((0, 0), dtype=np.float32),
            blast_reference_paths=(),
            blast_neighbor_k=0,
            quality_method=quality_method,
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
    blast_scores = np.zeros((num_tiles,), dtype=np.float32)
    blast_neighbor_indices = np.empty((num_tiles, 0), dtype=np.int32)
    blast_neighbor_sims = np.empty((num_tiles, 0), dtype=np.float32)
    blast_reference_paths: tuple[str, ...] = ()
    blast_neighbor_k = 0
    ref_root = Path(reference_tiles_root).resolve() if (use_reference_labels and reference_tiles_root) else None
    ref_records_all = _discover_reference_tiles(ref_root) if ref_root else []
    ref_records = _active_reference_records(ref_records_all)
    use_retrieval_ranking = False
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if use_reference_labels:
        good_n_raw = sum(1 for _, lbl in ref_records_all if lbl == "good")
        bad_n_raw = sum(1 for _, lbl in ref_records_all if lbl == "bad")
        good_n = sum(1 for _, lbl in ref_records if lbl == "good")
        bad_n = sum(1 for _, lbl in ref_records if lbl == "bad")
        reference_stats = {
            "reference_tiles_root": str(ref_root) if ref_root else None,
            "reference_tiles_total": len(ref_records),
            "reference_tiles_good": int(good_n),
            "reference_tiles_bad": int(bad_n),
            "reference_tiles_total_raw": len(ref_records_all),
            "reference_tiles_good_raw": int(good_n_raw),
            "reference_tiles_bad_raw": int(bad_n_raw),
            "bad_references_enabled": not AML_DISABLE_BAD_REFERENCES,
            "reference_neighbor_k": int(AML_REFERENCE_TOP_K),
            "reference_similarity": "cosine_hnsw" if (hnswlib is not None and AML_REFERENCE_USE_HNSW) else "cosine_exact",
            "ranking_strategy": "aggregate_knn_retrieval",
            "hnsw_enabled": hnswlib is not None and AML_REFERENCE_USE_HNSW,
            "hnsw_m": AML_REFERENCE_HNSW_M,
            "hnsw_ef_construction": AML_REFERENCE_HNSW_EF_CONSTRUCTION,
            "hnsw_ef_search": AML_REFERENCE_HNSW_EF_SEARCH,
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

            ref_feat_l2, ref_labels, ref_paths = _embed_reference_tiles(
                records=ref_records,
                extractor=extractor,
                device=run_device,
                batch_size=max(1, min(64, int(batch_size))),
                extractor_id=extractor.identifier,
                use_cache=True,
            )
            retrieval = _compute_reference_knn_scores(
                features_l2=features_l2,
                ref_features_l2=ref_feat_l2,
                ref_labels=ref_labels,
                ref_paths=ref_paths,
                top_k=AML_REFERENCE_TOP_K,
                row_block_size=AML_REFERENCE_QUERY_BLOCK_ROWS,
                use_hnsw=AML_REFERENCE_USE_HNSW,
                extractor_id=extractor.identifier,
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
                    "reference_query_block_rows": int(AML_REFERENCE_QUERY_BLOCK_ROWS),
                }
            )
            if progress_cb is not None:
                progress_cb(
                    {
                        "phase": "embed_reference_tiles",
                        "status": "done",
                        "reference_mode": reference_mode,
                    }
                )

        if not ref_records:
            reference_mode = "no_reference_tiles"
            reference_stats["reference_mode"] = reference_mode

        blast_loaded = None
        if AML_ENABLE_BLAST_REFERENCES:
            blast_loaded = _load_blast_cell_embeddings(
                AML_BLAST_CELLS_ROOT,
                extractor=extractor,
                device=run_device,
                batch_size=max(1, min(64, int(batch_size))),
            )
        if blast_loaded is not None:
            blast_feat_l2, blast_paths = blast_loaded
            blast_scores, blast_neighbor_indices, blast_neighbor_sims, blast_top1 = _compute_blast_reference_scores(
                features_l2=features_l2,
                blast_features_l2=blast_feat_l2,
                top_k=AML_BLAST_TOP_K,
                row_block_size=AML_REFERENCE_QUERY_BLOCK_ROWS,
            )
            blast_reference_paths = blast_paths
            blast_neighbor_k = int(min(AML_BLAST_TOP_K, blast_neighbor_sims.shape[1])) if blast_neighbor_sims.ndim == 2 else 0
            reference_stats.update(
                {
                    "blast_reference_tiles_root": str(Path(AML_BLAST_CELLS_ROOT).resolve()),
                    "blast_reference_tiles_total": len(blast_paths),
                    "blast_reference_neighbor_k": int(blast_neighbor_k),
                }
            )
            blast_rank_signal = _zscore(blast_scores)
            if use_retrieval_ranking:
                scores = (scores + (AML_BLAST_SIMILARITY_WEIGHT * blast_rank_signal)).astype(np.float32, copy=False)
            else:
                scores = blast_rank_signal.astype(np.float32, copy=False)
                use_retrieval_ranking = True

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
    dark_roi_scores = (
        result.dark_roi_scores.astype(np.float32, copy=False)
        if isinstance(result.dark_roi_scores, np.ndarray)
        else np.empty((num_tiles,), dtype=np.float32)
    )

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
        dark_roi_scores=dark_roi_scores,
        blast_scores=blast_scores,
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
        blast_neighbor_indices=blast_neighbor_indices,
        blast_neighbor_sims=blast_neighbor_sims,
        blast_reference_paths=blast_reference_paths,
        blast_neighbor_k=blast_neighbor_k,
        quality_method=quality_method,
    )


def _reference_matches_for_tile(
    *,
    index: UnsupervisedROIIndex,
    tile_idx: int,
    max_items: int = AML_REFERENCE_EVIDENCE_PER_CLASS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], float, float, float]:
    def _collect(
        *,
        paths: tuple[str, ...],
        labels: tuple[str, ...] | None,
        neighbor_indices: npt.NDArray[np.int32],
        neighbor_sims: npt.NDArray[np.float32],
    ) -> list[dict[str, Any]]:
        if neighbor_indices.ndim != 1 or neighbor_sims.ndim != 1:
            return []

        out: list[dict[str, Any]] = []
        limit = max(0, min(int(max_items), int(neighbor_indices.shape[0]), int(neighbor_sims.shape[0])))
        for ref_idx, sim in zip(neighbor_indices[:limit], neighbor_sims[:limit]):
            ref_i = int(ref_idx)
            if ref_i < 0 or ref_i >= len(paths):
                continue
            out.append(
                {
                    "path": paths[ref_i],
                    "name": Path(paths[ref_i]).name,
                    "label": labels[ref_i] if labels is not None and ref_i < len(labels) else "",
                    "similarity": float(sim),
                }
            )
        return out

    bad_refs: list[dict[str, Any]] = []
    good_refs: list[dict[str, Any]] = []
    blast_refs: list[dict[str, Any]] = []
    bad_top1 = 0.0
    good_top1 = 0.0
    blast_top1 = 0.0

    if index.bad_neighbor_sims.ndim == 2 and index.bad_neighbor_sims.shape[0] > tile_idx:
        bad_top1 = float(index.bad_neighbor_sims[tile_idx, 0]) if index.bad_neighbor_sims.shape[1] else 0.0
        bad_refs = _collect(
            paths=index.reference_tile_paths,
            labels=index.reference_tile_labels,
            neighbor_indices=index.bad_neighbor_indices[tile_idx],
            neighbor_sims=index.bad_neighbor_sims[tile_idx],
        )
    if index.good_neighbor_sims.ndim == 2 and index.good_neighbor_sims.shape[0] > tile_idx:
        good_top1 = float(index.good_neighbor_sims[tile_idx, 0]) if index.good_neighbor_sims.shape[1] else 0.0
        good_refs = _collect(
            paths=index.reference_tile_paths,
            labels=index.reference_tile_labels,
            neighbor_indices=index.good_neighbor_indices[tile_idx],
            neighbor_sims=index.good_neighbor_sims[tile_idx],
        )
    if index.blast_neighbor_sims.ndim == 2 and index.blast_neighbor_sims.shape[0] > tile_idx:
        blast_top1 = float(index.blast_neighbor_sims[tile_idx, 0]) if index.blast_neighbor_sims.shape[1] else 0.0
        blast_refs = _collect(
            paths=index.blast_reference_paths,
            labels=None,
            neighbor_indices=index.blast_neighbor_indices[tile_idx],
            neighbor_sims=index.blast_neighbor_sims[tile_idx],
        )

    return bad_refs, good_refs, blast_refs, bad_top1, good_top1, blast_top1


def _prefilter_candidates_for_vllm(
    candidates: list[dict[str, Any]],
    *,
    max_candidates: int = VLLM_MAX_CANDIDATES,
    min_dark_score: float = VLLM_MIN_DARK_SCORE,
    reject_bad_like: bool = VLLM_BAD_LIKE_REJECT,
) -> list[dict[str, Any]]:
    """Pre-filter ROI candidates before sending to VLM.

    This reduces token count and prevents VLM confusion from:
    - Low-cellularity tiles (acellular debris, empty background)
    - Bad-like tiles (already classified as non-diagnostic)
    - Redundant overlapping candidates

    Args:
        candidates: Raw ranked candidate list from select_topk_candidates_for_view
        max_candidates: Maximum number of candidates to send to VLM
        min_dark_score: Minimum dark_roi_score (cellularity) threshold
        reject_bad_like: Whether to reject candidates with quality_hint='bad_like'

    Returns:
        Filtered candidate list optimized for VLM processing
    """
    if not VLLM_PREFILTER_ENABLED or not candidates:
        return candidates[:max_candidates]

    filtered = []
    for cand in candidates:
        # HARD REJECT: bad_like tiles are non-diagnostic
        if reject_bad_like and cand.get("quality_hint") == "bad_like":
            continue

        # HARD REJECT: low cellularity tiles (acellular/background)
        dark_score = cand.get("dark_roi_score")
        if dark_score is not None and dark_score < min_dark_score:
            # Exception: keep if good_support evidence exists
            if cand.get("good_top1_similarity", 0.0) < AML_VLLM_GOOD_SUPPORT_EXEMPTION_TOP1_MIN:
                continue

        filtered.append(cand)

        # Stop once we have enough candidates
        if len(filtered) >= max_candidates:
            break

    # If filtering removed everything, fall back to top candidates
    if not filtered:
        filtered = candidates[:max_candidates]

    return filtered


def select_topk_candidates_for_view(
    *,
    index: UnsupervisedROIIndex,
    view_bbox_level0: tuple[int, int, int, int],
    top_k: int = 12,
    min_center_separation_px: int = 256,
    focus_boxes_level0: list[dict[str, Any]] | None = None,
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

    ranking_scores = index.scores[idxs].astype(np.float32, copy=False)
    dark_region_mode = "none"
    inside_dark_region_all: npt.NDArray[np.bool_] | None = None
    dark_box_prior_view: npt.NDArray[np.float32] | None = None
    if focus_boxes_level0:
        inside_dark_region_all = np.zeros((index.num_tiles,), dtype=bool)
        for box in focus_boxes_level0:
            try:
                bx0 = int(box["x0"])
                by0 = int(box["y0"])
                bw = int(box["w"])
                bh = int(box["h"])
            except Exception:
                continue
            bx1 = bx0 + max(0, bw)
            by1 = by0 + max(0, bh)
            inside_dark_region_all |= (
                (cx >= bx0)
                & (cx <= bx1)
                & (cy >= by0)
                & (cy <= by1)
            )

        inside_dark_view = inside_dark_region_all[idxs]
        # Dark-region boxes are a coarse thumbnail prior, not a hard exclusion mask.
        # Keep them as a ranking bonus so strong rescued tiles just outside the coarse
        # boxes can still survive.
        dark_box_prior_view = inside_dark_view.astype(np.float32, copy=False)
        dark_region_mode = "prioritized" if np.any(inside_dark_view) else "outside"

    quality_prior_view: npt.NDArray[np.float32] | None = None
    quality_penalty_view: npt.NDArray[np.float32] | None = None
    good_support_view: npt.NDArray[np.bool_] | None = None
    if index.bad_likelihood.size == index.num_tiles:
        bad_like_view = index.bad_likelihood[idxs].astype(np.float32, copy=False)
        bad_margin_view = (
            index.bad_margin[idxs].astype(np.float32, copy=False)
            if index.bad_margin.size == index.num_tiles
            else np.zeros_like(bad_like_view)
        )
        bad_top1_view = (
            _top1_sims(index.bad_neighbor_sims[idxs], rows=int(idxs.shape[0]))
            if index.bad_neighbor_sims.ndim == 2 and index.bad_neighbor_sims.shape[0] == index.num_tiles
            else np.zeros_like(bad_like_view)
        )
        good_top1_view = (
            _top1_sims(index.good_neighbor_sims[idxs], rows=int(idxs.shape[0]))
            if index.good_neighbor_sims.ndim == 2 and index.good_neighbor_sims.shape[0] == index.num_tiles
            else np.zeros_like(bad_like_view)
        )
        low_quality_mask = _bad_reference_reject_mask(
            bad_top1=bad_top1_view,
            good_top1=good_top1_view,
            bad_like=bad_like_view,
            bad_margin=bad_margin_view,
        )
        good_support_view = (
            (good_top1_view >= AML_SUPPORT_GOOD_TOP1_FLOOR)
            & (bad_top1_view <= AML_SUPPORT_BAD_TOP1_CEILING)
            & (bad_like_view <= AML_SUPPORT_BAD_LIKE_CEILING)
        )
        quality_keep = ~low_quality_mask
        if int(np.count_nonzero(quality_keep)) >= 2:
            idxs = idxs[quality_keep]
            ranking_scores = ranking_scores[quality_keep]
            if dark_box_prior_view is not None and dark_box_prior_view.size == quality_keep.size:
                dark_box_prior_view = dark_box_prior_view[quality_keep]
            bad_like_view = bad_like_view[quality_keep]
            bad_margin_view = bad_margin_view[quality_keep]
            bad_top1_view = bad_top1_view[quality_keep]
            good_top1_view = good_top1_view[quality_keep]
            if good_support_view is not None and good_support_view.size == quality_keep.size:
                good_support_view = good_support_view[quality_keep]

        borderline_bad_mask = (
            (bad_like_view >= AML_BORDERLINE_BAD_LIKELIHOOD)
            & (bad_margin_view <= AML_BORDERLINE_BAD_MARGIN_MAX)
            & (good_top1_view <= (bad_top1_view + AML_BORDERLINE_BAD_TOP1_GAP))
        )
        if good_support_view is not None and good_support_view.size == borderline_bad_mask.size:
            borderline_bad_mask = borderline_bad_mask & ~good_support_view
        borderline_keep = ~borderline_bad_mask
        if int(np.count_nonzero(borderline_keep)) >= 2 and int(np.count_nonzero(borderline_bad_mask)) > 0:
            idxs = idxs[borderline_keep]
            ranking_scores = ranking_scores[borderline_keep]
            if dark_box_prior_view is not None and dark_box_prior_view.size == borderline_keep.size:
                dark_box_prior_view = dark_box_prior_view[borderline_keep]
            bad_like_view = bad_like_view[borderline_keep]
            bad_margin_view = bad_margin_view[borderline_keep]
            bad_top1_view = bad_top1_view[borderline_keep]
            good_top1_view = good_top1_view[borderline_keep]
            if good_support_view is not None and good_support_view.size == borderline_keep.size:
                good_support_view = good_support_view[borderline_keep]

        similarity_gap_view = (good_top1_view - bad_top1_view).astype(np.float32, copy=False)
        quality_prior_view = (
            AML_QUALITY_PRIOR_BAD_MARGIN_WEIGHT * bad_margin_view
            + AML_QUALITY_PRIOR_SIMILARITY_GAP_WEIGHT * similarity_gap_view
        ).astype(np.float32, copy=False)
        bad_like_soft_penalty = np.clip(
            (
                bad_like_view - AML_BAD_LIKE_SOFT_PENALTY_BASELINE
            ) / max(AML_BAD_LIKE_REJECT_THRESHOLD - AML_BAD_LIKE_SOFT_PENALTY_BASELINE, 1e-6),
            0.0,
            1.0,
        )
        bad_match_soft_penalty = np.clip(
            (bad_top1_view - good_top1_view + AML_BAD_MATCH_SOFT_PENALTY_OFFSET) / AML_BAD_MATCH_SOFT_PENALTY_SCALE,
            0.0,
            1.0,
        )
        quality_penalty_view = (
            AML_QUALITY_PENALTY_BAD_LIKE_WEIGHT * bad_like_soft_penalty
            + AML_QUALITY_PENALTY_BAD_MATCH_WEIGHT * bad_match_soft_penalty
        ).astype(np.float32, copy=False)

    dark_view_scores: npt.NDArray[np.float32] | None = None
    # dark_roi_scores is the PRIMARY signal for AML tile selection.
    # Novelty/centroid scores prefer unusual (often sparse) tiles; dark_roi_scores
    # directly measure purple-basophilic cellularity which is what we want.
    if index.dark_roi_scores.size == index.num_tiles:
        dark_view_scores = index.dark_roi_scores[idxs].astype(np.float32, copy=False)

        # ABSOLUTE MINIMUM THRESHOLD: Reject tiles with very low cellularity
        # Score < 0.25 typically indicates acellular/light-stain material
        # This is a HARD FILTER — light stain-only tiles must not be candidates
        if dark_view_scores.size > 0:
            cellular_mask = dark_view_scores >= AML_ABSOLUTE_MIN_DARK_SCORE
            if good_support_view is not None and good_support_view.size == dark_view_scores.size:
                cellular_mask = cellular_mask | good_support_view
            if int(np.count_nonzero(cellular_mask)) >= 2:
                idxs = idxs[cellular_mask]
                ranking_scores = ranking_scores[cellular_mask]
                if dark_box_prior_view is not None and dark_box_prior_view.size == cellular_mask.size:
                    dark_box_prior_view = dark_box_prior_view[cellular_mask]
                dark_view_scores = dark_view_scores[cellular_mask]
                if quality_prior_view is not None and quality_prior_view.size == cellular_mask.size:
                    quality_prior_view = quality_prior_view[cellular_mask]
                if quality_penalty_view is not None and quality_penalty_view.size == cellular_mask.size:
                    quality_penalty_view = quality_penalty_view[cellular_mask]
                if good_support_view is not None and good_support_view.size == cellular_mask.size:
                    good_support_view = good_support_view[cellular_mask]

        # HARD MINIMUM THRESHOLD: Filter out tiles below minimum dark score percentile
        # This prevents empty/background tiles from being returned as candidates
        if dark_view_scores.size > 0:
            dark_min_floor = float(np.percentile(dark_view_scores, AML_MIN_DARK_SCORE_PERCENTILE))
            dark_min_keep = dark_view_scores >= dark_min_floor
            if good_support_view is not None and good_support_view.size == dark_view_scores.size:
                dark_min_keep = dark_min_keep | good_support_view
            # Only apply if we still have enough candidates after filtering
            if int(np.count_nonzero(dark_min_keep)) >= 2:
                idxs = idxs[dark_min_keep]
                ranking_scores = ranking_scores[dark_min_keep]
                if dark_box_prior_view is not None and dark_box_prior_view.size == dark_min_keep.size:
                    dark_box_prior_view = dark_box_prior_view[dark_min_keep]
                dark_view_scores = dark_view_scores[dark_min_keep]
                if quality_prior_view is not None and quality_prior_view.size == dark_min_keep.size:
                    quality_prior_view = quality_prior_view[dark_min_keep]
                if quality_penalty_view is not None and quality_penalty_view.size == dark_min_keep.size:
                    quality_penalty_view = quality_penalty_view[dark_min_keep]
                if good_support_view is not None and good_support_view.size == dark_min_keep.size:
                    good_support_view = good_support_view[dark_min_keep]

        # Percentile-based filtering for top candidates
        if dark_view_scores.size >= max(int(AML_DARK_VIEW_MIN_TILES), int(top_k) * 2):
            dark_floor = float(np.percentile(dark_view_scores, AML_DARK_VIEW_PERCENTILE))
            dark_keep = dark_view_scores >= dark_floor
            if good_support_view is not None and good_support_view.size == dark_view_scores.size:
                dark_keep = dark_keep | good_support_view
            if int(np.count_nonzero(dark_keep)) >= int(top_k):
                idxs = idxs[dark_keep]
                ranking_scores = ranking_scores[dark_keep]
                if dark_box_prior_view is not None and dark_box_prior_view.size == dark_keep.size:
                    dark_box_prior_view = dark_box_prior_view[dark_keep]
                dark_view_scores = dark_view_scores[dark_keep]
                if quality_prior_view is not None and quality_prior_view.size == dark_keep.size:
                    quality_prior_view = quality_prior_view[dark_keep]
                if quality_penalty_view is not None and quality_penalty_view.size == dark_keep.size:
                    quality_penalty_view = quality_penalty_view[dark_keep]
                if good_support_view is not None and good_support_view.size == dark_keep.size:
                    good_support_view = good_support_view[dark_keep]

        if dark_view_scores.size:
            # Cellularity remains primary, but allow quality evidence to keep gray-black
            # trash from dominating the candidate list.
            combined_rank = (
                AML_COMBINED_RANK_DARK_WEIGHT * _zscore(dark_view_scores)
                + AML_COMBINED_RANK_BASE_SCORE_WEIGHT * _zscore(ranking_scores)
            )
            if quality_prior_view is not None and quality_prior_view.size == dark_view_scores.size:
                combined_rank = combined_rank + (AML_COMBINED_RANK_QUALITY_PRIOR_WEIGHT * _zscore(quality_prior_view))
            if quality_penalty_view is not None and quality_penalty_view.size == dark_view_scores.size:
                combined_rank = combined_rank - (AML_COMBINED_RANK_QUALITY_PENALTY_WEIGHT * quality_penalty_view)
            if good_support_view is not None and good_support_view.size == dark_view_scores.size:
                combined_rank = combined_rank + (
                    AML_COMBINED_RANK_GOOD_SUPPORT_BONUS * good_support_view.astype(np.float32, copy=False)
                )
            if dark_box_prior_view is not None and dark_box_prior_view.size == dark_view_scores.size:
                combined_rank = combined_rank + (AML_DARK_REGION_BOX_PRIOR_WEIGHT * dark_box_prior_view)
            ranking_scores = combined_rank.astype(np.float32, copy=False)
    elif dark_box_prior_view is not None and dark_box_prior_view.size == ranking_scores.size:
        ranking_scores = ranking_scores + (AML_DARK_REGION_BOX_PRIOR_WEIGHT * dark_box_prior_view)

    order_local = np.argsort(ranking_scores)[::-1]
    order = idxs[order_local]
    ordered_rank_scores = ranking_scores[order_local]
    ordered_dark_scores = (
        dark_view_scores[order_local]
        if isinstance(dark_view_scores, np.ndarray) and dark_view_scores.size == order_local.size
        else None
    )
    # Keep candidates spatially distinct: require near-tile-sized center spacing.
    # This reduces heavy overlap even when tile_size_level0_px is larger than the
    # external min_center_separation_px setting.
    adaptive_min_sep_px = max(
        int(max(1, min_center_separation_px)),
        int(round(index.tile_size_level0_px * ROI_ADAPTIVE_MIN_SEPARATION_TILE_RATIO)),
    )
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

    selected: list[tuple[int, float, float | None]] = []
    selected_bboxes: list[tuple[float, float, float, float]] = []
    for pos, tile_idx in enumerate(order):
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
                prev_tile_idx = prev_idx[0]
                dx = cxi - float(cx[prev_tile_idx])
                dy = cyi - float(cy[prev_tile_idx])
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
        dark_score = float(ordered_dark_scores[pos]) if ordered_dark_scores is not None else None
        selected.append((int(tile_idx), float(ordered_rank_scores[pos]), dark_score))
        selected_bboxes.append((tx0, ty0, tx1, ty1))
        if len(selected) >= top_k:
            break

    out: list[dict[str, Any]] = []
    for rank, (tile_idx, combined_rank_score, dark_roi_score) in enumerate(selected, start=1):
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

        bad_refs, good_refs, blast_refs, bad_top1, good_top1, blast_top1 = _reference_matches_for_tile(
            index=index,
            tile_idx=tile_idx,
        )

        # Get embedding retrieval scores
        retrieval_score = float(index.scores[tile_idx])
        bad_like = (
            float(index.bad_likelihood[tile_idx])
            if index.bad_likelihood.size > tile_idx
            else 0.5
        )
        quality_margin = (
            float(index.bad_margin[tile_idx])
            if index.bad_margin.size > tile_idx
            else 0.0
        )

        good_only_reference_mode = str(index.reference_mode or "").startswith("good_only")

        if good_only_reference_mode:
            retrieval_good = quality_margin > AML_QUALITY_REJECT_MARGIN
            retrieval_bad = False
        else:
            retrieval_good = (
                quality_margin > AML_GOOD_LIKE_MARGIN_THRESHOLD
                and bad_like < AML_GOOD_LIKE_BAD_LIKELIHOOD_MAX
                and good_top1 >= (bad_top1 + AML_GOOD_LIKE_TOP1_GAP)
            )
            retrieval_bad = (
                quality_margin <= AML_BAD_MARGIN_REJECT_THRESHOLD
                or (
                    bad_like >= AML_BORDERLINE_BAD_LIKELIHOOD
                    and quality_margin <= AML_BORDERLINE_BAD_MARGIN_MAX
                    and good_top1 <= (bad_top1 + AML_BORDERLINE_BAD_TOP1_GAP)
                )
            )
        bad_reference_reject = False if good_only_reference_mode else _bad_reference_is_rejected(
            bad_top1=bad_top1,
            good_top1=good_top1,
            bad_like=bad_like,
            bad_margin=quality_margin,
        )

        if bad_reference_reject:
            quality_hint = "bad_like"
        elif retrieval_good:
            quality_hint = "good_like"
        elif not good_only_reference_mode and retrieval_bad:
            quality_hint = "bad_like"
        else:
            quality_hint = "uncertain"
        candidate = {
            "rank": rank,
            "tile_index": int(tile_idx),
            "score": float(combined_rank_score),
            "combined_rank_score": float(combined_rank_score),
            "retrieval_score": retrieval_score,
            "dark_roi_score": float(dark_roi_score) if dark_roi_score is not None else None,
            "inside_dark_region": bool(inside_dark_region_all[tile_idx]) if inside_dark_region_all is not None else None,
            "dark_region_mode": dark_region_mode,
            "bad_likelihood": bad_like,
            "bad_margin": quality_margin,
            "bad_top1_similarity": bad_top1,
            "good_top1_similarity": good_top1,
            "quality_hint": quality_hint,
            "reference_mode": index.reference_mode,
            "reference_neighbor_k": index.reference_neighbor_k,
            "retrieved_bad_refs": bad_refs[:3] if bad_refs else [],
            "retrieved_good_refs": good_refs[:3] if good_refs else [],
            "center_norm": [cx_norm, cy_norm],
            "bbox_norm": [bx0n, by0n, bx1n, by1n],
            "center_level0": [cxi, cyi],
            "tile_bbox_level0": [tile_x0, tile_y0, tile_x1, tile_y1],
        }
        if AML_ENABLE_BLAST_REFERENCES:
            candidate.update(
                {
                    "blast_similarity_score": (
                        float(index.blast_scores[tile_idx])
                        if index.blast_scores.size > tile_idx
                        else 0.0
                    ),
                    "blast_top1_similarity": blast_top1,
                    "blast_neighbor_k": index.blast_neighbor_k,
                    "retrieved_blast_refs": blast_refs[:3] if blast_refs else [],
                }
            )

        out.append(candidate)

    return out


__all__ = [
    "UnsupervisedROIIndex",
    "build_unsupervised_roi_index",
    "select_topk_candidates_for_view",
    "_prefilter_candidates_for_vllm",
    "clear_all_reference_caches",
    # VLLM optimization config
    "VLLM_PREFILTER_ENABLED",
    "VLLM_MAX_CANDIDATES",
    "VLLM_MIN_DARK_SCORE",
    "VLLM_BAD_LIKE_REJECT",
]
