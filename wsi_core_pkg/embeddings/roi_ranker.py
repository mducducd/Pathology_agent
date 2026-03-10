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
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.str_]]:
    if not records:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.str_)

    model = extractor.model.to(device)
    model.eval()

    label_buf: list[str] = []
    batch_tensors: list[torch.Tensor] = []
    chunks: list[torch.Tensor] = []

    for path, label in records:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            batch_tensors.append(_prepare_input_tensor(rgb, extractor.transform))
            label_buf.append(label)

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
    return feat_l2, labels


def _compute_bad_similarity_scores(
    *,
    features_l2: npt.NDArray[np.float32],
    ref_features_l2: npt.NDArray[np.float32],
    ref_labels: npt.NDArray[np.str_],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], str]:
    if features_l2.size == 0 or ref_features_l2.size == 0 or ref_labels.size == 0:
        n = int(features_l2.shape[0]) if features_l2.ndim == 2 else 0
        return np.zeros((n,), dtype=np.float32), np.full((n,), 0.5, dtype=np.float32), "none"

    bad_mask = ref_labels == "bad"
    good_mask = ref_labels == "good"

    if np.any(bad_mask) and np.any(good_mask):
        bad_proto = _l2_normalize_rows(np.mean(ref_features_l2[bad_mask], axis=0, keepdims=True))[0]
        good_proto = _l2_normalize_rows(np.mean(ref_features_l2[good_mask], axis=0, keepdims=True))[0]
        sim_bad = (features_l2 @ bad_proto).astype(np.float32, copy=False)
        sim_good = (features_l2 @ good_proto).astype(np.float32, copy=False)
        margin = (sim_bad - sim_good).astype(np.float32, copy=False)
        bad_like = (1.0 / (1.0 + np.exp(-np.clip(4.0 * margin, -30.0, 30.0)))).astype(np.float32, copy=False)
        return margin, bad_like, "good_bad_centroid"

    if np.any(bad_mask):
        bad_proto = _l2_normalize_rows(np.mean(ref_features_l2[bad_mask], axis=0, keepdims=True))[0]
        sim_bad = (features_l2 @ bad_proto).astype(np.float32, copy=False)
        margin = (sim_bad - float(np.mean(sim_bad))).astype(np.float32, copy=False)
        bad_like = (1.0 / (1.0 + np.exp(-np.clip(4.0 * margin, -30.0, 30.0)))).astype(np.float32, copy=False)
        return margin, bad_like, "bad_only_centroid"

    if np.any(good_mask):
        good_proto = _l2_normalize_rows(np.mean(ref_features_l2[good_mask], axis=0, keepdims=True))[0]
        sim_good = (features_l2 @ good_proto).astype(np.float32, copy=False)
        margin = (float(np.mean(sim_good)) - sim_good).astype(np.float32, copy=False)
        bad_like = (1.0 / (1.0 + np.exp(-np.clip(4.0 * margin, -30.0, 30.0)))).astype(np.float32, copy=False)
        return margin, bad_like, "good_only_centroid"

    n = int(features_l2.shape[0])
    return np.zeros((n,), dtype=np.float32), np.full((n,), 0.5, dtype=np.float32), "none"


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
    k_neighbors: int = 20,
    use_reference_labels: bool = False,
    reference_tiles_root: str | Path | None = None,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> UnsupervisedROIIndex:
    slide_path = Path(slide_path).resolve()
    if progress_cb is not None:
        progress_cb({"phase": "load_extractor", "status": "running"})
    extractor = uni2()
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
        progress_cb=progress_cb,
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
        )

    features_l2 = _l2_normalize_rows(features)
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

    centroid = np.mean(features_l2, axis=0, keepdims=True).astype(np.float32, copy=False)
    centroid = _l2_normalize_rows(centroid)[0]
    centroid_dist = (1.0 - (features_l2 @ centroid)).astype(np.float32, copy=False)

    # Default generic ranking.
    scores = (0.65 * _zscore(novelty) + 0.35 * _zscore(centroid_dist)).astype(np.float32, copy=False)
    bad_margin = np.zeros((num_tiles,), dtype=np.float32)
    bad_likelihood = np.full((num_tiles,), 0.5, dtype=np.float32)
    reference_mode = "none"
    reference_stats: dict[str, Any] = {}

    if use_reference_labels:
        ref_root = Path(reference_tiles_root).resolve() if reference_tiles_root else None
        ref_records = _discover_reference_tiles(ref_root) if ref_root else []
        good_n = sum(1 for _, lbl in ref_records if lbl == "good")
        bad_n = sum(1 for _, lbl in ref_records if lbl == "bad")
        reference_stats = {
            "reference_tiles_root": str(ref_root) if ref_root else None,
            "reference_tiles_total": len(ref_records),
            "reference_tiles_good": int(good_n),
            "reference_tiles_bad": int(bad_n),
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
            ref_feat_l2, ref_labels = _embed_reference_tiles(
                records=ref_records,
                extractor=extractor,
                device=run_device,
                batch_size=max(1, min(64, int(batch_size))),
            )
            bad_margin, bad_likelihood, reference_mode = _compute_bad_similarity_scores(
                features_l2=features_l2,
                ref_features_l2=ref_feat_l2,
                ref_labels=ref_labels,
            )

            if reference_mode != "none":
                # AML mode: bias ranking toward bad-like regions while still retaining novelty.
                scores = (
                    0.35 * _zscore(novelty)
                    + 0.15 * _zscore(centroid_dist)
                    + 0.50 * _zscore(bad_margin)
                ).astype(np.float32, copy=False)

            reference_stats.update(
                {
                    "reference_mode": reference_mode,
                    "wsi_bad_like_fraction": float(np.mean(bad_likelihood >= 0.5)),
                    "wsi_bad_like_strong_fraction": float(np.mean(bad_likelihood >= 0.65)),
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
                "phase": "build_knn",
                "status": "done",
                "num_tiles": num_tiles,
                "feature_dim": feature_dim,
            }
        )
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
    )


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
    adaptive_min_sep_px = max(int(max(1, min_center_separation_px)), int(round(index.tile_size_level0_px * 0.85)))
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
        if bad_like >= 0.60:
            quality_hint = "bad_like"
        elif bad_like <= 0.40:
            quality_hint = "good_like"
        else:
            quality_hint = "uncertain"

        out.append(
            {
                "rank": rank,
                "tile_index": int(tile_idx),
                "score": float(index.scores[tile_idx]),
                "bad_likelihood": bad_like,
                "bad_margin": bad_margin,
                "quality_hint": quality_hint,
                "reference_mode": index.reference_mode,
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
