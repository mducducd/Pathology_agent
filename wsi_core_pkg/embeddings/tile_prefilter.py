from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

try:
    from skimage.color import rgb2hed as _rgb2hed
except Exception:  # pragma: no cover
    _rgb2hed = None


@dataclass(frozen=True)
class TileQualitySelection:
    selected_indices: list[int]
    total_tiles: int
    pool_tiles: int
    hard_rejected_tiles: int
    reserved_tiles: int


def _rgb_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _tile_metrics(image: Image.Image) -> dict[str, float]:
    rgb = _rgb_float(image)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    padded = np.pad(gray, 1, mode="edge")
    lap = (
        padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        - 4.0 * padded[1:-1, 1:-1]
    )
    focus = float(np.var(lap))

    hist, _ = np.histogram(gray, bins=32, range=(0.0, 1.0))
    prob = hist.astype(np.float32)
    prob /= max(float(prob.sum()), 1.0)
    prob = prob[prob > 0]
    entropy = float(-(prob * np.log2(prob)).sum() / np.log2(32.0))

    channel_max = np.max(rgb, axis=2)
    channel_min = np.min(rgb, axis=2)
    chroma = channel_max - channel_min
    saturation = float(np.mean(chroma))
    brightness = float(np.mean(gray))

    od = -np.log(np.clip(rgb, 1.0 / 255.0, 1.0))
    stain_strength = float(np.mean(od))
    nuclear_signal = np.clip(
        0.65 * od[:, :, 2] + 0.35 * od[:, :, 0] - 0.55 * od[:, :, 1],
        0.0,
        None,
    )
    nuclear_stain = float(np.mean(nuclear_signal))

    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    edge_mag = np.pad(grad_x, ((0, 0), (0, 1)), mode="constant") + np.pad(grad_y, ((0, 1), (0, 0)), mode="constant")
    edge_density = float(np.mean(edge_mag > 0.08))

    grad_energy = float(np.mean((np.pad(grad_x, ((0, 0), (0, 1)), mode="constant") ** 2) + (np.pad(grad_y, ((0, 1), (0, 0)), mode="constant") ** 2)))

    pen_mask = (
        (chroma > 0.45)
        & (
            ((rgb[:, :, 2] > 0.62) & (rgb[:, :, 0] < 0.58))
            | ((rgb[:, :, 1] > 0.65) & (rgb[:, :, 0] < 0.58))
            | ((rgb[:, :, 0] > 0.74) & (rgb[:, :, 1] < 0.5) & (rgb[:, :, 2] < 0.5))
        )
    )
    dark_fold_mask = (gray < 0.08) & (chroma < 0.15) & (edge_mag < 0.03)

    if _rgb2hed is not None:
        try:
            hed = _rgb2hed(np.clip(rgb, 0.0, 1.0))
            hematoxylin_map = np.clip(hed[..., 0], 0.0, None)
        except Exception:
            hematoxylin_map = nuclear_signal
    else:
        hematoxylin_map = nuclear_signal

    hematoxylin = float(np.mean(hematoxylin_map))

    nuclear_floor = max(0.05, float(np.percentile(hematoxylin_map, 70)))
    nuclear_mask = hematoxylin_map >= nuclear_floor
    nuclear_fraction = float(np.mean(nuclear_mask))
    nuclear_detail = float(np.mean(edge_mag[nuclear_mask])) if np.any(nuclear_mask) else 0.0

    purple_cellular_mask = (
        (gray < 0.72)
        & (chroma > 0.08)
        & (hematoxylin_map >= max(0.05, nuclear_floor * 0.85))
        & (edge_mag > 0.04)
    )
    purple_cellular_fraction = float(np.mean(purple_cellular_mask))
    dark_cellular_mask = (
        (gray < 0.60)
        & (chroma > 0.08)
        & (hematoxylin_map >= max(0.05, nuclear_floor * 0.90))
        & (edge_mag > 0.04)
    )
    dark_cellular_fraction = float(np.mean(dark_cellular_mask))

    empty_fraction = float(np.mean((gray > 0.82) & (hematoxylin_map < 0.03)))
    red_dominant_fraction = float(
        np.mean(
            (gray < 0.72)
            & ((rgb[:, :, 0] - np.maximum(rgb[:, :, 1], rgb[:, :, 2])) > 0.06)
            & (edge_mag < 0.08)
        )
    )
    dark_blur_mask = (gray < 0.22) & (edge_mag < 0.025)
    dark_blur_fraction = float(np.mean(dark_blur_mask))
    crushed_dense_mask = (
        (gray < 0.55)
        & (hematoxylin_map >= max(0.08, nuclear_floor))
        & (edge_mag < 0.02)
    )
    crushed_dense_fraction = float(np.mean(crushed_dense_mask))
    artifact_fraction = float(
        np.mean(pen_mask | dark_fold_mask | dark_blur_mask | crushed_dense_mask)
    )

    return {
        "focus": focus,
        "focus_energy": grad_energy,
        "entropy": entropy,
        "saturation": saturation,
        "brightness": brightness,
        "stain_strength": stain_strength,
        "nuclear_stain": nuclear_stain,
        "hematoxylin": hematoxylin,
        "edge_density": edge_density,
        "nuclear_fraction": nuclear_fraction,
        "nuclear_detail": nuclear_detail,
        "purple_cellular_fraction": purple_cellular_fraction,
        "dark_cellular_fraction": dark_cellular_fraction,
        "empty_fraction": empty_fraction,
        "red_dominant_fraction": red_dominant_fraction,
        "dark_blur_fraction": dark_blur_fraction,
        "crushed_dense_fraction": crushed_dense_fraction,
        "artifact_fraction": artifact_fraction,
    }


def _robust_unit_scale(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32, copy=False)
    lo = float(np.percentile(values, 10))
    hi = float(np.percentile(values, 90))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = float(np.max(values))
        lo = float(np.min(values))
        if hi <= lo:
            return np.full(values.shape, 0.5, dtype=np.float32)
    scaled = (values - lo) / max(hi - lo, 1e-6)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32, copy=False)


def select_informative_tile_indices(
    images: Sequence[Image.Image],
    *,
    keep_ratio: float = 0.35,
    min_keep_tiles: int = 4,
    trigger_tile_count: int = 12,
    random_reserve_ratio: float = 0.08,
) -> TileQualitySelection:
    total_tiles = len(images)
    if total_tiles == 0:
        return TileQualitySelection(
            selected_indices=[],
            total_tiles=0,
            pool_tiles=0,
            hard_rejected_tiles=0,
            reserved_tiles=0,
        )

    metrics = [_tile_metrics(image) for image in images]
    focus = np.asarray([m["focus"] for m in metrics], dtype=np.float32)
    focus_energy = np.asarray([m["focus_energy"] for m in metrics], dtype=np.float32)
    entropy = np.asarray([m["entropy"] for m in metrics], dtype=np.float32)
    saturation = np.asarray([m["saturation"] for m in metrics], dtype=np.float32)
    brightness = np.asarray([m["brightness"] for m in metrics], dtype=np.float32)
    stain_strength = np.asarray([m["stain_strength"] for m in metrics], dtype=np.float32)
    nuclear_stain = np.asarray([m["nuclear_stain"] for m in metrics], dtype=np.float32)
    hematoxylin = np.asarray([m["hematoxylin"] for m in metrics], dtype=np.float32)
    edge_density = np.asarray([m["edge_density"] for m in metrics], dtype=np.float32)
    nuclear_fraction = np.asarray([m["nuclear_fraction"] for m in metrics], dtype=np.float32)
    nuclear_detail = np.asarray([m["nuclear_detail"] for m in metrics], dtype=np.float32)
    purple_cellular_fraction = np.asarray([m["purple_cellular_fraction"] for m in metrics], dtype=np.float32)
    dark_cellular_fraction = np.asarray([m["dark_cellular_fraction"] for m in metrics], dtype=np.float32)
    empty_fraction = np.asarray([m["empty_fraction"] for m in metrics], dtype=np.float32)
    red_dominant_fraction = np.asarray([m["red_dominant_fraction"] for m in metrics], dtype=np.float32)
    dark_blur_fraction = np.asarray([m["dark_blur_fraction"] for m in metrics], dtype=np.float32)
    crushed_dense_fraction = np.asarray([m["crushed_dense_fraction"] for m in metrics], dtype=np.float32)
    artifact_fraction = np.asarray([m["artifact_fraction"] for m in metrics], dtype=np.float32)

    focus_floor = max(float(np.percentile(focus, 15)) * 0.35, 1e-6)
    focus_energy_floor = max(float(np.percentile(focus_energy, 15)) * 0.35, 1e-6)
    hematoxylin_active = bool(np.any(hematoxylin > 1e-6))
    nuclear_fraction_floor = float(np.percentile(nuclear_fraction, 10)) if total_tiles > 1 else 0.0
    hard_keep = (
        (brightness <= 0.94)
        & (saturation >= 0.025)
        & (stain_strength >= 0.035)
        & (artifact_fraction <= 0.22)
        & (empty_fraction <= 0.72)
        & (dark_blur_fraction <= 0.35)
        & (focus >= focus_floor)
        & (focus_energy >= focus_energy_floor)
        & (nuclear_fraction >= nuclear_fraction_floor)
    )
    if hematoxylin_active:
        hematoxylin_floor = float(np.percentile(hematoxylin, 10))
        hard_keep = hard_keep & (hematoxylin >= hematoxylin_floor)
    pool = np.nonzero(hard_keep)[0]
    hard_rejected_tiles = int(total_tiles - int(pool.size))
    if pool.size == 0:
        pool = np.arange(total_tiles, dtype=np.int32)

    if pool.size <= max(1, int(trigger_tile_count)):
        return TileQualitySelection(
            selected_indices=pool.astype(int).tolist(),
            total_tiles=total_tiles,
            pool_tiles=int(pool.size),
            hard_rejected_tiles=hard_rejected_tiles,
            reserved_tiles=0,
        )

    focus_score = _robust_unit_scale(np.log1p(focus[pool] * 256.0))
    focus_energy_score = _robust_unit_scale(np.log1p(focus_energy[pool] * 32.0))
    entropy_score = _robust_unit_scale(entropy[pool])
    stain_score = _robust_unit_scale(stain_strength[pool])
    nuclear_score = _robust_unit_scale(nuclear_stain[pool])
    hematoxylin_score = _robust_unit_scale(hematoxylin[pool]) if hematoxylin_active else nuclear_score
    edge_score = _robust_unit_scale(edge_density[pool])
    nuclear_fraction_score = _robust_unit_scale(nuclear_fraction[pool])
    nuclear_detail_score = _robust_unit_scale(nuclear_detail[pool])
    purple_cellular_score = _robust_unit_scale(purple_cellular_fraction[pool])
    dark_cellular_score = _robust_unit_scale(dark_cellular_fraction[pool])
    empty_penalty = _robust_unit_scale(empty_fraction[pool])
    red_penalty = _robust_unit_scale(red_dominant_fraction[pool])
    dark_blur_penalty = _robust_unit_scale(dark_blur_fraction[pool])
    crushed_penalty = _robust_unit_scale(crushed_dense_fraction[pool])
    artifact_penalty = _robust_unit_scale(artifact_fraction[pool])

    # Blast-rich marrow fields are often darker and more basophilic. Keep that
    # bias, but only reward darker fields when they also preserve nuclear detail
    # and dense cellular structure rather than artifact-darkness alone.
    combined = (
        0.15 * focus_score
        + 0.05 * focus_energy_score
        + 0.06 * entropy_score
        + 0.10 * stain_score
        + 0.09 * edge_score
        + 0.09 * nuclear_score
        + 0.08 * hematoxylin_score
        + 0.12 * nuclear_fraction_score
        + 0.12 * nuclear_detail_score
        + 0.10 * purple_cellular_score
        + 0.12 * dark_cellular_score
        - 0.16 * artifact_penalty
        - 0.09 * empty_penalty
        - 0.06 * red_penalty
        - 0.09 * dark_blur_penalty
        - 0.07 * crushed_penalty
    )

    keep_top = max(int(min_keep_tiles), int(np.ceil(float(pool.size) * max(0.0, min(1.0, keep_ratio)))))
    keep_top = max(1, min(int(pool.size), keep_top))

    order = np.argsort(combined)[::-1]
    kept = pool[order[:keep_top]]
    reserve_count = min(
        max(0, int(np.ceil(float(pool.size) * max(0.0, random_reserve_ratio)))),
        max(0, int(pool.size) - keep_top),
    )

    reserved_tiles = 0
    if reserve_count > 0:
        rejected = pool[order[keep_top:]]
        seed = int((total_tiles * 1009) + (pool.size * 131) + np.round(float(np.sum(stain_strength)) * 1000.0)) % (2**32)
        rng = np.random.default_rng(seed)
        reserve = rng.choice(rejected, size=reserve_count, replace=False)
        kept = np.concatenate([kept, reserve])
        reserved_tiles = int(reserve_count)

    selected_indices = np.unique(kept).astype(int).tolist()
    selected_indices.sort()
    return TileQualitySelection(
        selected_indices=selected_indices,
        total_tiles=total_tiles,
        pool_tiles=int(pool.size),
        hard_rejected_tiles=hard_rejected_tiles,
        reserved_tiles=reserved_tiles,
    )


__all__ = ["TileQualitySelection", "select_informative_tile_indices"]
