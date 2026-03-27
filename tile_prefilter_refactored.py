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


def _mean_filter3(x: np.ndarray) -> np.ndarray:
    padded = np.pad(x, 1, mode="edge")
    acc = np.zeros_like(x, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            acc += padded[dy : dy + x.shape[0], dx : dx + x.shape[1]]
    return acc / 9.0


def _downsample_mask(mask: np.ndarray, factor: int = 4) -> np.ndarray:
    h, w = mask.shape
    f = max(1, int(factor))
    out_h, out_w = h // f, w // f
    if out_h <= 0 or out_w <= 0:
        return mask.astype(bool, copy=False)
    pooled = mask[: out_h * f, : out_w * f].astype(np.float32, copy=False).reshape(out_h, f, out_w, f).mean(axis=(1, 3))
    return pooled >= 0.25


def _component_shape_metrics(mask: np.ndarray) -> dict[str, float]:
    h, w = mask.shape
    if h == 0 or w == 0:
        return {
            "density": 0.0,
            "mean_area": 0.0,
            "area_cv": 1.0,
            "mean_circularity": 0.0,
        }

    visited = np.zeros((h, w), dtype=bool)
    areas: list[int] = []
    circularities: list[float] = []

    for y in range(h):
        for x in range(w):
            if visited[y, x] or not mask[y, x]:
                continue
            visited[y, x] = True
            stack = [(y, x)]
            area = 0
            perimeter = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w or not mask[ny, nx]:
                        perimeter += 1
                        continue
                    if not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if area >= 2:
                areas.append(area)
                circularities.append(float(4.0 * np.pi * area / max(perimeter * perimeter, 1.0)))

    if not areas:
        return {
            "density": 0.0,
            "mean_area": 0.0,
            "area_cv": 1.0,
            "mean_circularity": 0.0,
        }

    area_arr = np.asarray(areas, dtype=np.float32)
    mean_area = float(np.mean(area_arr))
    return {
        "density": float(len(areas) / max(float(h * w), 1.0)),
        "mean_area": mean_area,
        "area_cv": float(np.std(area_arr) / max(mean_area, 1e-6)),
        "mean_circularity": float(np.mean(np.asarray(circularities, dtype=np.float32))),
    }


def _focus_score_from_var(lap_var: float) -> float:
    return float(np.clip(np.log1p(lap_var * 256.0) / np.log1p(3.0), 0.0, 1.0))


def _u(value: float, scale: float) -> float:
    return float(np.clip(value / max(scale, 1e-6), 0.0, 1.0))


def _robust_unit_scale(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32, copy=False)
    lo = float(np.percentile(values, 10))
    hi = float(np.percentile(values, 90))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(values)), float(np.max(values))
        if hi <= lo:
            return np.full(values.shape, 0.5, dtype=np.float32)
    return np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32, copy=False)


def _tile_metrics(image: Image.Image) -> dict[str, float]:
    rgb = _rgb_float(image)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    padded = np.pad(gray, 1, mode="edge")
    lap = padded[1:-1, :-2] + padded[1:-1, 2:] + padded[:-2, 1:-1] + padded[2:, 1:-1] - 4.0 * padded[1:-1, 1:-1]
    focus_var = float(np.var(lap))

    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    edge_mag = np.pad(grad_x, ((0, 0), (0, 1)), mode="constant") + np.pad(grad_y, ((0, 1), (0, 0)), mode="constant")
    focus_energy = float(np.mean(np.pad(grad_x, ((0, 0), (0, 1)), mode="constant") ** 2 + np.pad(grad_y, ((0, 1), (0, 0)), mode="constant") ** 2))

    channel_max = np.max(rgb, axis=2)
    channel_min = np.min(rgb, axis=2)
    chroma = channel_max - channel_min
    brightness = float(np.mean(gray))
    stain_strength = float(np.mean(-np.log(np.clip(rgb, 1.0 / 255.0, 1.0))))

    od = -np.log(np.clip(rgb, 1.0 / 255.0, 1.0))
    nuclear_signal = np.clip(0.65 * od[..., 2] + 0.35 * od[..., 0] - 0.55 * od[..., 1], 0.0, None)
    if _rgb2hed is not None:
        try:
            hematoxylin_map = np.clip(_rgb2hed(np.clip(rgb, 0.0, 1.0))[..., 0], 0.0, None)
        except Exception:
            hematoxylin_map = nuclear_signal
    else:
        hematoxylin_map = nuclear_signal

    hematoxylin = float(np.mean(hematoxylin_map))
    focus_proxy = _focus_score_from_var(focus_var)
    blue_dominant = b >= r - 0.06
    red_dominant = r - np.maximum(g, b)

    tissue_mask = (gray < 0.85) | (chroma > 0.05) | (edge_mag > 0.02) | (hematoxylin_map > 0.05)
    tissue_fraction = float(np.mean(tissue_mask))

    purple_mask = (
        (gray < 0.72) & (chroma > 0.08) & (hematoxylin_map >= 0.08) & (edge_mag > 0.03) & blue_dominant
    )
    purple_fraction = float(np.mean(purple_mask))

    eosinophilic_cellular_mask = (
        (gray < 0.74) & (gray > 0.28) & (chroma > 0.10) & (hematoxylin_map >= 0.05) &
        (edge_mag > 0.045) & (r >= b - 0.03) & (r <= b + 0.18)
    )
    eosinophilic_cellular_fraction = float(np.mean(eosinophilic_cellular_mask))

    rbc_mask = (
        ((gray < 0.78) & (red_dominant > 0.05) & (edge_mag < 0.06) & (hematoxylin_map < 0.06)) |
        ((gray < 0.70) & (red_dominant > 0.05) & (chroma > 0.08) & (edge_mag < 0.045) & (hematoxylin_map < 0.06)) |
        ((gray < 0.62) & (red_dominant > 0.06) & (chroma < 0.12))
    )
    rbc_fraction = float(np.mean(rbc_mask))

    gray_black_mask = (
        ((gray < 0.40) & (chroma < 0.10) & (edge_mag < 0.05)) |
        ((gray < 0.52) & (chroma < 0.08) & ((edge_mag < 0.04) | (hematoxylin_map < 0.06)))
    )
    gray_black_fraction = float(np.mean(gray_black_mask))

    artifact_mask = (
        ((chroma > 0.45) & (((b > 0.62) & (r < 0.58)) | ((g > 0.65) & (r < 0.58)) | ((r > 0.74) & (g < 0.5) & (b < 0.5)))) |
        ((gray < 0.08) & (chroma < 0.15) & (edge_mag < 0.03)) |
        ((gray < 0.22) & (edge_mag < 0.025))
    )

    nuclear_floor = max(0.05, float(np.percentile(hematoxylin_map, 70)))
    crushed_dense_mask = (
        (gray < 0.55) & (hematoxylin_map >= max(0.08, nuclear_floor)) & (edge_mag < 0.02)
    )
    artifact_fraction = float(np.mean(artifact_mask | crushed_dense_mask))

    nuclear_mask = hematoxylin_map >= nuclear_floor
    nuclear_fraction = float(np.mean(nuclear_mask))
    nuclear_detail = float(np.mean(edge_mag[nuclear_mask])) if np.any(nuclear_mask) else 0.0
    nuclear_density_map = _mean_filter3(nuclear_mask.astype(np.float32, copy=False))
    packed_nuclear_mask = nuclear_mask & (nuclear_density_map >= 0.28) & (edge_mag > 0.03)
    packed_nuclear_fraction = float(np.mean(packed_nuclear_mask))

    dark_cellular_fraction = float(np.mean(
        (gray < 0.60) & (gray > 0.18) & (chroma > 0.10) &
        (hematoxylin_map >= max(0.05, nuclear_floor * 0.90)) & (edge_mag > 0.05) & blue_dominant
    ))
    very_dark_fraction = float(np.mean(
        (gray < 0.45) & (gray > 0.18) & (chroma > 0.10) &
        (hematoxylin_map >= max(0.08, nuclear_floor * 0.95)) & (edge_mag > 0.04) & blue_dominant
    ))
    edge_focus_floor = max(0.05, float(np.percentile(edge_mag, 70)))
    dark_in_focus_fraction = float(np.mean(
        (gray < 0.50) & (gray > 0.20) & (chroma > 0.08) &
        (hematoxylin_map >= max(0.10, nuclear_floor * 0.90)) &
        (edge_mag >= edge_focus_floor) & (focus_proxy >= 0.28) & blue_dominant
    ))

    nuclei_proxy_mask = (
        (hematoxylin_map >= max(0.04, nuclear_floor * 0.80)) & (gray < 0.80) & (edge_mag > 0.03) & (chroma > 0.05)
    )
    nuclei_stats = _component_shape_metrics(_downsample_mask(nuclei_proxy_mask, factor=4))

    gray_cluster_mask = (gray < 0.62) & (chroma < 0.12)
    gray_cluster_fraction = float(np.mean(gray_cluster_mask))
    gray_cluster_stats = _component_shape_metrics(_downsample_mask(gray_cluster_mask, factor=4))
    gray_cluster_penalty = float(
        np.clip(gray_cluster_fraction / 0.18, 0.0, 1.0) * np.clip(gray_cluster_stats["mean_area"] / 24.0, 0.0, 1.0)
    )

    stringy_artifact_score = float(
        np.clip((0.22 - nuclei_stats["mean_circularity"]) / 0.22, 0.0, 1.0) *
        np.clip((0.08 - packed_nuclear_fraction) / 0.08, 0.0, 1.0) *
        np.clip((dark_cellular_fraction + very_dark_fraction + purple_fraction + eosinophilic_cellular_fraction) / 0.35, 0.0, 1.0)
    )

    empty_fraction = float(np.mean((gray > 0.82) & (hematoxylin_map < 0.03)))
    nucleated_to_red_ratio = float(np.clip(purple_fraction / max(purple_fraction + rbc_fraction + 0.02, 1e-6), 0.0, 1.0))

    coarse_score = (
        0.26 * tissue_fraction +
        0.28 * purple_fraction +
        0.06 * eosinophilic_cellular_fraction +
        0.15 * focus_proxy -
        0.16 * rbc_fraction -
        0.13 * artifact_fraction -
        0.15 * gray_black_fraction -
        0.17 * gray_cluster_penalty
    )

    return {
        "coarse_score": float(np.clip(coarse_score, 0.0, 1.0)),
        "tissue_fraction": tissue_fraction,
        "purple_fraction": purple_fraction,
        "rbc_fraction": rbc_fraction,
        "focus_proxy": focus_proxy,
        "artifact_fraction": artifact_fraction,
        "brightness": brightness,
        "stain_strength": stain_strength,
        "hematoxylin": hematoxylin,
        "focus": focus_var,
        "focus_energy": focus_energy,
        "nuclear_fraction": nuclear_fraction,
        "nuclear_detail": nuclear_detail,
        "packed_nuclear_fraction": packed_nuclear_fraction,
        "dark_cellular_fraction": dark_cellular_fraction,
        "very_dark_fraction": very_dark_fraction,
        "dark_in_focus_fraction": dark_in_focus_fraction,
        "eosinophilic_cellular_fraction": eosinophilic_cellular_fraction,
        "gray_black_fraction": gray_black_fraction,
        "gray_cluster_penalty": gray_cluster_penalty,
        "nuclei_component_density": float(nuclei_stats["density"]),
        "nuclei_component_area_cv": float(nuclei_stats["area_cv"]),
        "nuclei_component_mean_circularity": float(nuclei_stats["mean_circularity"]),
        "stringy_artifact_score": stringy_artifact_score,
        "empty_fraction": empty_fraction,
        "red_dominant_fraction": rbc_fraction,
        "nucleated_to_red_ratio": nucleated_to_red_ratio,
    }


def score_dark_informative_roi(image: Image.Image) -> float:
    m = _tile_metrics(image)
    brightness_penalty = float(np.clip((m["brightness"] - 0.55) / 0.20, 0.0, 1.0))
    score = (
        0.34 * m["coarse_score"] +
        0.08 * _u(m["dark_cellular_fraction"], 0.30) +
        0.03 * _u(m["very_dark_fraction"], 0.20) +
        0.04 * _u(m["dark_in_focus_fraction"], 0.15) +
        0.10 * _u(m["nuclear_fraction"], 0.36) +
        0.09 * _u(m["packed_nuclear_fraction"], 0.28) +
        0.07 * _u(m["nuclear_detail"], 0.12) +
        0.05 * _u(np.log1p(m["focus"] * 256.0), np.log1p(3.0)) +
        0.05 * _u(np.log1p(m["focus_energy"] * 32.0), np.log1p(2.0)) +
        0.04 * _u(m["nuclei_component_density"], 0.07) +
        0.01 * float(np.clip(1.0 - (m["nuclei_component_area_cv"] / 1.5), 0.0, 1.0)) +
        0.04 * _u(m["nuclei_component_mean_circularity"], 0.28) +
        0.02 * _u(m["eosinophilic_cellular_fraction"], 0.18) +
        0.06 * _u(m["nucleated_to_red_ratio"], 1.0) -
        0.08 * _u(m["empty_fraction"], 0.55) -
        0.05 * _u(m["red_dominant_fraction"], 0.30) -
        0.12 * _u(m["artifact_fraction"], 0.22) -
        0.14 * _u(m["stringy_artifact_score"], 1.0) -
        0.18 * _u(m["gray_black_fraction"], 0.20) -
        0.16 * _u(m["gray_cluster_penalty"], 1.0) -
        0.35 * brightness_penalty
    )
    return float(np.clip(score, 0.0, 1.0))


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
        return TileQualitySelection([], 0, 0, 0, 0)

    metrics = [_tile_metrics(image) for image in images]

    def arr(key: str) -> np.ndarray:
        return np.asarray([m[key] for m in metrics], dtype=np.float32)

    coarse_scores = arr("coarse_score")
    tissue_fraction = arr("tissue_fraction")
    purple_fraction = arr("purple_fraction")
    eos_fraction = arr("eosinophilic_cellular_fraction")
    rbc_fraction = arr("rbc_fraction")
    artifact_fraction = arr("artifact_fraction")
    gray_black_fraction = arr("gray_black_fraction")
    gray_cluster_penalty = arr("gray_cluster_penalty")
    brightness = arr("brightness")
    hematoxylin = arr("hematoxylin")
    packed_nuclear_fraction = arr("packed_nuclear_fraction")
    nuclei_circularity = arr("nuclei_component_mean_circularity")
    stringy_artifact_score = arr("stringy_artifact_score")

    coarse_floor = float(np.percentile(coarse_scores, 20)) if total_tiles > 1 else -0.5
    tissue_floor = max(0.03, float(np.percentile(tissue_fraction, 8)) if total_tiles > 1 else 0.05)
    purple_floor = max(0.01, float(np.percentile(purple_fraction, 8)) if total_tiles > 1 else 0.03)
    eos_floor = max(0.01, float(np.percentile(eos_fraction, 8)) if total_tiles > 1 else 0.02)

    chromatic_cellular_indicator = (
        (brightness < 0.62) &
        (hematoxylin > 0.08) &
        ((purple_fraction > 0.01) | (eos_fraction > 0.04)) &
        (stringy_artifact_score < 0.55) &
        (gray_black_fraction < 0.28) &
        (gray_cluster_penalty < 0.35) &
        ((packed_nuclear_fraction > 0.02) | (nuclei_circularity > 0.18))
    )

    standard_keep = (
        (tissue_fraction >= tissue_floor) &
        ((purple_fraction >= purple_floor) | (eos_fraction >= max(0.04, eos_floor))) &
        (rbc_fraction <= 0.58) &
        (artifact_fraction <= 0.33) &
        (gray_black_fraction <= 0.30) &
        (gray_cluster_penalty <= 0.35) &
        (coarse_scores >= coarse_floor) &
        (brightness <= 0.78)
    )
    stringy_reject = (
        ((stringy_artifact_score >= 0.70) & (packed_nuclear_fraction <= 0.06) & (nuclei_circularity <= 0.18)) |
        ((gray_black_fraction >= 0.34) & (packed_nuclear_fraction <= 0.08) & (nuclei_circularity <= 0.20)) |
        (gray_cluster_penalty >= 0.45)
    )
    hard_keep = (standard_keep | chromatic_cellular_indicator) & ~stringy_reject

    pool = np.nonzero(hard_keep)[0]
    hard_rejected_tiles = int(total_tiles - pool.size)
    if pool.size == 0:
        pool = np.argsort(coarse_scores)[::-1][: max(min_keep_tiles, int(total_tiles * 0.2))]
        hard_rejected_tiles = total_tiles - len(pool)

    if pool.size <= max(1, int(trigger_tile_count)):
        return TileQualitySelection(pool.astype(int).tolist(), total_tiles, int(pool.size), hard_rejected_tiles, 0)

    def pool_arr(key: str) -> np.ndarray:
        return arr(key)[pool]

    combined = (
        0.38 * _robust_unit_scale(coarse_scores[pool]) +
        0.05 * _robust_unit_scale(pool_arr("dark_cellular_fraction")) +
        0.03 * _robust_unit_scale(pool_arr("very_dark_fraction")) +
        0.04 * _robust_unit_scale(pool_arr("dark_in_focus_fraction")) +
        0.09 * _robust_unit_scale(pool_arr("nuclear_fraction")) +
        0.10 * _robust_unit_scale(pool_arr("packed_nuclear_fraction")) +
        0.06 * _robust_unit_scale(np.log1p(pool_arr("focus") * 256.0)) +
        0.04 * _robust_unit_scale(np.log1p(pool_arr("focus_energy") * 32.0)) +
        0.04 * _robust_unit_scale(pool_arr("nuclei_component_density")) +
        0.01 * np.clip(1.0 - (pool_arr("nuclei_component_area_cv") / max(float(np.percentile(pool_arr("nuclei_component_area_cv"), 90)), 1.5)), 0.0, 1.0).astype(np.float32, copy=False) +
        0.04 * _robust_unit_scale(pool_arr("nuclei_component_mean_circularity")) +
        0.02 * _robust_unit_scale(pool_arr("eosinophilic_cellular_fraction")) +
        0.06 * _robust_unit_scale(pool_arr("nucleated_to_red_ratio")) -
        0.12 * _robust_unit_scale(pool_arr("artifact_fraction")) -
        0.12 * _robust_unit_scale(pool_arr("stringy_artifact_score")) -
        0.16 * _robust_unit_scale(pool_arr("gray_black_fraction")) -
        0.16 * _robust_unit_scale(pool_arr("gray_cluster_penalty")) -
        0.06 * _robust_unit_scale(pool_arr("red_dominant_fraction")) -
        0.07 * _robust_unit_scale(pool_arr("empty_fraction")) -
        0.30 * _robust_unit_scale(np.clip((pool_arr("brightness") - 0.55) / 0.20, 0.0, 1.0))
    )

    keep_top = max(int(min_keep_tiles), int(np.ceil(pool.size * max(0.0, min(1.0, keep_ratio)))))
    keep_top = max(1, min(int(pool.size), keep_top))
    order = np.argsort(combined)[::-1]
    kept = pool[order[:keep_top]]

    reserve_count = min(
        max(0, int(np.ceil(pool.size * max(0.0, random_reserve_ratio)))),
        max(0, int(pool.size) - keep_top),
    )
    reserved_tiles = 0
    if reserve_count > 0:
        rejected = pool[order[keep_top:]]
        seed = int((total_tiles * 1009) + (pool.size * 131) + np.round(float(np.sum(arr("stain_strength"))) * 1000.0)) % (2**32)
        reserve = np.random.default_rng(seed).choice(rejected, size=reserve_count, replace=False)
        kept = np.concatenate([kept, reserve])
        reserved_tiles = int(reserve_count)

    selected_indices = np.unique(kept).astype(int).tolist()
    selected_indices.sort()
    return TileQualitySelection(selected_indices, total_tiles, int(pool.size), hard_rejected_tiles, reserved_tiles)


__all__ = ["TileQualitySelection", "score_dark_informative_roi", "select_informative_tile_indices"]
