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
    fh = max(1, int(factor))
    fw = max(1, int(factor))
    out_h = h // fh
    out_w = w // fw
    if out_h <= 0 or out_w <= 0:
        return mask.astype(bool, copy=False)
    cropped = mask[: out_h * fh, : out_w * fw].astype(np.float32, copy=False)
    pooled = cropped.reshape(out_h, fh, out_w, fw).mean(axis=(1, 3))
    return pooled >= 0.25


def _component_shape_metrics(mask: np.ndarray) -> dict[str, float]:
    h, w = mask.shape
    if h == 0 or w == 0:
        return {
            "component_density": 0.0,
            "component_mean_area": 0.0,
            "component_area_cv": 1.0,
            "component_mean_circularity": 0.0,
        }

    visited = np.zeros((h, w), dtype=bool)
    areas: list[int] = []
    circularities: list[float] = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            perimeter = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        perimeter += 1
                        continue
                    if not mask[ny, nx]:
                        perimeter += 1
                        continue
                    if not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if area >= 2:
                areas.append(area)
                circularities.append(float((4.0 * np.pi * area) / max(perimeter * perimeter, 1.0)))

    if not areas:
        return {
            "component_density": 0.0,
            "component_mean_area": 0.0,
            "component_area_cv": 1.0,
            "component_mean_circularity": 0.0,
        }

    area_arr = np.asarray(areas, dtype=np.float32)
    mean_area = float(np.mean(area_arr))
    std_area = float(np.std(area_arr))
    area_cv = float(std_area / max(mean_area, 1e-6))
    return {
        "component_density": float(len(areas) / max(float(h * w), 1.0)),
        "component_mean_area": mean_area,
        "component_area_cv": area_cv,
        "component_mean_circularity": float(np.mean(np.asarray(circularities, dtype=np.float32))),
    }


def _tissue_fraction(rgb: np.ndarray, hematoxylin_map: np.ndarray, edge_mag: np.ndarray) -> float:
    """Compute tissue fraction - non-background areas.

    Background is typically very bright (>0.85 gray) with low hematoxylin and low edge content.
    """
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    channel_max = np.max(rgb, axis=2)
    channel_min = np.min(rgb, axis=2)
    chroma = channel_max - channel_min

    # Tissue: not too bright, has some color variation, or has edges
    tissue_mask = (
        (gray < 0.85) |
        (chroma > 0.05) |
        (edge_mag > 0.02) |
        (hematoxylin_map > 0.05)
    )
    return float(np.mean(tissue_mask))


def _purple_basophilic_fraction(
    rgb: np.ndarray,
    gray: np.ndarray,
    chroma: np.ndarray,
    hematoxylin_map: np.ndarray,
    edge_mag: np.ndarray,
) -> float:
    """Compute purple/basophilic fraction - nucleated cell rich regions.

    AML-relevant: captures purple-blue cellular regions that are not RBC-pink.
    Blue channel must be close to or exceed red channel (true basophilia).
    """
    # Purple/basophilic: blue >= red (not pink), moderate darkness, hematoxylin
    purple_mask = (
        (gray < 0.72) &
        (chroma > 0.08) &
        (hematoxylin_map >= 0.08) &
        (edge_mag > 0.03) &
        (rgb[:, :, 2] >= rgb[:, :, 0] - 0.06)  # blue >= red - tolerance
    )
    return float(np.mean(purple_mask))


def _rbc_red_fraction(
    rgb: np.ndarray,
    gray: np.ndarray,
    chroma: np.ndarray,
    hematoxylin_map: np.ndarray,
    edge_mag: np.ndarray,
) -> float:
    """Compute RBC-red fraction - eosinophilic pink regions dominated by red blood cells.

    For AML, smooth RBC/clot fields should be penalized, but red-pink cellular marrow
    should not be treated as trash. This function targets smooth or washed-out
    eosinophilic material rather than all red-pink tissue.
    """
    red_dominant = rgb[:, :, 0] - np.maximum(rgb[:, :, 1], rgb[:, :, 2])
    bright_rbc_mask = (
        (gray < 0.78) &
        (red_dominant > 0.05) &
        (edge_mag < 0.06) &
        (hematoxylin_map < 0.06)
    )
    smooth_red_mask = (
        (gray < 0.70) &
        (red_dominant > 0.05) &
        (chroma > 0.08) &
        (edge_mag < 0.045) &
        (hematoxylin_map < 0.06)
    )
    washed_eosinophilic_mask = (
        (gray < 0.62) &
        (red_dominant > 0.06) &
        (chroma < 0.12)
    )
    rbc_mask = bright_rbc_mask | smooth_red_mask | washed_eosinophilic_mask
    return float(np.mean(rbc_mask))


def _eosinophilic_cellular_fraction(
    rgb: np.ndarray,
    gray: np.ndarray,
    chroma: np.ndarray,
    hematoxylin_map: np.ndarray,
    edge_mag: np.ndarray,
) -> float:
    """Compute red-pink cellular fraction with preserved morphology.

    This captures dark rich eosinophilic or mixed pink-purple marrow that still has
    real cellular detail, separating it from smooth RBC/clot-dominant regions.
    """
    eos_mask = (
        (gray < 0.74) &
        (gray > 0.28) &
        (chroma > 0.10) &
        (hematoxylin_map >= 0.05) &
        (edge_mag > 0.045) &
        (rgb[:, :, 0] >= rgb[:, :, 2] - 0.03) &
        (rgb[:, :, 0] <= rgb[:, :, 2] + 0.18)
    )
    return float(np.mean(eos_mask))


def _gray_black_artifact_fraction(
    gray: np.ndarray,
    chroma: np.ndarray,
    edge_mag: np.ndarray,
    hematoxylin_map: np.ndarray,
) -> float:
    """Compute fraction of dull gray-black material likely to be junk.

    Useful marrow regions are colored and structured. Very dark low-chroma areas with
    weak texture are more often folds, debris, crushed smear, or other trash.
    """
    charcoal_mask = (
        (gray < 0.40) &
        (chroma < 0.10) &
        (edge_mag < 0.05)
    )
    muddy_dark_mask = (
        (gray < 0.52) &
        (chroma < 0.08) &
        ((edge_mag < 0.04) | (hematoxylin_map < 0.06))
    )
    return float(np.mean(charcoal_mask | muddy_dark_mask))


def _darkness_score(gray: np.ndarray) -> float:
    """Compute darkness score - inverse of mean brightness.

    Dark regions in AML often indicate dense cellularity.
    """
    return float(1.0 - np.mean(gray))


def _focus_score(lap_var: float) -> float:
    """Compute focus score from Laplacian variance.

    In-focus tiles have higher Laplacian variance.
    """
    return float(np.clip(np.log1p(lap_var * 256.0) / np.log1p(3.0), 0.0, 1.0))


def _artifact_fraction(
    rgb: np.ndarray,
    gray: np.ndarray,
    chroma: np.ndarray,
    edge_mag: np.ndarray,
    hematoxylin_map: np.ndarray,
) -> float:
    """Compute artifact fraction - pen marks, folds, dark blur, crushed dense regions.

    Artifacts should be penalized in ROI selection.
    """
    # Pen ink markers
    pen_mask = (
        (chroma > 0.45) &
        (
            ((rgb[:, :, 2] > 0.62) & (rgb[:, :, 0] < 0.58)) |
            ((rgb[:, :, 1] > 0.65) & (rgb[:, :, 0] < 0.58)) |
            ((rgb[:, :, 0] > 0.74) & (rgb[:, :, 1] < 0.5) & (rgb[:, :, 2] < 0.5))
        )
    )

    # Tissue folds: very dark, low chroma, low edge
    dark_fold_mask = (gray < 0.08) & (chroma < 0.15) & (edge_mag < 0.03)

    # Dark blur: very dark, extremely low edge
    dark_blur_mask = (gray < 0.22) & (edge_mag < 0.025)

    # Crushed dense: dark, high hematoxylin, but no edge detail (smudged)
    nuclear_floor = max(0.05, float(np.percentile(hematoxylin_map, 70)))
    crushed_dense_mask = (
        (gray < 0.55) &
        (hematoxylin_map >= max(0.08, nuclear_floor)) &
        (edge_mag < 0.02)
    )

    return float(np.mean(pen_mask | dark_fold_mask | dark_blur_mask | crushed_dense_mask))


def _tile_metrics(image: Image.Image) -> dict[str, float]:
    rgb = _rgb_float(image)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # Focus via Laplacian variance
    padded = np.pad(gray, 1, mode="edge")
    lap = (
        padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        - 4.0 * padded[1:-1, 1:-1]
    )
    focus_var = float(np.var(lap))
    focus = focus_var

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

    if _rgb2hed is not None:
        try:
            hed = _rgb2hed(np.clip(rgb, 0.0, 1.0))
            hematoxylin_map = np.clip(hed[..., 0], 0.0, None)
        except Exception:
            hematoxylin_map = nuclear_signal
    else:
        hematoxylin_map = nuclear_signal

    hematoxylin = float(np.mean(hematoxylin_map))

    # === AML CASCADE STAGE 1: Coarse screening features ===
    tissue_frac = _tissue_fraction(rgb, hematoxylin_map, edge_mag)
    purple_frac = _purple_basophilic_fraction(rgb, gray, chroma, hematoxylin_map, edge_mag)
    rbc_frac = _rbc_red_fraction(rgb, gray, chroma, hematoxylin_map, edge_mag)
    eosinophilic_cellular_frac = _eosinophilic_cellular_fraction(rgb, gray, chroma, hematoxylin_map, edge_mag)
    gray_black_frac = _gray_black_artifact_fraction(gray, chroma, edge_mag, hematoxylin_map)
    dark_score = _darkness_score(gray)
    focus_score_val = _focus_score(focus_var)
    artifact_frac = _artifact_fraction(rgb, gray, chroma, edge_mag, hematoxylin_map)

    # Prefer chromatic cellular marrow (blue-purple or red-pink with detail), not
    # bland gray-black darkness.
    coarse_score = (
        0.26 * tissue_frac +
        0.28 * purple_frac +
        0.06 * eosinophilic_cellular_frac +
        0.15 * focus_score_val -
        0.16 * rbc_frac -
        0.15 * artifact_frac -
        0.18 * gray_black_frac
    )

    # === AML CASCADE STAGE 2: Morphology refinement features ===
    nuclear_floor = max(0.05, float(np.percentile(hematoxylin_map, 70)))
    nuclear_mask = hematoxylin_map >= nuclear_floor
    nuclear_density_map = _mean_filter3(nuclear_mask.astype(np.float32, copy=False))
    nuclear_fraction = float(np.mean(nuclear_mask))
    nuclear_detail = float(np.mean(edge_mag[nuclear_mask])) if np.any(nuclear_mask) else 0.0
    packed_nuclear_mask = nuclear_mask & (nuclear_density_map >= 0.28) & (edge_mag > 0.03)
    packed_nuclear_fraction = float(np.mean(packed_nuclear_mask))

    # Enhanced dark cellular detection for AML
    # Must be blue-dominant (basophilic) — reject dark-pink/eosinophilic pixels
    blue_dominant = rgb[:, :, 2] >= rgb[:, :, 0] - 0.06
    dark_cellular_mask = (
        (gray < 0.60) &
        (gray > 0.18) &
        (chroma > 0.10) &
        (hematoxylin_map >= max(0.05, nuclear_floor * 0.90)) &
        (edge_mag > 0.05) &
        blue_dominant
    )
    dark_cellular_fraction = float(np.mean(dark_cellular_mask))

    # Very dark region detection - key for AML ROI
    # Require blue >= red to ensure basophilic (nuclei), not eosinophilic (RBC)
    very_dark_mask = (
        (gray < 0.45) &
        (gray > 0.18) &
        (chroma > 0.10) &
        (hematoxylin_map >= max(0.08, nuclear_floor * 0.95)) &
        (edge_mag > 0.04) &
        blue_dominant
    )
    very_dark_fraction = float(np.mean(very_dark_mask))

    # Dark-in-focus: the most promising AML regions (blue-dominant only)
    tile_in_focus = focus_score_val >= 0.28
    edge_focus_floor = max(0.05, float(np.percentile(edge_mag, 70)))
    dark_in_focus_mask = (
        (gray < 0.50) &
        (gray > 0.20) &
        (chroma > 0.08) &
        (hematoxylin_map >= max(0.10, nuclear_floor * 0.90)) &
        (edge_mag >= edge_focus_floor) &
        tile_in_focus &
        blue_dominant
    )
    dark_in_focus_fraction = float(np.mean(dark_in_focus_mask))

    purple_cellular_mask = (
        (gray < 0.72) &
        (gray > 0.18) &
        (chroma > 0.08) &
        (hematoxylin_map >= max(0.05, nuclear_floor * 0.85)) &
        (edge_mag > 0.04) &
        blue_dominant
    )
    purple_cellular_fraction = float(np.mean(purple_cellular_mask))

    if np.any(nuclear_mask):
        nuclear_values = hematoxylin_map[nuclear_mask]
        nuclear_median = float(np.median(nuclear_values))
        nuclear_mad = float(np.median(np.abs(nuclear_values - nuclear_median)))
    else:
        nuclear_median = 0.0
        nuclear_mad = 0.0
    monomorphic_nuclear_mask = (
        packed_nuclear_mask &
        (np.abs(hematoxylin_map - nuclear_median) <= max(0.025, 1.8 * nuclear_mad))
    )
    monomorphic_nuclear_fraction = float(np.mean(monomorphic_nuclear_mask))
    nuclei_proxy_mask = (
        (hematoxylin_map >= max(0.04, nuclear_floor * 0.80)) &
        (gray < 0.80) &
        (edge_mag > 0.03) &
        (chroma > 0.05)
    )
    nuclei_component_stats = _component_shape_metrics(_downsample_mask(nuclei_proxy_mask, factor=4))
    nuclei_component_density = float(nuclei_component_stats["component_density"])
    nuclei_component_mean_area = float(nuclei_component_stats["component_mean_area"])
    nuclei_component_area_cv = float(nuclei_component_stats["component_area_cv"])
    nuclei_component_mean_circularity = float(nuclei_component_stats["component_mean_circularity"])

    # Penalize dark, stringy, low-circularity material that can look "structured"
    # but is not a true cellular field. This catches smear streaks / fibrin-like /
    # precipitate-heavy regions that fooled the VLM in acellular ROIs.
    line_like_penalty = float(np.clip((0.22 - nuclei_component_mean_circularity) / 0.22, 0.0, 1.0))
    sparse_packing_penalty = float(np.clip((0.08 - packed_nuclear_fraction) / 0.08, 0.0, 1.0))
    dark_stain_presence = float(np.clip((dark_cellular_fraction + very_dark_fraction + purple_cellular_fraction + eosinophilic_cellular_frac) / 0.35, 0.0, 1.0))
    stringy_artifact_score = float(line_like_penalty * sparse_packing_penalty * dark_stain_presence)

    empty_fraction = float(np.mean((gray > 0.82) & (hematoxylin_map < 0.03)))
    nucleated_to_red_ratio = float(
        np.clip(
            purple_cellular_fraction / max(purple_cellular_fraction + rbc_frac + 0.02, 1e-6),
            0.0,
            1.0,
        )
    )

    return {
        # === CASCADE STAGE 1: Coarse screening (for fast prefilter) ===
        "coarse_score": coarse_score,
        "tissue_fraction": tissue_frac,
        "purple_fraction": purple_frac,
        "rbc_fraction": rbc_frac,
        "darkness": dark_score,
        "focus_proxy": focus_score_val,
        "artifact_fraction_coarse": artifact_frac,
        "eosinophilic_cellular_fraction": eosinophilic_cellular_frac,
        "gray_black_fraction": gray_black_frac,

        # === Original metrics for compatibility and refinement stage ===
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
        "packed_nuclear_fraction": packed_nuclear_fraction,
        "purple_cellular_fraction": purple_cellular_fraction,
        "dark_cellular_fraction": dark_cellular_fraction,
        "very_dark_fraction": very_dark_fraction,
        "dark_in_focus_fraction": dark_in_focus_fraction,
        "monomorphic_nuclear_fraction": monomorphic_nuclear_fraction,
        "nuclei_component_density": nuclei_component_density,
        "nuclei_component_mean_area": nuclei_component_mean_area,
        "nuclei_component_area_cv": nuclei_component_area_cv,
        "nuclei_component_mean_circularity": nuclei_component_mean_circularity,
        "stringy_artifact_score": stringy_artifact_score,
        "empty_fraction": empty_fraction,
        "red_dominant_fraction": rbc_frac,
        "nucleated_to_red_ratio": nucleated_to_red_ratio,
        "artifact_fraction": artifact_frac,
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


def score_dark_informative_roi(image: Image.Image) -> float:
    """Score a tile for AML-relevant dark informative ROI.

    Implements the cascade approach:
    1. Coarse screening: tissue + purple - RBC - artifact
    2. Refinement: dark-in-focus, very dark, nuclear packing
    """
    metrics = _tile_metrics(image)

    def _unit(value: float, scale: float) -> float:
        return float(np.clip(value / max(scale, 1e-6), 0.0, 1.0))

    # === CASCADE STAGE 1: Coarse score (primary driver) ===
    # Use the pre-computed coarse score directly
    coarse_score = float(np.clip(metrics["coarse_score"], 0.0, 1.0))

    # === CASCADE STAGE 2: Refinement features ===
    focus_score = _unit(np.log1p(metrics["focus"] * 256.0), np.log1p(3.0))
    focus_energy_score = _unit(np.log1p(metrics["focus_energy"] * 32.0), np.log1p(2.0))

    # Nuclear features
    nuclear_fraction_score = _unit(metrics["nuclear_fraction"], 0.36)
    nuclear_detail_score = _unit(metrics["nuclear_detail"], 0.12)
    packed_nuclear_score = _unit(metrics["packed_nuclear_fraction"], 0.28)

    # Dark region features - key for AML
    dark_cellular_score = _unit(metrics["dark_cellular_fraction"], 0.30)
    very_dark_score = _unit(metrics["very_dark_fraction"], 0.20)
    dark_in_focus_score = _unit(metrics["dark_in_focus_fraction"], 0.15)
    eosinophilic_cellular_score = _unit(metrics["eosinophilic_cellular_fraction"], 0.18)

    # Morphology
    component_density_score = _unit(metrics["nuclei_component_density"], 0.07)
    component_uniformity_score = float(np.clip(1.0 - (metrics["nuclei_component_area_cv"] / 1.5), 0.0, 1.0))
    round_nuclei_score = _unit(metrics["nuclei_component_mean_circularity"], 0.28)

    # Penalties
    empty_penalty = _unit(metrics["empty_fraction"], 0.55)
    red_penalty = _unit(metrics["red_dominant_fraction"], 0.30)
    nucleated_to_red_score = _unit(metrics["nucleated_to_red_ratio"], 1.0)
    artifact_penalty = _unit(metrics["artifact_fraction"], 0.22)
    stringy_penalty = _unit(metrics["stringy_artifact_score"], 1.0)
    gray_black_penalty = _unit(metrics["gray_black_fraction"], 0.20)

    # BRIGHTNESS PENALTY: Light tiles cannot be dense cellular AML regions.
    # brightness > 0.65 → increasingly penalized; brightness > 0.78 → max penalty
    brightness_val = metrics["brightness"]
    brightness_penalty = float(np.clip((brightness_val - 0.55) / 0.20, 0.0, 1.0))

    # === COMBINED SCORE ===
    # Heavily weight the coarse score (cascade stage 1)
    # Then add refinement features for discriminating among candidates
    score = (
        # Coarse screening (35% - the foundation)
        0.34 * coarse_score +

        # Dark region features (15% - AML key signal, but not blind darkness)
        0.08 * dark_cellular_score +
        0.03 * very_dark_score +
        0.04 * dark_in_focus_score +

        # Nuclear packing (26% - cellularity confirmation, boosted)
        0.10 * nuclear_fraction_score +
        0.09 * packed_nuclear_score +
        0.07 * nuclear_detail_score +

        # Focus/quality (10%)
        0.05 * focus_score +
        0.05 * focus_energy_score +

        # Morphology (6%)
        0.02 * component_density_score +
        0.01 * component_uniformity_score +
        0.03 * round_nuclei_score +

        # Rare fallback: dark red-pink cellular marrow can still be informative
        0.02 * eosinophilic_cellular_score +

        # Nucleated vs RBC ratio
        0.06 * nucleated_to_red_score +

        # Penalties (brightness is the BIGGEST penalty)
        - 0.08 * empty_penalty -
        0.05 * red_penalty -
        0.12 * artifact_penalty -
        0.14 * stringy_penalty -
        0.18 * gray_black_penalty -
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
        return TileQualitySelection(
            selected_indices=[],
            total_tiles=0,
            pool_tiles=0,
            hard_rejected_tiles=0,
            reserved_tiles=0,
        )

    metrics = [_tile_metrics(image) for image in images]

    # === CASCADE STAGE 1: Coarse screening features ===
    coarse_scores = np.asarray([m["coarse_score"] for m in metrics], dtype=np.float32)
    tissue_fraction = np.asarray([m["tissue_fraction"] for m in metrics], dtype=np.float32)
    purple_fraction = np.asarray([m["purple_fraction"] for m in metrics], dtype=np.float32)
    rbc_fraction = np.asarray([m["rbc_fraction"] for m in metrics], dtype=np.float32)
    focus_proxy = np.asarray([m["focus_proxy"] for m in metrics], dtype=np.float32)
    artifact_coarse = np.asarray([m["artifact_fraction_coarse"] for m in metrics], dtype=np.float32)

    # === CASCADE STAGE 2: Refinement features ===
    focus = np.asarray([m["focus"] for m in metrics], dtype=np.float32)
    focus_energy = np.asarray([m["focus_energy"] for m in metrics], dtype=np.float32)
    brightness = np.asarray([m["brightness"] for m in metrics], dtype=np.float32)
    stain_strength = np.asarray([m["stain_strength"] for m in metrics], dtype=np.float32)
    hematoxylin = np.asarray([m["hematoxylin"] for m in metrics], dtype=np.float32)
    nuclear_fraction = np.asarray([m["nuclear_fraction"] for m in metrics], dtype=np.float32)
    packed_nuclear_fraction = np.asarray([m["packed_nuclear_fraction"] for m in metrics], dtype=np.float32)
    dark_cellular_fraction = np.asarray([m["dark_cellular_fraction"] for m in metrics], dtype=np.float32)
    very_dark_fraction = np.asarray([m["very_dark_fraction"] for m in metrics], dtype=np.float32)
    dark_in_focus_fraction = np.asarray([m["dark_in_focus_fraction"] for m in metrics], dtype=np.float32)
    eosinophilic_cellular_fraction = np.asarray([m["eosinophilic_cellular_fraction"] for m in metrics], dtype=np.float32)
    gray_black_fraction = np.asarray([m["gray_black_fraction"] for m in metrics], dtype=np.float32)
    nuclei_component_density = np.asarray([m["nuclei_component_density"] for m in metrics], dtype=np.float32)
    nuclei_component_area_cv = np.asarray([m["nuclei_component_area_cv"] for m in metrics], dtype=np.float32)
    nuclei_component_mean_circularity = np.asarray([m["nuclei_component_mean_circularity"] for m in metrics], dtype=np.float32)
    stringy_artifact_score = np.asarray([m["stringy_artifact_score"] for m in metrics], dtype=np.float32)
    empty_fraction = np.asarray([m["empty_fraction"] for m in metrics], dtype=np.float32)
    red_dominant_fraction = np.asarray([m["red_dominant_fraction"] for m in metrics], dtype=np.float32)
    nucleated_to_red_ratio = np.asarray([m["nucleated_to_red_ratio"] for m in metrics], dtype=np.float32)
    artifact_fraction = np.asarray([m["artifact_fraction"] for m in metrics], dtype=np.float32)

    # === CASCADE HARD REJECTION: Background removal + color screening ===
    # AML ROI selection: background removal -> color screening -> morphology refinement
    # CRITICAL: Dark purple regions often have LOW saturation (all channels dark),
    # so we must NOT filter on saturation alone for dark tiles.
    coarse_score_floor = float(np.percentile(coarse_scores, 20)) if total_tiles > 1 else -0.5
    tissue_floor = max(0.03, float(np.percentile(tissue_fraction, 8)) if total_tiles > 1 else 0.05)
    purple_floor = max(0.01, float(np.percentile(purple_fraction, 8)) if total_tiles > 1 else 0.03)
    eosinophilic_floor = max(0.01, float(np.percentile(eosinophilic_cellular_fraction, 8)) if total_tiles > 1 else 0.02)

    # Keep dense chromatic cellular tiles, but deep blue-purple remains the primary
    # target. Dark red-pink is only a rare fallback when it is clearly cellular.
    chromatic_cellular_indicator = (
        (brightness < 0.62) &
        (hematoxylin > 0.08) &
        ((purple_fraction > 0.01) | (eosinophilic_cellular_fraction > 0.04)) &
        (stringy_artifact_score < 0.55) &
        (gray_black_fraction < 0.28) &
        (
            (packed_nuclear_fraction > 0.02) |
            (nuclei_component_mean_circularity > 0.18)
        )
    )

    # Hard reject: background (no tissue), no purple signal, too much RBC, severe artifacts
    # BUT: preserve dark purple regions even if they fail saturation check
    # BRIGHTNESS: tiles with brightness > 0.78 are too light for dense cellular regions
    standard_keep = (
        (tissue_fraction >= tissue_floor) &
        ((purple_fraction >= purple_floor) | (eosinophilic_cellular_fraction >= max(0.04, eosinophilic_floor))) &
        (rbc_fraction <= 0.58) &
        (artifact_coarse <= 0.33) &
        (gray_black_fraction <= 0.30) &
        (coarse_scores >= coarse_score_floor) &
        (brightness <= 0.78)
    )

    stringy_reject = (
        (
            (stringy_artifact_score >= 0.70) &
            (packed_nuclear_fraction <= 0.06) &
            (nuclei_component_mean_circularity <= 0.18)
        ) |
        (
            (gray_black_fraction >= 0.34) &
            (packed_nuclear_fraction <= 0.08) &
            (nuclei_component_mean_circularity <= 0.20)
        )
    )
    hard_keep = (standard_keep | chromatic_cellular_indicator) & ~stringy_reject

    pool = np.nonzero(hard_keep)[0]
    hard_rejected_tiles = int(total_tiles - int(pool.size))
    if pool.size == 0:
        # Fallback: keep at least some tiles if all rejected
        pool = np.argsort(coarse_scores)[::-1][: max(min_keep_tiles, int(total_tiles * 0.2))]
        hard_rejected_tiles = total_tiles - len(pool)

    if pool.size <= max(1, int(trigger_tile_count)):
        return TileQualitySelection(
            selected_indices=pool.astype(int).tolist(),
            total_tiles=total_tiles,
            pool_tiles=int(pool.size),
            hard_rejected_tiles=hard_rejected_tiles,
            reserved_tiles=0,
        )

    # === CASCADE SCORING: Use coarse score as foundation, refine with detailed features ===
    # Coarse score already encodes: 0.30*tissue + 0.30*purple + 0.15*focus - 0.20*RBC - 0.20*artifact
    coarse_score_pool = _robust_unit_scale(coarse_scores[pool])

    # Refinement features
    focus_score = _robust_unit_scale(np.log1p(focus[pool] * 256.0))
    focus_energy_score = _robust_unit_scale(np.log1p(focus_energy[pool] * 32.0))
    nuclear_fraction_score = _robust_unit_scale(nuclear_fraction[pool])
    packed_nuclear_score = _robust_unit_scale(packed_nuclear_fraction[pool])
    dark_cellular_score = _robust_unit_scale(dark_cellular_fraction[pool])
    very_dark_score = _robust_unit_scale(very_dark_fraction[pool])
    dark_in_focus_score = _robust_unit_scale(dark_in_focus_fraction[pool])
    eosinophilic_cellular_score = _robust_unit_scale(eosinophilic_cellular_fraction[pool])
    component_density_score = _robust_unit_scale(nuclei_component_density[pool])
    component_uniformity_score = np.clip(
        1.0 - (nuclei_component_area_cv[pool] / max(float(np.percentile(nuclei_component_area_cv[pool], 90)), 1.5)),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    round_nuclei_score = _robust_unit_scale(nuclei_component_mean_circularity[pool])
    empty_penalty = _robust_unit_scale(empty_fraction[pool])
    red_penalty = _robust_unit_scale(red_dominant_fraction[pool])
    nucleated_to_red_score = _robust_unit_scale(nucleated_to_red_ratio[pool])
    artifact_penalty = _robust_unit_scale(artifact_fraction[pool])
    stringy_penalty = _robust_unit_scale(stringy_artifact_score[pool])
    gray_black_penalty = _robust_unit_scale(gray_black_fraction[pool])
    brightness_penalty = _robust_unit_scale(np.clip((brightness[pool] - 0.55) / 0.20, 0.0, 1.0))

    # AML ROI ranking: cascade approach
    # Brightness penalty is critical — light tiles with stain are NOT cellular
    combined = (
        # Coarse screening foundation (40%)
        0.38 * coarse_score_pool +

        # Dark region refinement (12%) - keep color-rich dark tissue, not bland blackness
        0.05 * dark_cellular_score +
        0.03 * very_dark_score +
        0.04 * dark_in_focus_score +

        # Nuclear packing (19%)
        0.09 * nuclear_fraction_score +
        0.10 * packed_nuclear_score +

        # Focus (10%)
        0.06 * focus_score +
        0.04 * focus_energy_score +

        # Morphology (6%)
        0.02 * component_density_score +
        0.01 * component_uniformity_score +
        0.03 * round_nuclei_score +

        # Rare fallback only; blue-purple should dominate candidate ranking
        0.02 * eosinophilic_cellular_score +

        # Nucleated vs RBC (6%)
        0.06 * nucleated_to_red_score +

        # Penalties (brightness heavily penalized)
        - 0.12 * artifact_penalty -
        0.12 * stringy_penalty -
        0.16 * gray_black_penalty -
        0.06 * red_penalty -
        0.07 * empty_penalty -
        0.30 * brightness_penalty
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


__all__ = ["TileQualitySelection", "score_dark_informative_roi", "select_informative_tile_indices"]
