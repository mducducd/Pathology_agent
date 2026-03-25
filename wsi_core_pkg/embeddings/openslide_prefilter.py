from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import openslide
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "openslide-python is required. Install it with your environment's package manager or pip."
    ) from exc

try:
    from skimage.color import rgb2hed
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-image is required. Install it with your environment's package manager or pip."
    ) from exc


@dataclass(frozen=True)
class TileCoord:
    """Top-left tile coordinate in level-0 reference pixels."""

    x0: int
    y0: int


@dataclass(frozen=True)
class SelectedTile:
    """Selected tile with raw metrics and final combined score."""

    x0: int
    y0: int
    score: float
    focus: float
    entropy: float
    h_mean: float


@dataclass(frozen=True)
class FilterStats:
    n_input: int
    n_selected: int
    selected_fraction: float
    prefilter_level: int
    prefilter_downsample: float
    proxy_size: int
    score_mean_all: float
    score_mean_selected: float


def shannon_entropy_u8(gray_u8: np.ndarray) -> float:
    """Compute Shannon entropy for a grayscale uint8 image."""
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())



def tile_metrics(rgb: np.ndarray) -> tuple[float, float, float]:
    """
    Compute cheap raw-tile metrics.

    Returns:
        focus: gradient energy (higher is sharper)
        entropy: grayscale Shannon entropy
        h_mean: mean hematoxylin signal in HED color space
    """
    gray = (
        0.299 * rgb[..., 0] +
        0.587 * rgb[..., 1] +
        0.114 * rgb[..., 2]
    ).astype(np.float32)

    gy, gx = np.gradient(gray)
    focus = float(np.mean(gx * gx + gy * gy))

    entropy = shannon_entropy_u8(np.clip(gray, 0, 255).astype(np.uint8))

    hed = rgb2hed(rgb)
    h = np.clip(hed[..., 0], 0.0, None)
    h_mean = float(np.mean(h))

    return focus, entropy, h_mean



def robust_unit_scale(x: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> np.ndarray:
    """Robustly scale values to [0, 1] using percentiles."""
    a, b = np.percentile(x, [lo, hi])
    if b <= a + 1e-12:
        return np.full_like(x, 0.5, dtype=np.float32)
    y = (x - a) / (b - a)
    return np.clip(y, 0.0, 1.0).astype(np.float32)



def choose_prefilter_level(
    slide: openslide.OpenSlide,
    *,
    target_level: int,
    target_tile_size: int,
    desired_proxy_size: int = 64,
) -> int:
    """
    Choose a coarse level so the proxy tile is around `desired_proxy_size` pixels.
    """
    ds_target = float(slide.level_downsamples[target_level])
    desired_prefilter_ds = ds_target * (target_tile_size / desired_proxy_size)
    return int(slide.get_best_level_for_downsample(desired_prefilter_ds))



def same_region_proxy_size(
    slide: openslide.OpenSlide,
    *,
    target_level: int,
    target_tile_size: int,
    prefilter_level: int,
) -> int:
    """
    Return the size at `prefilter_level` that covers the same physical region as
    `target_tile_size` at `target_level`.
    """
    ds_target = float(slide.level_downsamples[target_level])
    ds_pref = float(slide.level_downsamples[prefilter_level])
    return max(8, int(round(target_tile_size * ds_target / ds_pref)))



def load_coords_csv(path: str | Path) -> list[TileCoord]:
    """
    Load candidate tile coordinates from a CSV.

    Expected header names:
      - x0,y0
      - or x,y

    Coordinates must be top-left tile positions in level-0 pixels.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Coordinate CSV not found: {path}")

    coords: list[TileCoord] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Coordinate CSV is missing a header row")

        fieldnames = {name.strip().lower(): name for name in reader.fieldnames}
        x_name = fieldnames.get("x0") or fieldnames.get("x")
        y_name = fieldnames.get("y0") or fieldnames.get("y")
        if x_name is None or y_name is None:
            raise ValueError(
                "Coordinate CSV must contain x0,y0 columns or x,y columns"
            )

        for row_idx, row in enumerate(reader, start=2):
            try:
                x0 = int(float(row[x_name]))
                y0 = int(float(row[y_name]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid coordinate on CSV line {row_idx}: {row}"
                ) from exc
            coords.append(TileCoord(x0=x0, y0=y0))

    return coords



def save_selected_csv(path: str | Path, selected: Iterable[SelectedTile]) -> None:
    """Write selected tiles and their scores to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x0", "y0", "score", "focus", "entropy", "h_mean"])
        for tile in selected:
            writer.writerow(
                [
                    tile.x0,
                    tile.y0,
                    f"{tile.score:.8f}",
                    f"{tile.focus:.8f}",
                    f"{tile.entropy:.8f}",
                    f"{tile.h_mean:.8f}",
                ]
            )



def prefilter_tiles(
    slide_path: str | Path,
    candidates_level0: Sequence[TileCoord],
    *,
    target_level: int = 0,
    target_tile_size: int = 224,
    desired_proxy_size: int = 64,
    keep_top_fraction: float = 0.30,
    reserve_fraction_from_rejects: float = 0.05,
    min_percentile: float = 10.0,
    focus_weight: float = 0.50,
    entropy_weight: float = 0.20,
    h_weight: float = 0.30,
    random_seed: int = 0,
) -> tuple[list[SelectedTile], FilterStats]:
    """
    Prefilter raw-tissue tiles before expensive foundation-model embedding.

    The filter uses a cheap read at a coarser pyramid level and scores tiles by:
      - focus (gradient energy)
      - entropy (information content)
      - hematoxylin signal (nuclear/stain content)

    A conservative low-tail rejection is applied, then the top-scoring survivor
    tiles are kept, plus a small random reserve from rejected tiles to preserve
    recall.
    """
    if not 0.0 < keep_top_fraction <= 1.0:
        raise ValueError("keep_top_fraction must be in (0, 1]")
    if not 0.0 <= reserve_fraction_from_rejects <= 1.0:
        raise ValueError("reserve_fraction_from_rejects must be in [0, 1]")
    if not 0.0 <= min_percentile < 100.0:
        raise ValueError("min_percentile must be in [0, 100)")

    rng = np.random.default_rng(random_seed)
    slide = openslide.OpenSlide(str(slide_path))

    try:
        if not 0 <= target_level < slide.level_count:
            raise ValueError(
                f"target_level={target_level} is out of range for slide with "
                f"{slide.level_count} levels"
            )

        prefilter_level = choose_prefilter_level(
            slide,
            target_level=target_level,
            target_tile_size=target_tile_size,
            desired_proxy_size=desired_proxy_size,
        )
        proxy_size = same_region_proxy_size(
            slide,
            target_level=target_level,
            target_tile_size=target_tile_size,
            prefilter_level=prefilter_level,
        )
        prefilter_downsample = float(slide.level_downsamples[prefilter_level])

        coords: list[TileCoord] = []
        focus_list: list[float] = []
        entropy_list: list[float] = []
        h_list: list[float] = []

        for tile in candidates_level0:
            rgba = slide.read_region(
                (tile.x0, tile.y0),
                prefilter_level,
                (proxy_size, proxy_size),
            )
            rgb = np.asarray(rgba, dtype=np.uint8)[..., :3]

            focus, entropy, h_mean = tile_metrics(rgb)
            coords.append(tile)
            focus_list.append(focus)
            entropy_list.append(entropy)
            h_list.append(h_mean)
    finally:
        slide.close()

    if len(coords) == 0:
        empty_stats = FilterStats(
            n_input=0,
            n_selected=0,
            selected_fraction=0.0,
            prefilter_level=0,
            prefilter_downsample=1.0,
            proxy_size=0,
            score_mean_all=0.0,
            score_mean_selected=0.0,
        )
        return [], empty_stats

    focus_arr = np.asarray(focus_list, dtype=np.float32)
    entropy_arr = np.asarray(entropy_list, dtype=np.float32)
    h_arr = np.asarray(h_list, dtype=np.float32)

    keep_mask = (
        (focus_arr >= np.percentile(focus_arr, min_percentile))
        & (entropy_arr >= np.percentile(entropy_arr, min_percentile))
        & (h_arr >= np.percentile(h_arr, min_percentile))
    )

    focus_n = robust_unit_scale(focus_arr)
    entropy_n = robust_unit_scale(entropy_arr)
    h_n = robust_unit_scale(h_arr)
    score = focus_weight * focus_n + entropy_weight * entropy_n + h_weight * h_n

    survivor_idx = np.flatnonzero(keep_mask)
    reject_idx = np.flatnonzero(~keep_mask)
    if len(survivor_idx) == 0:
        survivor_idx = np.arange(len(coords), dtype=np.int64)

    n_keep = max(1, int(np.ceil(len(survivor_idx) * keep_top_fraction)))
    ranked_survivors = survivor_idx[np.argsort(score[survivor_idx])[::-1]]
    top_survivors = ranked_survivors[:n_keep]

    n_reserve = int(np.ceil(len(coords) * reserve_fraction_from_rejects))
    if len(reject_idx) > 0 and n_reserve > 0:
        reserve = rng.choice(
            reject_idx,
            size=min(n_reserve, len(reject_idx)),
            replace=False,
        )
        final_idx = np.concatenate([top_survivors, reserve])
    else:
        final_idx = top_survivors

    final_idx = np.unique(final_idx)
    final_idx = final_idx[np.argsort(score[final_idx])[::-1]]

    selected_tiles = [
        SelectedTile(
            x0=coords[i].x0,
            y0=coords[i].y0,
            score=float(score[i]),
            focus=float(focus_arr[i]),
            entropy=float(entropy_arr[i]),
            h_mean=float(h_arr[i]),
        )
        for i in final_idx
    ]

    stats = FilterStats(
        n_input=len(coords),
        n_selected=len(selected_tiles),
        selected_fraction=len(selected_tiles) / len(coords),
        prefilter_level=prefilter_level,
        prefilter_downsample=prefilter_downsample,
        proxy_size=proxy_size,
        score_mean_all=float(np.mean(score)),
        score_mean_selected=float(np.mean(score[final_idx])),
    )
    return selected_tiles, stats



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cheap OpenSlide prefilter for WSI tiles before expensive foundation-model embedding. "
            "Input coordinates must already be tissue-only candidates in level-0 pixels."
        )
    )
    parser.add_argument("--slide", required=True, help="Path to input WSI (.svs, .ndpi, .mrxs, ...)")
    parser.add_argument("--coords-csv", required=True, help="CSV with x0,y0 or x,y columns")
    parser.add_argument("--out-csv", required=True, help="Output CSV for selected tiles")
    parser.add_argument("--target-level", type=int, default=0, help="Embedding read level")
    parser.add_argument("--tile-size", type=int, default=224, help="Embedding tile size at target level")
    parser.add_argument(
        "--proxy-size",
        type=int,
        default=64,
        help="Approximate proxy tile size used for cheap scoring",
    )
    parser.add_argument(
        "--keep-top-fraction",
        type=float,
        default=0.30,
        help="Fraction of survivor tiles to keep by score",
    )
    parser.add_argument(
        "--reserve-fraction",
        type=float,
        default=0.05,
        help="Random reserve fraction sampled from rejected tiles",
    )
    parser.add_argument(
        "--min-percentile",
        type=float,
        default=10.0,
        help="Low-tail percentile cutoff applied to each raw metric",
    )
    parser.add_argument("--focus-weight", type=float, default=0.50, help="Weight for focus score")
    parser.add_argument("--entropy-weight", type=float, default=0.20, help="Weight for entropy score")
    parser.add_argument("--h-weight", type=float, default=0.30, help="Weight for hematoxylin score")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed for reserve sampling")
    return parser



def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    coords = load_coords_csv(args.coords_csv)
    selected, stats = prefilter_tiles(
        slide_path=args.slide,
        candidates_level0=coords,
        target_level=args.target_level,
        target_tile_size=args.tile_size,
        desired_proxy_size=args.proxy_size,
        keep_top_fraction=args.keep_top_fraction,
        reserve_fraction_from_rejects=args.reserve_fraction,
        min_percentile=args.min_percentile,
        focus_weight=args.focus_weight,
        entropy_weight=args.entropy_weight,
        h_weight=args.h_weight,
        random_seed=args.random_seed,
    )
    save_selected_csv(args.out_csv, selected)

    print(f"Input tiles:           {stats.n_input}")
    print(f"Selected tiles:        {stats.n_selected}")
    print(f"Selected fraction:     {stats.selected_fraction:.4f}")
    print(f"Prefilter level:       {stats.prefilter_level}")
    print(f"Prefilter downsample:  {stats.prefilter_downsample:.4f}")
    print(f"Proxy size:            {stats.proxy_size}")
    print(f"Mean score (all):      {stats.score_mean_all:.6f}")
    print(f"Mean score (selected): {stats.score_mean_selected:.6f}")
    print(f"Saved:                 {args.out_csv}")


if __name__ == "__main__":
    main()
