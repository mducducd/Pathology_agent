from __future__ import annotations

import hashlib
import json
import logging
import re
import xml.dom.minidom as minidom
from collections.abc import Callable, Iterator
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Final, Generic, Literal, NamedTuple, NewType, TypeAlias, TypeVar, TypedDict, cast
from zipfile import ZipFile

import numpy as np
import numpy.typing as npt
import openslide
import torch
from PIL import Image

from . import Extractor
from .openslide_prefilter import (
    choose_screening_level,
    get_slide_nonempty_bounds_level0,
    same_region_proxy_size_level0,
    screening_roi_score,
    screening_tile_metrics,
)
from .tile_prefilter import score_dark_informative_roi, select_informative_tile_indices

ImageExtension: TypeAlias = Literal["png", "jpg"]
EXTENSION_TO_FORMAT: Final[dict[ImageExtension, str]] = {"png": "PNG", "jpg": "JPEG"}

Microns = NewType("Microns", float)
SlidePixels = NewType("SlidePixels", int)
TilePixels = NewType("TilePixels", int)
SlideMPP = NewType("SlideMPP", float)
DeviceLikeType: TypeAlias = str | torch.device | int

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2025 Marko van Treeck"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

with open(__file__, "rb") as this_file_fp:
    _CODE_HASH: Final[str] = hashlib.file_digest(this_file_fp, "sha256").hexdigest()

_Unit = TypeVar("_Unit")


@dataclass(frozen=True)
class _XYCoords(Generic[_Unit]):
    x: _Unit
    y: _Unit


class _Tile(NamedTuple, Generic[_Unit]):
    image: Image.Image
    coordinates: _XYCoords[_Unit]
    size: _Unit


class _TilerParams(TypedDict):
    slide_path: str
    tile_size_um: Microns
    tile_size_px: TilePixels
    max_supertile_size_slide_px: SlidePixels
    brightness_cutoff: int | None
    tile_prefilter_method: str
    coarse_trigger_supertile_count: int | None
    coarse_keep_ratio: float | None
    coarse_min_keep_supertile_count: int
    coarse_max_keep_supertile_count: int | None
    quality_keep_ratio: float | None
    quality_min_keep_tile_count: int
    quality_trigger_tile_count: int | None
    quality_random_reserve_ratio: float | None
    code_sha256: str
    tile_ext: ImageExtension


class MPPExtractionError(Exception):
    """Raised when MPP extraction from slide metadata fails and no default is provided."""


@dataclass(frozen=True)
class TileFeatureMatrix:
    features: torch.Tensor
    coordinates_um: npt.NDArray[np.float32]
    dark_roi_scores: npt.NDArray[np.float32]
    tile_size_um: float
    tile_size_px: int
    slide_path: str
    extractor_id: str


def tiles_with_cache(
    slide_path: Path | str,
    *,
    cache_dir: Path | None,
    cache_tiles_ext: ImageExtension,
    tile_size_um: Microns,
    tile_size_px: TilePixels,
    max_supertile_size_slide_px: SlidePixels,
    max_workers: int,
    brightness_cutoff: int | None,
    canny_cutoff: float | None,
    default_slide_mpp: SlideMPP | None,
    tile_prefilter_method: str,
    coarse_trigger_supertile_count: int | None,
    coarse_keep_ratio: float | None,
    coarse_min_keep_supertile_count: int,
    coarse_max_keep_supertile_count: int | None,
    quality_keep_ratio: float | None,
    quality_min_keep_tile_count: int,
    quality_trigger_tile_count: int | None,
    quality_random_reserve_ratio: float | None,
    dark_region_boxes_level0: list[dict[str, int]] | None = None,
    progress_cb: Callable[[dict[str, Any]], None] | None,
) -> Iterator[_Tile[Microns]]:
    """Iterate over tiles in a WSI, using cache if configured.

    Args:
        dark_region_boxes_level0: If provided, only yield tiles whose centers fall
            within these bounding boxes (level-0 coordinates). Enables strict dark-region
            gating for AML mode.
    """
    slide_path = Path(slide_path)

    if cache_dir is None:
        slide = openslide.open_slide(str(slide_path))
        try:
            yield from _tiles_with_tissue(
                slide=slide,
                tile_size_um=tile_size_um,
                tile_size_px=tile_size_px,
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
            )
        finally:
            slide.close()
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    tiler_params: _TilerParams = {
        "slide_path": str(slide_path),
        "tile_size_um": tile_size_um,
        "tile_size_px": tile_size_px,
        "max_supertile_size_slide_px": max_supertile_size_slide_px,
        "brightness_cutoff": brightness_cutoff,
        "tile_prefilter_method": tile_prefilter_method,
        "coarse_trigger_supertile_count": coarse_trigger_supertile_count,
        "coarse_keep_ratio": coarse_keep_ratio,
        "coarse_min_keep_supertile_count": coarse_min_keep_supertile_count,
        "coarse_max_keep_supertile_count": coarse_max_keep_supertile_count,
        "quality_keep_ratio": quality_keep_ratio,
        "quality_min_keep_tile_count": quality_min_keep_tile_count,
        "quality_trigger_tile_count": quality_trigger_tile_count,
        "quality_random_reserve_ratio": quality_random_reserve_ratio,
        "code_sha256": _CODE_HASH,
        "tile_ext": cache_tiles_ext,
    }
    tiler_params_hash = hashlib.sha256(json.dumps(tiler_params, sort_keys=True).encode()).hexdigest()
    cache_file_path = cache_dir / slide_path.with_suffix(f".{tiler_params_hash}.zip").name

    if cache_file_path.exists():
        yield from _tiles_from_cache_file(cache_file_path)
        return

    with (
        NamedTemporaryFile(dir=cache_file_path.parent, delete=False) as tmp_cache_file,
        ZipFile(tmp_cache_file.name, "w") as zip_fp,
    ):
        try:
            with zip_fp.open("tiler_params.json", "w") as tiler_params_json_fp:
                tiler_params_json_fp.write(json.dumps(tiler_params).encode())

            slide = openslide.open_slide(str(slide_path))
            try:
                for tile in _tiles_with_tissue(
                    slide=slide,
                    tile_size_um=tile_size_um,
                    tile_size_px=tile_size_px,
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
                ):
                    with zip_fp.open(
                        f"tile_({float(tile.coordinates.x)}, {float(tile.coordinates.y)}).{cache_tiles_ext}",
                        "w",
                    ) as tile_zip_fp:
                        save_kwargs = {"format": EXTENSION_TO_FORMAT[cache_tiles_ext]}
                        if cache_tiles_ext == "png":
                            save_kwargs["icc_profile"] = None
                        tile.image.save(tile_zip_fp, **save_kwargs)
                    yield tile
            finally:
                slide.close()
        except Exception:
            _logger.exception("error while processing %s", slide_path)
            Path(tmp_cache_file.name).unlink(missing_ok=True)
            raise

        Path(tmp_cache_file.name).rename(cache_file_path)


def _tiles_with_tissue(
    slide: openslide.AbstractSlide,
    *,
    tile_size_um: Microns,
    tile_size_px: TilePixels,
    max_supertile_size_slide_px: SlidePixels,
    max_workers: int,
    brightness_cutoff: int | None,
    canny_cutoff: float | None,
    default_slide_mpp: SlideMPP | None,
    tile_prefilter_method: str,
    coarse_trigger_supertile_count: int | None,
    coarse_keep_ratio: float | None,
    coarse_min_keep_supertile_count: int,
    coarse_max_keep_supertile_count: int | None,
    quality_keep_ratio: float | None,
    quality_min_keep_tile_count: int,
    quality_trigger_tile_count: int | None,
    quality_random_reserve_ratio: float | None,
    dark_region_boxes_level0: list[dict[str, int]] | None = None,
    progress_cb: Callable[[dict[str, Any]], None] | None,
) -> Iterator[_Tile[Microns]]:
    """Iterate over tiles in tissue regions, optionally restricted to dark regions.

    Args:
        dark_region_boxes_level0: If provided, only yield tiles whose centers fall
            within these bounding boxes (level-0 coordinates).
    """
    use_quality_prefilter = (
        quality_keep_ratio is not None
        and quality_keep_ratio > 0.0
        and tile_prefilter_method in {"quality", "hybrid"}
    )
    use_dark_region_gating = dark_region_boxes_level0 is not None and len(dark_region_boxes_level0) > 0
    if use_dark_region_gating:
        # In AML mode we want all tiles from the detected dark regions to be embedded.
        # The dark-region boxes already define the candidate area, so do not prune again
        # with the per-tile quality prefilter here.
        use_quality_prefilter = False
    quality_total_tiles = 0
    quality_kept_tiles = 0
    quality_pool_tiles = 0
    quality_hard_rejected_tiles = 0
    dark_region_filtered_tiles = 0

    if use_quality_prefilter and progress_cb is not None:
        progress_cb(
            {
                "phase": "quality_prefilter",
                "status": "running",
                "quality_total_tiles": 0,
                "quality_kept_tiles": 0,
                "quality_pool_tiles": 0,
                "quality_hard_rejected_tiles": 0,
            }
        )

    for supertile, supertile_coords_um, _ in _supertiles(
        slide=slide,
        tile_size_um=tile_size_um,
        tile_size_px=tile_size_px,
        max_supertile_size_slide_px=max_supertile_size_slide_px,
        max_workers=max_workers,
        brightness_cutoff=brightness_cutoff,
        default_slide_mpp=default_slide_mpp,
        coarse_trigger_supertile_count=coarse_trigger_supertile_count,
        coarse_keep_ratio=coarse_keep_ratio,
        coarse_min_keep_supertile_count=coarse_min_keep_supertile_count,
        coarse_max_keep_supertile_count=coarse_max_keep_supertile_count,
        progress_cb=progress_cb,
    ):
        tiles = _split_supertile_into_tiles(
            supertile=supertile,
            supertile_coords_um=supertile_coords_um,
            tile_size_um=tile_size_um,
            tile_size_px=tile_size_px,
        )

        # STRICT DARK REGION GATING: Filter tiles whose centers fall outside dark regions
        if use_dark_region_gating:
            slide_mpp = cast(SlideMPP, get_slide_mpp_(slide, default_mpp=default_slide_mpp))
            filtered_tiles: list[_Tile[Microns]] = []
            for tile in tiles:
                # tile.coordinates.x/y are absolute level-0 micron coordinates (top-left corner)
                # Tile center in microns
                tile_center_x_um = float(tile.coordinates.x) + (float(tile_size_um) / 2.0)
                tile_center_y_um = float(tile.coordinates.y) + (float(tile_size_um) / 2.0)
                # Convert to level-0 pixels
                if slide_mpp is not None:
                    tile_center_x_px = int(tile_center_x_um / float(slide_mpp))
                    tile_center_y_px = int(tile_center_y_um / float(slide_mpp))
                    # Check if tile center falls within any dark region box
                    for box in dark_region_boxes_level0:
                        bx0, by0 = int(box["x0"]), int(box["y0"])
                        bw, bh = int(box["w"]), int(box["h"])
                        bx1, by1 = bx0 + bw, by0 + bh
                        if bx0 <= tile_center_x_px < bx1 and by0 <= tile_center_y_px < by1:
                            filtered_tiles.append(tile)
                            break
            dark_region_filtered_tiles += len(tiles) - len(filtered_tiles)
            tiles = filtered_tiles

        if not tiles:
            continue

        if canny_cutoff is not None:
            tiles = [tile for tile in tiles if _has_enough_texture(tile.image, cutoff=canny_cutoff)]

        if use_quality_prefilter and tiles:
            selection = select_informative_tile_indices(
                [tile.image for tile in tiles],
                keep_ratio=float(quality_keep_ratio),
                min_keep_tiles=int(quality_min_keep_tile_count),
                trigger_tile_count=max(1, int(quality_trigger_tile_count or 1)),
                random_reserve_ratio=float(quality_random_reserve_ratio or 0.0),
            )
            quality_total_tiles += selection.total_tiles
            quality_kept_tiles += len(selection.selected_indices)
            quality_pool_tiles += selection.pool_tiles
            quality_hard_rejected_tiles += selection.hard_rejected_tiles
            if progress_cb is not None:
                progress_cb(
                    {
                        "phase": "quality_prefilter",
                        "status": "running",
                        "quality_total_tiles": quality_total_tiles,
                        "quality_kept_tiles": quality_kept_tiles,
                        "quality_pool_tiles": quality_pool_tiles,
                        "quality_hard_rejected_tiles": quality_hard_rejected_tiles,
                    }
                )
            tiles = [tiles[idx] for idx in selection.selected_indices]

        for tile in tiles:
            yield tile

    if use_quality_prefilter and progress_cb is not None:
        progress_cb(
            {
                "phase": "quality_prefilter",
                "status": "done",
                "quality_total_tiles": quality_total_tiles,
                "quality_kept_tiles": quality_kept_tiles,
                "quality_pool_tiles": quality_pool_tiles,
                "quality_hard_rejected_tiles": quality_hard_rejected_tiles,
            }
        )


def _split_supertile_into_tiles(
    *,
    supertile: Image.Image,
    supertile_coords_um: _XYCoords[Microns],
    tile_size_um: Microns,
    tile_size_px: TilePixels,
) -> list[_Tile[Microns]]:
    assert supertile.size[0] == supertile.size[1], "supertile must be square"
    assert supertile.size[0] % tile_size_px == 0, "supertile must divide into tiles"
    no_tiles = supertile.size[0] // tile_size_px
    out: list[_Tile[Microns]] = []
    for y in range(no_tiles):
        for x in range(no_tiles):
            tile = supertile.crop(
                (
                    x * tile_size_px,
                    y * tile_size_px,
                    (x + 1) * tile_size_px,
                    (y + 1) * tile_size_px,
                )
            )
            out.append(
                _Tile(
                    image=tile,
                    coordinates=_XYCoords(
                        x=Microns(supertile_coords_um.x + x * tile_size_um),
                        y=Microns(supertile_coords_um.y + y * tile_size_um),
                    ),
                    size=tile_size_um,
                )
            )
    return out


def _tiles(
    slide: openslide.AbstractSlide,
    *,
    tile_size_um: Microns,
    tile_size_px: TilePixels,
    max_supertile_size_slide_px: SlidePixels,
    max_workers: int,
    brightness_cutoff: int | None,
    default_slide_mpp: SlideMPP | None,
    coarse_trigger_supertile_count: int | None,
    coarse_keep_ratio: float | None,
    coarse_min_keep_supertile_count: int,
    coarse_max_keep_supertile_count: int | None,
    progress_cb: Callable[[dict[str, Any]], None] | None,
) -> Iterator[_Tile[Microns]]:
    for supertile, supertile_coords_um, supertile_size_um in _supertiles(
        slide=slide,
        tile_size_um=tile_size_um,
        tile_size_px=tile_size_px,
        max_supertile_size_slide_px=max_supertile_size_slide_px,
        max_workers=max_workers,
        brightness_cutoff=brightness_cutoff,
        default_slide_mpp=default_slide_mpp,
        coarse_trigger_supertile_count=coarse_trigger_supertile_count,
        coarse_keep_ratio=coarse_keep_ratio,
        coarse_min_keep_supertile_count=coarse_min_keep_supertile_count,
        coarse_max_keep_supertile_count=coarse_max_keep_supertile_count,
        progress_cb=progress_cb,
    ):
        yield from _split_supertile_into_tiles(
            supertile=supertile,
            supertile_coords_um=supertile_coords_um,
            tile_size_um=tile_size_um,
            tile_size_px=tile_size_px,
        )


def _foreground_grid(
    slide: openslide.AbstractSlide,
    tile_size_slide_px: SlidePixels,
    brightness_cutoff: int | None,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int32]]:
    supertile_thumb_size = np.ceil(np.array(slide.dimensions) / int(tile_size_slide_px)).astype(np.uint32)

    thumb_grayscale = np.array(
        slide.get_thumbnail(tuple((supertile_thumb_size * 2).astype(np.uint32)))
        .resize(tuple(supertile_thumb_size))
        .convert("I"),
        dtype=np.int32,
    )

    is_foreground = (
        thumb_grayscale < brightness_cutoff
        if brightness_cutoff is not None
        else cast(npt.NDArray[np.bool_], np.full_like(thumb_grayscale, True, dtype=bool))
    )

    bounds = get_slide_nonempty_bounds_level0(cast(openslide.OpenSlide, slide))
    if bounds is not None:
        x0, y0, w, h = bounds
        x_start = max(0, int(x0 // int(tile_size_slide_px)))
        y_start = max(0, int(y0 // int(tile_size_slide_px)))
        x_end = min(
            is_foreground.shape[1],
            int(np.ceil((x0 + w) / max(int(tile_size_slide_px), 1))),
        )
        y_end = min(
            is_foreground.shape[0],
            int(np.ceil((y0 + h) / max(int(tile_size_slide_px), 1))),
        )
        bounded_mask = np.zeros_like(is_foreground, dtype=bool)
        if x_end > x_start and y_end > y_start:
            bounded_mask[y_start:y_end, x_start:x_end] = True
        is_foreground = is_foreground & bounded_mask
        thumb_grayscale = np.where(bounded_mask, thumb_grayscale, 255).astype(np.int32, copy=False)

    return is_foreground, thumb_grayscale


def _foreground_coords(
    slide: openslide.AbstractSlide,
    tile_size_slide_px: SlidePixels,
    brightness_cutoff: int | None,
) -> Iterator[_XYCoords[SlidePixels]]:
    is_foreground, _ = _foreground_grid(
        slide=slide,
        tile_size_slide_px=tile_size_slide_px,
        brightness_cutoff=brightness_cutoff,
    )

    for y_slide_px in range(0, slide.dimensions[1], int(tile_size_slide_px)):
        for x_slide_px in range(0, slide.dimensions[0], int(tile_size_slide_px)):
            if is_foreground[y_slide_px // int(tile_size_slide_px), x_slide_px // int(tile_size_slide_px)]:
                yield _XYCoords(SlidePixels(x_slide_px), SlidePixels(y_slide_px))


def _mean_filter3(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    padded = np.pad(x, 1, mode="edge")
    acc = np.zeros_like(x, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            acc += padded[dy : dy + x.shape[0], dx : dx + x.shape[1]]
    return acc / 9.0


def _select_supertile_coords(
    slide: openslide.AbstractSlide,
    *,
    supertile_size_slide_px: SlidePixels,
    brightness_cutoff: int | None,
    coarse_trigger_supertile_count: int | None,
    coarse_keep_ratio: float | None,
    coarse_min_keep_supertile_count: int,
    coarse_max_keep_supertile_count: int | None,
    progress_cb: Callable[[dict[str, Any]], None] | None,
) -> list[_XYCoords[SlidePixels]]:
    is_foreground, thumb_grayscale = _foreground_grid(
        slide=slide,
        tile_size_slide_px=supertile_size_slide_px,
        brightness_cutoff=brightness_cutoff,
    )

    ys, xs = np.nonzero(is_foreground)
    total = int(ys.size)
    if total == 0:
        if progress_cb is not None:
            progress_cb(
                {
                    "phase": "coarse_prefilter",
                    "status": "done",
                    "coarse_total_supertile_count": 0,
                    "coarse_selected_supertile_count": 0,
                    "coarse_prefilter_used": False,
                }
            )
        return []

    trigger = max(0, int(coarse_trigger_supertile_count or 0))
    ratio = float(coarse_keep_ratio) if coarse_keep_ratio is not None else 1.0
    min_keep = max(1, int(coarse_min_keep_supertile_count))
    max_keep = None if coarse_max_keep_supertile_count is None or coarse_max_keep_supertile_count <= 0 else int(coarse_max_keep_supertile_count)

    use_prefilter = trigger > 0 and total >= trigger and (ratio < 1.0 or (max_keep is not None and total > max_keep))
    keep_count = total
    if use_prefilter:
        keep_count = max(min_keep, int(np.ceil(total * max(0.0, min(1.0, ratio)))))
        if max_keep is not None:
            keep_count = min(keep_count, max_keep)
        keep_count = max(1, min(total, keep_count))
        screen_slide = cast(openslide.OpenSlide, slide)
        screening_level = choose_screening_level(
            screen_slide,
            region_size_level0=int(supertile_size_slide_px),
            desired_proxy_size=64,
        )
        screening_proxy_size = same_region_proxy_size_level0(
            screen_slide,
            region_size_level0=int(supertile_size_slide_px),
            screening_level=screening_level,
        )

        screen_scores = np.full((total,), -np.inf, dtype=np.float32)
        valid_mask = np.zeros((total,), dtype=bool)
        for idx, (yy, xx) in enumerate(zip(ys, xs, strict=False)):
            x0 = int(xx * int(supertile_size_slide_px))
            y0 = int(yy * int(supertile_size_slide_px))
            rgb = np.asarray(
                slide.read_region(
                    (x0, y0),
                    screening_level,
                    (screening_proxy_size, screening_proxy_size),
                ).convert("RGB"),
                dtype=np.uint8,
            )
            metrics = screening_tile_metrics(rgb)
            if metrics.tissue_fraction < 0.03:
                continue
            screen_scores[idx] = screening_roi_score(metrics)
            valid_mask[idx] = True

        if np.any(valid_mask):
            candidate_ys = ys[valid_mask]
            candidate_xs = xs[valid_mask]
            candidate_scores = screen_scores[valid_mask]
            order = np.argsort(candidate_scores)[::-1]
        else:
            gray = thumb_grayscale.astype(np.float32, copy=False)
            fg_float = is_foreground.astype(np.float32, copy=False)
            darkness = np.clip((255.0 - gray) / 255.0, 0.0, 1.0)
            dark_core = np.clip((200.0 - gray) / 200.0, 0.0, 1.0)
            density = _mean_filter3(fg_float)
            local_contrast = np.abs(gray - _mean_filter3(gray)) / 255.0
            score_map = (
                0.56 * darkness
                + 0.20 * dark_core
                + 0.18 * density
                + 0.06 * local_contrast
            ).astype(np.float32, copy=False)
            candidate_ys = ys
            candidate_xs = xs
            candidate_scores = score_map[ys, xs]
            order = np.argsort(candidate_scores)[::-1]

        min_sep = 2 if total > keep_count * 2 else 1
        selected: list[tuple[int, int]] = []
        for idx in order:
            yy = int(candidate_ys[idx])
            xx = int(candidate_xs[idx])
            if all(max(abs(yy - sy), abs(xx - sx)) >= min_sep for sy, sx in selected):
                selected.append((yy, xx))
                if len(selected) >= keep_count:
                    break
        if len(selected) < keep_count:
            seen = set(selected)
            for idx in order:
                yy = int(candidate_ys[idx])
                xx = int(candidate_xs[idx])
                point = (yy, xx)
                if point in seen:
                    continue
                selected.append(point)
                seen.add(point)
                if len(selected) >= keep_count:
                    break
        selected.sort()
        coords = [
            _XYCoords(
                SlidePixels(int(xx * int(supertile_size_slide_px))),
                SlidePixels(int(yy * int(supertile_size_slide_px))),
            )
            for yy, xx in selected
        ]
    else:
        coords = [
            _XYCoords(
                SlidePixels(int(x_slide_px)),
                SlidePixels(int(y_slide_px)),
            )
            for y_slide_px in range(0, slide.dimensions[1], int(supertile_size_slide_px))
            for x_slide_px in range(0, slide.dimensions[0], int(supertile_size_slide_px))
            if is_foreground[y_slide_px // int(supertile_size_slide_px), x_slide_px // int(supertile_size_slide_px)]
        ]

    if progress_cb is not None:
        progress_cb(
            {
                "phase": "coarse_prefilter",
                "status": "done",
                "coarse_total_supertile_count": total,
                "coarse_selected_supertile_count": len(coords),
                "coarse_prefilter_used": bool(use_prefilter),
            }
        )
    return coords


def _has_enough_texture(tile: Image.Image, cutoff: float) -> bool:
    """Heuristic edge score in [0, 1] based on grayscale gradients."""
    gray = np.asarray(tile.convert("L"), dtype=np.float32) / 255.0
    if gray.size == 0:
        return False
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    edge_score = float((grad_x.mean() + grad_y.mean()) / 2.0)
    return edge_score >= cutoff


def _supertiles(
    slide: openslide.AbstractSlide,
    *,
    tile_size_um: Microns,
    tile_size_px: TilePixels,
    max_supertile_size_slide_px: SlidePixels,
    max_workers: int,
    brightness_cutoff: int | None,
    default_slide_mpp: SlideMPP | None,
    coarse_trigger_supertile_count: int | None,
    coarse_keep_ratio: float | None,
    coarse_min_keep_supertile_count: int,
    coarse_max_keep_supertile_count: int | None,
    progress_cb: Callable[[dict[str, Any]], None] | None,
) -> Iterator[_Tile[Microns]]:
    slide_mpp = cast(SlideMPP, get_slide_mpp_(slide, default_mpp=default_slide_mpp))

    max_supertile_um = float(max_supertile_size_slide_px) * float(slide_mpp)
    len_of_supertile_in_tiles = max(int(max_supertile_um // float(tile_size_um)), 1)

    tile_size_slide_px = int(np.ceil(float(tile_size_um) / float(slide_mpp)))
    supertile_size_slide_px = SlidePixels(tile_size_slide_px * len_of_supertile_in_tiles)
    supertile_size_tile_px = TilePixels(int(tile_size_px) * len_of_supertile_in_tiles)
    supertile_size_um = Microns(float(supertile_size_slide_px) * float(slide_mpp))

    def _read_supertile(coords_slide_px: _XYCoords[SlidePixels]) -> _Tile[Microns]:
        x_slide_px, y_slide_px = int(coords_slide_px.x), int(coords_slide_px.y)
        image = (
            slide.read_region((x_slide_px, y_slide_px), 0, (int(supertile_size_slide_px), int(supertile_size_slide_px)))
            .resize((int(supertile_size_tile_px), int(supertile_size_tile_px)))
            .convert("RGB")
        )
        return _Tile(
            image=image,
            coordinates=_XYCoords(
                x=Microns(x_slide_px * float(slide_mpp)),
                y=Microns(y_slide_px * float(slide_mpp)),
            ),
            size=supertile_size_um,
        )

    selected_coords = _select_supertile_coords(
        slide=slide,
        supertile_size_slide_px=supertile_size_slide_px,
        brightness_cutoff=brightness_cutoff,
        coarse_trigger_supertile_count=coarse_trigger_supertile_count,
        coarse_keep_ratio=coarse_keep_ratio,
        coarse_min_keep_supertile_count=coarse_min_keep_supertile_count,
        coarse_max_keep_supertile_count=coarse_max_keep_supertile_count,
        progress_cb=progress_cb,
    )

    if max_workers > 1:
        with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            yield from executor.map(_read_supertile, selected_coords)
    else:
        for coords in selected_coords:
            yield _read_supertile(coords)


def _tiles_from_cache_file(cache_file_path: Path) -> Iterator[_Tile[Microns]]:
    with ZipFile(cache_file_path, "r") as zip_fp:
        tiler_params: _TilerParams = json.loads(zip_fp.read("tiler_params.json").decode())
        cache_tiles_ext = tiler_params.get("tile_ext", "jpg")
        pattern = re.compile(rf"tile_\(([-+]?\d*\.?\d+), ([-+]?\d*\.?\d+)\)\.{re.escape(cache_tiles_ext)}$")

        for name in zip_fp.namelist():
            match = pattern.match(name)
            if match is None:
                continue

            x_um_str, y_um_str = match.groups()
            x_um, y_um = Microns(float(x_um_str)), Microns(float(y_um_str))

            with zip_fp.open(name, "r") as tile_fp:
                image = Image.open(tile_fp).convert("RGB")
                image.load()
                yield _Tile(
                    image=image,
                    coordinates=_XYCoords(x_um, y_um),
                    size=tiler_params["tile_size_um"],
                )


def get_slide_mpp_(
    slide: openslide.AbstractSlide | Path,
    *,
    default_mpp: SlideMPP | None,
) -> SlideMPP | None:
    if isinstance(slide, Path):
        slide = openslide.open_slide(str(slide))
        close_slide = True
    else:
        close_slide = False

    try:
        if openslide.PROPERTY_NAME_MPP_X in slide.properties:
            slide_mpp = SlideMPP(float(slide.properties[openslide.PROPERTY_NAME_MPP_X]))
        elif slide_mpp := _extract_mpp_from_comments(slide):
            pass
        elif slide_mpp := _extract_mpp_from_metadata(slide):
            pass
        else:
            slide_mpp = None
    finally:
        if close_slide:
            slide.close()

    if slide_mpp is None and default_mpp is not None:
        _logger.warning("Could not infer slide MPP from metadata; using default=%s", default_mpp)
    elif slide_mpp is None and default_mpp is None:
        raise MPPExtractionError("Could not infer slide MPP from metadata and no default was provided.")

    return slide_mpp or default_mpp


def _extract_mpp_from_comments(slide: openslide.AbstractSlide) -> SlideMPP | None:
    slide_comment = slide.properties.get("openslide.comment", "")
    match = re.search(r"<PixelSizeMicrons>(.*?)</PixelSizeMicrons>", slide_comment)
    if match is None:
        return None
    return SlideMPP(float(match.group(1)))


def _extract_mpp_from_metadata(slide: openslide.AbstractSlide) -> SlideMPP | None:
    try:
        xml_text = slide.properties.get("tiff.ImageDescription")
        if not xml_text:
            return None
        doc = minidom.parseString(xml_text)
        images = doc.documentElement.getElementsByTagName("Image")
        pixels = images[0].getElementsByTagName("Pixels")
        return SlideMPP(float(pixels[0].getAttribute("PhysicalSizeX")))
    except Exception:
        _logger.exception("Failed to extract MPP from TIFF image description.")
        return None


def _transform_tile_to_tensor(image: Image.Image, transform: Any) -> torch.Tensor:
    transformed = transform(image)
    if isinstance(transformed, np.ndarray):
        transformed = torch.from_numpy(transformed)
    if not isinstance(transformed, torch.Tensor):
        raise TypeError(f"Extractor transform must return torch.Tensor or ndarray, got {type(transformed)!r}.")

    if transformed.ndim == 4 and transformed.shape[0] == 1:
        transformed = transformed[0]
    if transformed.ndim != 3:
        raise ValueError(f"Expected transformed tile to have shape [C,H,W], got {tuple(transformed.shape)}.")
    if not torch.is_floating_point(transformed):
        transformed = transformed.float()
    return transformed


def _pick_first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _pick_first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, dict):
        for key in ("features", "embeddings", "x", "logits"):
            candidate = value.get(key)
            tensor = _pick_first_tensor(candidate)
            if tensor is not None:
                return tensor
        for candidate in value.values():
            tensor = _pick_first_tensor(candidate)
            if tensor is not None:
                return tensor
        return None
    return None


def _normalize_feature_output(output: Any) -> torch.Tensor:
    tensor = _pick_first_tensor(output)
    if tensor is None:
        raise TypeError("Model output does not contain a tensor feature representation.")

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor[:, 0, :]
    elif tensor.ndim > 3:
        tensor = tensor.reshape(tensor.shape[0], -1)

    if tensor.ndim != 2:
        raise ValueError(f"Expected feature matrix with shape [B, D], got {tuple(tensor.shape)}.")
    return tensor


def extract_wsi_features_by_tiles(
    slide_path: Path | str,
    *,
    extractor: Extractor[Any],
    tile_size_um: float = 256.0,
    tile_size_px: int = 224,
    batch_size: int = 32,
    device: DeviceLikeType | None = None,
    cache_dir: Path | None = None,
    cache_tiles_ext: ImageExtension = "jpg",
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
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    use_amp: bool = True,
) -> TileFeatureMatrix:
    """Read a WSI, tile it, and extract tile-level feature vectors with the given extractor.

    Optimizations:
    - use_amp: Automatic Mixed Precision for ~1.5-2x faster GPU inference (default: True)
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if tile_size_px <= 0:
        raise ValueError("tile_size_px must be > 0.")
    if tile_size_um <= 0:
        raise ValueError("tile_size_um must be > 0.")

    slide_path = Path(slide_path)
    default_mpp = SlideMPP(default_slide_mpp) if default_slide_mpp is not None else None

    tile_iter = tiles_with_cache(
        slide_path=slide_path,
        cache_dir=cache_dir,
        cache_tiles_ext=cache_tiles_ext,
        tile_size_um=Microns(tile_size_um),
        tile_size_px=TilePixels(tile_size_px),
        max_supertile_size_slide_px=SlidePixels(max_supertile_size_slide_px),
        max_workers=max_workers,
        brightness_cutoff=brightness_cutoff,
        canny_cutoff=canny_cutoff,
        default_slide_mpp=default_mpp,
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
    )

    if device is None:
        run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        run_device = torch.device(device)

    model = extractor.model
    if hasattr(model, "to"):
        model = model.to(run_device)
    if hasattr(model, "eval"):
        model.eval()

    batch_tensors: list[torch.Tensor] = []
    batch_coords: list[tuple[float, float]] = []
    feature_chunks: list[torch.Tensor] = []
    all_coords: list[tuple[float, float]] = []
    batch_dark_scores: list[float] = []
    all_dark_scores: list[float] = []
    processed_tiles = 0
    processed_batches = 0

    if progress_cb is not None:
        progress_cb(
            {
                "phase": "extract_embeddings",
                "status": "running",
                "processed_tiles": 0,
                "processed_batches": 0,
            }
        )

    def _flush_batch() -> None:
        nonlocal processed_tiles, processed_batches
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors, dim=0).to(run_device, non_blocking=True)
        with torch.no_grad(), torch.autocast(run_device.type, enabled=use_amp and run_device.type == "cuda"):
            output = model(x)
        features = _normalize_feature_output(output).detach().cpu()
        feature_chunks.append(features)
        all_coords.extend(batch_coords)
        all_dark_scores.extend(batch_dark_scores)
        processed_tiles += int(features.shape[0])
        processed_batches += 1
        if progress_cb is not None:
            progress_cb(
                {
                    "phase": "extract_embeddings",
                    "status": "running",
                    "processed_tiles": processed_tiles,
                    "processed_batches": processed_batches,
                }
            )
        batch_tensors.clear()
        batch_coords.clear()
        batch_dark_scores.clear()

    for tile in tile_iter:
        batch_tensors.append(_transform_tile_to_tensor(tile.image, extractor.transform))
        batch_coords.append((float(tile.coordinates.x), float(tile.coordinates.y)))
        batch_dark_scores.append(score_dark_informative_roi(tile.image))
        if len(batch_tensors) >= batch_size:
            _flush_batch()
    _flush_batch()

    if feature_chunks:
        feature_matrix = torch.cat(feature_chunks, dim=0)
    else:
        feature_matrix = torch.empty((0, 0), dtype=torch.float32)

    coordinates_um = np.asarray(all_coords, dtype=np.float32).reshape(-1, 2)
    dark_roi_scores = np.asarray(all_dark_scores, dtype=np.float32).reshape(-1)
    if progress_cb is not None:
        progress_cb(
            {
                "phase": "extract_embeddings",
                "status": "done",
                "processed_tiles": int(feature_matrix.shape[0]) if feature_matrix.ndim == 2 else 0,
                "processed_batches": processed_batches,
            }
        )

    return TileFeatureMatrix(
        features=feature_matrix,
        coordinates_um=coordinates_um,
        dark_roi_scores=dark_roi_scores,
        tile_size_um=float(tile_size_um),
        tile_size_px=int(tile_size_px),
        slide_path=str(slide_path),
        extractor_id=extractor.identifier,
    )


def save_tile_features_npz(result: TileFeatureMatrix, output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=result.features.numpy(),
        coordinates_um=result.coordinates_um,
        dark_roi_scores=result.dark_roi_scores,
        tile_size_um=np.float32(result.tile_size_um),
        tile_size_px=np.int32(result.tile_size_px),
        slide_path=np.array(result.slide_path),
        extractor_id=np.array(result.extractor_id),
    )
    return output_path


__all__ = [
    "ImageExtension",
    "Microns",
    "SlidePixels",
    "TilePixels",
    "SlideMPP",
    "TileFeatureMatrix",
    "tiles_with_cache",
    "get_slide_mpp_",
    "extract_wsi_features_by_tiles",
    "save_tile_features_npz",
]
