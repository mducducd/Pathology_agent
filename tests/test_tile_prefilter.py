from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image

MODULE_PATH = Path(__file__).resolve().parents[1] / "wsi_core_pkg" / "embeddings" / "tile_prefilter.py"
SPEC = importlib.util.spec_from_file_location("tile_prefilter_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
tile_prefilter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tile_prefilter
SPEC.loader.exec_module(tile_prefilter)

select_informative_tile_indices = tile_prefilter.select_informative_tile_indices
tile_touches_tissue_edge = tile_prefilter.tile_touches_tissue_edge


def _make_tile(*, size: int = 224, tissue_x_stop: int | None = None) -> Image.Image:
    yy, xx = np.indices((size, size))
    arr = np.full((size, size, 3), 255, dtype=np.uint8)

    pattern_a = ((xx // 12) + (yy // 12)) % 2
    pattern_b = (xx // 8) % 2
    tissue = np.stack(
        [
            92 + (18 * pattern_a),
            58 + (10 * pattern_b),
            148 + (24 * pattern_a),
        ],
        axis=-1,
    ).astype(np.uint8)

    stop = size if tissue_x_stop is None else max(0, min(size, int(tissue_x_stop)))
    arr[:, :stop] = tissue[:, :stop]
    return Image.fromarray(arr, mode="RGB")


def _make_corner_wedge_tile(*, size: int = 224) -> Image.Image:
    yy, xx = np.indices((size, size))
    arr = np.asarray(_make_tile(size=size), dtype=np.uint8).copy()
    wedge = (xx + yy) < int(size * 0.42)
    arr[wedge] = 255
    return Image.fromarray(arr, mode="RGB")


def test_tile_touches_tissue_edge_detects_mixed_tissue_and_background() -> None:
    full_tissue_tile = _make_tile()
    edge_touching_tile = _make_tile(tissue_x_stop=140)
    corner_wedge_tile = _make_corner_wedge_tile()

    assert tile_touches_tissue_edge(full_tissue_tile) is False
    assert tile_touches_tissue_edge(edge_touching_tile) is True
    assert tile_touches_tissue_edge(corner_wedge_tile) is True


def test_select_informative_tile_indices_rejects_tissue_edge_tiles() -> None:
    full_tissue_tile = _make_tile()
    edge_touching_tile = _make_tile(tissue_x_stop=140)

    selection = select_informative_tile_indices(
        [full_tissue_tile, edge_touching_tile],
        keep_ratio=1.0,
        min_keep_tiles=1,
        trigger_tile_count=1,
        random_reserve_ratio=0.0,
    )

    assert selection.selected_indices == [0]
    assert selection.hard_rejected_tiles == 1


def test_select_informative_tile_indices_does_not_fallback_to_edge_tiles() -> None:
    edge_touching_tile = _make_tile(tissue_x_stop=140)

    selection = select_informative_tile_indices(
        [edge_touching_tile],
        keep_ratio=1.0,
        min_keep_tiles=1,
        trigger_tile_count=1,
        random_reserve_ratio=0.0,
    )

    assert selection.selected_indices == []
    assert selection.hard_rejected_tiles == 1
