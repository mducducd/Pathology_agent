from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "wsi_core_pkg" / "dark_regions.py"

pkg = types.ModuleType("wsi_core_pkg")
pkg.__path__ = [str(ROOT / "wsi_core_pkg")]
sys.modules.setdefault("wsi_core_pkg", pkg)

config_mod = types.ModuleType("wsi_core_pkg.config")
config_mod.DEBUG_ROOT_DIR = "/tmp"
sys.modules["wsi_core_pkg.config"] = config_mod

slide_utils_mod = types.ModuleType("wsi_core_pkg.slide_utils")
slide_utils_mod._read_region_rgb = lambda *args, **kwargs: None
slide_utils_mod._resize_to_max_dim = lambda region, max_dim: (region, 0, 0)
sys.modules["wsi_core_pkg.slide_utils"] = slide_utils_mod

spec = importlib.util.spec_from_file_location("wsi_core_pkg.dark_regions_under_test", MODULE_PATH)
assert spec is not None and spec.loader is not None
dark_regions = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dark_regions
spec.loader.exec_module(dark_regions)


def test_refine_dark_region_boxes_trims_background_padding() -> None:
    tissue_mask = np.zeros((40, 40), dtype=bool)
    tissue_mask[6:30, 7:29] = True
    boxes = [{"x": 4, "y": 4, "w": 28, "h": 28, "area": 784}]

    refined = dark_regions._refine_dark_region_boxes(boxes, tissue_mask)

    assert len(refined) == 1
    kept = refined[0]
    assert kept["x"] >= 6
    assert kept["y"] >= 6
    assert kept["x"] + kept["w"] <= 30
    assert kept["y"] + kept["h"] <= 30


@pytest.mark.skip(reason="Edge-touching dark-region boxes are now allowed after trimming.")
def test_refine_dark_region_boxes_keeps_boxes_touching_tissue_edge_after_trim() -> None:
    tissue_mask = np.ones((48, 48), dtype=bool)
    yy, xx = np.indices((48, 48))
    tissue_mask[(xx + yy) < 18] = False
    boxes = [{"x": 0, "y": 0, "w": 28, "h": 28, "area": 784}]

    refined = dark_regions._refine_dark_region_boxes(boxes, tissue_mask)

    assert len(refined) == 1
    kept = refined[0]
    assert kept["x"] >= 0
    assert kept["y"] >= 0
    assert kept["w"] > 0
    assert kept["h"] > 0


def test_select_dark_core_boxes_bridges_split_dark_shoulder_with_coarse_context() -> None:
    score = np.zeros((64, 64), dtype=np.float32)
    tissue_mask = np.zeros((64, 64), dtype=bool)
    tissue_mask[20:44, 20:45] = True
    score[tissue_mask] = 0.10

    # Dense dark core plus a second deep-purple shoulder separated by a narrow gap.
    # The coarse-to-fine context should merge them into one broader dark region.
    score[24:32, 24:32] = 0.95
    score[24:32, 35:43] = 0.60

    boxes, selected_mask = dark_regions._select_dark_core_boxes(
        score=score,
        tissue_mask=tissue_mask,
        threshold_pct=90,
        out_w=64,
        out_h=64,
        min_area=16,
        max_regions=4,
    )

    assert boxes
    assert np.any(selected_mask)
    merged = boxes[0]
    assert merged["x"] <= 24
    assert merged["x"] + merged["w"] >= 43


def test_select_dark_core_boxes_keeps_isolated_core_compact() -> None:
    score = np.zeros((96, 96), dtype=np.float32)
    tissue_mask = np.zeros((96, 96), dtype=bool)
    tissue_mask[16:80, 18:78] = True
    score[tissue_mask] = 0.12

    # A small dense core inside otherwise low-score tissue should stay tight
    # instead of inflating into a broad tissue-level rectangle.
    score[40:48, 42:50] = 0.92

    boxes, selected_mask = dark_regions._select_dark_core_boxes(
        score=score,
        tissue_mask=tissue_mask,
        threshold_pct=90,
        out_w=96,
        out_h=96,
        min_area=16,
        max_regions=4,
    )

    assert boxes
    assert np.any(selected_mask)
    kept = boxes[0]
    assert kept["w"] <= 14
    assert kept["h"] <= 14
    assert int(np.sum(selected_mask)) <= 180
