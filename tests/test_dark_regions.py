from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


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


def test_refine_dark_region_boxes_rejects_boxes_touching_tissue_edge() -> None:
    tissue_mask = np.ones((48, 48), dtype=bool)
    yy, xx = np.indices((48, 48))
    tissue_mask[(xx + yy) < 18] = False
    boxes = [{"x": 0, "y": 0, "w": 28, "h": 28, "area": 784}]

    refined = dark_regions._refine_dark_region_boxes(boxes, tissue_mask)

    assert refined == []
