from __future__ import annotations

import importlib.util
import numpy as np
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "wsi_core_pkg" / "embeddings" / "roi_ranker.py"

pkg = types.ModuleType("wsi_core_pkg")
pkg.__path__ = [str(ROOT / "wsi_core_pkg")]
sys.modules.setdefault("wsi_core_pkg", pkg)

emb_pkg = types.ModuleType("wsi_core_pkg.embeddings")
emb_pkg.__path__ = [str(ROOT / "wsi_core_pkg" / "embeddings")]
sys.modules.setdefault("wsi_core_pkg.embeddings", emb_pkg)

extractors_pkg = types.ModuleType("wsi_core_pkg.embeddings.extractors")
extractors_pkg.__path__ = [str(ROOT / "wsi_core_pkg" / "embeddings" / "extractors")]
sys.modules.setdefault("wsi_core_pkg.embeddings.extractors", extractors_pkg)

torch_mod = types.ModuleType("torch")
torch_mod.device = object
sys.modules.setdefault("torch", torch_mod)

uni2_mod = types.ModuleType("wsi_core_pkg.embeddings.extractors.uni2")
uni2_mod.uni2 = lambda *args, **kwargs: None
sys.modules["wsi_core_pkg.embeddings.extractors.uni2"] = uni2_mod

tiling_mod = types.ModuleType("wsi_core_pkg.embeddings.tiling")
tiling_mod.SlideMPP = float
tiling_mod.extract_wsi_features_by_tiles = lambda *args, **kwargs: None
tiling_mod.get_slide_mpp_ = lambda *args, **kwargs: None
sys.modules["wsi_core_pkg.embeddings.tiling"] = tiling_mod

spec = importlib.util.spec_from_file_location("wsi_core_pkg.embeddings.roi_ranker_under_test", MODULE_PATH)
assert spec is not None and spec.loader is not None
roi_ranker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = roi_ranker
spec.loader.exec_module(roi_ranker)

UnsupervisedROIIndex = roi_ranker.UnsupervisedROIIndex
select_topk_candidates_for_view = roi_ranker.select_topk_candidates_for_view


def _make_index(
    *,
    scores: list[float],
    dark_roi_scores: list[float],
    bad_likelihood: list[float],
    bad_margin: list[float],
    bad_top1: list[float],
    good_top1: list[float],
) -> UnsupervisedROIIndex:
    num_tiles = len(scores)
    coords = np.asarray([[float(i * 400), 0.0] for i in range(num_tiles)], dtype=np.float32)
    return UnsupervisedROIIndex(
        slide_path="dummy.svs",
        extractor_id="dummy",
        tile_size_um=256.0,
        tile_size_px=224,
        tile_size_level0_px=224,
        coordinates_level0_xy=coords,
        scores=np.asarray(scores, dtype=np.float32),
        dark_roi_scores=np.asarray(dark_roi_scores, dtype=np.float32),
        num_tiles=num_tiles,
        feature_dim=2,
        bad_margin=np.asarray(bad_margin, dtype=np.float32),
        bad_likelihood=np.asarray(bad_likelihood, dtype=np.float32),
        reference_mode="good_bad_exact_knn",
        bad_neighbor_indices=np.zeros((num_tiles, 1), dtype=np.int32),
        bad_neighbor_sims=np.asarray(bad_top1, dtype=np.float32).reshape(num_tiles, 1),
        good_neighbor_indices=np.zeros((num_tiles, 1), dtype=np.int32),
        good_neighbor_sims=np.asarray(good_top1, dtype=np.float32).reshape(num_tiles, 1),
    )


def test_select_topk_candidates_for_view_filters_borderline_bad_tiles_when_good_alternatives_exist() -> None:
    index = _make_index(
        scores=[0.60, 0.45, 0.43],
        dark_roi_scores=[0.82, 0.74, 0.72],
        bad_likelihood=[0.58, 0.35, 0.37],
        bad_margin=[-0.01, 0.10, 0.08],
        bad_top1=[0.45, 0.28, 0.30],
        good_top1=[0.43, 0.56, 0.54],
    )

    candidates = select_topk_candidates_for_view(
        index=index,
        view_bbox_level0=(0, 0, 2000, 1000),
        top_k=3,
        min_center_separation_px=128,
    )

    assert [candidate["tile_index"] for candidate in candidates] == [1, 2]


def test_select_topk_candidates_for_view_marks_borderline_matches_uncertain() -> None:
    index = _make_index(
        scores=[0.55, 0.50],
        dark_roi_scores=[0.74, 0.72],
        bad_likelihood=[0.49, 0.33],
        bad_margin=[0.01, 0.12],
        bad_top1=[0.40, 0.26],
        good_top1=[0.42, 0.55],
    )

    candidates = select_topk_candidates_for_view(
        index=index,
        view_bbox_level0=(0, 0, 1200, 1000),
        top_k=2,
        min_center_separation_px=128,
    )

    by_tile = {int(candidate["tile_index"]): candidate for candidate in candidates}
    assert by_tile[0]["quality_hint"] == "uncertain"
    assert by_tile[1]["quality_hint"] == "good_like"
