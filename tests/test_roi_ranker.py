from __future__ import annotations

import importlib.util
import json
import numpy as np
import os
from pathlib import Path
import sys
import tempfile
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
tiling_mod.TileFeatureMatrix = object
tiling_mod.extract_wsi_features_by_tiles = lambda *args, **kwargs: None
tiling_mod.get_slide_mpp_ = lambda *args, **kwargs: None
tiling_mod.load_tile_features_npz = lambda *args, **kwargs: None
tiling_mod.resolve_feature_cache_file_path = lambda *args, **kwargs: None
tiling_mod.save_tile_features_npz = lambda *args, **kwargs: None
sys.modules["wsi_core_pkg.embeddings.tiling"] = tiling_mod

spec = importlib.util.spec_from_file_location("wsi_core_pkg.embeddings.roi_ranker_under_test", MODULE_PATH)
assert spec is not None and spec.loader is not None
roi_ranker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = roi_ranker
spec.loader.exec_module(roi_ranker)

UnsupervisedROIIndex = roi_ranker.UnsupervisedROIIndex
select_topk_candidates_for_view = roi_ranker.select_topk_candidates_for_view


def _l2_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32, copy=False)


def _make_index(
    *,
    scores: list[float],
    dark_roi_scores: list[float],
    bad_likelihood: list[float],
    bad_margin: list[float],
    bad_top1: list[float],
    good_top1: list[float],
    reference_candidate_mask: list[bool] | None = None,
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
        reference_candidate_mask=(
            np.asarray(reference_candidate_mask, dtype=np.bool_)
            if reference_candidate_mask is not None
            else np.ones((num_tiles,), dtype=np.bool_)
        ),
    )


def test_select_topk_candidates_for_view_keeps_retrieval_order_without_quality_pruning() -> None:
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

    assert [candidate["tile_index"] for candidate in candidates] == [0, 1, 2]


def test_select_topk_candidates_for_view_marks_borderline_matches_uncertain() -> None:
    index = _make_index(
        scores=[0.55, 0.50],
        dark_roi_scores=[0.74, 0.72],
        bad_likelihood=[0.39, 0.33],
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


def test_select_topk_candidates_for_view_keeps_good_support_above_bad_like_hard_reject() -> None:
    index = _make_index(
        scores=[0.58, 0.50],
        dark_roi_scores=[0.82, 0.76],
        bad_likelihood=[0.50, 0.30],
        bad_margin=[0.07, 0.11],
        bad_top1=[0.30, 0.26],
        good_top1=[0.62, 0.57],
    )

    candidates = select_topk_candidates_for_view(
        index=index,
        view_bbox_level0=(0, 0, 1200, 1000),
        top_k=2,
        min_center_separation_px=128,
    )

    by_tile = {int(candidate["tile_index"]): candidate for candidate in candidates}
    assert 0 in by_tile
    assert by_tile[0]["quality_hint"] == "uncertain"


def test_select_topk_candidates_for_view_uses_dark_boxes_as_prior_not_hard_gate() -> None:
    index = _make_index(
        scores=[0.52, 0.78],
        dark_roi_scores=[0.58, 0.86],
        bad_likelihood=[0.22, 0.24],
        bad_margin=[0.15, 0.16],
        bad_top1=[0.18, 0.20],
        good_top1=[0.62, 0.66],
    )

    candidates = select_topk_candidates_for_view(
        index=index,
        view_bbox_level0=(0, 0, 1200, 1000),
        top_k=2,
        min_center_separation_px=128,
        focus_boxes_level0=[{"x0": 0, "y0": 0, "w": 300, "h": 300}],
    )

    assert [candidate["tile_index"] for candidate in candidates] == [1, 0]
    by_tile = {int(candidate["tile_index"]): candidate for candidate in candidates}
    assert by_tile[0]["inside_dark_region"] is True
    assert by_tile[1]["inside_dark_region"] is False
    assert all(candidate["dark_region_mode"] == "prioritized" for candidate in candidates)


def test_select_topk_candidates_for_view_uses_reference_candidate_mask() -> None:
    index = _make_index(
        scores=[0.20, 0.55, 0.95],
        dark_roi_scores=[0.95, 0.76, 0.98],
        bad_likelihood=[0.20, 0.18, 0.22],
        bad_margin=[0.15, 0.14, 0.16],
        bad_top1=[0.12, 0.10, 0.11],
        good_top1=[0.62, 0.60, 0.66],
        reference_candidate_mask=[False, True, False],
    )

    candidates = select_topk_candidates_for_view(
        index=index,
        view_bbox_level0=(0, 0, 1600, 1000),
        top_k=3,
        min_center_separation_px=128,
    )

    assert [candidate["tile_index"] for candidate in candidates] == [1]


def test_select_topk_candidates_for_view_uses_quality_prior_and_penalty_in_ranking() -> None:
    index = _make_index(
        scores=[0.75, 0.75, 0.10],
        dark_roi_scores=[0.22, 0.68, 0.20],
        bad_likelihood=[0.78, 0.20, 0.25],
        bad_margin=[-0.08, 0.14, 0.01],
        bad_top1=[0.66, 0.18, 0.30],
        good_top1=[0.30, 0.64, 0.31],
    )

    candidates = select_topk_candidates_for_view(
        index=index,
        view_bbox_level0=(0, 0, 1600, 1000),
        top_k=2,
        min_center_separation_px=128,
    )

    assert [candidate["tile_index"] for candidate in candidates] == [1, 0]
    assert candidates[0]["quality_hint"] == "good_like"
    assert candidates[1]["quality_hint"] == "bad_like"


def test_select_topk_candidates_for_view_applies_dark_floor_when_enough_candidates_remain() -> None:
    index = _make_index(
        scores=[0.95, 0.86, 0.80],
        dark_roi_scores=[0.10, 0.26, 0.24],
        bad_likelihood=[0.18, 0.20, 0.22],
        bad_margin=[0.16, 0.14, 0.12],
        bad_top1=[0.10, 0.14, 0.16],
        good_top1=[0.74, 0.68, 0.65],
    )

    candidates = select_topk_candidates_for_view(
        index=index,
        view_bbox_level0=(0, 0, 1600, 1000),
        top_k=2,
        min_center_separation_px=128,
    )

    assert [candidate["tile_index"] for candidate in candidates] == [1, 2]


def test_select_topk_candidates_for_view_skips_dark_floor_when_it_would_drop_below_topk() -> None:
    index = _make_index(
        scores=[0.91, 0.86, 0.80],
        dark_roi_scores=[0.17, 0.17, 0.28],
        bad_likelihood=[0.20, 0.22, 0.18],
        bad_margin=[0.12, 0.10, 0.16],
        bad_top1=[0.16, 0.18, 0.12],
        good_top1=[0.70, 0.66, 0.72],
    )

    candidates = select_topk_candidates_for_view(
        index=index,
        view_bbox_level0=(0, 0, 1600, 1000),
        top_k=3,
        min_center_separation_px=128,
    )

    assert [candidate["tile_index"] for candidate in candidates] == [0, 1, 2]


def test_compute_reference_knn_scores_uses_tile_to_reference_knn() -> None:
    features = _l2_rows(
        np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.1, 0.9],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    ref_features = _l2_rows(
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    ref_labels = np.asarray(["good", "good", "bad"], dtype=np.str_)

    scoring = roi_ranker._compute_reference_knn_scores(
        features_l2=features,
        ref_features_l2=ref_features,
        ref_labels=ref_labels,
        top_k=2,
        row_block_size=16,
        use_hnsw=False,
    )

    assert scoring.reference_mode == "good_bad_exact_knn"
    assert scoring.candidate_mask.tolist() == [False, False, False, False]
    assert scoring.good_neighbor_indices[0].tolist() == [0, 1]
    assert scoring.good_neighbor_indices[2].tolist() == [1, 0]
    assert scoring.bad_neighbor_indices[3].tolist() == [2]
    assert float(scoring.rank_scores[1]) > float(scoring.rank_scores[0])
    assert float(scoring.bad_likelihood[3]) > 0.90


def test_reference_embedding_cache_round_trip_uses_persistent_files() -> None:
    features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels = np.asarray(["good", "bad"], dtype=np.str_)

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_a = Path(tmpdir) / "good_a.jpg"
        ref_b = Path(tmpdir) / "bad_b.jpg"
        ref_a.write_bytes(b"a")
        ref_b.write_bytes(b"b")
        ref_paths = (str(ref_a), str(ref_b))

        old_cache_dir = roi_ranker._persistent_cache_dir
        old_mem_cache = roi_ranker._reference_embeddings_cache
        old_env = os.environ.get("AML_REFERENCE_CACHE_DIR")
        try:
            os.environ["AML_REFERENCE_CACHE_DIR"] = tmpdir
            roi_ranker._persistent_cache_dir = None
            roi_ranker._reference_embeddings_cache = None

            saved = roi_ranker._save_reference_embeddings_to_cache(
                features,
                labels,
                ref_paths,
                "DummyExtractor",
            )
            assert saved is not None

            roi_ranker._reference_embeddings_cache = None
            loaded = roi_ranker._load_reference_embeddings_from_cache(
                ref_paths,
                "DummyExtractor",
            )
            assert loaded is not None

            loaded_features, loaded_labels, loaded_paths = loaded
            assert np.array_equal(loaded_features, features)
            assert np.array_equal(loaded_labels, labels)
            assert loaded_paths == ref_paths
        finally:
            if old_env is None:
                os.environ.pop("AML_REFERENCE_CACHE_DIR", None)
            else:
                os.environ["AML_REFERENCE_CACHE_DIR"] = old_env
            roi_ranker._persistent_cache_dir = old_cache_dir
            roi_ranker._reference_embeddings_cache = old_mem_cache


def test_reference_embedding_cache_loads_legacy_named_entry_by_metadata() -> None:
    features = np.asarray([[0.25, 0.75]], dtype=np.float32)
    labels = np.asarray(["good"], dtype=np.str_)

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_a = Path(tmpdir) / "good_a.jpg"
        ref_a.write_bytes(b"a")
        ref_paths = (str(ref_a),)

        cache_prefix = Path(tmpdir) / "DummyExtractor_embed_legacyfingerprint"
        np.save(str(cache_prefix) + "_features.npy", features, allow_pickle=False)
        np.save(str(cache_prefix) + "_labels.npy", labels, allow_pickle=True)
        (Path(str(cache_prefix) + "_meta.json")).write_text(
            json.dumps(
                {
                    "extractor_id": "DummyExtractor",
                    "count": 1,
                    "dim": 2,
                    "fingerprint": "legacyfingerprint",
                    "paths": list(ref_paths),
                }
            )
        )

        old_cache_dir = roi_ranker._persistent_cache_dir
        old_mem_cache = roi_ranker._reference_embeddings_cache
        old_env = os.environ.get("AML_REFERENCE_CACHE_DIR")
        try:
            os.environ["AML_REFERENCE_CACHE_DIR"] = tmpdir
            roi_ranker._persistent_cache_dir = None
            roi_ranker._reference_embeddings_cache = None

            loaded = roi_ranker._load_reference_embeddings_from_cache(
                ref_paths,
                "DummyExtractor",
            )
            assert loaded is not None

            loaded_features, loaded_labels, loaded_paths = loaded
            assert np.array_equal(loaded_features, features)
            assert np.array_equal(loaded_labels, labels)
            assert loaded_paths == ref_paths
        finally:
            if old_env is None:
                os.environ.pop("AML_REFERENCE_CACHE_DIR", None)
            else:
                os.environ["AML_REFERENCE_CACHE_DIR"] = old_env
            roi_ranker._persistent_cache_dir = old_cache_dir
            roi_ranker._reference_embeddings_cache = old_mem_cache
