from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "wsi_core_pkg" / "embeddings" / "__init__.py"

pkg = types.ModuleType("wsi_core_pkg")
pkg.__path__ = [str(ROOT / "wsi_core_pkg")]
sys.modules.setdefault("wsi_core_pkg", pkg)

emb_pkg = types.ModuleType("wsi_core_pkg.embeddings")
emb_pkg.__path__ = [str(ROOT / "wsi_core_pkg" / "embeddings")]
sys.modules["wsi_core_pkg.embeddings"] = emb_pkg

extractors_pkg = types.ModuleType("wsi_core_pkg.embeddings.extractors")
extractors_pkg.__path__ = [str(ROOT / "wsi_core_pkg" / "embeddings" / "extractors")]
sys.modules["wsi_core_pkg.embeddings.extractors"] = extractors_pkg


def _make_builder(name: str):
    def _builder() -> str:
        return name

    _builder.__name__ = name
    return _builder


dinobloom_mod = types.ModuleType("wsi_core_pkg.embeddings.extractors.dinobloom")
dinobloom_mod.dino_bloom = _make_builder("dino_bloom")
dinobloom_mod.dinobloom = _make_builder("dinobloom")
dinobloom_mod.dinobloom_base = _make_builder("dinobloom_base")
dinobloom_mod.dinobloom_large = _make_builder("dinobloom_large")
dinobloom_mod.dinobloom_giant = _make_builder("dinobloom_giant")
sys.modules["wsi_core_pkg.embeddings.extractors.dinobloom"] = dinobloom_mod

reddino_mod = types.ModuleType("wsi_core_pkg.embeddings.extractors.reddino")
reddino_mod.red_dino = _make_builder("red_dino")
reddino_mod.reddino = _make_builder("reddino")
reddino_mod.reddino_base = _make_builder("reddino_base")
reddino_mod.reddino_large = _make_builder("reddino_large")
sys.modules["wsi_core_pkg.embeddings.extractors.reddino"] = reddino_mod

uni2_mod = types.ModuleType("wsi_core_pkg.embeddings.extractors.uni2")
uni2_mod.uni2 = _make_builder("uni2")
sys.modules["wsi_core_pkg.embeddings.extractors.uni2"] = uni2_mod

virchow2_mod = types.ModuleType("wsi_core_pkg.embeddings.extractors.virchow2")
virchow2_mod.virchow2 = _make_builder("virchow2")
sys.modules["wsi_core_pkg.embeddings.extractors.virchow2"] = virchow2_mod

h_optimus_1_mod = types.ModuleType("wsi_core_pkg.embeddings.extractors.h_optimus_1")
h_optimus_1_mod.h_optimus_1 = _make_builder("h_optimus_1")
sys.modules["wsi_core_pkg.embeddings.extractors.h_optimus_1"] = h_optimus_1_mod

index_mod = types.ModuleType("wsi_core_pkg.embeddings.index_tiles_hnsw")
index_mod.embed_tiles_to_hnsw = object()
sys.modules["wsi_core_pkg.embeddings.index_tiles_hnsw"] = index_mod

tiling_mod = types.ModuleType("wsi_core_pkg.embeddings.tiling")
tiling_mod.TileFeatureMatrix = object
tiling_mod.extract_wsi_features_by_tiles = object()
tiling_mod.save_tile_features_npz = object()
sys.modules["wsi_core_pkg.embeddings.tiling"] = tiling_mod

spec = importlib.util.spec_from_file_location("wsi_core_pkg.embeddings", MODULE_PATH)
assert spec is not None and spec.loader is not None
embeddings = importlib.util.module_from_spec(spec)
sys.modules["wsi_core_pkg.embeddings"] = embeddings
spec.loader.exec_module(embeddings)


def test_available_embedding_extractors_include_all_dinobloom_variants() -> None:
    extractors = embeddings.available_embedding_extractors()

    assert "dinobloom" in extractors
    assert "dinobloom_base" in extractors
    assert "dinobloom_large" in extractors
    assert "dinobloom_giant" in extractors
    assert "virchow2" in extractors
    assert "h_optimus_1" in extractors


def test_display_names_cover_all_dinobloom_variants() -> None:
    assert embeddings.embedding_extractor_display_name("dinobloom") == "DinoBloom-S"
    assert embeddings.embedding_extractor_display_name("dinobloom_base") == "DinoBloom-B"
    assert embeddings.embedding_extractor_display_name("dinobloom_large") == "DinoBloom-L"
    assert embeddings.embedding_extractor_display_name("dinobloom_giant") == "DinoBloom-G"
    assert embeddings.embedding_extractor_display_name("virchow2") == "Virchow2"
    assert embeddings.embedding_extractor_display_name("h_optimus_1") == "H-optimus-1"


def test_get_embedding_extractor_dispatches_new_dinobloom_variants() -> None:
    assert embeddings.get_embedding_extractor("dinobloom_base") == "dinobloom_base"
    assert embeddings.get_embedding_extractor("dinobloom_large") == "dinobloom_large"
    assert embeddings.get_embedding_extractor("dinobloom_giant") == "dinobloom_giant"
    assert embeddings.get_embedding_extractor("virchow2") == "virchow2"
    assert embeddings.get_embedding_extractor("h_optimus_1") == "h_optimus_1"
