import json
import importlib.util
from pathlib import Path


def _load_aml_auto_fallback_module():
    module_path = Path(__file__).resolve().parents[1] / "wsi_core_pkg" / "aml_auto_fallback.py"
    spec = importlib.util.spec_from_file_location("test_aml_auto_fallback_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aml_auto_fallback = _load_aml_auto_fallback_module()


def test_is_max_turns_exceeded_error_matches_expected_failure() -> None:
    exc = RuntimeError("Max turns (140) exceeded")
    assert aml_auto_fallback.is_max_turns_exceeded_error(exc, 140)
    assert not aml_auto_fallback.is_max_turns_exceeded_error(exc, 80)


def test_aml_case_output_dir_and_collection_path(tmp_path: Path) -> None:
    case_out = aml_auto_fallback.aml_case_output_dir(
        run_id="run-1",
        case_output_dir=str(tmp_path / "custom_case"),
        outputs_root_dir=str(tmp_path / "outputs"),
    )
    roi_path = aml_auto_fallback.aml_roi_collection_path_for_case(
        case_output_dir=case_out,
        roi_collection_filename="roi_collection.json",
    )
    assert case_out == (tmp_path / "custom_case").resolve()
    assert roi_path == (tmp_path / "custom_case" / "roi_collection.json").resolve()


def test_load_saved_aml_roi_collection_summary_returns_summary_for_saved_rois(tmp_path: Path) -> None:
    roi_collection_path = tmp_path / "aml_case" / "roi_collection.json"
    roi_collection_path.parent.mkdir(parents=True, exist_ok=True)
    roi_collection_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "aml_roi_collection",
                "slide_name": "case1.svs",
                "accepted_rois": [
                    {"roi_id": "1", "image_path": "images/roi_1.jpg"},
                    {"roi_id": "2", "image_path": "images/roi_2.jpg"},
                ],
            }
        )
    )

    summary = aml_auto_fallback.load_saved_aml_roi_collection_summary(roi_collection_path)

    assert summary is not None
    assert summary["accepted_roi_count"] == 2
    assert summary["slide_name"] == "case1.svs"
    assert summary["roi_collection_path"] == str(roi_collection_path.resolve())


def test_load_saved_aml_roi_collection_summary_rejects_empty_or_invalid_manifests(tmp_path: Path) -> None:
    empty_manifest = tmp_path / "aml_case" / "empty.json"
    empty_manifest.parent.mkdir(parents=True, exist_ok=True)
    empty_manifest.write_text(json.dumps({"accepted_rois": []}))
    invalid_manifest = tmp_path / "aml_case" / "broken.json"
    invalid_manifest.write_text("{")

    assert aml_auto_fallback.load_saved_aml_roi_collection_summary(empty_manifest) is None
    assert aml_auto_fallback.load_saved_aml_roi_collection_summary(invalid_manifest) is None
