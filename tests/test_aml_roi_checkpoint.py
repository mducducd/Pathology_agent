import json
from pathlib import Path

from PIL import Image

from wsi_core_pkg import state
from wsi_core_pkg.aml_output import persist_current_aml_roi_collection


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _configure_state(tmp_path: Path) -> Path:
    slide_path = tmp_path / "case1.svs"
    slide_path.write_text("stub")
    state.RUN_ID = "run-1"
    state.SLIDE_PATH = str(slide_path)
    state.MODEL_NAME = "test-model"
    state.EXTRACTOR_NAME = "extractor-x"
    state.TILE_PREFILTER_METHOD = "quality"
    state.TILE_SIZE_PX = 224
    state.ROI_OUTPUT_SIZE_PX = 1024
    state.TARGET_ACCEPTED_ROIS = 5
    state.CASE_OUTPUT_DIR = str(tmp_path / "aml_case")
    state._last_overview_with_box_path = None
    state._last_roi_candidate_overlay_path = None
    return slide_path


def test_persist_current_aml_roi_collection_writes_checkpoint(tmp_path: Path) -> None:
    _configure_state(tmp_path)
    roi_debug = tmp_path / "debug" / "roi_source.jpg"
    _make_image(roi_debug, (255, 0, 0))
    state._roi_marks = [
        {
            "roi_id": 1,
            "label": "accepted roi",
            "view_bbox_level0": [1, 2, 3, 4],
            "field_width_um": 100.0,
            "field_height_um": 120.0,
            "tissue_fraction": 0.8,
            "effective_magnification": 40.0,
            "candidate_rank": 3,
            "requested_bbox_norm": [0, 0, 999, 999],
            "debug_path": str(roi_debug),
        }
    ]

    output_path = persist_current_aml_roi_collection()

    assert output_path is not None
    payload = json.loads(output_path.read_text())
    assert payload["accepted_roi_count"] == 1
    assert payload["accepted_rois"][0]["accepted_reason"] == "accepted roi"
    assert payload["accepted_rois"][0]["image_path"] == "images/roi_1.jpg"
    assert (tmp_path / "aml_case" / "images" / "roi_1.jpg").is_file()


def test_persist_current_aml_roi_collection_removes_discarded_roi_files(tmp_path: Path) -> None:
    _configure_state(tmp_path)
    roi1_debug = tmp_path / "debug" / "roi1.jpg"
    roi2_debug = tmp_path / "debug" / "roi2.jpg"
    _make_image(roi1_debug, (255, 0, 0))
    _make_image(roi2_debug, (0, 255, 0))

    state._roi_marks = [
        {"roi_id": 1, "label": "roi1", "debug_path": str(roi1_debug)},
        {"roi_id": 2, "label": "roi2", "debug_path": str(roi2_debug)},
    ]
    persist_current_aml_roi_collection()

    state._roi_marks = [
        {"roi_id": 1, "label": "roi1", "debug_path": str(roi1_debug)},
    ]
    output_path = persist_current_aml_roi_collection()

    assert output_path is not None
    payload = json.loads(output_path.read_text())
    assert payload["accepted_roi_count"] == 1
    assert (tmp_path / "aml_case" / "images" / "roi_1.jpg").is_file()
    assert not (tmp_path / "aml_case" / "images" / "roi_2.jpg").exists()
