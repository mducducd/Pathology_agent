import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


def is_max_turns_exceeded_error(exc: Exception, max_turns: int) -> bool:
    text = str(exc or "")
    lowered = text.lower()
    expected = f"max turns ({int(max_turns)}) exceeded"
    if expected in lowered or "max turns exceeded" in lowered:
        return True
    match = re.search(r"max turns\s*\((\d+)\)\s*exceeded", lowered)
    if match:
        return int(match.group(1)) == int(max_turns)
    return "max turns" in lowered and "exceeded" in lowered and "(" not in lowered


def aml_case_output_dir(
    *,
    run_id: str,
    case_output_dir: Optional[str],
    outputs_root_dir: str,
) -> Path:
    case_out = Path(case_output_dir) if case_output_dir else (Path(outputs_root_dir) / run_id / "aml_case")
    return case_out.expanduser().resolve()


def aml_roi_collection_path_for_case(
    *,
    case_output_dir: Path,
    roi_collection_filename: str,
) -> Path:
    return case_output_dir / roi_collection_filename


def load_saved_aml_roi_collection_summary(roi_collection_path: Path) -> Optional[Dict[str, Any]]:
    if not roi_collection_path.is_file():
        return None
    try:
        collection = json.loads(roi_collection_path.read_text())
    except Exception:
        return None
    accepted_rois = collection.get("accepted_rois", [])
    if not isinstance(accepted_rois, list) or not accepted_rois:
        return None
    return {
        "roi_collection_path": str(roi_collection_path.resolve()),
        "slide_name": str(collection.get("slide_name") or collection.get("case_id") or ""),
        "accepted_roi_count": len(accepted_rois),
        "collection": collection,
    }
