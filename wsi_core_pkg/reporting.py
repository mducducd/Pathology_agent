import os
import shutil
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from . import state
from .config import REPORT_ROOT_DIR


def _copy_image_for_report(
    src_path: Optional[str],
    images_dir: str,
    run_dir: str,
    copied_map: Dict[str, str],
) -> Optional[str]:
    if not src_path or not os.path.exists(src_path):
        return None
    if src_path in copied_map:
        return copied_map[src_path]

    base = os.path.basename(src_path)
    dst = os.path.join(images_dir, base)
    i = 1
    name, ext = os.path.splitext(base)
    while os.path.exists(dst):
        dst = os.path.join(images_dir, f"{name}_{i}{ext}")
        i += 1

    shutil.copy2(src_path, dst)
    rel = os.path.relpath(dst, run_dir)
    copied_map[src_path] = rel
    return rel


def _as_dict(value: Any) -> Optional[Dict[str, Any]]:
    return value if isinstance(value, dict) else None


def _field(
    lines: List[str],
    label: str,
    value: Any,
    fmt: Callable[[Any], str] = str,
) -> None:
    if value is not None:
        lines.append(f"- **{label}**: {fmt(value)}")


def _append_debug_image(
    lines: List[str],
    debug_path: Optional[str],
    images_dir: str,
    run_dir: str,
    copied_paths: Dict[str, str],
    alt: str,
) -> None:
    if not debug_path:
        return
    rel_img = _copy_image_for_report(debug_path, images_dir, run_dir, copied_paths)
    if rel_img:
        lines.append("")
        lines.append(f"![{alt}]({rel_img})")


def _render_view_metadata(lines: List[str], obj: Dict[str, Any]) -> None:
    _field(lines, "View level", obj.get("view_level"))
    bbox = obj.get("view_bbox_level0")
    if bbox is not None:
        x0, y0, w, h = bbox
        lines.append(f"- **BBox (level 0)**: x={x0}, y={y0}, w={w}, h={h}")
    field_w = obj.get("field_width_um")
    field_h = obj.get("field_height_um")
    if field_w is not None and field_h is not None:
        lines.append(f"- **Approx field size**: ~{field_w:.0f} × {field_h:.0f} µm")
    _field(lines, "Tissue fraction", obj.get("tissue_fraction"), lambda v: f"{v:.2f}")


def _render_header(
    lines: List[str],
    ts: str,
    run_prompt: str,
    final_text: str,
    reasoning_content: Optional[str],
) -> None:
    lines.append(f"# WSI Agent Report ({ts})\n")

    lines.append("## Prompt\n")
    lines.append("```text")
    lines.append(run_prompt)
    lines.append("```")
    lines.append("")

    lines.append("## Final Report\n")
    lines.append(final_text)
    lines.append("")

    if reasoning_content:
        lines.append("## Model Reasoning\n")
        lines.append("```text")
        lines.append(reasoning_content)
        lines.append("```")
        lines.append("")


def _render_aml_summary(lines: List[str], rois: List[Dict[str, Any]]) -> None:
    if str(getattr(state, "AGENT_TYPE", "") or "").lower() != "aml":
        return

    counts: Counter = Counter()
    for roi in rois:
        ref = _as_dict(roi.get("aml_reference_evidence"))
        counts[ref.get("match_label") if ref else None] += 1

    closer_to_bad = counts.get("closer_to_bad", 0)
    closer_to_good = counts.get("closer_to_good", 0)
    uncertain = len(rois) - closer_to_bad - closer_to_good

    lines.append("### AML Retrieval Summary\n")
    lines.append(f"- **ROIs closer to bad-quality ROI examples**: {closer_to_bad}")
    lines.append(f"- **ROIs closer to good-quality ROI examples**: {closer_to_good}")
    lines.append(f"- **Uncertain ROIs**: {uncertain}")
    lines.append("")


def _render_aml_reference(lines: List[str], aml_ref: Dict[str, Any]) -> None:
    match_label = aml_ref.get("match_label")
    if match_label:
        lines.append(f"- **AML reference match**: {str(match_label).replace('_', ' ')}")
    _field(lines, "AML reference summary", aml_ref.get("summary") or None)
    _field(lines, "Retrieval score", aml_ref.get("retrieval_score"), lambda v: f"{float(v):.3f}")
    _field(lines, "Nearest bad similarity", aml_ref.get("bad_top1_similarity"), lambda v: f"{float(v):.3f}")
    _field(lines, "Nearest good similarity", aml_ref.get("good_top1_similarity"), lambda v: f"{float(v):.3f}")

    nearest_bad = _as_dict(aml_ref.get("nearest_bad_ref"))
    if nearest_bad and nearest_bad.get("name"):
        lines.append(f"- **Nearest bad exemplar**: {nearest_bad['name']}")
    nearest_good = _as_dict(aml_ref.get("nearest_good_ref"))
    if nearest_good and nearest_good.get("name"):
        lines.append(f"- **Nearest good exemplar**: {nearest_good['name']}")


def _render_roi(
    lines: List[str],
    roi: Dict[str, Any],
    images_dir: str,
    run_dir: str,
    copied_paths: Dict[str, str],
) -> None:
    rid = roi["roi_id"]
    lines.append(f"### ROI {rid}: {roi['label']}\n")
    lines.append(f"- **Importance**: {roi.get('importance', 1)}")
    _field(lines, "Note", roi.get("note") or None)
    _render_view_metadata(lines, roi)
    _field(
        lines,
        "Effective magnification (approx)",
        roi.get("effective_magnification"),
        lambda v: f"~{v:.1f}x",
    )

    aml_ref = _as_dict(roi.get("aml_reference_evidence"))
    if aml_ref:
        _render_aml_reference(lines, aml_ref)

    _append_debug_image(
        lines, roi.get("debug_path"), images_dir, run_dir, copied_paths, f"ROI {rid}"
    )
    lines.append("")


def _render_rois(
    lines: List[str],
    images_dir: str,
    run_dir: str,
    copied_paths: Dict[str, str],
) -> None:
    lines.append("## Regions of Interest (ROIs)\n")
    if not state._roi_marks:
        lines.append("_No ROIs were kept in this run._\n")
        return

    sorted_rois = sorted(
        state._roi_marks,
        key=lambda r: (-int(r.get("importance", 1)), r["roi_id"]),
    )
    _render_aml_summary(lines, sorted_rois)
    for roi in sorted_rois:
        _render_roi(lines, roi, images_dir, run_dir, copied_paths)


def _render_step(
    lines: List[str],
    step: Dict[str, Any],
    images_dir: str,
    run_dir: str,
    copied_paths: Dict[str, str],
) -> None:
    idx = step["step_index"]
    lines.append(f"### Step {idx}: `{step['tool']}`\n")
    lines.append(f"- **Reason**: {step['nav_reason'] or '(no reason provided)'}")

    _render_view_metadata(lines, step)
    _field(lines, "ROI candidate stage", step.get("roi_candidate_stage") or None)
    _field(lines, "ROI candidate pipeline", step.get("roi_candidate_pipeline") or None)
    _field(lines, "Top-K candidates in this view", step.get("roi_candidate_count"))
    _field(lines, "Candidate source", step.get("roi_candidate_source") or None)
    _field(lines, "Candidate warning", step.get("roi_candidate_warning") or None)

    index_meta = _as_dict(step.get("roi_candidate_index_meta"))
    if index_meta:
        nt = index_meta.get("num_tiles")
        fd = index_meta.get("feature_dim")
        ex = index_meta.get("extractor_id")
        if nt is not None or fd is not None or ex:
            lines.append(
                f"- **Candidate index meta**: extractor={ex}, tiles={nt}, feature_dim={fd}"
            )

    dims = step.get("view_image_dims")
    if dims is not None:
        lines.append(f"- **View image size**: {dims[0]}×{dims[1]} px")

    _append_debug_image(
        lines, step.get("debug_path"), images_dir, run_dir, copied_paths, f"Step {idx}"
    )
    lines.append("")


def _render_steps(
    lines: List[str],
    images_dir: str,
    run_dir: str,
    copied_paths: Dict[str, str],
) -> None:
    lines.append("## Navigation Steps\n")
    if not state._step_log:
        lines.append("_No navigation steps recorded._\n")
        return
    for step in state._step_log:
        _render_step(lines, step, images_dir, run_dir, copied_paths)


def _write_text_report(path: str, run_prompt: str, final_text: str) -> None:
    with open(path, "w") as f:
        f.write("Prompt\n")
        f.write(run_prompt.strip() + "\n\n")
        f.write("Final Report\n")
        f.write(final_text.strip() + "\n")


def write_markdown_report(
    run_prompt: str,
    final_text: str,
    run_id: Optional[str] = None,
    reasoning_content: Optional[str] = None,
) -> str:
    ts = run_id if run_id is not None else datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(REPORT_ROOT_DIR, ts, "wsi_reports")
    images_dir = os.path.join(run_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    report_path = os.path.join(run_dir, "report.md")
    text_report_path = os.path.join(run_dir, "report.txt")

    copied_paths: Dict[str, str] = {}
    lines: List[str] = []

    _render_header(lines, ts, run_prompt, final_text, reasoning_content)
    _render_rois(lines, images_dir, run_dir, copied_paths)
    _render_steps(lines, images_dir, run_dir, copied_paths)

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    _write_text_report(text_report_path, run_prompt, final_text)

    return report_path
