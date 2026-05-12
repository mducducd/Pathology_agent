import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import state
from .config import REPORT_ROOT_DIR

ReportRow = Dict[str, Any]


@dataclass
class ReportImageStore:
    images_dir: str
    run_dir: str
    copied_paths: Dict[str, str] = field(default_factory=dict)

    def copy(self, src_path: Optional[str]) -> Optional[str]:
        if not src_path or not os.path.exists(src_path):
            return None
        if src_path in self.copied_paths:
            return self.copied_paths[src_path]

        base = os.path.basename(src_path)
        dst = os.path.join(self.images_dir, base)
        i = 1
        name, ext = os.path.splitext(base)
        while os.path.exists(dst):
            dst = os.path.join(self.images_dir, f"{name}_{i}{ext}")
            i += 1

        shutil.copy2(src_path, dst)
        rel = os.path.relpath(dst, self.run_dir)
        self.copied_paths[src_path] = rel
        return rel


def _as_dict(value: Any) -> Optional[Dict[str, Any]]:
    return value if isinstance(value, dict) else None


def _text(value: Any) -> str:
    return str(value or "")


def _fmt_float(value: Any, digits: int) -> str:
    return f"{float(value):.{digits}f}"


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
    image_store: ReportImageStore,
    alt: str,
) -> None:
    if not debug_path:
        return
    rel_img = image_store.copy(debug_path)
    if rel_img:
        lines.append("")
        lines.append(f"![{alt}]({rel_img})")


def _render_view_metadata(lines: List[str], obj: ReportRow) -> None:
    _field(lines, "View level", obj.get("view_level"))
    bbox = obj.get("view_bbox_level0")
    if bbox is not None:
        x0, y0, w, h = bbox
        lines.append(f"- **BBox (level 0)**: x={x0}, y={y0}, w={w}, h={h}")
    field_w = obj.get("field_width_um")
    field_h = obj.get("field_height_um")
    if field_w is not None and field_h is not None:
        lines.append(f"- **Approx field size**: ~{field_w:.0f} × {field_h:.0f} µm")
    _field(
        lines, "Tissue fraction", obj.get("tissue_fraction"), lambda v: _fmt_float(v, 2)
    )


def _append_fenced_text(lines: List[str], value: Any) -> None:
    lines.append("```text")
    lines.append(_text(value))
    lines.append("```")
    lines.append("")


def _render_header(
    lines: List[str],
    ts: str,
    run_prompt: str,
    final_text: str,
    reasoning_content: Optional[str],
) -> None:
    reasoning_text = _text(reasoning_content) if reasoning_content is not None else ""

    lines.append(f"# WSI Agent Report ({ts})\n")

    lines.append("## Prompt\n")
    _append_fenced_text(lines, run_prompt)

    lines.append("## Final Report\n")
    lines.append(_text(final_text))
    lines.append("")

    if reasoning_text:
        lines.append("## Model Reasoning\n")
        _append_fenced_text(lines, reasoning_text)


def _is_aml_report() -> bool:
    return _text(getattr(state, "AGENT_TYPE", "")).lower() == "aml"


def _render_aml_summary(lines: List[str], rois: Sequence[ReportRow]) -> None:
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


def _render_aml_reference(lines: List[str], aml_ref: ReportRow) -> None:
    match_label = aml_ref.get("match_label")
    if match_label:
        lines.append(f"- **AML reference match**: {str(match_label).replace('_', ' ')}")
    _field(lines, "AML reference summary", aml_ref.get("summary") or None)
    _field(
        lines,
        "Retrieval score",
        aml_ref.get("retrieval_score"),
        lambda v: _fmt_float(v, 3),
    )
    _field(
        lines,
        "Nearest bad similarity",
        aml_ref.get("bad_top1_similarity"),
        lambda v: _fmt_float(v, 3),
    )
    _field(
        lines,
        "Nearest good similarity",
        aml_ref.get("good_top1_similarity"),
        lambda v: _fmt_float(v, 3),
    )

    nearest_bad = _as_dict(aml_ref.get("nearest_bad_ref"))
    if nearest_bad and nearest_bad.get("name"):
        lines.append(f"- **Nearest bad exemplar**: {nearest_bad['name']}")
    nearest_good = _as_dict(aml_ref.get("nearest_good_ref"))
    if nearest_good and nearest_good.get("name"):
        lines.append(f"- **Nearest good exemplar**: {nearest_good['name']}")


def _render_roi(
    lines: List[str],
    roi: ReportRow,
    image_store: ReportImageStore,
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

    _append_debug_image(lines, roi.get("debug_path"), image_store, f"ROI {rid}")
    lines.append("")


def _sorted_rois(rois: Sequence[ReportRow]) -> List[ReportRow]:
    return sorted(
        rois,
        key=lambda r: (-int(r.get("importance", 1)), r["roi_id"]),
    )


def _render_rois(
    lines: List[str],
    rois: Sequence[ReportRow],
    image_store: ReportImageStore,
) -> None:
    lines.append("## Regions of Interest (ROIs)\n")
    if not rois:
        lines.append("_No ROIs were kept in this run._\n")
        return

    sorted_rois = _sorted_rois(rois)
    if _is_aml_report():
        _render_aml_summary(lines, sorted_rois)
    for roi in sorted_rois:
        _render_roi(lines, roi, image_store)


def _render_step(
    lines: List[str],
    step: ReportRow,
    image_store: ReportImageStore,
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

    _append_debug_image(lines, step.get("debug_path"), image_store, f"Step {idx}")
    lines.append("")


def _render_steps(
    lines: List[str],
    steps: Sequence[ReportRow],
    image_store: ReportImageStore,
) -> None:
    lines.append("## Navigation Steps\n")
    if not steps:
        lines.append("_No navigation steps recorded._\n")
        return
    for step in steps:
        _render_step(lines, step, image_store)


def _write_text_report(path: str, run_prompt: str, final_text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Prompt\n")
        f.write(_text(run_prompt).strip() + "\n\n")
        f.write("Final Report\n")
        f.write(_text(final_text).strip() + "\n")


def _render_report_lines(
    ts: str,
    run_prompt: str,
    final_text: str,
    reasoning_content: Optional[str],
    image_store: ReportImageStore,
) -> List[str]:
    lines: List[str] = []
    _render_header(lines, ts, run_prompt, final_text, reasoning_content)
    _render_rois(lines, list(state._roi_marks or []), image_store)
    _render_steps(lines, list(state._step_log or []), image_store)
    return lines


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

    image_store = ReportImageStore(images_dir=images_dir, run_dir=run_dir)
    lines = _render_report_lines(
        ts,
        run_prompt,
        final_text,
        reasoning_content,
        image_store,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    _write_text_report(text_report_path, run_prompt, final_text)

    return report_path
