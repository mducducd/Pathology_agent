import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

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


def write_markdown_report(
    run_prompt: str,
    final_text: str,
    run_id: Optional[str] = None,
    reasoning_content: Optional[str] = None,
) -> str:
    if run_id is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        ts = run_id

    run_dir = os.path.join(REPORT_ROOT_DIR, ts, "wsi_reports")
    os.makedirs(run_dir, exist_ok=True)

    images_dir = os.path.join(run_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    report_path = os.path.join(run_dir, "report.md")
    text_report_path = os.path.join(run_dir, "report.txt")

    copied_paths: Dict[str, str] = {}
    lines: List[str] = []

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

    lines.append("## Regions of Interest (ROIs)\n")
    if not state._roi_marks:
        lines.append("_No ROIs were kept in this run._\n")
    else:
        sorted_rois = sorted(
            state._roi_marks,
            key=lambda r: (-int(r.get("importance", 1)), r["roi_id"]),
        )
        if str(getattr(state, "AGENT_TYPE", "") or "").lower() == "aml":
            closer_to_bad = sum(
                1
                for roi in sorted_rois
                if isinstance(roi.get("aml_reference_evidence"), dict)
                and roi["aml_reference_evidence"].get("match_label") == "closer_to_bad"
            )
            closer_to_good = sum(
                1
                for roi in sorted_rois
                if isinstance(roi.get("aml_reference_evidence"), dict)
                and roi["aml_reference_evidence"].get("match_label") == "closer_to_good"
            )
            uncertain = len(sorted_rois) - closer_to_bad - closer_to_good
            lines.append("### AML Retrieval Summary\n")
            lines.append(f"- **ROIs closer to bad-quality ROI examples**: {closer_to_bad}")
            lines.append(f"- **ROIs closer to good-quality ROI examples**: {closer_to_good}")
            lines.append(f"- **Uncertain ROIs**: {uncertain}")
            lines.append("")
        for roi in sorted_rois:
            rid = roi["roi_id"]
            label = roi["label"]
            note = roi.get("note", "")
            importance = roi.get("importance", 1)
            bbox0 = roi.get("view_bbox_level0")
            level = roi.get("view_level")
            eff_mag = roi.get("effective_magnification")
            field_w = roi.get("field_width_um")
            field_h = roi.get("field_height_um")
            tf = roi.get("tissue_fraction")
            debug_path = roi.get("debug_path")
            aml_ref = roi.get("aml_reference_evidence") if isinstance(roi.get("aml_reference_evidence"), dict) else None

            lines.append(f"### ROI {rid}: {label}\n")
            lines.append(f"- **Importance**: {importance}")
            if note:
                lines.append(f"- **Note**: {note}")
            if level is not None:
                lines.append(f"- **View level**: {level}")
            if bbox0 is not None:
                x0, y0, w, h = bbox0
                lines.append(f"- **BBox (level 0)**: x={x0}, y={y0}, w={w}, h={h}")
            if field_w is not None and field_h is not None:
                lines.append(
                    f"- **Approx field size**: ~{field_w:.0f} × {field_h:.0f} µm"
                )
            if tf is not None:
                lines.append(f"- **Tissue fraction**: {tf:.2f}")
            if eff_mag is not None:
                lines.append(f"- **Effective magnification (approx)**: ~{eff_mag:.1f}x")
            if aml_ref:
                match_label = aml_ref.get("match_label")
                summary = aml_ref.get("summary")
                bad_top1 = aml_ref.get("bad_top1_similarity")
                good_top1 = aml_ref.get("good_top1_similarity")
                retrieval_score = aml_ref.get("retrieval_score")
                nearest_bad = aml_ref.get("nearest_bad_ref") if isinstance(aml_ref.get("nearest_bad_ref"), dict) else None
                nearest_good = aml_ref.get("nearest_good_ref") if isinstance(aml_ref.get("nearest_good_ref"), dict) else None
                if match_label:
                    lines.append(f"- **AML reference match**: {str(match_label).replace('_', ' ')}")
                if summary:
                    lines.append(f"- **AML reference summary**: {summary}")
                if retrieval_score is not None:
                    lines.append(f"- **Retrieval score**: {float(retrieval_score):.3f}")
                if bad_top1 is not None:
                    lines.append(f"- **Nearest bad similarity**: {float(bad_top1):.3f}")
                if good_top1 is not None:
                    lines.append(f"- **Nearest good similarity**: {float(good_top1):.3f}")
                if nearest_bad and nearest_bad.get("name"):
                    lines.append(f"- **Nearest bad exemplar**: {nearest_bad['name']}")
                if nearest_good and nearest_good.get("name"):
                    lines.append(f"- **Nearest good exemplar**: {nearest_good['name']}")

            if debug_path:
                rel_img = _copy_image_for_report(
                    debug_path, images_dir, run_dir, copied_paths
                )
                if rel_img:
                    lines.append("")
                    lines.append(f"![ROI {rid}]({rel_img})")
            lines.append("")

    lines.append("## Navigation Steps\n")
    if not state._step_log:
        lines.append("_No navigation steps recorded._\n")
    else:
        for step in state._step_log:
            idx = step["step_index"]
            tool = step["tool"]
            nav_reason = step["nav_reason"] or "(no reason provided)"
            bbox = step.get("view_bbox_level0")
            view_level = step.get("view_level")
            dims = step.get("view_image_dims")
            debug_path = step.get("debug_path")
            field_w = step.get("field_width_um")
            field_h = step.get("field_height_um")
            tf = step.get("tissue_fraction")
            roi_candidate_count = step.get("roi_candidate_count")
            roi_candidate_source = step.get("roi_candidate_source")
            roi_candidate_warning = step.get("roi_candidate_warning")
            roi_candidate_stage = step.get("roi_candidate_stage")
            roi_candidate_pipeline = step.get("roi_candidate_pipeline")
            roi_candidate_index_meta = step.get("roi_candidate_index_meta")

            lines.append(f"### Step {idx}: `{tool}`\n")
            lines.append(f"- **Reason**: {nav_reason}")
            if view_level is not None:
                lines.append(f"- **View level**: {view_level}")
            if bbox is not None:
                x0, y0, w, h = bbox
                lines.append(f"- **BBox (level 0)**: x={x0}, y={y0}, w={w}, h={h}")
            if field_w is not None and field_h is not None:
                lines.append(
                    f"- **Approx field size**: ~{field_w:.0f} × {field_h:.0f} µm"
                )
            if tf is not None:
                lines.append(f"- **Tissue fraction**: {tf:.2f}")
            if roi_candidate_stage:
                lines.append(f"- **ROI candidate stage**: {roi_candidate_stage}")
            if roi_candidate_pipeline:
                lines.append(f"- **ROI candidate pipeline**: {roi_candidate_pipeline}")
            if roi_candidate_count is not None:
                lines.append(f"- **Top-K candidates in this view**: {roi_candidate_count}")
            if roi_candidate_source:
                lines.append(f"- **Candidate source**: {roi_candidate_source}")
            if roi_candidate_warning:
                lines.append(f"- **Candidate warning**: {roi_candidate_warning}")
            if isinstance(roi_candidate_index_meta, dict):
                nt = roi_candidate_index_meta.get("num_tiles")
                fd = roi_candidate_index_meta.get("feature_dim")
                ex = roi_candidate_index_meta.get("extractor_id")
                if nt is not None or fd is not None or ex:
                    lines.append(
                        f"- **Candidate index meta**: extractor={ex}, tiles={nt}, feature_dim={fd}"
                    )
            if dims is not None:
                lines.append(f"- **View image size**: {dims[0]}×{dims[1]} px")

            if debug_path:
                rel_img = _copy_image_for_report(
                    debug_path, images_dir, run_dir, copied_paths
                )
                if rel_img:
                    lines.append("")
                    lines.append(f"![Step {idx}]({rel_img})")
            lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    with open(text_report_path, "w") as f:
        f.write("Prompt\n")
        f.write(run_prompt.strip() + "\n\n")
        f.write("Final Report\n")
        f.write(final_text.strip() + "\n")

    return report_path
