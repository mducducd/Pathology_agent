#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


AML_PREFIX = "aml"
NORMAL_PREFIX = "normaleskm"
AML_DECISION = "Acute leukemia"
NORMAL_DECISION = "Normal marrow"
CALL_MORE_DECISION = "Call for more diagnostics"


def infer_ground_truth(slide_name: str) -> str:
    key = slide_name.lower()
    if key.startswith(AML_PREFIX):
        return "AML"
    if key.startswith(NORMAL_PREFIX):
        return "Normal marrow"
    return "Unknown"


def normalize_prediction(final_decision: str | None) -> str:
    if final_decision == AML_DECISION:
        return "AML"
    if final_decision == NORMAL_DECISION:
        return "Normal marrow"
    if final_decision == CALL_MORE_DECISION:
        return CALL_MORE_DECISION
    return "Unknown"


def iter_slide_dirs(root: Path):
    for path in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        yield path


def load_report_json(slide_dir: Path) -> dict:
    for path in (slide_dir / "report.json", slide_dir / "artifacts" / "wsi_reports" / "report.json"):
        if path.exists():
            return json.loads(path.read_text())
    return {}


def report_tile_size_px(report: dict) -> str:
    tile_size = report.get("tile_size")
    if isinstance(tile_size, dict):
        px = tile_size.get("px")
        if px not in (None, ""):
            return str(px)
    px = report.get("tile_size_px")
    return "" if px in (None, "") else str(px)


def pct(numerator: int, denominator: int) -> str:
    return "" if denominator == 0 else f"{(100.0 * numerator / denominator):.2f}"


def build_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for slide_dir in iter_slide_dirs(root):
        slide_name = slide_dir.name
        ground_truth = infer_ground_truth(slide_name)

        summary_path = slide_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        report = load_report_json(slide_dir)

        status = summary.get("status", "missing")
        final_decision = summary.get("final_decision") or report.get("final_decision")
        predicted_label = normalize_prediction(final_decision)
        if status == "ok" and predicted_label == "Unknown":
            predicted_label = CALL_MORE_DECISION

        included_in_statistics = status != "error"
        is_correct = (
            status == "ok"
            and predicted_label in {"AML", "Normal marrow"}
            and predicted_label == ground_truth
        )

        rows.append(
            {
                "slide_name": slide_name,
                "ground_truth": ground_truth,
                "summary_status": status,
                "tile_size_px": summary.get("tile_size_px", report_tile_size_px(report)),
                "predicted_label": predicted_label,
                "included_in_statistics": included_in_statistics,
                "is_correct": is_correct,
                "error": summary.get("error", ""),
                "elapsed_sec": summary.get("elapsed_sec", ""),
            }
        )
    return rows


def build_summary(rows: list[dict]) -> list[dict]:
    rows_in_stats = [row for row in rows if row["included_in_statistics"]]

    total_all = len(rows)
    total = len(rows_in_stats)
    aml_total = sum(row["ground_truth"] == "AML" for row in rows_in_stats)
    normal_total = sum(row["ground_truth"] == "Normal marrow" for row in rows_in_stats)

    ok_runs = sum(row["summary_status"] == "ok" for row in rows)
    error_runs = sum(row["summary_status"] == "error" for row in rows)
    call_more_predictions = sum(
        row["predicted_label"] == CALL_MORE_DECISION for row in rows_in_stats
    )
    correct = sum(bool(row["is_correct"]) for row in rows_in_stats)

    tp = sum(row["ground_truth"] == "AML" and row["predicted_label"] == "AML" for row in rows_in_stats)
    tn = sum(
        row["ground_truth"] == "Normal marrow" and row["predicted_label"] == "Normal marrow"
        for row in rows_in_stats
    )
    fp = sum(
        row["ground_truth"] == "Normal marrow" and row["predicted_label"] == "AML"
        for row in rows_in_stats
    )
    fn = sum(
        row["ground_truth"] == "AML" and row["predicted_label"] == "Normal marrow"
        for row in rows_in_stats
    )

    return [
        {"metric": "total_slides_all", "value": total_all},
        {"metric": "slides_in_statistics", "value": total},
        {"metric": "aml_slides", "value": aml_total},
        {"metric": "normal_slides", "value": normal_total},
        {"metric": "ok_runs", "value": ok_runs},
        {"metric": "error_runs", "value": error_runs},
        {"metric": "call_for_more_diagnostics_predictions", "value": call_more_predictions},
        {"metric": "correct_predictions", "value": correct},
        {"metric": "accuracy_all_nonerror_slides_pct", "value": pct(correct, total)},
        {"metric": "tp_aml_pred_aml", "value": tp},
        {"metric": "tn_normal_pred_normal", "value": tn},
        {"metric": "fp_normal_pred_aml", "value": fp},
        {"metric": "fn_aml_pred_normal", "value": fn},
        {"metric": "sensitivity_recall_aml_pct", "value": pct(tp, tp + fn)},
        {"metric": "specificity_normal_pct", "value": pct(tn, tn + fp)},
        {"metric": "precision_aml_pct", "value": pct(tp, tp + fp)},
        {"metric": "npv_normal_pct", "value": pct(tn, tn + fn)},
        {
            "metric": "f1_aml_pct",
            "value": "" if (2 * tp + fp + fn) == 0 else f"{(100.0 * (2 * tp) / (2 * tp + fp + fn)):.2f}",
        },
        {
            "metric": "balanced_accuracy_pct",
            "value": (
                ""
                if (tp + fn) == 0 or (tn + fp) == 0
                else f"{((100.0 * tp / (tp + fn)) + (100.0 * tn / (tn + fp))) / 2:.2f}"
            ),
        },
    ]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _detect_common_settings(root: Path) -> dict:
    """Detect common agent/model/extractor/tile_size across all slides."""
    models = set()
    extractors = set()
    agents = set()
    tile_sizes = set()

    for slide_dir in iter_slide_dirs(root):
        summary_path = slide_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        report = load_report_json(slide_dir)

        m = summary.get("model") or report.get("model_name") or ""
        if m:
            models.add(str(m))
        ex = summary.get("extractor") or (report.get("feature_extractor") or {}).get("key") or ""
        if ex:
            extractors.add(str(ex))
        ag = summary.get("agent") or summary.get("agent_type") or ""
        if ag:
            agents.add(str(ag))
        ts = summary.get("tile_size_px") or report_tile_size_px(report) or ""
        if ts:
            tile_sizes.add(str(ts))

    result: dict = {}
    if len(models) == 1:
        result["model"] = next(iter(models))
    if len(extractors) == 1:
        result["extractor"] = next(iter(extractors))
    if len(agents) == 1:
        result["agent"] = next(iter(agents))
    if len(tile_sizes) == 1:
        result["tile_size_px"] = next(iter(tile_sizes))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Export batch-result statistics for AML vs normal marrow runs.")
    parser.add_argument(
        "--root",
        default="/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/batch_results",
        help="Batch-results root directory",
    )
    parser.add_argument(
        "--output-prefix",
        default="batch_results",
        help="Prefix for the generated CSV files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write the generated CSV files. Defaults to the batch-results root.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(root)
    summary = build_summary(rows)

    detected = _detect_common_settings(root)
    suffix_parts = []
    if detected.get("model"):
        suffix_parts.append(detected["model"].replace(" ", "_"))
    if detected.get("extractor"):
        suffix_parts.append(detected["extractor"])
    if detected.get("tile_size_px"):
        suffix_parts.append(str(detected["tile_size_px"]))

    composed_prefix = f"{args.output_prefix}_" + "_".join(suffix_parts) if suffix_parts else args.output_prefix

    slides_csv = output_dir / f"{composed_prefix}_slides.csv"
    summary_csv = output_dir / f"{composed_prefix}_summary.csv"

    write_csv(
        slides_csv,
        rows,
        [
            "slide_name",
            "ground_truth",
            "summary_status",
            "tile_size_px",
            "predicted_label",
            "included_in_statistics",
            "is_correct",
            "error",
            "elapsed_sec",
        ],
    )
    write_csv(summary_csv, summary, ["metric", "value"])

    print(slides_csv)
    print(summary_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
