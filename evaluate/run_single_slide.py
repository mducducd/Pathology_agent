#!/usr/bin/env python3
"""Run the AML detector agent on a single MIRAX slide (headless, no GUI).

Usage:
    python evaluate/run_single_slide.py \
        --slide /path/to/patient.mrxs \
        --output-dir ./batch_outputs \
        [--model GPT-OSS-120B] \
        [--extractor uni2] \
        [--tile-filter hybrid]
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wsi_core_pkg.final_decision import ensure_report_json_final_decision
from wsi_core_pkg.slide_validation import validate_mirax_slide_package


def _make_run_id(patient_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = patient_name.replace("/", "_").replace(" ", "_")[:60]
    return f"{ts}_{safe}"


def _strip_tile_cache(root: Path) -> None:
    for cache_dir in (root / "tile_cache", root / "_tile_cache"):
        if cache_dir.exists() and cache_dir.is_dir():
            shutil.rmtree(cache_dir)


def _configure_fresh_embedding_cache(run_id: str) -> dict[str, str]:
    cache_root = REPO_ROOT / "outputs" / "_fresh_embedding_cache" / run_id
    reference_cache_dir = cache_root / "reference_hnsw"
    roi_tile_cache_dir = cache_root / "roi_tiles"
    reference_cache_dir.mkdir(parents=True, exist_ok=True)
    roi_tile_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AML_REFERENCE_CACHE_DIR"] = str(reference_cache_dir)
    os.environ["ROI_TILE_CACHE_DIR"] = str(roi_tile_cache_dir)
    return {
        "cache_root": str(cache_root),
        "reference_cache_dir": str(reference_cache_dir),
        "roi_tile_cache_dir": str(roi_tile_cache_dir),
    }


def _cleanup_fresh_embedding_cache(cache_info: dict[str, str] | None) -> None:
    if not cache_info:
        return
    cache_root = Path(cache_info["cache_root"])
    if cache_root.exists() and cache_root.is_dir():
        shutil.rmtree(cache_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AML agent on a single slide")
    parser.add_argument("--slide", required=True, help="Path to .mrxs file")
    parser.add_argument("--output-dir", required=True, help="Directory to collect results")
    parser.add_argument("--model", default="GPT-OSS-120B", help="VLM model name")
    parser.add_argument("--extractor", default="uni2", help="Feature extractor key")
    parser.add_argument("--tile-filter", default="hybrid", help="Tile prefilter method")
    parser.add_argument("--agent", default="aml", help="Agent mode (e.g. aml, wsi)")
    parser.add_argument("--tile-size-um", type=float, default=256.0)
    parser.add_argument("--tile-size-px", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--fresh-embedding-cache",
        action="store_true",
        help="Use a fresh per-run embedding/reference cache namespace and delete it after the run.",
    )
    args = parser.parse_args()

    slide_path = os.path.abspath(args.slide)
    if not os.path.exists(slide_path):
        print(f"[ERROR] Slide not found: {slide_path}", file=sys.stderr)
        return 1

    patient_name = Path(slide_path).stem
    run_id = _make_run_id(patient_name)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fresh_cache_info = _configure_fresh_embedding_cache(run_id) if args.fresh_embedding_cache else None

    print(f"[SLIDE] {slide_path}")
    print(f"[RUN]   {run_id}")
    print(f"[MODEL] {args.model}  [EXTRACTOR] {args.extractor}  [FILTER] {args.tile_filter}")

    # Import inside main so env vars (like OPENAI_API_KEY) can be set before import
    from wsi_core_pkg.runtime import run_wsi_agent_for_web

    t0 = time.time()
    try:
        validate_mirax_slide_package(slide_path)

        result = run_wsi_agent_for_web(
            slide_path=slide_path,
            prompt=None,            # uses DEFAULT_AML_PROMPT
            agent_type="aml",
            run_id=run_id,
            model_name=args.model,
            extractor_name=args.extractor,
            tile_size_um=args.tile_size_um,
            tile_size_px=args.tile_size_px,
            batch_size=args.batch_size,
            tile_prefilter_method=args.tile_filter,
        )
        elapsed = time.time() - t0

        # ── Collect outputs ──────────────────────────────────────────────
        patient_out = out_dir / patient_name
        patient_out.mkdir(parents=True, exist_ok=True)

        # Save final agent text
        (patient_out / "final_output.txt").write_text(result.get("final_output") or "")

        # Copy markdown report if it exists
        report_src = result.get("report_path")
        if report_src and os.path.isfile(report_src):
            shutil.copy2(report_src, patient_out / "report.md")

        # Save state snapshot as JSON
        state_data = result.get("state") or {}
        (patient_out / "state.json").write_text(json.dumps(state_data, indent=2, default=str))

        # Copy run artifacts (debug images, ROI snapshots) if present
        run_artifacts = Path("./outputs") / run_id
        if run_artifacts.is_dir():
            dst = patient_out / "artifacts"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(run_artifacts, dst, dirs_exist_ok=True)
            _strip_tile_cache(dst)
            _strip_tile_cache(run_artifacts)

        copied_report_json = patient_out / "artifacts" / "wsi_reports" / "report.json"
        source_report_json = Path(report_src).with_suffix(".json") if report_src else None
        if copied_report_json.is_file():
            final_decision = ensure_report_json_final_decision(copied_report_json)
        elif source_report_json and source_report_json.is_file():
            final_decision = ensure_report_json_final_decision(source_report_json)
        else:
            raise RuntimeError("Completed run did not produce report.json")

        summary = {
            "patient": patient_name,
            "slide": slide_path,
            "run_id": run_id,
            "agent": args.agent,
            "model": args.model,
            "extractor": args.extractor,
            "tile_filter": args.tile_filter,
            "tile_size_px": args.tile_size_px,
            "elapsed_sec": round(elapsed, 1),
            "status": "ok",
            "final_decision": final_decision,
            "fresh_embedding_cache": bool(args.fresh_embedding_cache),
        }
        (patient_out / "summary.json").write_text(json.dumps(summary, indent=2))

        print(f"[OK]    {patient_name}  ({elapsed:.0f}s)  -> {patient_out}")
        return 0

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"[FAIL]  {patient_name}  ({elapsed:.0f}s)  {exc}", file=sys.stderr)

        patient_out = out_dir / patient_name
        patient_out.mkdir(parents=True, exist_ok=True)
        summary = {
            "patient": patient_name,
            "slide": slide_path,
            "run_id": run_id,
            "agent": args.agent,
            "model": args.model,
            "extractor": args.extractor,
            "tile_filter": args.tile_filter,
            "tile_size_px": args.tile_size_px,
            "elapsed_sec": round(elapsed, 1),
            "status": "error",
            "error": str(exc),
            "fresh_embedding_cache": bool(args.fresh_embedding_cache),
        }
        (patient_out / "summary.json").write_text(json.dumps(summary, indent=2))
        return 1
    finally:
        _cleanup_fresh_embedding_cache(fresh_cache_info)


if __name__ == "__main__":
    sys.exit(main())
