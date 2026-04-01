#!/usr/bin/env python3
"""Benchmark end-to-end AML runtime on a single slide across models.

Usage:
    python evaluate/benchmark_aml_vllm_speed.py \
        --slide /mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs/AML_Box11_OT79.mrxs
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SINGLE_SLIDE = REPO_ROOT / "evaluate" / "run_single_slide.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "evaluate" / "outputs" / "aml_runtime_benchmarks"
DEFAULT_MODELS = ("GPT-OSS-120B", "GLM-4.6V-FP8")
MODEL_ALIASES = {
    "GPT-OSS": "GPT-OSS-120B",
    "GPT-OSS-120B": "GPT-OSS-120B",
    "GLM-4.6V": "GLM-4.6V-FP8",
    "GLM-4.6V-FP8": "GLM-4.6V-FP8",
}


def canonical_model_name(raw: str) -> str:
    key = raw.strip()
    return MODEL_ALIASES.get(key, key)


def run_once(
    *,
    python_bin: str,
    slide_path: Path,
    output_root: Path,
    model: str,
    extractor: str | None,
    tile_filter: str | None,
    tile_size_px: int | None,
    tile_size_um: float | None,
    batch_size: int | None,
    keep_existing: bool,
) -> dict:
    patient_name = slide_path.stem
    model_dir = output_root / model.replace("/", "_")
    patient_out = model_dir / patient_name

    if patient_out.exists() and not keep_existing:
        shutil.rmtree(patient_out)

    cmd = [
        python_bin,
        str(RUN_SINGLE_SLIDE),
        "--slide",
        str(slide_path),
        "--output-dir",
        str(model_dir),
        "--model",
        model,
        "--fresh-embedding-cache",
    ]
    if extractor is not None:
        cmd.extend(["--extractor", extractor])
    if tile_filter is not None:
        cmd.extend(["--tile-filter", tile_filter])
    if tile_size_px is not None:
        cmd.extend(["--tile-size-px", str(tile_size_px)])
    if tile_size_um is not None:
        cmd.extend(["--tile-size-um", str(tile_size_um)])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    wall_sec = time.perf_counter() - t0

    summary_path = patient_out / "summary.json"
    summary = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())

    return {
        "model": model,
        "slide": str(slide_path),
        "status_code": proc.returncode,
        "wall_sec": round(wall_sec, 1),
        "summary_elapsed_sec": summary.get("elapsed_sec"),
        "status": summary.get("status", "missing"),
        "final_decision": summary.get("final_decision", ""),
        "run_id": summary.get("run_id", ""),
        "extractor": summary.get("extractor", extractor or ""),
        "tile_filter": summary.get("tile_filter", tile_filter or ""),
        "tile_size_px": summary.get("tile_size_px", tile_size_px if tile_size_px is not None else ""),
        "tile_size_um": summary.get("tile_size_um", tile_size_um if tile_size_um is not None else ""),
        "batch_size": summary.get("batch_size", batch_size if batch_size is not None else ""),
        "output_dir": str(patient_out),
        "stdout_log": str(patient_out / "benchmark_stdout.log"),
        "stderr_log": str(patient_out / "benchmark_stderr.log"),
        "error": summary.get("error", ""),
        "_stdout": proc.stdout,
        "_stderr": proc.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark AML runtime on one slide across VLM models.")
    parser.add_argument(
        "--slide",
        default="/mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs/AML_Box11_OT79.mrxs",
        help="Path to the benchmark slide",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model names or aliases to benchmark",
    )
    parser.add_argument("--extractor", default=None, help="Override the runner default extractor")
    parser.add_argument("--tile-filter", default=None, help="Override the runner default tile filter")
    parser.add_argument("--tile-size-px", type=int, default=None, help="Override the runner default tile size px")
    parser.add_argument("--tile-size-um", type=float, default=None, help="Override the runner default tile size um")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the runner default batch size")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    slide_path = Path(args.slide).resolve()
    if not slide_path.is_file():
        raise SystemExit(f"Slide not found: {slide_path}")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = output_root / f"{slide_path.stem}_{started_at}"
    bench_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for raw_model in args.models:
        model = canonical_model_name(raw_model)
        print(f"[BENCH] Running {model} on {slide_path.name}")
        result = run_once(
            python_bin=args.python_bin,
            slide_path=slide_path,
            output_root=bench_dir,
            model=model,
            extractor=args.extractor,
            tile_filter=args.tile_filter,
            tile_size_px=args.tile_size_px,
            tile_size_um=args.tile_size_um,
            batch_size=args.batch_size,
            keep_existing=args.keep_existing,
        )

        patient_out = Path(result["output_dir"])
        patient_out.mkdir(parents=True, exist_ok=True)
        Path(result["stdout_log"]).write_text(result.pop("_stdout"))
        Path(result["stderr_log"]).write_text(result.pop("_stderr"))
        results.append(result)
        print(
            f"[BENCH] {model}: status={result['status']} "
            f"wall={result['wall_sec']}s summary_elapsed={result['summary_elapsed_sec']}"
        )

    csv_path = bench_dir / "benchmark_results.csv"
    json_path = bench_dir / "benchmark_results.json"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "slide",
                "status_code",
                "wall_sec",
                "summary_elapsed_sec",
                "status",
                "final_decision",
                "run_id",
                "extractor",
                "tile_filter",
                "tile_size_px",
                "tile_size_um",
                "batch_size",
                "output_dir",
                "stdout_log",
                "stderr_log",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    json_path.write_text(json.dumps(results, indent=2) + "\n")

    print(f"[BENCH] CSV:  {csv_path}")
    print(f"[BENCH] JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
