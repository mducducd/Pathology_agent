#!/usr/bin/env python3
"""Prewarm the shared AML hybrid tile cache used by batch evaluation.

This script builds the same cache layout reused by:

    bash evaluate/run_batch_aml_suite.sh --use-tile-cache --resume ...

The cache is written under:

    <experiment-root>/_cache/tile_cache/<extractor>/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


_EXTRACTOR_ALIASES = {
    "uni2": "uni2",
    "uni_2": "uni2",
    "virchow2": "virchow2",
    "virchow_2": "virchow2",
    "h_optimus_1": "h_optimus_1",
    "dinobloom": "dinobloom",
    "dino_bloom": "dinobloom",
    "dinobloom_s": "dinobloom",
    "dinobloom_small": "dinobloom",
    "dinobloom_g": "dinobloom_giant",
    "dino_bloom_g": "dinobloom_giant",
    "dinobloom_giant": "dinobloom_giant",
    "dino_bloom_giant": "dinobloom_giant",
}


def _extract_cuda_device_arg(argv: list[str]) -> str | None:
    for index, arg in enumerate(argv):
        if arg == "--cuda-device" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--cuda-device="):
            return arg.split("=", 1)[1]
    return None


def _extract_cli_value(argv: list[str], flag: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def _sanitize_stem(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return safe.strip("._-") or "item"


def _canonicalize_extractor_alias(name: str | None, default: str = "reddino") -> str:
    raw = str(name or default).strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    return _EXTRACTOR_ALIASES.get(raw, raw or default)


def _shared_cache_dir(experiment_root: Path, extractor_name: str) -> Path:
    return experiment_root / "_cache" / "tile_cache" / _sanitize_stem(extractor_name)


def _shared_reference_cache_dir(experiment_root: Path, extractor_name: str) -> Path:
    return experiment_root / "_cache" / "reference_hnsw" / _sanitize_stem(extractor_name)


_EARLY_CUDA_DEVICE = _extract_cuda_device_arg(sys.argv[1:])
if _EARLY_CUDA_DEVICE not in (None, ""):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_EARLY_CUDA_DEVICE)

_EARLY_EXPERIMENT_ROOT = _extract_cli_value(sys.argv[1:], "--experiment-root")
_EARLY_EXTRACTOR = _canonicalize_extractor_alias(_extract_cli_value(sys.argv[1:], "--extractor"))
if _EARLY_EXPERIMENT_ROOT not in (None, ""):
    early_experiment_root = Path(_EARLY_EXPERIMENT_ROOT).expanduser().resolve()
    os.environ["ROI_TILE_CACHE_DIR"] = str(
        _shared_cache_dir(early_experiment_root, _EARLY_EXTRACTOR)
    )
    os.environ["AML_REFERENCE_CACHE_DIR"] = str(
        _shared_reference_cache_dir(early_experiment_root, _EARLY_EXTRACTOR)
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from wsi_core_pkg import state
    from wsi_core_pkg.embeddings import available_embedding_extractors, normalize_embedding_extractor_name
    from wsi_core_pkg.state import reset_wsi_state, set_slide_path
    from wsi_core_pkg.tools import _ensure_unsupervised_roi_index
except ModuleNotFoundError as exc:
    hint_python = REPO_ROOT / ".venv" / "bin" / "python"
    missing = exc.name or "dependency"
    raise SystemExit(
        f"Missing Python dependency '{missing}'. "
        f"Run this script with {hint_python} or activate the repo virtualenv first."
    ) from exc


DEFAULT_CSV = "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/random_100_Normal_AML_Patients.csv"
DEFAULT_SLIDES_ROOT = "/mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs"
MIRAX_EXTS = {".mrxs", ".mrsx"}


def _make_run_id(patient_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"precache_{stamp}_{_sanitize_stem(patient_name)[:60]}"


def _normalize_extractor_name(name: str | None) -> str:
    candidate = _canonicalize_extractor_alias(name)
    try:
        return normalize_embedding_extractor_name(candidate)
    except ValueError as exc:
        extractor_options = ", ".join(sorted(available_embedding_extractors()))
        raise SystemExit(
            f"Unknown extractor '{name}'. Use one of: {extractor_options}"
        ) from exc


def _iter_entries(csv_path: Path) -> list[str]:
    entries: list[str] = []
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            entry = str(row[0] or "").strip().strip('"')
            if entry:
                entries.append(entry)
    return entries


def _resolve_slide(entry: str, slides_root: Path) -> tuple[str, Path]:
    raw = str(entry or "").strip().strip('"')
    candidate = Path(raw)
    if candidate.suffix.lower() in MIRAX_EXTS:
        slide_path = candidate if candidate.is_absolute() else (slides_root / candidate)
        return slide_path.stem, slide_path.resolve()

    patient_name = candidate.name
    slide_path = (slides_root / f"{patient_name}.mrxs").resolve()
    return patient_name, slide_path


def _cleanup_temp_outputs(run_id: str) -> None:
    temp_dir = REPO_ROOT / "outputs" / run_id
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


def _close_loaded_slide() -> None:
    slide = getattr(state, "_slide", None)
    if slide is None:
        return
    try:
        slide.close()
    except Exception:
        pass
    state._slide = None


def _count_cache_zips(cache_dir: Path) -> int:
    try:
        return sum(1 for _ in cache_dir.glob("*.zip"))
    except Exception:
        return 0


def _has_existing_slide_cache(cache_dir: Path, slide_path: Path) -> bool:
    slide_stem = slide_path.stem
    try:
        return any(cache_dir.glob(f"{slide_stem}.*.zip"))
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prewarm shared AML hybrid tile cache for batch runs.",
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="CSV listing patient ids / slide paths")
    parser.add_argument("--slides-root", default=DEFAULT_SLIDES_ROOT, help="Root containing slide files")
    parser.add_argument(
        "--experiment-root",
        required=True,
        help="Experiment root whose _cache/ directory should be populated",
    )
    parser.add_argument(
        "--extractor",
        default="reddino",
        help="Extractor key, e.g. uni2, virchow2, h_optimus_1, dinobloom, dinobloom_giant (aliases like h-optimus-1, dino-bloom-s, and dino-bloom-g also work)",
    )
    parser.add_argument("--tile-filter", default="hybrid", help="Tile prefilter method")
    parser.add_argument("--agent", default="aml", help="Agent mode used for cache prep")
    parser.add_argument("--cuda-device", default=None, help="Set CUDA_VISIBLE_DEVICES, e.g. 1")
    parser.add_argument("--tile-size-um", type=float, default=256.0)
    parser.add_argument("--tile-size-px", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of slides")
    parser.add_argument(
        "--skip-existing-cache",
        action="store_true",
        help="Skip slides that already have at least one tile-cache zip in the target extractor cache dir",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    slides_root = Path(args.slides_root).resolve()
    experiment_root = Path(args.experiment_root).resolve()
    extractor_key = _normalize_extractor_name(args.extractor)
    cache_dir = _shared_cache_dir(experiment_root, extractor_key)
    reference_cache_dir = _shared_reference_cache_dir(experiment_root, extractor_key)
    manifest_path = (
        experiment_root
        / "_cache"
        / "preextract_manifests"
        / f"{_sanitize_stem(extractor_key)}.json"
    )

    experiment_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    reference_cache_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    entries = _iter_entries(csv_path)
    if args.limit > 0:
        entries = entries[: args.limit]

    print("═══════════════════════════════════════════════════════════════")
    print(" Preextract Hybrid Cache")
    print(f" CSV:             {csv_path}")
    print(f" Slides root:     {slides_root}")
    print(f" Experiment root: {experiment_root}")
    print(f" Cache dir:       {cache_dir}")
    print(f" Ref cache dir:   {reference_cache_dir}")
    print(f" Extractor:       {extractor_key}")
    print(f" Agent:           {args.agent}")
    print(f" CUDA devices:    {os.getenv('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f" Tile filter:     {args.tile_filter}")
    print(f" Tile size:       {args.tile_size_px}px / {args.tile_size_um:.1f}um")
    print(f" Batch size:      {args.batch_size}")
    print(f" Slides:          {len(entries)}")
    print("═══════════════════════════════════════════════════════════════")

    results: list[dict[str, object]] = []
    ok = 0
    failed = 0
    skipped = 0

    for idx, entry in enumerate(entries, start=1):
        patient_name, slide_path = _resolve_slide(entry, slides_root)
        print(f"[{idx}/{len(entries)}] {patient_name}")

        if not slide_path.is_file():
            print(f"  SKIP  slide not found: {slide_path}")
            skipped += 1
            results.append(
                {
                    "patient": patient_name,
                    "slide": str(slide_path),
                    "status": "skip",
                    "error": "slide not found",
                }
            )
            continue

        if args.skip_existing_cache and _has_existing_slide_cache(cache_dir, slide_path):
            print(f"  SKIP  cache already exists for {slide_path.stem}")
            skipped += 1
            results.append(
                {
                    "extractor": extractor_key,
                    "patient": patient_name,
                    "slide": str(slide_path),
                    "status": "skip",
                    "error": "cache already exists",
                }
            )
            continue

        run_id = _make_run_id(patient_name)
        started = time.time()
        os.environ.pop("ROI_DISABLE_TILE_CACHE", None)
        os.environ["ROI_TILE_CACHE_DIR"] = str(cache_dir)
        os.environ["AML_REFERENCE_CACHE_DIR"] = str(reference_cache_dir)

        try:
            set_slide_path(str(slide_path))
            reset_wsi_state(
                run_id=run_id,
                extractor_name=extractor_key,
                tile_size_um=args.tile_size_um,
                tile_size_px=args.tile_size_px,
                batch_size=args.batch_size,
                tile_prefilter_method=args.tile_filter,
            )
            state.AGENT_TYPE = str(args.agent or "aml").strip().lower()

            index = _ensure_unsupervised_roi_index()
            elapsed = round(time.time() - started, 1)
            num_tiles = int(getattr(index, "num_tiles", 0) or 0)
            feature_dim = int(getattr(index, "feature_dim", 0) or 0)
            cache_zip_count = _count_cache_zips(cache_dir)
            print(f"  OK    tiles={num_tiles} dim={feature_dim} elapsed={elapsed}s cache_zips={cache_zip_count}")
            ok += 1
            results.append(
                {
                    "extractor": extractor_key,
                    "patient": patient_name,
                    "slide": str(slide_path),
                    "status": "ok",
                    "elapsed_sec": elapsed,
                    "num_tiles": num_tiles,
                    "feature_dim": feature_dim,
                }
            )
        except Exception as exc:
            elapsed = round(time.time() - started, 1)
            cache_zip_count = _count_cache_zips(cache_dir)
            print(f"  FAIL  {type(exc).__name__}: {exc} cache_zips={cache_zip_count}")
            failed += 1
            results.append(
                {
                    "extractor": extractor_key,
                    "patient": patient_name,
                    "slide": str(slide_path),
                    "status": "error",
                    "elapsed_sec": elapsed,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            _close_loaded_slide()
            _cleanup_temp_outputs(run_id)

    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(results, indent=2) + "\n")
    except Exception as exc:
        print(f"[WARN] Could not write manifest: {manifest_path} ({type(exc).__name__}: {exc})")

    print("")
    print("═══════════════════════════════════════════════════════════════")
    print(f" COMPLETE  ok={ok} fail={failed} skip={skipped}")
    print(f" Manifest: {manifest_path}")
    print("═══════════════════════════════════════════════════════════════")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
