#!/usr/bin/env python3
"""Run the AML detector agent on a single MIRAX slide (headless, no GUI).

Usage:
    python evaluate/run_single_slide.py \
        --slide /path/to/patient.mrxs \
        --output-dir ./batch_outputs \
        [--model GPT-OSS-120B] \
        [--extractor uni2] \
        [--experiment-root ./batch_outputs] \
        [--tile-filter hybrid]
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MIRAX_EXTS = {".mrxs", ".mrsx"}
FINAL_DECISIONS = (
    "Normal marrow",
    "Acute leukemia",
    "Call for more diagnostics",
)
FINAL_DECISION_LOOKUP = {label.lower(): label for label in FINAL_DECISIONS}


def _make_run_id(patient_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = patient_name.replace("/", "_").replace(" ", "_")[:60]
    return f"{ts}_{safe}"


def _strip_tile_cache(root: Path) -> None:
    for cache_dir in (root / "tile_cache", root / "_tile_cache"):
        if cache_dir.exists() and cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)


def _configure_fresh_embedding_cache(run_id: str) -> dict[str, str]:
    cache_root = REPO_ROOT / "outputs" / "_fresh_embedding_cache" / run_id
    reference_cache_dir = cache_root / "_cache" / "reference_hnsw"
    reference_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AML_REFERENCE_CACHE_DIR"] = str(reference_cache_dir)
    cache_info = {
        "cache_root": str(cache_root),
        "reference_cache_dir": str(reference_cache_dir),
    }
    return cache_info


def _cleanup_fresh_embedding_cache(cache_info: dict[str, str] | None) -> None:
    if not cache_info:
        return
    cache_root = Path(cache_info["cache_root"])
    if cache_root.exists() and cache_root.is_dir():
        shutil.rmtree(cache_root, ignore_errors=True)


def _configure_persistent_reference_cache(
    *,
    output_dir: Path,
    experiment_root: Path | None,
    extractor_name: str,
) -> Path:
    cache_root = (experiment_root or output_dir).resolve()
    reference_cache_dir = cache_root / "_cache" / "reference_hnsw" / _sanitize_stem(extractor_name)
    reference_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AML_REFERENCE_CACHE_DIR"] = str(reference_cache_dir)
    return reference_cache_dir


def _configure_experiment_tile_cache(
    *,
    output_dir: Path,
    experiment_root: Path | None,
    extractor_name: str,
    tile_filter: str,
    enable_tile_cache: bool,
) -> Path | None:
    if not enable_tile_cache:
        os.environ["ROI_DISABLE_TILE_CACHE"] = "1"
        os.environ.pop("ROI_TILE_CACHE_DIR", None)
        return None

    os.environ.pop("ROI_DISABLE_TILE_CACHE", None)
    cache_root = (experiment_root or output_dir).resolve()
    tile_cache_dir = (
        cache_root
        / "_cache"
        / "tile_cache"
        / _sanitize_stem(extractor_name)
        / _sanitize_stem(_normalize_tile_filter_name(tile_filter))
    )
    tile_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ROI_TILE_CACHE_DIR"] = str(tile_cache_dir)
    return tile_cache_dir


def _normalize_tile_filter_name(value: str | None) -> str:
    raw = str(value or "hybrid").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"coarse_to_fine", "coarse2fine"}:
        return "hybrid"
    if raw not in {"none", "coarse", "quality", "hybrid"}:
        return "hybrid"
    return raw


def _configure_experiment_feature_cache(
    *,
    output_dir: Path,
    experiment_root: Path | None,
    extractor_name: str,
    tile_filter: str,
) -> Path:
    cache_root = (experiment_root or output_dir).resolve()
    feature_cache_dir = (
        cache_root
        / "_cache"
        / "feature_cache"
        / _sanitize_stem(extractor_name)
        / _sanitize_stem(_normalize_tile_filter_name(tile_filter))
    )
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ROI_FEATURE_CACHE_DIR"] = str(feature_cache_dir)
    return feature_cache_dir


def _sanitize_stem(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return safe.strip("._-") or "image"


def _debug_image_order_key(path: Path) -> tuple[int, str]:
    match = re.match(r"(\d+)", path.stem)
    order = int(match.group(1)) if match else 10**9
    return (order, path.name)


def _iter_roi_candidate_images(state_data: dict, run_artifacts: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def _add(path_value: object) -> None:
        if not isinstance(path_value, str) or not path_value:
            return
        path = Path(path_value)
        if path.exists() and path.is_file() and path not in seen:
            seen.add(path)
            candidates.append(path)

    _add(state_data.get("last_roi_candidate_overlay_path"))
    for step in state_data.get("step_log") or []:
        debug_path = step.get("debug_path") if isinstance(step, dict) else None
        if isinstance(debug_path, str) and "roi_candidates" in Path(debug_path).name.lower():
            _add(debug_path)

    wsi_debug_dir = run_artifacts / "wsi_debug"
    if wsi_debug_dir.is_dir():
        for path in sorted(wsi_debug_dir.glob("*roi_candidates*")):
            _add(str(path))
    return candidates


def _coerce_bbox_level0(bbox: object) -> tuple[int, int, int, int] | None:
    try:
        x0, y0, w, h = bbox
        x0_i = int(round(float(x0)))
        y0_i = int(round(float(y0)))
        w_i = int(round(float(w)))
        h_i = int(round(float(h)))
    except Exception:
        return None
    if w_i <= 0 or h_i <= 0:
        return None
    return (x0_i, y0_i, w_i, h_i)


def _selected_roi_ids_for_export(state_data: dict, roi_marks: list[dict]) -> list[int]:
    roi_by_id: dict[int, dict] = {}
    roi_by_bbox: dict[tuple[int, int, int, int], int] = {}
    for roi in roi_marks:
        if not isinstance(roi, dict):
            continue
        try:
            roi_id = int(roi.get("roi_id"))
        except Exception:
            continue
        roi_by_id[roi_id] = roi
        bbox = _coerce_bbox_level0(roi.get("view_bbox_level0"))
        if bbox is not None:
            roi_by_bbox[bbox] = roi_id

    selected_roi_ids: list[int] = []
    seen_ids: set[int] = set()
    for tile in state_data.get("saved_good_tiles") or []:
        if not isinstance(tile, dict):
            continue

        roi_id: int | None
        try:
            raw_roi_id = tile.get("source_roi_id")
            roi_id = int(raw_roi_id) if raw_roi_id not in (None, "") else None
        except Exception:
            roi_id = None

        if roi_id is None:
            bbox = _coerce_bbox_level0(tile.get("current_view_bbox_level0"))
            roi_id = roi_by_bbox.get(bbox) if bbox is not None else None

        if roi_id is None or roi_id in seen_ids or roi_id not in roi_by_id:
            continue
        seen_ids.add(roi_id)
        selected_roi_ids.append(roi_id)

    return selected_roi_ids


def _copy_eval_images(patient_out: Path, state_data: dict, run_artifacts: Path) -> None:
    images_dir = patient_out / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    roi_candidate_images = _iter_roi_candidate_images(state_data, run_artifacts)
    if roi_candidate_images:
        earliest = min(roi_candidate_images, key=_debug_image_order_key)
        dst = images_dir / f"roi_candidates{earliest.suffix or '.jpg'}"
        shutil.copy2(earliest, dst)

    roi_marks = state_data.get("roi_marks") or []
    if not isinstance(roi_marks, list):
        roi_marks = []
    saved_good_tiles = state_data.get("saved_good_tiles") or []
    if not isinstance(saved_good_tiles, list):
        saved_good_tiles = []

    def _roi_sort_key(item: object) -> tuple[int, str]:
        if not isinstance(item, dict):
            return (10**9, "")
        try:
            return (int(item.get("roi_id") or 10**9), str(item.get("label") or ""))
        except Exception:
            return (10**9, str(item.get("label") or ""))

    sorted_rois = sorted((r for r in roi_marks if isinstance(r, dict)), key=_roi_sort_key)
    selected_roi_ids = _selected_roi_ids_for_export(state_data, sorted_rois)
    if selected_roi_ids:
        roi_lookup = {
            int(roi.get("roi_id")): roi
            for roi in sorted_rois
            if isinstance(roi, dict) and str(roi.get("roi_id") or "").strip()
        }
        rois_to_copy = [roi_lookup[roi_id] for roi_id in selected_roi_ids if roi_id in roi_lookup]
    else:
        # Fallback: copy all kept ROIs when the run snapshot does not carry
        # saved_good_tiles linkage or the model chose not to save key tiles.
        rois_to_copy = sorted_rois

    for index, roi in enumerate(rois_to_copy, start=1):
        debug_path = roi.get("debug_path")
        if not isinstance(debug_path, str):
            continue
        src = Path(debug_path)
        if not src.exists() or not src.is_file():
            continue
        roi_id = roi.get("roi_id") or index
        dst_name = f"roi_{_sanitize_stem(str(roi_id))}{src.suffix or '.jpg'}"
        shutil.copy2(src, images_dir / dst_name)


def _write_summary(patient_out: Path, **kwargs) -> None:
    (patient_out / "summary.json").write_text(json.dumps(kwargs, indent=2))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def _normalize_final_decision(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return FINAL_DECISION_LOOKUP.get(text.lower(), text)


def _extract_final_decision(final_output: str) -> str:
    text = str(final_output or "").strip()
    if not text:
        return ""

    label_group = "|".join(re.escape(label) for label in FINAL_DECISIONS)

    # Priority 1: Explicit "Final decision:" or "Final diagnosis:" statement (most reliable)
    # These are the most common and reliable formats
    final_decision_pattern = rf"final\s+(?:decision|diagnosis)\s*[:\-]\s*({label_group})"
    match = re.search(final_decision_pattern, text, flags=re.IGNORECASE)
    if match:
        return _normalize_final_decision(match.group(1))

    # Priority 2: "Diagnosis:" or "Decision:" without "Final" prefix
    diagnosis_pattern = rf"(?:diagnosis|decision)\s*[:\-]\s*({label_group})"
    match = re.search(diagnosis_pattern, text, flags=re.IGNORECASE)
    if match:
        return _normalize_final_decision(match.group(1))

    # Priority 3: "Consistent with X" or "Findings consistent with X" patterns
    consistent_pattern = rf"(?:findings\s+)?(?:are\s+)?(?:most\s+)?(?:consistent\s+(?:with|:)\s+)({label_group})"
    match = re.search(consistent_pattern, text, flags=re.IGNORECASE)
    if match:
        return _normalize_final_decision(match.group(1))

    # Priority 4: "Conclusion:" statements
    conclusion_pattern = rf"conclusion\s*[:\-]\s*({label_group})"
    match = re.search(conclusion_pattern, text, flags=re.IGNORECASE)
    if match:
        return _normalize_final_decision(match.group(1))

    # Priority 5: "X is the diagnosis/decision/conclusion" patterns
    is_diagnosis_pattern = rf"({label_group})\s+(?:is\s+)?(?:the\s+)?(?:final\s+)?(?:diagnosis|decision|conclusion)"
    match = re.search(is_diagnosis_pattern, text, flags=re.IGNORECASE)
    if match:
        return _normalize_final_decision(match.group(1))

    # Priority 6: "I conclude/diagnose/decide X" patterns
    conclude_pattern = rf"(?:I\s+(?:conclude|diagnose|decide|determine)\s+(?:that\s+)?(?:this\s+)?(?:is\s+)?(?:a\s+)?)(?:case\s+of\s+)?({label_group})"
    match = re.search(conclude_pattern, text, flags=re.IGNORECASE)
    if match:
        return _normalize_final_decision(match.group(1))

    # Priority 7: "Therefore/Thus/Hence X" patterns (sentence-ending conclusions)
    therefore_pattern = rf"(?:therefore|thus|hence),?\s+(?:this\s+)?(?:is\s+)?(?:a\s+)?(?:case\s+of\s+)?({label_group})"
    match = re.search(therefore_pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        return _normalize_final_decision(match.group(1))

    # Last resort: Find the last mentioned label
    # This is a fallback when no explicit decision statement format is detected
    # Only labels appearing in explicit decision-like contexts should be considered
    seen_labels = [
        (text.lower().rfind(label.lower()), label)
        for label in FINAL_DECISIONS
        if text.lower().rfind(label.lower()) >= 0
    ]
    if seen_labels:
        _, label = max(seen_labels, key=lambda item: item[0])
        return label

    return ""


def _copy_file_if_exists(src: Path | None, dst: Path) -> None:
    if src is None or not src.is_file():
        return
    shutil.copy2(src, dst)


def _persist_report_artifacts(
    *,
    patient_out: Path,
    report_src: object,
    state_data: dict,
    final_output: str,
    final_decision: str,
    patient_name: str,
    slide_path: str,
    run_id: str,
    args: argparse.Namespace,
    elapsed: float,
) -> None:
    report_src_path = Path(report_src) if isinstance(report_src, str) and report_src else None
    if report_src_path is not None:
        _copy_file_if_exists(report_src_path, patient_out / "report.md")
        _copy_file_if_exists(report_src_path.with_suffix(".txt"), patient_out / "report.txt")

    _write_json(patient_out / "state.json", state_data)
    _write_json(
        patient_out / "report.json",
        {
            "patient": patient_name,
            "slide": slide_path,
            "run_id": run_id,
            "agent_type": args.agent,
            "model_name": args.model,
            "feature_extractor": {"key": args.extractor},
            "tile_filter": args.tile_filter,
            "tile_size": {"px": args.tile_size_px, "um": args.tile_size_um},
            "tile_size_px": args.tile_size_px,
            "tile_size_um": args.tile_size_um,
            "batch_size": args.batch_size,
            "elapsed_sec": round(elapsed, 1),
            "final_decision": final_decision,
            "final_output": final_output,
            "roi_count": len(state_data.get("roi_marks") or []),
            "step_count": len(state_data.get("step_log") or []),
            "source_report_path": str(report_src_path) if report_src_path else "",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def _validate_slide_package(slide_path: str) -> None:
    slide = Path(slide_path).resolve()
    if not slide.is_file():
        raise FileNotFoundError(f"Slide not found: {slide}")

    if slide.suffix.lower() in MIRAX_EXTS:
        companion_dir = slide.parent / slide.stem
        if not companion_dir.is_dir():
            raise RuntimeError(
                "MIRAX slide detected (.mrxs/.mrsx) but companion data directory was not found "
                f"next to {slide.name}."
            )

    try:
        import openslide
    except Exception as exc:
        raise RuntimeError(f"OpenSlide import failed: {type(exc).__name__}: {exc}") from exc

    try:
        handle = openslide.open_slide(str(slide))
        try:
            _ = handle.level_count
            _ = handle.level_dimensions
        finally:
            handle.close()
    except openslide.lowlevel.OpenSlideUnsupportedFormatError as exc:
        raise RuntimeError(f"Unsupported or missing image file: {slide.name}") from exc
    except openslide.OpenSlideError as exc:
        raise RuntimeError(f"Failed to open slide '{slide.name}': {exc}") from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open slide '{slide.name}': {type(exc).__name__}: {exc}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AML agent on a single slide")
    parser.add_argument("--slide", required=True, help="Path to .mrxs file")
    parser.add_argument("--output-dir", required=True, help="Directory to collect results")
    parser.add_argument(
        "--experiment-root",
        default=None,
        help="Experiment root used for shared tile/reference cache placement across repeat runs",
    )
    parser.add_argument(
        "--cuda-device",
        default=None,
        help="Set CUDA_VISIBLE_DEVICES for this run, e.g. 1",
    )
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
        help="Use a fresh per-run reference cache namespace and delete it after the run.",
    )
    parser.add_argument(
        "--use-tile-cache",
        action="store_true",
        help="Allow on-disk ROI tile caching during evaluation. Disabled by default to save space.",
    )
    args = parser.parse_args()
    args.tile_filter = _normalize_tile_filter_name(args.tile_filter)

    slide_path = os.path.abspath(args.slide)
    if not os.path.exists(slide_path):
        print(f"[ERROR] Slide not found: {slide_path}", file=sys.stderr)
        return 1

    patient_name = Path(slide_path).stem
    run_id = _make_run_id(patient_name)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment_root = Path(args.experiment_root).resolve() if args.experiment_root else None

    if args.cuda_device not in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    tile_cache_dir = _configure_experiment_tile_cache(
        output_dir=out_dir,
        experiment_root=experiment_root,
        extractor_name=args.extractor,
        tile_filter=args.tile_filter,
        enable_tile_cache=args.use_tile_cache,
    )
    feature_cache_dir = _configure_experiment_feature_cache(
        output_dir=out_dir,
        experiment_root=experiment_root,
        extractor_name=args.extractor,
        tile_filter=args.tile_filter,
    )

    fresh_cache_info = (
        _configure_fresh_embedding_cache(run_id)
        if args.fresh_embedding_cache
        else None
    )
    reference_cache_dir = None
    if fresh_cache_info is None:
        reference_cache_dir = _configure_persistent_reference_cache(
            output_dir=out_dir,
            experiment_root=experiment_root,
            extractor_name=args.extractor,
        )
    run_artifacts = REPO_ROOT / "outputs" / run_id

    print(f"[SLIDE] {slide_path}")
    print(f"[RUN]   {run_id}")
    print(f"[MODEL] {args.model}  [EXTRACTOR] {args.extractor}  [FILTER] {args.tile_filter}")
    if reference_cache_dir is not None:
        print(f"[REFCACHE] {reference_cache_dir}")
    print(f"[FEATCACHE] {feature_cache_dir}")
    if os.getenv("CUDA_VISIBLE_DEVICES"):
        print(f"[CUDA]  CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")

    from wsi_core_pkg.runtime import run_wsi_agent_for_web

    t0 = time.time()
    try:
        _validate_slide_package(slide_path)

        result = run_wsi_agent_for_web(
            slide_path=slide_path,
            prompt=None,
            agent_type=args.agent,
            run_id=run_id,
            model_name=args.model,
            extractor_name=args.extractor,
            tile_size_um=args.tile_size_um,
            tile_size_px=args.tile_size_px,
            batch_size=args.batch_size,
            tile_prefilter_method=args.tile_filter,
        )
        elapsed = time.time() - t0

        patient_out = out_dir / patient_name
        patient_out.mkdir(parents=True, exist_ok=True)

        for legacy_dir in ("artifacts", "images"):
            dst = patient_out / legacy_dir
            if dst.exists() and dst.is_dir():
                shutil.rmtree(dst)
        for legacy_file in ("report.json", "report.md", "state.json"):
            path = patient_out / legacy_file
            if path.exists():
                path.unlink()

        final_output = result.get("final_output") or ""
        (patient_out / "final_output.txt").write_text(final_output)
        report_src = result.get("report_path")
        state_data = result.get("state") or {}
        final_decision = _extract_final_decision(final_output)
        if not final_decision:
            raise RuntimeError("Completed run did not include a parseable final decision")

        if run_artifacts.is_dir():
            _strip_tile_cache(run_artifacts)
        _copy_eval_images(patient_out, state_data, run_artifacts)
        _persist_report_artifacts(
            patient_out=patient_out,
            report_src=report_src,
            state_data=state_data,
            final_output=final_output,
            final_decision=final_decision,
            patient_name=patient_name,
            slide_path=slide_path,
            run_id=run_id,
            args=args,
            elapsed=elapsed,
        )

        _write_summary(
            patient_out,
            patient=patient_name,
            slide=slide_path,
            run_id=run_id,
            agent=args.agent,
            model=args.model,
            extractor=args.extractor,
            tile_filter=args.tile_filter,
            tile_size_px=args.tile_size_px,
            tile_size_um=args.tile_size_um,
            batch_size=args.batch_size,
            elapsed_sec=round(elapsed, 1),
            status="ok",
            final_decision=final_decision,
            fresh_embedding_cache=bool(args.fresh_embedding_cache),
            tile_cache_enabled=bool(args.use_tile_cache),
            tile_cache_dir=str(tile_cache_dir) if tile_cache_dir else "",
            feature_cache_dir=str(feature_cache_dir),
        )

        print(f"[OK]    {patient_name}  ({elapsed:.0f}s)  -> {patient_out}")
        return 0

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"[FAIL]  {patient_name}  ({elapsed:.0f}s)  {exc}", file=sys.stderr)

        patient_out = out_dir / patient_name
        patient_out.mkdir(parents=True, exist_ok=True)
        _write_summary(
            patient_out,
            patient=patient_name,
            slide=slide_path,
            run_id=run_id,
            agent=args.agent,
            model=args.model,
            extractor=args.extractor,
            tile_filter=args.tile_filter,
            tile_size_px=args.tile_size_px,
            tile_size_um=args.tile_size_um,
            batch_size=args.batch_size,
            elapsed_sec=round(elapsed, 1),
            status="error",
            error=str(exc),
            fresh_embedding_cache=bool(args.fresh_embedding_cache),
            tile_cache_enabled=bool(args.use_tile_cache),
            tile_cache_dir=str(tile_cache_dir) if tile_cache_dir else "",
            feature_cache_dir=str(feature_cache_dir),
        )
        return 1
    finally:
        if run_artifacts.exists() and run_artifacts.is_dir():
            shutil.rmtree(run_artifacts, ignore_errors=True)
        _cleanup_fresh_embedding_cache(fresh_cache_info)


if __name__ == "__main__":
    sys.exit(main())
