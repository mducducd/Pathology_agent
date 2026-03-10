import os
import uuid
import shutil
import zipfile
import traceback
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import openslide
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from wsi_core_pkg.embeddings import (
    extract_wsi_features_by_tiles,
    save_tile_features_npz,
    uni2,
)
from wsi_core import (
    run_wsi_agent_for_web,
    clear_wsi_outputs_state,
    get_public_state_snapshot,
    OUTPUTS_ROOT_DIR,
    DEBUG_ROOT_DIR,
    REPORT_ROOT_DIR,
    detect_dark_regions,
    MODEL_NAME,
)

# Primary slide types OpenSlide can open directly, plus MIRAX zip bundle
ALLOWED_SLIDE_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".mrsx", ".zip"}
SUPPORTED_PRIMARY_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".mrsx"}
MIRAX_EXTS = {".mrxs", ".mrsx"}
STD_EXTS = {".svs", ".tif", ".tiff", ".ndpi"}
ALLOWED_MODEL_NAMES = {
    "GLM-4.6V-FP8",
    "GPT-OSS-120B",
    "qwen3.5-35b-a3b",
    "Qwen3.5-397B-A17B-FP8",
}
ALLOWED_EMBEDDING_EXTRACTORS = {"uni2"}

app = FastAPI(title="WSI Agent Prototype")

OUTPUTS_ROOT = Path("./outputs")
OUTPUTS_ROOT.mkdir(exist_ok=True)

BASE_RUN_DIR = OUTPUTS_ROOT

STATIC_DIR = Path("./static").resolve()
if not STATIC_DIR.exists():
    raise RuntimeError(f"Missing static dir at {STATIC_DIR}. Create ./static with index.html/app.js/styles.css")


class RunStatus(BaseModel):
    run_id: str
    status: str               # created | uploading | pending | running | done | error | terminated
    created_at: datetime
    agent_type: str
    model_name: str
    prompt: Optional[str]
    slide_filename: str       # filled after finalize
    slide_path: Optional[str] = None
    final_output: Optional[str] = None
    reasoning_content: Optional[str] = None
    report_path: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None

    # upload bookkeeping
    upload_count: int = 0
    upload_bytes: int = 0
    uploaded_files: List[str] = []


RUNS: Dict[str, RunStatus] = {}
RUN_TERMINATE_FLAGS: Dict[str, threading.Event] = {}
RUN_THREADS: Dict[str, threading.Thread] = {}
_EMBEDDING_EXTRACTOR_CACHE: Dict[str, object] = {}
_EMBEDDING_EXTRACTOR_LOCK = threading.Lock()

# --- Auto-cleanup config ---
DELETE_UPLOADS_AFTER_RUN = os.getenv("DELETE_UPLOADS_AFTER_RUN", "1").strip().lower() in {"1", "true", "yes", "y"}


def _safe_cleanup_run_upload_dir(run_id: str) -> None:
    """
    Deletes ./outputs/<run_id>/uploads (uploaded slide data + extracted bundles, etc.)
    Does NOT touch report/debug/tile folders.
    """
    try:
        run_dir = (BASE_RUN_DIR / run_id / "uploads").resolve()
        base_dir = BASE_RUN_DIR.resolve()

        # Safety: ensure run_dir is inside BASE_RUN_DIR and exists
        if not str(run_dir).startswith(str(base_dir) + os.sep) and run_dir != base_dir:
            print(f"[CLEANUP] Refusing to delete outside base dir: {run_dir}")
            return

        if run_dir.exists() and run_dir.is_dir():
            shutil.rmtree(run_dir, ignore_errors=False)
            print(f"[CLEANUP] Deleted upload dir for run={run_id}: {run_dir}")
    except Exception as exc:
        # Never fail the run because cleanup failed
        print(f"[CLEANUP] Failed to delete upload dir for run={run_id}: {exc}")


def _safe_cleanup_run_generated_outputs(run_id: str) -> None:
    """
    Delete generated output artifacts for a run, keeping uploads untouched.
    """
    try:
        run_dir = (BASE_RUN_DIR / run_id).resolve()
        base_dir = BASE_RUN_DIR.resolve()

        if not str(run_dir).startswith(str(base_dir) + os.sep):
            print(f"[CLEANUP] Refusing to delete outside base dir: {run_dir}")
            return
        if not run_dir.exists() or not run_dir.is_dir():
            return

        for sub in ("wsi_reports", "wsi_debug", "dark", "traces", "Selected_Tiles", "embeddings", "tile_cache"):
            p = run_dir / sub
            if p.exists() and p.is_dir():
                shutil.rmtree(p, ignore_errors=False)
                print(f"[CLEANUP] Deleted generated dir for run={run_id}: {p}")
    except Exception as exc:
        print(f"[CLEANUP] Failed to delete generated outputs for run={run_id}: {exc}")


def _is_safe_relpath(p: Path) -> bool:
    if p.is_absolute():
        return False
    if ".." in p.parts:
        return False
    return True


def _get_embedding_extractor(extractor_name: str):
    key = (extractor_name or "").strip().lower()
    if key not in ALLOWED_EMBEDDING_EXTRACTORS:
        raise HTTPException(
            status_code=400,
            detail=f"extractor_name must be one of: {', '.join(sorted(ALLOWED_EMBEDDING_EXTRACTORS))}",
        )

    with _EMBEDDING_EXTRACTOR_LOCK:
        cached = _EMBEDDING_EXTRACTOR_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            if key == "uni2":
                extractor = uni2()
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported extractor: {extractor_name}")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load embedding extractor '{extractor_name}': {type(exc).__name__}: {exc}",
            ) from exc

        _EMBEDDING_EXTRACTOR_CACHE[key] = extractor
        return extractor


async def _save_uploadfile_to_path(up: UploadFile, out_path: Path, chunk_size: int = 8 * 1024 * 1024) -> int:
    """
    Stream UploadFile to disk. Returns bytes written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("wb") as f:
        while True:
            chunk = await up.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
    return written


def _safe_extract_zip(zip_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if not _is_safe_relpath(member_path):
                raise HTTPException(status_code=400, detail=f"Unsafe zip entry: {member.filename}")
            out_path = (extract_to / member.filename).resolve()
            if not str(out_path).startswith(str(extract_to.resolve())):
                raise HTTPException(status_code=400, detail=f"Unsafe zip entry: {member.filename}")
        zf.extractall(extract_to)


def _find_mirax_candidates(root: Path) -> List[Path]:
    cands = list(root.rglob("*.mrxs")) + list(root.rglob("*.mrsx"))
    return [p for p in cands if p.is_file()]


def _score_mirax_candidate(p: Path) -> Tuple[int, int, int]:
    """
    Higher is better:
      1) has sibling directory (likely data dir)
      2) shallower path
      3) shorter string (tie-break)
    """
    try:
        sibling_dirs = [d for d in p.parent.iterdir() if d.is_dir()]
        has_sibling_dir = 1 if sibling_dirs else 0
    except Exception:
        has_sibling_dir = 0
    depth = len(p.parts)
    return (has_sibling_dir, -depth, -len(str(p)))


def _find_best_mirax_file(root: Path) -> Path:
    cands = _find_mirax_candidates(root)
    if not cands:
        raise HTTPException(
            status_code=400,
            detail=(
                "No .mrxs/.mrsx file found. For MIRAX: upload a .zip containing the .mrxs/.mrsx "
                "and its data folder, or upload the MIRAX folder (Chrome/Edge), or upload the "
                ".mrxs/.mrsx together with all companion files from its folder."
            ),
        )
    cands.sort(key=_score_mirax_candidate, reverse=True)
    return cands[0]


def _maybe_relocate_mirax_into_single_top_dir(run_dir: Path, mirax_path: Path) -> Path:
    """
    If .mrxs landed at run_dir root but the folder content is under a single top-level dir,
    move .mrxs into that dir.
    """
    try:
        if mirax_path.parent.resolve() != run_dir.resolve():
            return mirax_path
        top_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name not in {"extracted"}]
        if len(top_dirs) != 1:
            return mirax_path
        target_dir = top_dirs[0]
        target_path = target_dir / mirax_path.name
        if target_path.exists():
            return mirax_path
        shutil.move(str(mirax_path), str(target_path))
        return target_path
    except Exception:
        return mirax_path

def _maybe_lift_mirax_out_of_same_stem_dir(mirax_path: Path) -> Path:
    """
    If the .mrxs/.mrsx is inside a folder with the same stem, move it one level up.

    Example (your case):
      run_dir/AML_Box1_OT02/AML_Box1_OT02.mrxs  ->  run_dir/AML_Box1_OT02.mrxs
      run_dir/AML_Box1_OT02/ (data dir stays)
    """
    try:
        parent_dir = mirax_path.parent
        if parent_dir.name != mirax_path.stem:
            return mirax_path

        target = parent_dir.parent / mirax_path.name
        if target.exists():
            return mirax_path  # don't clobber

        shutil.move(str(mirax_path), str(target))
        return target
    except Exception:
        return mirax_path



def _select_primary_slide(files: List[Path]) -> Path:
    # Prefer MIRAX file if present
    mirax = [p for p in files if p.suffix.lower() in MIRAX_EXTS]
    if mirax:
        mirax.sort(key=_score_mirax_candidate, reverse=True)
        return mirax[0]

    # Else prefer standard slide
    std = [p for p in files if p.suffix.lower() in STD_EXTS]
    if std:
        return std[0]

    raise HTTPException(status_code=400, detail="No supported slide file found.")


def _list_all_files(run_dir: Path) -> List[Path]:
    return [p for p in run_dir.rglob("*") if p.is_file()]


def _validate_final_bundle(run_dir: Path) -> Tuple[Path, str]:
    """
    After all per-file uploads, validate what we have and return (primary_slide_path, slide_filename).

    Supports:
      - Standard slide: exactly one .svs/.tif/.ndpi file (and no other files)
      - MIRAX folder/files: contains .mrxs/.mrsx + sibling directory with data
      - MIRAX zip: contains single .zip; we extract and pick .mrxs/.mrsx within
    """
    all_files = _list_all_files(run_dir)
    if not all_files:
        raise HTTPException(status_code=400, detail="No uploaded files found for this run.")

    # If there is a zip at root (or anywhere) and it's the only uploaded file: treat as bundle
    zips = [p for p in all_files if p.suffix.lower() == ".zip"]
    if zips and len(all_files) == 1:
        zip_path = zips[0]
        extract_dir = run_dir / "extracted"
        _safe_extract_zip(zip_path, extract_dir)
        mirax_path = _find_best_mirax_file(extract_dir)
        return mirax_path, mirax_path.name

    # Otherwise: gather primary candidates
    primaries = [p for p in all_files if p.suffix.lower() in SUPPORTED_PRIMARY_EXTS]
    if not primaries:
        raise HTTPException(
            status_code=400,
            detail="Upload contained no supported primary slide file (.svs/.tif/.ndpi/.mrxs/.mrsx) and was not a single zip.",
        )

    # Standard slide must be single file total
    std = [p for p in primaries if p.suffix.lower() in STD_EXTS]
    mir = [p for p in primaries if p.suffix.lower() in MIRAX_EXTS]
    if std and mir:
        raise HTTPException(status_code=400, detail="Do not mix standard slides and MIRAX in one run.")

    if std:
        if len(std) != 1 or len(all_files) != 1:
            raise HTTPException(status_code=400, detail="Standard slides must be uploaded as a single file only.")
        return std[0], std[0].name

    # MIRAX
    mir.sort(key=_score_mirax_candidate, reverse=True)
    m = mir[0]

    # Case 1: user uploaded .mrxs at run root, but folder content lives under a single top dir
    m = _maybe_relocate_mirax_into_single_top_dir(run_dir, m)

    # Case 2 (your case): user uploaded a folder that contains both the .mrxs and Data*.dat
    # inside the same folder; move the .mrxs one level up so it becomes a sibling of that folder.
    m = _maybe_lift_mirax_out_of_same_stem_dir(m)

    # Now enforce canonical MIRAX layout: sibling dir with same stem
    companion_dir = m.parent / m.stem
    if not companion_dir.is_dir():
        raise HTTPException(
            status_code=400,
            detail=(
                "MIRAX slide detected (.mrxs/.mrsx) but companion data directory not found. "
                "Expected a folder named like the slide (same stem) next to the .mrxs/.mrsx."
            ),
        )

    return m, m.name


def _assert_slide_openable(slide_path: Path) -> None:
    """
    Quick sanity check before starting the background run.
    """
    slide_path = slide_path.resolve()
    if not slide_path.exists():
        raise HTTPException(status_code=400, detail=f"Slide file not found: {slide_path}")

    try:
        # Use open_slide() (not OpenSlide()) so single-resolution TIFFs can fall back to ImageSlide.
        slide = openslide.open_slide(str(slide_path))
        try:
            _ = slide.level_count
            _ = slide.level_dimensions
        finally:
            slide.close()
    except openslide.lowlevel.OpenSlideUnsupportedFormatError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported or missing image file: {slide_path.name}",
        ) from exc
    except openslide.OpenSlideError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to open slide '{slide_path.name}': {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to open slide '{slide_path.name}': {type(exc).__name__}: {exc}",
        ) from exc



def run_worker(
    run_id: str,
    slide_path: str,
    prompt: Optional[str],
    agent_type: str,
    model_name: str,
    terminate_event: threading.Event,
) -> None:
    run = RUNS.get(run_id)
    if run is None:
        return
    if terminate_event.is_set() or run.status == "terminated":
        run.status = "terminated"
        if not run.error_message:
            run.error_message = "Run terminated by user."
        return
    run.status = "running"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        result = run_wsi_agent_for_web(
            slide_path=slide_path,
            prompt=prompt,
            agent_type=agent_type,
            run_id=run_id,
            model_name=model_name,
        )
        fatal_error: Optional[str] = None
        if isinstance(result, dict):
            st = result.get("state")
            if isinstance(st, dict) and st.get("has_fatal_error"):
                fatal_error = str(st.get("last_fatal_error") or "Unsupported or missing image file")
            final_output = result.get("final_output")
            if not fatal_error and isinstance(final_output, str):
                final_output_lc = final_output.lower()
                if (
                    "unsupported or missing image file" in final_output_lc
                    or "openslideunsupportedformaterror" in final_output_lc
                ):
                    fatal_error = "Unsupported or missing image file"
        if fatal_error:
            raise RuntimeError(fatal_error)
        if terminate_event.is_set() or run.status == "terminated":
            run.status = "terminated"
            if not run.error_message:
                run.error_message = "Run terminated by user."
        else:
            run.status = "done"
            run.final_output = result["final_output"]
            run.reasoning_content = result.get("reasoning_content")
            run.report_path = result.get("report_path")

    except Exception as exc:
        if terminate_event.is_set() or run.status == "terminated":
            run.status = "terminated"
            if not run.error_message:
                run.error_message = "Run terminated by user."
        else:
            run.status = "error"
            run.error_message = str(exc)
            run.traceback = traceback.format_exc()
            print("=== RUN ERROR ===")
            print(run.error_message)
            print(run.traceback)

    finally:
        RUN_THREADS.pop(run_id, None)
        try:
            loop.close()
        except Exception:
            pass

        # Delete uploaded slide data after the run finishes (success or error)
        if DELETE_UPLOADS_AFTER_RUN:
            _safe_cleanup_run_upload_dir(run_id)



def make_debug_image_url(abs_path: Optional[str]) -> Optional[str]:
    if not abs_path:
        return None
    if not os.path.exists(abs_path):
        return None
    debug_root = os.path.abspath(DEBUG_ROOT_DIR)
    abs_path_norm = os.path.abspath(abs_path)
    if not abs_path_norm.startswith(debug_root):
        return None
    rel = os.path.relpath(abs_path_norm, debug_root).replace("\\", "/")
    return f"/debug/{rel}"


# -----------------------------
# NEW API: create -> upload -> finalize
# -----------------------------

@app.post("/api/runs/create")
async def create_run(
    prompt: str = Form(""),
    agent_type: str = Form("wsi"),
    model_name: str = Form(MODEL_NAME),
):
    agent_type_lower = agent_type.lower()
    if agent_type_lower not in {"tile", "wsi", "aml"}:
        raise HTTPException(status_code=400, detail="agent_type must be 'tile', 'wsi', or 'aml'")
    if model_name not in ALLOWED_MODEL_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"model_name must be one of: {', '.join(sorted(ALLOWED_MODEL_NAMES))}",
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    run_dir = BASE_RUN_DIR / run_id / "uploads"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_status = RunStatus(
        run_id=run_id,
        status="created",
        created_at=datetime.utcnow(),
        agent_type=agent_type_lower,
        model_name=model_name,
        prompt=prompt or None,
        slide_filename="(upload pending)",
        slide_path=None,
        upload_count=0,
        upload_bytes=0,
        uploaded_files=[],
    )
    RUNS[run_id] = run_status
    RUN_TERMINATE_FLAGS[run_id] = threading.Event()
    return {"run_id": run_id, "model_name": model_name}


@app.post("/api/runs/{run_id}/upload")
async def upload_one_file(
    run_id: str,
    file: UploadFile = File(...),
    relpath: str = Form(""),
):
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status in {"pending", "running", "done", "terminated"}:
        raise HTTPException(status_code=400, detail=f"Run is already {run.status}; uploads are closed.")
    if run.status == "error":
        raise HTTPException(status_code=400, detail="Run is in error state; create a new run.")

    run_dir = BASE_RUN_DIR / run_id / "uploads"
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_name = relpath.strip() or (file.filename or "upload")
    rel = Path(raw_name)
    if not _is_safe_relpath(rel):
        raise HTTPException(status_code=400, detail=f"Unsafe filename/path: {raw_name}")

    ext = rel.suffix.lower()
    # Allow companion files with arbitrary extensions (MIRAX data). But block obviously weird absolute/parent paths above.
    if ext in ALLOWED_SLIDE_EXTS or ext == "" or True:
        pass

    out_path = (run_dir / rel)
    written = await _save_uploadfile_to_path(file, out_path)

    # Update bookkeeping
    run.status = "uploading"
    run.upload_count += 1
    run.upload_bytes += int(written)
    run.uploaded_files.append(raw_name)

    print(f"[UPLOAD] run={run_id} saved {raw_name} ({written} bytes) -> {out_path}")
    return {"ok": True, "saved_as": raw_name, "bytes": written}


@app.post("/api/runs/{run_id}/finalize")
async def finalize_and_start(run_id: str):
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status in {"pending", "running", "done", "terminated"}:
        raise HTTPException(status_code=400, detail=f"Run is already {run.status}.")
    if run.status == "error":
        raise HTTPException(status_code=400, detail="Run is in error state; create a new run.")

    run_dir = BASE_RUN_DIR / run_id / "uploads"
    if not run_dir.exists():
        raise HTTPException(status_code=400, detail="Run directory missing; nothing to finalize.")

    slide_path, slide_filename = _validate_final_bundle(run_dir)
    try:
        _assert_slide_openable(slide_path)
    except HTTPException as exc:
        run.status = "error"
        run.error_message = str(exc.detail)
        run.traceback = None
        raise

    slide_path = slide_path.resolve()
    run.slide_filename = slide_filename
    run.slide_path = str(slide_path)
    run.status = "pending"
    terminate_event = RUN_TERMINATE_FLAGS.setdefault(run_id, threading.Event())
    terminate_event.clear()

    thread = threading.Thread(
        target=run_worker,
        args=(run_id, str(slide_path), run.prompt or None, run.agent_type, run.model_name, terminate_event),
        daemon=True,
    )
    RUN_THREADS[run_id] = thread
    thread.start()

    print(f"[FINALIZE] run={run_id} primary={slide_path}")
    return {"ok": True, "run_id": run_id, "primary": slide_filename}


@app.post("/api/runs/{run_id}/terminate")
async def terminate_run(run_id: str):
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status in {"done", "error", "terminated"}:
        return {"ok": True, "run_id": run_id, "status": run.status}

    terminate_event = RUN_TERMINATE_FLAGS.setdefault(run_id, threading.Event())
    terminate_event.set()
    run.status = "terminated"
    run.error_message = "Run terminated by user."
    run.traceback = None
    return {"ok": True, "run_id": run_id, "status": run.status}


@app.post("/api/runs/{run_id}/clear_outputs")
async def clear_run_outputs(run_id: str):
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status != "done":
        raise HTTPException(status_code=400, detail="Clear outputs is only available when run is done.")

    RUN_THREADS.pop(run_id, None)
    _safe_cleanup_run_generated_outputs(run_id)
    clear_wsi_outputs_state()

    run.final_output = None
    run.reasoning_content = None
    run.report_path = None
    run.error_message = None
    run.traceback = None

    return {"ok": True, "run_id": run_id, "status": run.status}


# -----------------------------
# Run polling endpoint (unchanged)
# -----------------------------

@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    try:
        wsi_state = get_public_state_snapshot()
    except Exception:
        wsi_state = None

    if wsi_state and wsi_state.get("current_view"):
        debug_path = wsi_state["current_view"].get("debug_path")
        wsi_state["current_view"]["image_url"] = make_debug_image_url(debug_path)

    if wsi_state and wsi_state.get("roi_marks"):
        for roi in wsi_state["roi_marks"]:
            dp = roi.get("debug_path")
            roi["image_url"] = make_debug_image_url(dp)

    if wsi_state and wsi_state.get("last_overview_with_box_path"):
        overview_path = wsi_state["last_overview_with_box_path"]
        wsi_state["overview_image_url"] = make_debug_image_url(overview_path)
    elif wsi_state is not None:
        wsi_state["overview_image_url"] = None

    if wsi_state and wsi_state.get("last_roi_candidate_overlay_path"):
        candidate_path = wsi_state["last_roi_candidate_overlay_path"]
        wsi_state["roi_candidates_image_url"] = make_debug_image_url(candidate_path)
    elif wsi_state is not None:
        wsi_state["roi_candidates_image_url"] = None

    return {"run": run, "wsi_state": wsi_state}


@app.get("/api/runs/{run_id}/dark_regions")
def get_dark_regions(
    run_id: str,
    max_regions: int = 30,
    threshold_pct: int = 85,
    min_area: int = 800,
    max_dim: int = 1024,
):
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if not run.slide_path:
        raise HTTPException(status_code=400, detail="Slide path not available for this run.")
    if not os.path.exists(run.slide_path):
        raise HTTPException(status_code=400, detail="Slide file not found (maybe cleaned up).")

    result = detect_dark_regions(
        slide_path=run.slide_path,
        run_id=run_id,
        max_dim=max_dim,
        threshold_pct=threshold_pct,
        min_area=min_area,
        max_regions=max_regions,
    )
    result["image_url"] = make_debug_image_url(result.get("image_path"))
    return result


@app.post("/api/runs/{run_id}/embed_wsi")
def embed_wsi(
    run_id: str,
    extractor_name: str = "uni2",
    tile_size_um: float = 256.0,
    patch_size_px: int = 512,
    tile_size_px: Optional[int] = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    use_tile_cache: bool = True,
    cache_tiles_ext: str = "jpg",
    max_supertile_size_slide_px: int = 4096,
    max_workers: int = 4,
    brightness_cutoff: Optional[int] = 240,
    canny_cutoff: Optional[float] = 0.02,
    default_slide_mpp: Optional[float] = None,
):
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if not run.slide_path:
        raise HTTPException(status_code=400, detail="Slide path not available for this run.")

    slide_path = Path(run.slide_path).resolve()
    if not slide_path.exists():
        raise HTTPException(status_code=400, detail=f"Slide file not found: {slide_path}")

    cache_tiles_ext_l = cache_tiles_ext.strip().lower()
    if cache_tiles_ext_l not in {"jpg", "png"}:
        raise HTTPException(status_code=400, detail="cache_tiles_ext must be 'jpg' or 'png'.")
    if tile_size_um <= 0:
        raise HTTPException(status_code=400, detail="tile_size_um must be > 0.")
    effective_tile_size_px = int(tile_size_px if tile_size_px is not None else patch_size_px)
    if effective_tile_size_px <= 0:
        raise HTTPException(status_code=400, detail="patch_size_px/tile_size_px must be > 0.")
    if batch_size <= 0:
        raise HTTPException(status_code=400, detail="batch_size must be > 0.")
    if max_supertile_size_slide_px <= 0:
        raise HTTPException(status_code=400, detail="max_supertile_size_slide_px must be > 0.")
    if max_workers <= 0:
        raise HTTPException(status_code=400, detail="max_workers must be > 0.")

    # Sanity-check with the same slide opening path used by the WSI runtime.
    _assert_slide_openable(slide_path)

    extractor = _get_embedding_extractor(extractor_name)
    tile_cache_dir = (BASE_RUN_DIR / run_id / "tile_cache") if use_tile_cache else None

    try:
        result = extract_wsi_features_by_tiles(
            slide_path=slide_path,
            extractor=extractor,
            tile_size_um=tile_size_um,
            tile_size_px=effective_tile_size_px,
            batch_size=batch_size,
            device=(device.strip() if device else None),
            cache_dir=tile_cache_dir,
            cache_tiles_ext=cache_tiles_ext_l,  # type: ignore[arg-type]
            max_supertile_size_slide_px=max_supertile_size_slide_px,
            max_workers=max_workers,
            brightness_cutoff=brightness_cutoff,
            canny_cutoff=canny_cutoff,
            default_slide_mpp=default_slide_mpp,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"WSI embedding failed: {type(exc).__name__}: {exc}",
        ) from exc

    embed_dir = BASE_RUN_DIR / run_id / "embeddings"
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = embed_dir / f"{slide_path.stem}_{extractor_name}_{ts}.npz"
    save_tile_features_npz(result, out_path)

    feature_shape = list(result.features.shape)
    num_tiles = int(feature_shape[0]) if len(feature_shape) > 0 else 0
    feature_dim = int(feature_shape[1]) if len(feature_shape) > 1 else 0

    return {
        "ok": True,
        "run_id": run_id,
        "slide_path": str(slide_path),
        "extractor_name": extractor_name,
        "extractor_id": result.extractor_id,
        "tile_size_um": result.tile_size_um,
        "tile_size_px": result.tile_size_px,
        "patch_size_px": result.tile_size_px,
        "features_shape": feature_shape,
        "coordinates_shape": list(result.coordinates_um.shape),
        "num_tiles": num_tiles,
        "feature_dim": feature_dim,
        "output_path": str(out_path.resolve()),
    }


# Static mounts
app.mount("/debug", StaticFiles(directory=DEBUG_ROOT_DIR), name="debug")
app.mount("/reports", StaticFiles(directory=REPORT_ROOT_DIR), name="reports")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "wsi-agent-web", "model_name": MODEL_NAME}


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    reload_enabled = os.getenv("UVICORN_RELOAD", "0").strip().lower() in {"1", "true", "yes", "y"}
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=3008,
        reload=reload_enabled,
        reload_excludes=["outputs/*", "wsi_debug/*", "wsi_reports/*"],
    )
