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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from wsi_core import (
    run_wsi_agent_for_web,
    get_public_state_snapshot,
    DEBUG_ROOT_DIR,
    REPORT_ROOT_DIR,
)

# Primary slide types OpenSlide can open directly, plus MIRAX zip bundle
ALLOWED_SLIDE_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".mrsx", ".zip"}
SUPPORTED_PRIMARY_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".mrsx"}
MIRAX_EXTS = {".mrxs", ".mrsx"}
STD_EXTS = {".svs", ".tif", ".tiff", ".ndpi"}

app = FastAPI(title="WSI Agent Prototype")

BASE_RUN_DIR = Path("./runs")
BASE_RUN_DIR.mkdir(exist_ok=True)

STATIC_DIR = Path("./static").resolve()
if not STATIC_DIR.exists():
    raise RuntimeError(f"Missing static dir at {STATIC_DIR}. Create ./static with index.html/app.js/styles.css")


class RunStatus(BaseModel):
    run_id: str
    status: str               # created | uploading | pending | running | done | error
    created_at: datetime
    agent_type: str
    prompt: Optional[str]
    slide_filename: str       # filled after finalize
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

# --- Auto-cleanup config ---
DELETE_UPLOADS_AFTER_RUN = os.getenv("DELETE_UPLOADS_AFTER_RUN", "1").strip().lower() in {"1", "true", "yes", "y"}


def _safe_cleanup_run_upload_dir(run_id: str) -> None:
    """
    Deletes ./runs/<run_id>/ (uploaded slide data + extracted bundles, etc.)
    Does NOT touch REPORT_ROOT_DIR or DEBUG_ROOT_DIR.
    """
    try:
        run_dir = (BASE_RUN_DIR / run_id).resolve()
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


def _is_safe_relpath(p: Path) -> bool:
    if p.is_absolute():
        return False
    if ".." in p.parts:
        return False
    return True


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



def run_worker(run_id: str, slide_path: str, prompt: Optional[str], agent_type: str) -> None:
    run = RUNS[run_id]
    run.status = "running"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        result = run_wsi_agent_for_web(
            slide_path=slide_path,
            prompt=prompt,
            agent_type=agent_type,
            run_id=run_id,
        )
        run.status = "done"
        run.final_output = result["final_output"]
        run.reasoning_content = result.get("reasoning_content")
        run.report_path = result.get("report_path")

    except Exception as exc:
        run.status = "error"
        run.error_message = str(exc)
        run.traceback = traceback.format_exc()
        print("=== RUN ERROR ===")
        print(run.error_message)
        print(run.traceback)

    finally:
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
    agent_type: str = Form("msi"),
):
    agent_type_lower = agent_type.lower()
    if agent_type_lower not in {"msi", "wsi"}:
        raise HTTPException(status_code=400, detail="agent_type must be 'msi' or 'wsi'")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    run_dir = BASE_RUN_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_status = RunStatus(
        run_id=run_id,
        status="created",
        created_at=datetime.utcnow(),
        agent_type=agent_type_lower,
        prompt=prompt or None,
        slide_filename="(upload pending)",
        upload_count=0,
        upload_bytes=0,
        uploaded_files=[],
    )
    RUNS[run_id] = run_status
    return {"run_id": run_id}


@app.post("/api/runs/{run_id}/upload")
async def upload_one_file(
    run_id: str,
    file: UploadFile = File(...),
    relpath: str = Form(""),
):
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status in {"running", "done"}:
        raise HTTPException(status_code=400, detail=f"Run is already {run.status}; uploads are closed.")
    if run.status == "error":
        raise HTTPException(status_code=400, detail="Run is in error state; create a new run.")

    run_dir = BASE_RUN_DIR / run_id
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
    if run.status in {"running", "done"}:
        raise HTTPException(status_code=400, detail=f"Run is already {run.status}.")
    if run.status == "error":
        raise HTTPException(status_code=400, detail="Run is in error state; create a new run.")

    run_dir = BASE_RUN_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=400, detail="Run directory missing; nothing to finalize.")

    slide_path, slide_filename = _validate_final_bundle(run_dir)

    run.slide_filename = slide_filename
    run.status = "pending"

    thread = threading.Thread(
        target=run_worker,
        args=(run_id, str(slide_path), run.prompt or None, run.agent_type),
        daemon=True,
    )
    thread.start()

    print(f"[FINALIZE] run={run_id} primary={slide_path}")
    return {"ok": True, "run_id": run_id, "primary": slide_filename}


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

    return {"run": run, "wsi_state": wsi_state}


# Static mounts
app.mount("/debug", StaticFiles(directory=DEBUG_ROOT_DIR), name="debug")
app.mount("/reports", StaticFiles(directory=REPORT_ROOT_DIR), name="reports")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3007, reload=True)
