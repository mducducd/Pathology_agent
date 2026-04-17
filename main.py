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
from pydantic import BaseModel, Field

from wsi_core_pkg.embeddings import (
    available_embedding_extractors,
    embedding_extractor_display_name,
    extract_wsi_features_by_tiles,
    save_tile_features_npz,
    uni2,
)
from wsi_core_pkg.prompts import DEFAULT_AML_PROMPT, DEFAULT_TILE_PROMPT, DEFAULT_WSI_PROMPT
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
    "gpt-oss-20b",
    "gemma-4-31B-it",
}
EMBEDDING_EXTRACTOR_OPTIONS = [
    {
        "name": name,
        "label": embedding_extractor_display_name(name),
    }
    for name in available_embedding_extractors()
    if not str(name).endswith("_onnx")
]
ALLOWED_EMBEDDING_EXTRACTORS = {item["name"] for item in EMBEDDING_EXTRACTOR_OPTIONS}
DEFAULT_EMBEDDING_EXTRACTOR = "uni2"
ALLOWED_TILE_PREFILTER_METHODS = {"none", "coarse", "quality", "hybrid"}
DEFAULT_SERVER_SLIDE_ROOTS = [
    Path("/mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs/"),
]
SERVER_SELECTION_LABELS = {
    "slide_file": "Server slide file",
    "mirax_file": "Server MIRAX file",
    "directory": "Server folder",
    "mirax_directory": "Server MIRAX folder",
}

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
    extractor_name: str = "uni2"
    tile_size_px: int = 224
    tile_size_um: float = 256.0
    batch_size: int = 128
    tile_prefilter_method: str = "quality"
    slide_filename: str       # filled after finalize
    slide_path: Optional[str] = None
    final_output: Optional[str] = None
    reasoning_content: Optional[str] = None
    report_path: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    source_mode: str = "upload"  # upload | server
    selected_source_path: Optional[str] = None
    selected_source_label: Optional[str] = None

    # upload bookkeeping
    upload_count: int = 0
    upload_bytes: int = 0
    uploaded_files: List[str] = Field(default_factory=list)


RUNS: Dict[str, RunStatus] = {}
RUN_TERMINATE_FLAGS: Dict[str, threading.Event] = {}
RUN_THREADS: Dict[str, threading.Thread] = {}
_EMBEDDING_EXTRACTOR_CACHE: Dict[str, object] = {}
_EMBEDDING_EXTRACTOR_LOCK = threading.Lock()

# --- Auto-cleanup config ---
DELETE_UPLOADS_AFTER_RUN = os.getenv("DELETE_UPLOADS_AFTER_RUN", "1").strip().lower() in {"1", "true", "yes", "y"}


def _load_server_slide_roots() -> List[Path]:
    raw = os.getenv("SERVER_SLIDE_ROOTS", "").strip()
    candidates: List[Path]
    if raw:
        candidates = [Path(chunk).expanduser() for chunk in raw.split(os.pathsep) if chunk.strip()]
    else:
        candidates = list(DEFAULT_SERVER_SLIDE_ROOTS)

    roots: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
        except Exception:
            resolved = candidate
        if not resolved.is_absolute():
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        roots.append(resolved)
    return roots


SERVER_SLIDE_ROOTS = _load_server_slide_roots()


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


def _get_server_root_for_path(path: Path) -> Optional[Path]:
    resolved_path = path.resolve(strict=False)
    for root in SERVER_SLIDE_ROOTS:
        try:
            resolved_path.relative_to(root)
            return root
        except ValueError:
            continue
    return None


def _resolve_allowed_server_path(raw_path: str) -> Tuple[Path, Path]:
    raw = (raw_path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="A server path is required.")

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise HTTPException(status_code=400, detail="Server paths must be absolute.")

    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Path not found: {candidate}") from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to resolve path '{candidate}': {type(exc).__name__}: {exc}",
        ) from exc

    root = _get_server_root_for_path(resolved)
    if root is None:
        raise HTTPException(
            status_code=403,
            detail="Selected path is outside the allowed server slide roots.",
        )
    return resolved, root


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
            from wsi_core_pkg.embeddings import get_embedding_extractor

            extractor = get_embedding_extractor(key)
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


def _matching_mirax_sibling_for_dir(dir_path: Path) -> Optional[Path]:
    for ext in sorted(MIRAX_EXTS):
        candidate = dir_path.parent / f"{dir_path.name}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _discover_slide_candidates(root: Path, limit: int = 4) -> List[Path]:
    found: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for filename in sorted(filenames):
            ext = Path(filename).suffix.lower()
            if ext not in SUPPORTED_PRIMARY_EXTS:
                continue
            found.append(Path(dirpath) / filename)
            if len(found) >= limit:
                return found
    return found


def _resolve_slide_file_path(slide_path: Path) -> Tuple[Path, str]:
    ext = slide_path.suffix.lower()
    if ext in STD_EXTS:
        return slide_path, slide_path.name

    if ext in MIRAX_EXTS:
        companion_dir = slide_path.parent / slide_path.stem
        if not companion_dir.is_dir():
            raise HTTPException(
                status_code=400,
                detail=(
                    "MIRAX slide detected (.mrxs/.mrsx) but companion data directory not found. "
                    "Expected a folder named like the slide (same stem) next to the .mrxs/.mrsx."
                ),
            )
        return slide_path, slide_path.name

    raise HTTPException(
        status_code=400,
        detail="Selected file is not a supported slide. Choose .svs/.tif/.tiff/.ndpi/.mrxs/.mrsx.",
    )


def _resolve_server_slide_selection(source_path: Path) -> Tuple[Path, str, str]:
    source_path = source_path.resolve(strict=True)

    if source_path.is_file():
        slide_path, slide_filename = _resolve_slide_file_path(source_path)
        selection_kind = "mirax_file" if slide_path.suffix.lower() in MIRAX_EXTS else "slide_file"
        return slide_path, slide_filename, selection_kind

    if not source_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Selected path is neither a file nor a directory: {source_path}")

    sibling_mirax = _matching_mirax_sibling_for_dir(source_path)
    if sibling_mirax is not None:
        slide_path, slide_filename = _resolve_slide_file_path(sibling_mirax)
        return slide_path, slide_filename, "mirax_directory"

    candidates = _discover_slide_candidates(source_path, limit=4)
    if not candidates:
        raise HTTPException(
            status_code=400,
            detail=(
                "Selected folder does not contain a supported slide bundle. "
                "Choose a slide file or a MIRAX companion folder."
            ),
        )
    if len(candidates) > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Selected folder contains multiple slide candidates. "
                "Choose a specific slide file or navigate into a single-slide folder."
            ),
        )

    slide_path, slide_filename = _resolve_slide_file_path(candidates[0])
    selection_kind = "mirax_directory" if slide_path.suffix.lower() in MIRAX_EXTS else "directory"
    return slide_path, slide_filename, selection_kind


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


def _server_root_label(root: Path) -> str:
    return root.name or str(root)


def _build_server_breadcrumbs(current_dir: Path, root: Path) -> List[dict]:
    crumbs = [{"label": _server_root_label(root), "path": str(root)}]
    if current_dir == root:
        return crumbs

    rel_parts = current_dir.relative_to(root).parts
    acc = root
    for part in rel_parts:
        acc = acc / part
        crumbs.append({"label": part, "path": str(acc)})
    return crumbs


def _list_server_directory_entries(current_dir: Path) -> List[dict]:
    try:
        children = sorted(
            current_dir.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied: {current_dir}") from exc

    entries: List[dict] = []
    for child in children:
        name = child.name
        if name.startswith("."):
            continue

        if child.is_dir():
            hint = "MIRAX companion folder" if _matching_mirax_sibling_for_dir(child) is not None else None
            entries.append(
                {
                    "name": name,
                    "path": str(child),
                    "kind": "dir",
                    "hint": hint,
                }
            )
            continue

        ext = child.suffix.lower()
        if ext not in SUPPORTED_PRIMARY_EXTS:
            continue

        try:
            size_bytes = child.stat().st_size
        except OSError:
            size_bytes = None

        entries.append(
            {
                "name": name,
                "path": str(child),
                "kind": "file",
                "ext": ext,
                "size_bytes": size_bytes,
                "slide_kind": "mirax" if ext in MIRAX_EXTS else "standard",
            }
        )

    return entries


class ServerPathRequest(BaseModel):
    path: str


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
            extractor_name=run.extractor_name,
            tile_size_um=run.tile_size_um,
            tile_size_px=run.tile_size_px,
            batch_size=run.batch_size,
            tile_prefilter_method=run.tile_prefilter_method,
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

@app.get("/api/default_prompts")
def get_default_prompts():
    return {
        "prompts": {
            "tile": DEFAULT_TILE_PROMPT,
            "aml": DEFAULT_AML_PROMPT,
            "wsi": DEFAULT_WSI_PROMPT,
        }
    }


@app.get("/api/embedding_extractors")
def get_embedding_extractors():
    return {
        "default_extractor": DEFAULT_EMBEDDING_EXTRACTOR,
        "extractors": EMBEDDING_EXTRACTOR_OPTIONS,
    }

@app.post("/api/runs/create")
async def create_run(
    prompt: str = Form(""),
    agent_type: str = Form("wsi"),
    model_name: str = Form(MODEL_NAME),
    extractor_name: str = Form(DEFAULT_EMBEDDING_EXTRACTOR),
    tile_size_px: int = Form(224),
    tile_size_um: float = Form(256.0),
    batch_size: int = Form(128),
    tile_prefilter_method: str = Form("quality"),
):
    agent_type_lower = agent_type.lower()
    if agent_type_lower not in {"tile", "wsi", "aml"}:
        raise HTTPException(status_code=400, detail="agent_type must be 'tile', 'wsi', or 'aml'")
    if model_name not in ALLOWED_MODEL_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"model_name must be one of: {', '.join(sorted(ALLOWED_MODEL_NAMES))}",
        )
    if extractor_name not in ALLOWED_EMBEDDING_EXTRACTORS:
        raise HTTPException(
            status_code=400,
            detail=f"extractor_name must be one of: {', '.join(sorted(ALLOWED_EMBEDDING_EXTRACTORS))}",
        )
    tile_prefilter_method = tile_prefilter_method.strip().lower()
    if tile_prefilter_method not in ALLOWED_TILE_PREFILTER_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"tile_prefilter_method must be one of: {', '.join(sorted(ALLOWED_TILE_PREFILTER_METHODS))}",
        )
    if tile_size_px <= 0:
        raise HTTPException(status_code=400, detail="tile_size_px must be > 0.")
    if tile_size_um <= 0:
        raise HTTPException(status_code=400, detail="tile_size_um must be > 0.")
    if batch_size <= 0:
        raise HTTPException(status_code=400, detail="batch_size must be > 0.")

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
        extractor_name=extractor_name,
        tile_size_px=tile_size_px,
        tile_size_um=tile_size_um,
        batch_size=batch_size,
        tile_prefilter_method=tile_prefilter_method,
        slide_filename="(upload pending)",
        slide_path=None,
        upload_count=0,
        upload_bytes=0,
        uploaded_files=[],
    )
    RUNS[run_id] = run_status
    RUN_TERMINATE_FLAGS[run_id] = threading.Event()
    return {"run_id": run_id, "model_name": model_name}


@app.get("/api/server_fs/roots")
def list_server_roots():
    roots = []
    for root in SERVER_SLIDE_ROOTS:
        roots.append(
            {
                "label": _server_root_label(root),
                "path": str(root),
                "exists": root.exists() and root.is_dir(),
            }
        )
    return {"roots": roots}


@app.get("/api/server_fs/list")
def list_server_directory(path: str):
    current_dir, root = _resolve_allowed_server_path(path)
    if not current_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Selected path is not a directory: {current_dir}")

    parent_path: Optional[str] = None
    if current_dir != root:
        parent = current_dir.parent
        if _get_server_root_for_path(parent) == root:
            parent_path = str(parent)

    return {
        "current_path": str(current_dir),
        "root_path": str(root),
        "root_label": _server_root_label(root),
        "parent_path": parent_path,
        "breadcrumbs": _build_server_breadcrumbs(current_dir, root),
        "entries": _list_server_directory_entries(current_dir),
    }


@app.post("/api/server_fs/resolve")
def resolve_server_path(payload: ServerPathRequest):
    source_path, root = _resolve_allowed_server_path(payload.path)
    slide_path, slide_filename, selection_kind = _resolve_server_slide_selection(source_path)
    return {
        "requested_path": str(source_path),
        "root_path": str(root),
        "slide_path": str(slide_path),
        "slide_filename": slide_filename,
        "selection_kind": selection_kind,
        "selection_label": SERVER_SELECTION_LABELS.get(selection_kind, "Server selection"),
    }


@app.post("/api/runs/{run_id}/select_server_path")
def select_server_path_for_run(run_id: str, payload: ServerPathRequest):
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status in {"pending", "running", "done", "terminated"}:
        raise HTTPException(status_code=400, detail=f"Run is already {run.status}; source selection is closed.")
    if run.status == "error":
        raise HTTPException(status_code=400, detail="Run is in error state; create a new run.")
    if run.upload_count > 0 or run.uploaded_files:
        raise HTTPException(status_code=400, detail="This run already has uploaded files. Create a new run to use server browsing.")

    source_path, _ = _resolve_allowed_server_path(payload.path)
    _, slide_filename, selection_kind = _resolve_server_slide_selection(source_path)

    run.source_mode = "server"
    run.selected_source_path = str(source_path)
    run.selected_source_label = SERVER_SELECTION_LABELS.get(selection_kind, "Server selection")
    run.slide_filename = slide_filename
    run.slide_path = None
    run.status = "created"

    return {
        "ok": True,
        "run_id": run_id,
        "selected_source_path": run.selected_source_path,
        "selected_source_label": run.selected_source_label,
        "slide_filename": slide_filename,
    }


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
    if run.source_mode == "server" or run.selected_source_path:
        raise HTTPException(status_code=400, detail="This run is configured for server browsing. Create a new run to upload files.")

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

    if run.source_mode == "server":
        if not run.selected_source_path:
            raise HTTPException(status_code=400, detail="No server path has been selected for this run.")
        source_path, _ = _resolve_allowed_server_path(run.selected_source_path)
        slide_path, slide_filename, selection_kind = _resolve_server_slide_selection(source_path)
        run.selected_source_label = SERVER_SELECTION_LABELS.get(selection_kind, run.selected_source_label)
    else:
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
    result["mask_url"] = make_debug_image_url(result.get("mask_path"))
    return result


@app.post("/api/runs/{run_id}/embed_wsi")
def embed_wsi(
    run_id: str,
    extractor_name: Optional[str] = None,
    tile_size_um: Optional[float] = None,
    patch_size_px: int = 512,
    tile_size_px: Optional[int] = None,
    batch_size: int = 128,
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

    # Use run's stored parameters if not provided
    if extractor_name is None:
        extractor_name = run.extractor_name
    if tile_size_um is None:
        tile_size_um = run.tile_size_um

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
        port=1234,
        reload=reload_enabled,
        reload_excludes=["outputs/*", "wsi_debug/*", "wsi_reports/*"],
    )
