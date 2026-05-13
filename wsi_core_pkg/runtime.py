import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from agents import ModelBehaviorError, Runner

from . import context_injection
from . import state
from .agents import (
    WSIAmlDiagnosisAgent,
    WSIAmlRoiCollectorAgent,
    WSIPathologyAgent,
    WSITileSelectorAgent,
    _agent_with_model,
)
from .aml_auto_fallback import (
    aml_case_output_dir as _aml_case_output_dir,
    aml_roi_collection_path_for_case as _aml_roi_collection_path_for_case,
    is_max_turns_exceeded_error as _is_max_turns_exceeded_error,
    load_saved_aml_roi_collection_summary as _load_saved_aml_roi_collection_summary,
)
from .aml_output import (
    FINAL_DECISION_LOOKUP,
    materialize_aml_case_collection,
    persist_current_aml_roi_collection,
    write_roi_collection_json,
)
from .config import MAX_TURNS, MODEL_NAME, OUTPUTS_ROOT_DIR, client_sync
from .context_injection import (
    CONTEXT_IMAGE_JPEG_QUALITY,
    CONTEXT_IMAGE_MAX_DIM,
    _encode_image_as_data_url,
)
from .config import AmlRoiCollectionComplete
from .prompts import (
    DEFAULT_AML_PROMPT,
    DEFAULT_AML_DIAGNOSIS_PROMPT,
    DEFAULT_AML_ROI_COLLECTION_PROMPT,
    DEFAULT_TILE_PROMPT,
    DEFAULT_WSI_PROMPT,
)
from .reporting import write_markdown_report
from .state import get_public_state_snapshot, reset_wsi_state, set_slide_path
from .tuning_config import tuning_value

_ROI_COLLECTION_FILENAMES = ("roi_collection.json", "roi_location.json")
_ROI_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class NavigationRunConfig:
    run_id: str
    extractor_name: str
    tile_size_um: float
    tile_size_px: int
    batch_size: int
    tile_prefilter_method: str
    roi_output_size_px: int
    max_accepted_rois: int
    target_accepted_rois: int
    default_mpp_um: Optional[float]
    candidate_nav_field_um: Optional[float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_aml_pipeline_config(key: str, default: Any) -> Any:
    try:
        return tuning_value("aml_pipeline", key)
    except Exception:
        return default


def _extract_json_from_text(text: str) -> str:
    """Extract the first JSON object from a possibly prose-wrapped response."""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if match:
        candidate = match.group(1)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    match = re.search(r"(\{.*\})", stripped, re.DOTALL)
    if match:
        candidate = match.group(1)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    return text


def _message_content_to_text(content: Any) -> Optional[str]:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text") is not None:
                parts.append(str(item.get("text")))
        return "\n".join(parts) if parts else None
    if isinstance(content, dict) and content.get("text") is not None:
        return str(content.get("text"))
    return str(content)


def _chat_response_text(response: Any) -> Optional[str]:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    if message is None:
        return None
    for attr in ("content", "reasoning_content", "reasoning", "text"):
        text = _message_content_to_text(getattr(message, attr, None))
        if text and text.strip():
            return text
    return None


def _normalize_recovered_tool_name(raw_name: str) -> str:
    cleaned = re.sub(r"<\|.*?\|>", "", str(raw_name or ""))
    cleaned = cleaned.replace('"', " ").replace("'", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n`*:_-")
    return cleaned


def _recover_final_output_from_tool_error(
    exc: ModelBehaviorError,
    *,
    agent_type: str,
) -> Optional[str]:
    if str(agent_type or "").lower() not in {"aml", "aml_auto", "aml_roi"}:
        return None

    match = re.search(r"Tool\s+(.+?)\s+not found in agent\s+", str(exc), flags=re.IGNORECASE)
    if not match:
        return None

    recovered_name = _normalize_recovered_tool_name(match.group(1))
    canonical = FINAL_DECISION_LOOKUP.get(recovered_name.lower())
    if canonical is None:
        return None

    return f"Final decision: {canonical}"


def _is_context_window_exceeded_error(exc: Exception) -> bool:
    text = str(exc or "")
    exc_name = type(exc).__name__
    return (
        "ContextWindowExceededError" in exc_name
        or "ContextWindowExceededError" in text
        or "maximum context length" in text
        or "exceeds model's maximum context length" in text
        or ("Input length" in text and "context length" in text)
    )


@contextmanager
def _ultra_lean_context_mode():
    saved = {
        "CONTEXT_MAX_INLINE_IMAGES": context_injection.CONTEXT_MAX_INLINE_IMAGES,
        "CONTEXT_MAX_INLINE_IMAGE_URL_CHARS": context_injection.CONTEXT_MAX_INLINE_IMAGE_URL_CHARS,
        "CONTEXT_PREVIOUS_VIEWS_MAX": context_injection.CONTEXT_PREVIOUS_VIEWS_MAX,
        "CONTEXT_ROI_CANDIDATE_LINES_MAX": context_injection.CONTEXT_ROI_CANDIDATE_LINES_MAX,
        "EXAMPLE_TILES_MAX_PER_CLASS": context_injection.EXAMPLE_TILES_MAX_PER_CLASS,
        "EXAMPLE_ROIS_MAX_PER_CLASS": context_injection.EXAMPLE_ROIS_MAX_PER_CLASS,
    }
    try:
        context_injection.CONTEXT_MAX_INLINE_IMAGES = 0
        context_injection.CONTEXT_MAX_INLINE_IMAGE_URL_CHARS = 0
        context_injection.CONTEXT_PREVIOUS_VIEWS_MAX = 0
        context_injection.CONTEXT_ROI_CANDIDATE_LINES_MAX = 2
        context_injection.EXAMPLE_TILES_MAX_PER_CLASS = 0
        context_injection.EXAMPLE_ROIS_MAX_PER_CLASS = 0
        yield
    finally:
        for key, value in saved.items():
            setattr(context_injection, key, value)


def _select_runtime_agent(agent_type_l: str, model_name: Optional[str]):
    if agent_type_l == "tile":
        base_agent = WSITileSelectorAgent
    elif agent_type_l == "aml_roi":
        base_agent = WSIAmlRoiCollectorAgent
    elif agent_type_l == "aml_diagnosis":
        base_agent = WSIAmlDiagnosisAgent
    else:
        base_agent = WSIPathologyAgent
    return _agent_with_model(base_agent, model_name)


# ---------------------------------------------------------------------------
# ROI collection JSON helpers
# ---------------------------------------------------------------------------

def resolve_roi_collection_path(path_or_dir: str) -> Path:
    """Return a manifest path from a case dir, or the original path if no manifest exists."""
    p = Path(path_or_dir)
    if p.is_file():
        return p
    configured = _get_aml_pipeline_config("ROI_COLLECTION_FILENAME", "roi_collection.json")
    for roi_filename in (configured, *_ROI_COLLECTION_FILENAMES):
        candidate = p / str(roi_filename)
        if candidate.is_file():
            return candidate
    return p


def _natural_path_sort_key(path: Path) -> List[Any]:
    parts = re.split(r"(\d+)", path.name.lower())
    key: List[Any] = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part)
    return key


def _build_roi_collection_from_directory(case_dir: Path) -> Tuple[Dict[str, Any], Optional[Path], str]:
    """Build an in-memory ROI collection from a legacy AML output folder."""
    roi_paths: List[Path] = []
    seen_paths: set[str] = set()
    for base_dir in (case_dir / "images", case_dir):
        if not base_dir.is_dir():
            continue
        for candidate in sorted(base_dir.iterdir(), key=_natural_path_sort_key):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in _ROI_IMAGE_EXTS:
                continue
            lower_name = candidate.name.lower()
            if not lower_name.startswith("roi") or lower_name == "roi_candidates.jpg":
                continue
            key = str(candidate.resolve())
            if key in seen_paths:
                continue
            seen_paths.add(key)
            roi_paths.append(candidate)

    if not roi_paths:
        raise FileNotFoundError(f"No ROI image files matching 'roi*.jpg' were found in: {case_dir}")

    overview_path: Optional[Path] = None
    for candidate in (
        case_dir / "images" / "roi_candidates.jpg",
        case_dir / "roi_candidates.jpg",
    ):
        if candidate.is_file():
            overview_path = candidate
            break

    accepted_rois: List[Dict[str, Any]] = []
    for idx, image_path in enumerate(roi_paths, start=1):
        match = re.search(r"roi[_-]?(\d+)", image_path.stem, re.IGNORECASE)
        roi_id = match.group(1) if match else str(idx)
        rel_path: Optional[str] = None
        try:
            rel_path = image_path.relative_to(case_dir).as_posix()
        except ValueError:
            rel_path = image_path.name
        accepted_rois.append(
            {
                "roi_id": roi_id,
                "image_path": rel_path,
                "absolute_image_path": str(image_path.resolve()),
            }
        )

    collection: Dict[str, Any] = {
        "schema_version": 1,
        "task": "aml_roi_collection",
        "slide_name": case_dir.name,
        "case_id": case_dir.name,
        "accepted_rois": accepted_rois,
    }
    if overview_path is not None:
        try:
            collection["roi_candidates_image"] = overview_path.relative_to(case_dir).as_posix()
        except ValueError:
            collection["roi_candidates_image"] = str(overview_path.resolve())
    return collection, case_dir, case_dir.name


def _select_earliest_overlay_image(path_text: Optional[str], suffix: str) -> Optional[str]:
    if not path_text:
        return None
    current = Path(path_text)
    if not current.is_file():
        return None

    candidates = sorted(current.parent.glob(f"*{suffix}"), key=_natural_path_sort_key)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return str(current.resolve())


def _roi_image_path_from_entry(roi: Dict[str, Any], collection_dir: Optional[Path]) -> Optional[str]:
    abs_path = roi.get("absolute_image_path")
    if abs_path and Path(str(abs_path)).is_file():
        return str(abs_path)
    if collection_dir is None:
        return None
    rel = roi.get("image_path")
    if not rel:
        return None
    candidate = collection_dir / str(rel)
    if candidate.is_file():
        return str(candidate)
    return None


def load_aml_diagnosis_input(path_or_dir: str) -> Tuple[Dict[str, Any], Optional[Path], str, Path]:
    """Resolve AML diagnosis input from json, legacy folder, or a single ROI image."""
    resolved = Path(path_or_dir).expanduser().resolve()

    if resolved.is_dir():
        manifest_or_dir = resolve_roi_collection_path(str(resolved))
        if manifest_or_dir.is_file():
            resolved = manifest_or_dir
        else:
            collection, collection_dir, slide_name = _build_roi_collection_from_directory(resolved)
            return collection, collection_dir, slide_name, resolved

    if not resolved.is_file():
        raise FileNotFoundError(f"ROI input not found: {resolved}")

    if resolved.suffix.lower() == ".json":
        collection = json.loads(resolved.read_text())
        collection_dir = resolved.parent
        slide_name = collection.get("slide_name") or collection.get("case_id") or resolved.parent.name
        return collection, collection_dir, slide_name, resolved

    collection, collection_dir, slide_name = _build_single_roi_collection_from_image(resolved)
    return collection, collection_dir, slide_name, resolved


def _build_single_roi_collection_from_image(image_path: Path) -> Tuple[Dict[str, Any], Optional[Path], str]:
    """Build a minimal in-memory collection from a single ROI image file."""
    collection = {
        "schema_version": 1,
        "task": "aml_roi_collection",
        "slide_name": image_path.parent.parent.name if image_path.parent.name == "images" else image_path.parent.name,
        "case_id": image_path.parent.parent.name if image_path.parent.name == "images" else image_path.parent.name,
        "accepted_rois": [
            {
                "roi_id": "1",
                "image_path": image_path.name,
                "absolute_image_path": str(image_path.resolve()),
            }
        ],
    }
    return collection, image_path.parent, collection["slide_name"]


def build_roi_collection_from_state(
    *,
    run_id: str,
    slide_path: str,
    model_name: str,
    extractor_name: str,
    tile_filter: str,
    tile_size_px: int,
    roi_size_px: int,
    target_accepted_rois: int,
) -> Dict[str, Any]:
    """Build an in-memory ROI collection manifest from current agent state."""
    images_subdir = _get_aml_pipeline_config("ROI_IMAGES_SUBDIR", "images")
    roi_marks = list(state._roi_marks or [])

    accepted_rois = []
    for i, roi in enumerate(roi_marks, start=1):
        roi_id = str(roi.get("roi_id", i))
        img_rel = f"{images_subdir}/roi_{roi_id}.jpg"
        accepted_rois.append({
            "roi_id": roi_id,
            "image_path": img_rel,
            "absolute_image_path": None,   # filled after export
            "bbox_level0": roi.get("view_bbox_level0"),
            "field_width_um": roi.get("field_width_um"),
            "field_height_um": roi.get("field_height_um"),
            "tissue_fraction": roi.get("tissue_fraction"),
            "effective_magnification": roi.get("effective_magnification"),
            "candidate_rank": roi.get("candidate_rank"),
            "requested_bbox_norm": roi.get("requested_bbox_norm"),
            "accepted_reason": roi.get("label", ""),
            "_debug_path": roi.get("debug_path", ""),
        })

    case_id = Path(slide_path).stem if slide_path else run_id
    slide_name = Path(slide_path).name if slide_path else ""
    return {
        "schema_version": 1,
        "task": "aml_roi_collection",
        "slide_path": slide_path,
        "slide_name": slide_name,
        "run_id": run_id,
        "case_id": case_id,
        "model_name": model_name,
        "extractor_name": extractor_name,
        "tile_filter": tile_filter,
        "tile_size_px": tile_size_px,
        "roi_size_px": roi_size_px,
        "roi_images_dir": images_subdir,
        "accepted_roi_count": len(roi_marks),
        "target_accepted_rois": target_accepted_rois,
        "slide_overview_image": None,
        "roi_candidates_image": None,
        "accepted_rois": accepted_rois,
        "discard_summary": [],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def _export_roi_images_to_case_dir(
    collection: Dict[str, Any],
    case_output_dir: Path,
) -> Dict[str, Any]:
    """Copy ROI images into <case_output_dir>/images and update manifest paths."""
    collection = dict(collection)
    candidates_src = _select_earliest_overlay_image(
        state._last_roi_candidate_overlay_path,
        "_roi_candidates.jpg",
    )
    if candidates_src and Path(candidates_src).is_file():
        collection["roi_candidates_image"] = str(Path(candidates_src).resolve())

    overview_src = state._last_overview_with_box_path
    if overview_src and Path(overview_src).is_file():
        collection["slide_overview_image"] = str(Path(overview_src).resolve())

    return materialize_aml_case_collection(collection, case_output_dir)


def build_aml_diagnosis_messages_from_roi_collection(
    collection: Dict[str, Any],
    *,
    collection_dir: Optional[Path] = None,
    prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build messages for diagnosis agent: ONLY encoded images + prompt, nothing else.

    Completely blind assessment - no ROI labels, no filenames, no metadata.
    Agent sees only: images (base64-encoded) + system prompt.
    """
    accepted_rois = [roi for roi in collection.get("accepted_rois", []) if isinstance(roi, dict)]
    content: List[Dict[str, Any]] = []

    # Inject slide overview first if available.
    overview_path: Optional[str] = None
    overview_rel = collection.get("slide_overview_image")
    if collection_dir is not None and overview_rel:
        overview_candidate = collection_dir / str(overview_rel)
        if overview_candidate.is_file():
            overview_path = str(overview_candidate)
    if overview_path:
        overview_url = _encode_image_as_data_url(overview_path, max_dim=512, jpeg_quality=80)
        if overview_url:
            content.append(
                {
                    "type": "text",
                    "text": "Slide overview (all marked ROI positions visible).",
                }
            )
            content.append({"type": "image_url", "image_url": {"url": overview_url}})

    # Build content: ONLY images, NO text labels or metadata
    for roi in accepted_rois:
        # Resolve paths internally only; the model sees the encoded image.
        img_path = _roi_image_path_from_entry(roi, collection_dir)
        if img_path:
            url = _encode_image_as_data_url(img_path, max_dim=768, jpeg_quality=85)
            if url:
                content.append({"type": "image_url", "image_url": {"url": url}})

    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Sub-runners
# ---------------------------------------------------------------------------

def _make_state_initializer(
    *,
    slide_path: str,
    config: NavigationRunConfig,
    agent_type_for_state: str,
):
    """Return a zero-argument callable that (re)initialises WSI agent state."""
    def _init():
        set_slide_path(slide_path)
        reset_wsi_state(
            config.run_id,
            extractor_name=config.extractor_name,
            tile_size_um=config.tile_size_um,
            tile_size_px=config.tile_size_px,
            batch_size=config.batch_size,
            tile_prefilter_method=config.tile_prefilter_method,
            roi_output_size_px=config.roi_output_size_px,
            max_accepted_rois=config.max_accepted_rois,
            target_accepted_rois=config.target_accepted_rois,
            default_mpp_um=config.default_mpp_um,
            candidate_nav_field_um=config.candidate_nav_field_um,
        )
        state.AGENT_TYPE = agent_type_for_state
    return _init


def _agent_result(final_output: str, reasoning_content: Optional[str] = None) -> SimpleNamespace:
    return SimpleNamespace(final_output=final_output, reasoning_content=reasoning_content)


def _run_aml_roi(
    *,
    slide_path: str,
    prompt: Optional[str],
    model_name: Optional[str],
    config: NavigationRunConfig,
    max_turns: int,
    case_output_dir: Optional[str],
    write_report: bool = True,
) -> Dict[str, Any]:
    """ROI collection stage: navigate WSI, collect N ROIs, export images + JSON."""
    effective_prompt = prompt or DEFAULT_AML_ROI_COLLECTION_PROMPT
    case_out = _aml_case_output_dir(
        run_id=config.run_id,
        case_output_dir=case_output_dir,
        outputs_root_dir=OUTPUTS_ROOT_DIR,
    )

    # Use AGENT_TYPE="aml" so AML-specific ranking, context injection, and tools all work.
    _initialize_state = _make_state_initializer(
        slide_path=slide_path,
        config=config,
        agent_type_for_state="aml",
    )
    _initialize_state()
    state.MODEL_NAME = str(model_name or "")
    state.CASE_OUTPUT_DIR = str(case_out)

    runtime_agent = _agent_with_model(WSIAmlRoiCollectorAgent, model_name)

    def _run_once() -> Any:
        return Runner.run_sync(runtime_agent, effective_prompt, max_turns=max_turns)

    try:
        try:
            result = _run_once()
        except AmlRoiCollectionComplete as exc:
            persist_current_aml_roi_collection(case_output_dir=case_out, model_name=model_name)
            result = _agent_result(str(exc))
        except ModelBehaviorError as exc:
            recovered = _recover_final_output_from_tool_error(exc, agent_type="aml_roi")
            if recovered is None:
                persist_current_aml_roi_collection(case_output_dir=case_out, model_name=model_name)
                raise
            result = _agent_result(recovered)
    except Exception as exc:
        if not _is_context_window_exceeded_error(exc):
            persist_current_aml_roi_collection(case_output_dir=case_out, model_name=model_name)
            raise
        persist_current_aml_roi_collection(case_output_dir=case_out, model_name=model_name)
        _initialize_state()
        state.MODEL_NAME = str(model_name or "")
        state.CASE_OUTPUT_DIR = str(case_out)
        with _ultra_lean_context_mode():
            try:
                result = _run_once()
            except AmlRoiCollectionComplete as lean_exc:
                persist_current_aml_roi_collection(case_output_dir=case_out, model_name=model_name)
                result = _agent_result(str(lean_exc))

    if state.HAS_FATAL_ERROR:
        raise RuntimeError(state.LAST_FATAL_ERROR or "WSI run failed due to a fatal slide error.")

    case_out.mkdir(parents=True, exist_ok=True)

    # Build manifest from state
    collection = build_roi_collection_from_state(
        run_id=config.run_id,
        slide_path=slide_path,
        model_name=model_name or "",
        extractor_name=config.extractor_name,
        tile_filter=config.tile_prefilter_method,
        tile_size_px=config.tile_size_px,
        roi_size_px=config.roi_output_size_px,
        target_accepted_rois=config.target_accepted_rois,
    )

    # Export images and update absolute paths
    collection = _export_roi_images_to_case_dir(collection, case_out)

    roi_collection_path = _aml_roi_collection_path_for_case(
        case_output_dir=case_out,
        roi_collection_filename=str(_get_aml_pipeline_config("ROI_COLLECTION_FILENAME", "roi_collection.json")),
    )
    write_roi_collection_json(collection, roi_collection_path)

    final_text = result.final_output
    reasoning = getattr(result, "reasoning_content", None)

    report_path = None
    if write_report:
        report_path = write_markdown_report(
            effective_prompt,
            final_text,
            run_id=config.run_id,
            reasoning_content=reasoning,
        )

    state_snapshot = get_public_state_snapshot()

    return {
        "final_output": final_text,
        "reasoning_content": reasoning,
        "report_path": report_path,
        "roi_collection_path": str(roi_collection_path),
        "slide_name": collection.get("slide_name", ""),
        "roi_images_dir": str(case_out / _get_aml_pipeline_config("ROI_IMAGES_SUBDIR", "images")),
        "accepted_roi_count": collection.get("accepted_roi_count", 0),
        "state": state_snapshot,
    }


def _resolve_roi_image_path(roi: Dict[str, Any], collection_dir: Optional[Path]) -> Optional[str]:
    """Resolve ROI image path from a roi_collection.json entry."""
    return _roi_image_path_from_entry(roi, collection_dir)


def _reset_diagnosis_state(run_id: str) -> None:
    state.RUN_ID = run_id
    state.AGENT_TYPE = "aml_diagnosis"
    state.HAS_FATAL_ERROR = False
    state.LAST_FATAL_ERROR = None
    state._roi_marks = []
    state._step_log = []
    state._example_tiles_injected = False
    state._example_rois_injected = False


def _build_aml_diagnosis_content(
    accepted_rois: List[Dict[str, Any]],
    collection_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    diagnosis_content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Run AML morphology-only diagnosis using the ROI images below. "
                "Each image is labeled by ROI id. Follow the system prompt JSON schema exactly."
            ),
        }
    ]
    for i, roi in enumerate(accepted_rois, start=1):
        roi_id = str(roi.get("roi_id", i))
        img_path = _resolve_roi_image_path(roi, collection_dir)
        diagnosis_content.append({"type": "text", "text": f"ROI #{roi_id} image:"})
        if img_path:
            url = _encode_image_as_data_url(
                img_path,
                max_dim=CONTEXT_IMAGE_MAX_DIM,
                jpeg_quality=CONTEXT_IMAGE_JPEG_QUALITY,
            )
            if url:
                diagnosis_content.append({"type": "image_url", "image_url": {"url": url}})
            else:
                diagnosis_content.append({"type": "text", "text": "(image encoding failed)"})
        else:
            diagnosis_content.append({"type": "text", "text": "(image not available)"})
    return diagnosis_content


def _strict_json_system_prompt(prompt: Optional[str]) -> str:
    base_system_prompt = (prompt or DEFAULT_AML_DIAGNOSIS_PROMPT).strip()
    return "\n\n".join(
        part for part in (
            base_system_prompt,
            "Return one strict JSON object only. Do not return markdown, prose, or code fences.",
        )
        if part
    )


def _format_diagnosis_response_text(raw_text: str) -> str:
    final_text = _extract_json_from_text(raw_text)
    try:
        parsed_final = json.loads(final_text)
        return json.dumps(parsed_final, ensure_ascii=False, indent=2)
    except Exception:
        return raw_text


def _run_aml_diagnosis(
    *,
    run_id: str,
    prompt: Optional[str],
    model_name: Optional[str],
    max_turns: int,
    roi_collection_path: Optional[str],
    roi_input_path: Optional[str],
    reset_runtime_state: bool = True,
) -> Dict[str, Any]:
    """Diagnosis-only stage: inject ROI images and return the model JSON."""
    _ = max_turns  # Direct chat-completion path does not use Runner turns.
    if reset_runtime_state:
        _reset_diagnosis_state(run_id)
    else:
        state.AGENT_TYPE = "aml_diagnosis"
        state.HAS_FATAL_ERROR = False
        state.LAST_FATAL_ERROR = None
        state.CURRENT_AGENT_ACTION = "Running AML diagnosis from saved ROI collection."

    input_path = roi_input_path or roi_collection_path
    if not input_path:
        raise ValueError("aml_diagnosis requires roi_input_path or roi_collection_path")

    collection, collection_dir, slide_name, resolved = load_aml_diagnosis_input(input_path)
    accepted_rois = [roi for roi in collection.get("accepted_rois", []) if isinstance(roi, dict)]
    selected_model = model_name or MODEL_NAME
    system_prompt = _strict_json_system_prompt(prompt)

    response = client_sync.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_aml_diagnosis_content(accepted_rois, collection_dir)},
        ],
        temperature=0.0,
    )
    raw_final_text = _chat_response_text(response) or "{}"
    final_text = _format_diagnosis_response_text(raw_final_text)

    report_path = write_markdown_report(
        system_prompt,
        final_text,
        run_id=run_id,
        reasoning_content=None,
    )

    return {
        "final_output": final_text,
        "raw_final_output": raw_final_text,
        "reasoning_content": None,
        "report_path": report_path,
        "roi_collection_path": str(resolved),
        "slide_name": slide_name,
        "state": {"run_id": run_id, "agent_type": "aml_diagnosis"},
    }


def _resolve_aml_auto_prompts(
    *,
    prompt: Optional[str],
    aml_auto_roi_prompt: Optional[str],
    aml_auto_diagnosis_prompt: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    roi_prompt = aml_auto_roi_prompt if str(aml_auto_roi_prompt or "").strip() else None

    diagnosis_prompt = aml_auto_diagnosis_prompt
    if not str(diagnosis_prompt or "").strip():
        diagnosis_prompt = prompt
    if not str(diagnosis_prompt or "").strip():
        diagnosis_prompt = None
    elif str(diagnosis_prompt).strip() == DEFAULT_AML_PROMPT.strip():
        # Preserve split-stage defaults for legacy callers that still send the
        # old combined AML prompt in aml_auto mode.
        diagnosis_prompt = None
    return roi_prompt, diagnosis_prompt


def _run_aml_auto(
    *,
    slide_path: str,
    prompt: Optional[str],
    aml_auto_roi_prompt: Optional[str],
    aml_auto_diagnosis_prompt: Optional[str],
    model_name: Optional[str],
    config: NavigationRunConfig,
    max_turns: int,
    case_output_dir: Optional[str],
) -> Dict[str, Any]:
    """Full AML pipeline: ROI collection then diagnosis."""
    roi_prompt, diagnosis_prompt = _resolve_aml_auto_prompts(
        prompt=prompt,
        aml_auto_roi_prompt=aml_auto_roi_prompt,
        aml_auto_diagnosis_prompt=aml_auto_diagnosis_prompt,
    )

    case_out = _aml_case_output_dir(
        run_id=config.run_id,
        case_output_dir=case_output_dir,
        outputs_root_dir=OUTPUTS_ROOT_DIR,
    )
    roi_result: Dict[str, Any]
    fallback_reason: Optional[str] = None
    try:
        roi_result = _run_aml_roi(
            slide_path=slide_path,
            prompt=roi_prompt,
            model_name=model_name,
            config=config,
            max_turns=max_turns,
            case_output_dir=case_output_dir,
            write_report=False,   # diagnosis report is the primary one
        )
    except Exception as exc:
        if not _is_max_turns_exceeded_error(exc, max_turns):
            raise
        saved_summary = _load_saved_aml_roi_collection_summary(
            _aml_roi_collection_path_for_case(
                case_output_dir=case_out,
                roi_collection_filename=str(_get_aml_pipeline_config("ROI_COLLECTION_FILENAME", "roi_collection.json")),
            )
        )
        if saved_summary is None:
            raise
        fallback_reason = str(exc)
        roi_result = {
            "final_output": (
                "ROI collection terminated after the turn budget was exceeded. "
                "Continuing AML diagnosis from the saved accepted ROI collection."
            ),
            "reasoning_content": None,
            "report_path": None,
            "roi_collection_path": saved_summary["roi_collection_path"],
            "slide_name": saved_summary.get("slide_name", ""),
            "roi_images_dir": str(case_out / _get_aml_pipeline_config("ROI_IMAGES_SUBDIR", "images")),
            "accepted_roi_count": saved_summary["accepted_roi_count"],
            "state": {
                "run_id": config.run_id,
                "agent_type": "aml",
                "fallback_reason": fallback_reason,
            },
        }

    roi_collection_path = roi_result["roi_collection_path"]
    diag_run_id = f"{config.run_id}_diag"

    diagnosis_result = _run_aml_diagnosis(
        run_id=diag_run_id,
        prompt=diagnosis_prompt,
        model_name=model_name,
        max_turns=max_turns,
        roi_collection_path=roi_collection_path,
        roi_input_path=None,
        reset_runtime_state=False,
    )

    return {
        "final_output": diagnosis_result["final_output"],
        "reasoning_content": diagnosis_result.get("reasoning_content"),
        "report_path": diagnosis_result.get("report_path"),
        "roi_collection_output": roi_result.get("final_output"),
        "diagnosis_output": diagnosis_result["final_output"],
        "roi_collection_path": roi_collection_path,
        "roi_images_dir": roi_result.get("roi_images_dir"),
        "accepted_roi_count": roi_result.get("accepted_roi_count"),
        "state": roi_result.get("state"),
        "roi_stage_fallback_reason": fallback_reason,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_wsi_agent_for_web(
    slide_path: Optional[str] = None,
    prompt: Optional[str] = None,
    agent_type: str = "wsi",
    run_id: str = "",
    model_name: Optional[str] = None,
    aml_auto_roi_prompt: Optional[str] = None,
    aml_auto_diagnosis_prompt: Optional[str] = None,
    max_turns: int = MAX_TURNS,
    extractor_name: str = "uni2",
    tile_size_um: float = 256.0,
    tile_size_px: int = 224,
    batch_size: int = 128,
    tile_prefilter_method: str = "quality",
    roi_output_size_px: int = 1024,
    max_accepted_rois: int = 10,
    target_accepted_rois: int = 5,
    default_mpp_um: float | None = None,
    candidate_nav_field_um: float | None = None,
    roi_collection_path: str | None = None,
    roi_input_path: str | None = None,
    case_output_dir: str | None = None,
) -> Dict[str, Any]:
    agent_type_l = (agent_type or "wsi").lower()
    nav_config = NavigationRunConfig(
        run_id=run_id,
        extractor_name=extractor_name,
        tile_size_um=tile_size_um,
        tile_size_px=tile_size_px,
        batch_size=batch_size,
        tile_prefilter_method=tile_prefilter_method,
        roi_output_size_px=roi_output_size_px,
        max_accepted_rois=max_accepted_rois,
        target_accepted_rois=target_accepted_rois,
        default_mpp_um=default_mpp_um,
        candidate_nav_field_um=candidate_nav_field_um,
    )

    # Legacy alias
    if agent_type_l == "aml":
        agent_type_l = "aml_auto"

    # ── AML pipeline modes ────────────────────────────────────────────
    if agent_type_l == "aml_auto":
        if not slide_path:
            raise ValueError("aml_auto requires slide_path")
        return _run_aml_auto(
            slide_path=slide_path,
            prompt=prompt,
            aml_auto_roi_prompt=aml_auto_roi_prompt,
            aml_auto_diagnosis_prompt=aml_auto_diagnosis_prompt,
            model_name=model_name,
            config=nav_config,
            max_turns=max_turns,
            case_output_dir=case_output_dir,
        )

    if agent_type_l == "aml_roi":
        if not slide_path:
            raise ValueError("aml_roi requires slide_path")
        return _run_aml_roi(
            slide_path=slide_path,
            prompt=prompt,
            model_name=model_name,
            config=nav_config,
            max_turns=max_turns,
            case_output_dir=case_output_dir,
        )

    if agent_type_l == "aml_diagnosis":
        return _run_aml_diagnosis(
            run_id=run_id,
            prompt=prompt,
            model_name=model_name,
            max_turns=max_turns,
            roi_collection_path=roi_collection_path,
            roi_input_path=roi_input_path,
        )

    # ── Standard WSI / tile agents ────────────────────────────────────
    if not slide_path:
        raise ValueError(f"agent_type={agent_type_l!r} requires slide_path")

    if not prompt:
        if agent_type_l == "tile":
            prompt = DEFAULT_TILE_PROMPT
        else:
            prompt = DEFAULT_WSI_PROMPT

    _initialize_state = _make_state_initializer(
        slide_path=slide_path,
        config=nav_config,
        agent_type_for_state=agent_type_l,
    )

    def _run_once() -> Any:
        runtime_agent = _select_runtime_agent(agent_type_l, model_name)
        return Runner.run_sync(runtime_agent, prompt, max_turns=max_turns)

    _initialize_state()
    try:
        try:
            result = _run_once()
        except ModelBehaviorError as exc:
            recovered_final_text = _recover_final_output_from_tool_error(exc, agent_type=agent_type_l)
            if recovered_final_text is None:
                raise
            report_path = write_markdown_report(
                prompt,
                recovered_final_text,
                run_id=run_id,
                reasoning_content=None,
            )
            state_snapshot = get_public_state_snapshot()
            return {
                "final_output": recovered_final_text,
                "reasoning_content": None,
                "report_path": report_path,
                "state": state_snapshot,
            }
    except Exception as exc:
        if not _is_context_window_exceeded_error(exc):
            raise
        _initialize_state()
        with _ultra_lean_context_mode():
            result = _run_once()

    if state.HAS_FATAL_ERROR:
        raise RuntimeError(state.LAST_FATAL_ERROR or "WSI run failed due to a fatal slide error.")

    final_text = result.final_output
    reasoning = getattr(result, "reasoning_content", None)

    report_path = write_markdown_report(
        prompt,
        final_text,
        run_id=run_id,
        reasoning_content=reasoning,
    )

    state_snapshot = get_public_state_snapshot()

    return {
        "final_output": final_text,
        "reasoning_content": reasoning,
        "report_path": report_path,
        "state": state_snapshot,
    }
