import re
from contextlib import contextmanager
from typing import Any, Dict, Optional

from agents import ModelBehaviorError, Runner

from . import state
from . import context_injection
from .agents import (
    WSIAmlDetectorAgent,
    WSIPathologyAgent,
    WSITileSelectorAgent,
    _agent_with_model,
)
from .config import MAX_TURNS
from .prompts import DEFAULT_AML_PROMPT, DEFAULT_TILE_PROMPT, DEFAULT_WSI_PROMPT
from .reporting import write_markdown_report
from .state import get_public_state_snapshot, reset_wsi_state, set_slide_path

_FINAL_DIAGNOSIS_LABELS = (
    "Normal marrow",
    "Acute leukemia",
)
_FINAL_DIAGNOSIS_LOOKUP = {label.lower(): label for label in _FINAL_DIAGNOSIS_LABELS}


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
    if str(agent_type or "").lower() != "aml":
        return None

    match = re.search(r"Tool\s+(.+?)\s+not found in agent\s+", str(exc), flags=re.IGNORECASE)
    if not match:
        return None

    recovered_name = _normalize_recovered_tool_name(match.group(1))
    canonical = _FINAL_DIAGNOSIS_LOOKUP.get(recovered_name.lower())
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
    elif agent_type_l == "aml":
        base_agent = WSIAmlDetectorAgent
    else:
        base_agent = WSIPathologyAgent
    return _agent_with_model(base_agent, model_name)


def run_wsi_agent_for_web(
    slide_path: str,
    prompt: Optional[str],
    agent_type: str,
    run_id: str,
    model_name: Optional[str] = None,
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
) -> Dict[str, Any]:
    agent_type_l = (agent_type or "wsi").lower()

    if not prompt:
        if agent_type_l == "tile":
            prompt = DEFAULT_TILE_PROMPT
        elif agent_type_l == "aml":
            prompt = DEFAULT_AML_PROMPT
        else:
            prompt = DEFAULT_WSI_PROMPT

    def _initialize_state() -> None:
        set_slide_path(slide_path)
        reset_wsi_state(
            run_id,
            extractor_name=extractor_name,
            tile_size_um=tile_size_um,
            tile_size_px=tile_size_px,
            batch_size=batch_size,
            tile_prefilter_method=tile_prefilter_method,
            roi_output_size_px=roi_output_size_px,
            max_accepted_rois=max_accepted_rois,
            target_accepted_rois=target_accepted_rois,
            default_mpp_um=default_mpp_um,
        )
        state.AGENT_TYPE = agent_type_l

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
