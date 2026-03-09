from typing import Any, Dict, Optional

from agents import Runner

from . import state
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


def run_wsi_agent_for_web(
    slide_path: str,
    prompt: Optional[str],
    agent_type: str,
    run_id: str,
    model_name: Optional[str] = None,
    max_turns: int = MAX_TURNS,
) -> Dict[str, Any]:
    set_slide_path(slide_path)
    reset_wsi_state(run_id)

    if not prompt:
        if agent_type.lower() == "tile":
            prompt = DEFAULT_TILE_PROMPT
        elif agent_type.lower() == "aml":
            prompt = DEFAULT_AML_PROMPT
        else:
            prompt = DEFAULT_WSI_PROMPT

    if agent_type.lower() == "tile":
        base_agent = WSITileSelectorAgent
    elif agent_type.lower() == "aml":
        base_agent = WSIAmlDetectorAgent
    else:
        base_agent = WSIPathologyAgent
    runtime_agent = _agent_with_model(base_agent, model_name)
    result = Runner.run_sync(runtime_agent, prompt, max_turns=max_turns)
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
