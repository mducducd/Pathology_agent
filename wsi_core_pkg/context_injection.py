import base64
import json
import mimetypes
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import state
from .config import (
    EXAMPLE_ROIS_MAX_PER_CLASS,
    EXAMPLE_ROIS_NEG_DIR,
    EXAMPLE_ROIS_POS_DIR,
    EXAMPLE_TILES_BAD_DIR,
    EXAMPLE_TILES_GOOD_DIR,
    EXAMPLE_TILES_MAX_PER_CLASS,
    MODEL_NAME,
    client_async,
    client_sync,
)

_real_async_chat_create = client_async.chat.completions.create
_real_sync_chat_create = client_sync.chat.completions.create
_patch_installed = False


def _encode_image_as_data_url(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        img_bytes = f.read()
    image_b64 = base64.b64encode(img_bytes).decode("ascii")
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/jpeg"
    return f"data:{mime};base64,{image_b64}"


def _redact_messages_for_trace(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    redacted = deepcopy(messages)
    for msg in redacted:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image_url":
                if "image_url" in part:
                    part["image_url"]["url"] = "redacted"
    return redacted


def _append_trace(entry: Dict[str, Any]) -> None:
    if not state.TRACE_FILE_PATH:
        return
    try:
        with open(state.TRACE_FILE_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _collect_example_tiles(dir_path: str, max_count: int) -> List[str]:
    if not dir_path or not os.path.isdir(dir_path):
        return []
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    paths = []
    for name in os.listdir(dir_path):
        if os.path.splitext(name)[1].lower() in exts:
            paths.append(os.path.join(dir_path, name))
    paths = sorted(paths)
    return paths[:max_count]


def _inject_example_tiles(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if state._example_tiles_injected:
        return messages

    good_paths = _collect_example_tiles(EXAMPLE_TILES_GOOD_DIR, EXAMPLE_TILES_MAX_PER_CLASS)
    bad_paths = _collect_example_tiles(EXAMPLE_TILES_BAD_DIR, EXAMPLE_TILES_MAX_PER_CLASS)

    new_messages = list(messages)
    insert_pos = 0

    if good_paths:
        content = [{"type": "text", "text": "Example GOOD tiles (use as guidance for selection)."}]
        for p in good_paths:
            url = _encode_image_as_data_url(p)
            if url:
                content.append({"type": "image_url", "image_url": {"url": url}})
        new_messages.insert(insert_pos, {"role": "user", "content": content})
        insert_pos += 1

    if bad_paths:
        content = [{"type": "text", "text": "Example BAD tiles (avoid these)."}]
        for p in bad_paths:
            url = _encode_image_as_data_url(p)
            if url:
                content.append({"type": "image_url", "image_url": {"url": url}})
        new_messages.insert(insert_pos, {"role": "user", "content": content})

    print(f"[EXAMPLES] Injected {len(good_paths)} good and {len(bad_paths)} bad tiles.")

    state._example_tiles_injected = True
    return new_messages


def _inject_example_rois(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if state._example_rois_injected:
        return messages

    roi_paths = _collect_example_tiles(EXAMPLE_ROIS_POS_DIR, EXAMPLE_ROIS_MAX_PER_CLASS)
    non_roi_paths = _collect_example_tiles(EXAMPLE_ROIS_NEG_DIR, EXAMPLE_ROIS_MAX_PER_CLASS)

    new_messages = list(messages)
    insert_pos = 0

    if roi_paths:
        content = [{"type": "text", "text": "Example ROI images (diagnostic regions to keep)."}]
        for p in roi_paths:
            url = _encode_image_as_data_url(p)
            if url:
                content.append({"type": "image_url", "image_url": {"url": url}})
        new_messages.insert(insert_pos, {"role": "user", "content": content})
        insert_pos += 1

    if non_roi_paths:
        content = [{"type": "text", "text": "Example NON-ROI images (background/non-diagnostic regions to avoid)."}]
        for p in non_roi_paths:
            url = _encode_image_as_data_url(p)
            if url:
                content.append({"type": "image_url", "image_url": {"url": url}})
        new_messages.insert(insert_pos, {"role": "user", "content": content})

    print(f"[EXAMPLES] Injected {len(roi_paths)} ROI and {len(non_roi_paths)} non-ROI examples.")

    state._example_rois_injected = True
    return new_messages


def _format_field_width_caption(field_width_um: Optional[float]) -> str:
    if field_width_um is None:
        return ""
    approx = int(round(field_width_um / 50.0) * 50)
    return f", field ~{approx} µm wide"


def _inject_wsi_images(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    last_tool_idx = None
    tool_name = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "tool":
            last_tool_idx = i
            tool_name = messages[i].get("name")
            break
    if last_tool_idx is None:
        return messages

    new_messages = list(messages)
    insert_pos = last_tool_idx + 1

    curr_path = state._current_view.get("debug_path") if state._current_view else None
    curr_url = _encode_image_as_data_url(curr_path) if curr_path else None
    if curr_url:
        fw = state._current_view.get("field_width_um")
        extra = _format_field_width_caption(fw)

        if tool_name == "wsi_mark_roi_norm" and state._roi_marks:
            last_roi = state._roi_marks[-1]
            roi_id = last_roi.get("roi_id")
            label = last_roi.get("label", "")
            text = (
                f"CURRENT VIEW = NEWLY MARKED ROI (ROI #{roi_id}: {label}{extra}). "
                "Carefully inspect this high-power field. If it is mostly background or "
                "not diagnostic, your very next action should be to call "
                "wsi_discard_last_roi. All coordinates for any subsequent tool call must "
                "be chosen relative to THIS image."
            )
        else:
            text = (
                f"CURRENT VIEW for navigation{extra}. "
                "All coordinates for your NEXT tool call must be chosen relative to THIS "
                "image. Do NOT select boxes centered on blank/white background; always "
                "place boxes tightly around tissue."
            )

        current_view_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": curr_url}},
            ],
        }
        new_messages.insert(insert_pos, current_view_msg)
        insert_pos += 1

    if state._last_roi_candidates:
        source = state._last_roi_candidate_source or "unknown"
        cand_lines = []
        for c in state._last_roi_candidates[:8]:
            rank = c.get("rank")
            center = c.get("center_norm", [0, 0])
            score = c.get("score")
            score_txt = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
            quality_hint = c.get("quality_hint")
            bad_like = c.get("bad_likelihood")
            extras = []
            if isinstance(quality_hint, str) and quality_hint:
                extras.append(f"hint={quality_hint}")
            if isinstance(bad_like, (int, float)):
                extras.append(f"bad_like={float(bad_like):.2f}")
            suffix = f", {', '.join(extras)}" if extras else ""
            cand_lines.append(f"#{rank}: center=({int(center[0])},{int(center[1])}), score={score_txt}{suffix}")
        aml_meta_line = ""
        meta = state._roi_ranker_meta if isinstance(state._roi_ranker_meta, dict) else {}
        ref_stats = meta.get("reference_stats") if isinstance(meta.get("reference_stats"), dict) else None
        if ref_stats and str(getattr(state, "AGENT_TYPE", "") or "").lower() == "aml":
            mode = ref_stats.get("reference_mode")
            bad_frac = ref_stats.get("wsi_bad_like_fraction")
            strong_bad_frac = ref_stats.get("wsi_bad_like_strong_fraction")
            parts = []
            if mode:
                parts.append(f"mode={mode}")
            if isinstance(bad_frac, (int, float)):
                parts.append(f"bad_like_fraction={float(bad_frac):.2f}")
            if isinstance(strong_bad_frac, (int, float)):
                parts.append(f"strong_bad_like_fraction={float(strong_bad_frac):.2f}")
            if parts:
                aml_meta_line = "\nAML quality summary: " + ", ".join(parts)
        cand_text = (
            "Top ROI candidates for CURRENT VIEW (normalized 0-999 coordinates). "
            f"Candidate source: {source}. "
            "Expected source is 'uni2_knn' from UNI2 tile embeddings + kNN ranking. "
            "For wsi_mark_roi_norm, choose one of these candidate centers/bboxes; arbitrary ROI coords are rejected:\n"
            + "\n".join(cand_lines)
            + aml_meta_line
        )
        candidate_content = [{"type": "text", "text": cand_text}]
        cand_overlay_url = _encode_image_as_data_url(state._last_roi_candidate_overlay_path or "")
        if cand_overlay_url:
            candidate_content.append({"type": "image_url", "image_url": {"url": cand_overlay_url}})
        new_messages.insert(insert_pos, {"role": "user", "content": candidate_content})
        insert_pos += 1

    if state._last_overview_with_box_path:
        url = _encode_image_as_data_url(state._last_overview_with_box_path)
        if url:
            fw = state._current_view.get("field_width_um") if state._current_view else None
            extra = _format_field_width_caption(fw)
            overview_msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Whole-slide overview (red box = current view{extra}).",
                    },
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
            new_messages.insert(insert_pos, overview_msg)
            insert_pos += 1

    for view in state._view_history[-4:]:
        url = _encode_image_as_data_url(view["debug_path"])
        if not url:
            continue
        fw = view.get("field_width_um")
        extra = _format_field_width_caption(fw)
        tag = view.get("tag", "view")
        view_msg = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Previous view ({tag}{extra}).",
                },
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
        new_messages.append(view_msg)

    return new_messages


async def _patched_async_chat_create(*args, **kwargs):
    msgs = kwargs.get("messages")
    if isinstance(msgs, list):
        msgs = _inject_example_rois(msgs)
        msgs = _inject_example_tiles(msgs)
        kwargs["messages"] = _inject_wsi_images(msgs)
    if isinstance(msgs, list):
        _append_trace(
            {
                "type": "request",
                "timestamp": datetime.utcnow().isoformat(),
                "model": kwargs.get("model", MODEL_NAME),
                "messages": _redact_messages_for_trace(msgs),
            }
        )
    resp = await _real_async_chat_create(*args, **kwargs)
    try:
        choice = resp.choices[0] if resp and resp.choices else None
        _append_trace(
            {
                "type": "response",
                "timestamp": datetime.utcnow().isoformat(),
                "model": kwargs.get("model", MODEL_NAME),
                "content": getattr(choice.message, "content", None) if choice else None,
                "reasoning": getattr(choice.message, "reasoning", None) if choice else None,
            }
        )
    except Exception:
        pass
    return resp


def _patched_sync_chat_create(*args, **kwargs):
    msgs = kwargs.get("messages")
    if isinstance(msgs, list):
        msgs = _inject_example_rois(msgs)
        msgs = _inject_example_tiles(msgs)
        kwargs["messages"] = _inject_wsi_images(msgs)
    if isinstance(msgs, list):
        _append_trace(
            {
                "type": "request",
                "timestamp": datetime.utcnow().isoformat(),
                "model": kwargs.get("model", MODEL_NAME),
                "messages": _redact_messages_for_trace(msgs),
            }
        )
    resp = _real_sync_chat_create(*args, **kwargs)
    try:
        choice = resp.choices[0] if resp and resp.choices else None
        _append_trace(
            {
                "type": "response",
                "timestamp": datetime.utcnow().isoformat(),
                "model": kwargs.get("model", MODEL_NAME),
                "content": getattr(choice.message, "content", None) if choice else None,
                "reasoning": getattr(choice.message, "reasoning", None) if choice else None,
            }
        )
    except Exception:
        pass
    return resp


def install_chat_patches() -> None:
    global _patch_installed
    if _patch_installed:
        return
    client_async.chat.completions.create = _patched_async_chat_create
    client_sync.chat.completions.create = _patched_sync_chat_create
    _patch_installed = True
