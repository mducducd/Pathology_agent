import base64
import io
import json
import logging
import mimetypes
import os
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from PIL import Image

from . import state
from .embeddings import embedding_extractor_display_name

_logger = logging.getLogger(__name__)
from .config import (
    CONTEXT_PREVIOUS_VIEWS_MAX,
    CONTEXT_ROI_CANDIDATE_LINES_MAX,
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
from .tuning_config import tuning_value

_real_async_chat_create = client_async.chat.completions.create
_real_sync_chat_create = client_sync.chat.completions.create
_patch_installed = False
_data_url_cache: OrderedDict[tuple[str, int, int], Optional[str]] = OrderedDict()


def _ci_int(key: str, default: int) -> int:
    env = os.getenv(key)
    if env is not None:
        try:
            return int(env)
        except Exception:
            pass
    try:
        return int(tuning_value("context_injection.images", key))
    except Exception:
        return default


CONTEXT_IMAGE_MAX_DIM = _ci_int("CONTEXT_IMAGE_MAX_DIM", 1024)
CONTEXT_IMAGE_JPEG_QUALITY = _ci_int("CONTEXT_IMAGE_JPEG_QUALITY", 55)
CONTEXT_MAX_INLINE_IMAGES = _ci_int("CONTEXT_MAX_INLINE_IMAGES", 6)
CONTEXT_MAX_INLINE_IMAGE_URL_CHARS = _ci_int(
    "CONTEXT_MAX_INLINE_IMAGE_URL_CHARS", 90000
)
_INJECTED_CONTEXT_TAG = "_wsi_context_tag"
_LEGACY_INJECTED_TEXT_PREFIXES = (
    "Example GOOD tiles",
    "Example BAD tiles",
    "Example ROI images",
    "Example NON-ROI images",
    "CURRENT VIEW = NEWLY MARKED ROI",
    "CURRENT VIEW for navigation",
    "AML coverage reminder:",
    "AML efficiency reminder:",
    "Top ROI candidates for CURRENT VIEW",
    "Whole-slide overview",
    "Previous view (",
)


@dataclass
class _InlineImageBudget:
    max_images: int
    max_url_chars: int
    used_images: int = 0
    used_url_chars: int = 0

    def try_take(self, url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        size = len(url)
        if self.used_images >= self.max_images:
            return None
        if self.used_url_chars + size > self.max_url_chars:
            return None
        self.used_images += 1
        self.used_url_chars += size
        return url


def _agent_type() -> str:
    return str(getattr(state, "AGENT_TYPE", "") or "").lower()


def _selected_extractor_name() -> str:
    return str(getattr(state, "EXTRACTOR_NAME", "uni2") or "uni2").strip().lower()


def _selected_candidate_nav_field_um() -> float:
    try:
        override = getattr(state, "CANDIDATE_NAV_FIELD_UM_OVERRIDE", None)
        if override is not None:
            return max(100.0, float(override))
        return max(
            100.0, float(tuning_value("tools.navigation", "CANDIDATE_NAV_FIELD_UM"))
        )
    except Exception:
        return 1200.0


def _encode_image_as_data_url(
    path: str,
    *,
    max_dim: int = CONTEXT_IMAGE_MAX_DIM,
    jpeg_quality: int = CONTEXT_IMAGE_JPEG_QUALITY,
) -> Optional[str]:
    if not path:
        return None
    key = (path, int(max_dim), int(jpeg_quality))
    cached = _data_url_cache.get(key)
    if cached is not None:
        return cached
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if max_dim > 0:
                resampling = getattr(
                    getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS
                )
                img.thumbnail((max_dim, max_dim), resampling)
            buf = io.BytesIO()
            img.save(
                buf,
                format="JPEG",
                quality=max(20, min(95, int(jpeg_quality))),
                optimize=True,
            )
            img_bytes = buf.getvalue()
            mime = "image/jpeg"
    except Exception:
        try:
            with open(path, "rb") as f:
                img_bytes = f.read()
            mime, _ = mimetypes.guess_type(path)
            if not mime:
                mime = "image/jpeg"
        except Exception:
            return None
    image_b64 = base64.b64encode(img_bytes).decode("ascii")
    url = f"data:{mime};base64,{image_b64}"
    if key in _data_url_cache:
        _data_url_cache.move_to_end(key)
    else:
        if len(_data_url_cache) >= 512:
            _data_url_cache.popitem(last=False)
        _data_url_cache[key] = url
    return url


def _first_text_part(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str):
                return text
    return ""


def _is_legacy_injected_context_message(message: Dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    first_text = _first_text_part(message)
    return any(
        first_text.startswith(prefix) for prefix in _LEGACY_INJECTED_TEXT_PREFIXES
    )


def _strip_injected_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get(_INJECTED_CONTEXT_TAG):
            continue
        if _is_legacy_injected_context_message(message):
            continue
        cleaned.append(message)
    return cleaned


def _sanitize_messages_for_api(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean_messages: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        msg = {k: v for k, v in message.items() if k != _INJECTED_CONTEXT_TAG}
        content = msg.get("content")
        if isinstance(content, list):
            clean_parts = []
            for part in content:
                if isinstance(part, dict):
                    clean_parts.append(
                        {k: v for k, v in part.items() if not str(k).startswith("_")}
                    )
                else:
                    clean_parts.append(part)
            msg["content"] = clean_parts
        clean_messages.append(msg)
    return clean_messages


def _tag_context_message(message: Dict[str, Any], tag: str) -> Dict[str, Any]:
    tagged = dict(message)
    tagged[_INJECTED_CONTEXT_TAG] = tag
    return tagged


def _make_image_part(
    path: str, budget: Optional[_InlineImageBudget]
) -> Optional[Dict[str, Any]]:
    url = _encode_image_as_data_url(path)
    if budget is not None:
        url = budget.try_take(url)
    if not url:
        return None
    return {"type": "image_url", "image_url": {"url": url}}


def _prepare_messages_for_request(
    messages: List[Dict[str, Any]], *, minimal: bool = False
) -> List[Dict[str, Any]]:
    msgs = _strip_injected_messages(messages)

    # Diagnosis-only mode receives explicit ROI images from runtime.
    # Do not inject current view, candidates, overview, previous views, or examples.
    if _agent_type() == "aml_diagnosis":
        return _sanitize_messages_for_api(msgs)

    max_images = 1 if minimal else CONTEXT_MAX_INLINE_IMAGES
    max_url_chars = (
        max(18000, CONTEXT_MAX_INLINE_IMAGE_URL_CHARS // 3)
        if minimal
        else CONTEXT_MAX_INLINE_IMAGE_URL_CHARS
    )
    budget = _InlineImageBudget(max_images=max_images, max_url_chars=max_url_chars)

    msgs = _inject_wsi_images(
        msgs,
        budget=budget,
        include_candidate_overlay=not minimal,
        include_overview=not minimal,
        include_previous_views=not minimal,
    )

    include_examples = not minimal and len(getattr(state, "_roi_marks", []) or []) == 0
    if include_examples:
        msgs = _inject_example_rois(msgs, budget=budget)
        msgs = _inject_example_tiles(msgs, budget=budget)

    return _sanitize_messages_for_api(msgs)


def _is_context_length_error(exc: Exception) -> bool:
    text = str(exc)
    return (
        "maximum context length" in text
        or "exceeds model's maximum context length" in text
        or ("Input length" in text and "context length" in text)
    )


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


def _inject_example_tiles(
    messages: List[Dict[str, Any]],
    *,
    budget: Optional[_InlineImageBudget] = None,
) -> List[Dict[str, Any]]:
    if state._example_tiles_injected:
        return messages

    good_paths = _collect_example_tiles(
        EXAMPLE_TILES_GOOD_DIR, EXAMPLE_TILES_MAX_PER_CLASS
    )
    bad_paths = _collect_example_tiles(
        EXAMPLE_TILES_BAD_DIR, EXAMPLE_TILES_MAX_PER_CLASS
    )

    new_messages = list(messages)
    insert_pos = 0

    if good_paths:
        content = [
            {
                "type": "text",
                "text": (
                    "Example GOOD tiles (high-quality diagnostic ROI examples, not AML-vs-normal labels). "
                    "Use them as ROI-quality references for tissue vs background, nucleated-cell richness, focus, and artifact rejection. "
                    "Good AML/marrow tiles are hypercellular, deep blue-purple/basophilic, nucleated, in focus, low artifact, and morphologically informative."
                ),
            }
        ]
        for p in good_paths:
            image_part = _make_image_part(p, budget)
            if image_part:
                content.append(image_part)
        new_messages.insert(
            insert_pos,
            _tag_context_message(
                {"role": "user", "content": content}, "example_good_tiles"
            ),
        )
        insert_pos += 1

    if bad_paths:
        content = [
            {
                "type": "text",
                "text": (
                    "Example BAD tiles (low-quality/non-diagnostic ROI examples, not AML-vs-normal labels). "
                    "Use them to recognize empty/background-heavy, RBC/clot-dominant, gray-black low-chroma junk, artifact-dark, blurred, crushed, or non-representative edge/debris regions."
                ),
            }
        ]
        for p in bad_paths:
            image_part = _make_image_part(p, budget)
            if image_part:
                content.append(image_part)
        new_messages.insert(
            insert_pos,
            _tag_context_message(
                {"role": "user", "content": content}, "example_bad_tiles"
            ),
        )

    _logger.info(
        "[EXAMPLES] Injected %d good and %d bad tiles.", len(good_paths), len(bad_paths)
    )

    state._example_tiles_injected = True
    return new_messages


def _inject_example_rois(
    messages: List[Dict[str, Any]],
    *,
    budget: Optional[_InlineImageBudget] = None,
) -> List[Dict[str, Any]]:
    if state._example_rois_injected:
        return messages

    roi_paths = _collect_example_tiles(EXAMPLE_ROIS_POS_DIR, EXAMPLE_ROIS_MAX_PER_CLASS)
    non_roi_paths = _collect_example_tiles(
        EXAMPLE_ROIS_NEG_DIR, EXAMPLE_ROIS_MAX_PER_CLASS
    )

    new_messages = list(messages)
    insert_pos = 0

    if roi_paths:
        content = [
            {
                "type": "text",
                "text": (
                    "Example ROI images (diagnostic regions to keep). "
                    "Good AML ROIs are hypercellular, deep blue-purple/basophilic, blast-suspected, in focus, low artifact, and representative. "
                    "These examples help find visually informative ROIs, not prove AML by themselves."
                ),
            }
        ]
        for p in roi_paths:
            image_part = _make_image_part(p, budget)
            if image_part:
                content.append(image_part)
        new_messages.insert(
            insert_pos,
            _tag_context_message(
                {"role": "user", "content": content}, "example_roi_tiles"
            ),
        )
        insert_pos += 1

    if non_roi_paths:
        content = [
            {
                "type": "text",
                "text": (
                    "Example NON-ROI images (background/non-diagnostic regions to avoid). "
                    "Avoid empty, RBC-heavy, gray-black low-chroma junk, artifact-dark, blurred, crushed, or edge/debris dominated regions."
                ),
            }
        ]
        for p in non_roi_paths:
            image_part = _make_image_part(p, budget)
            if image_part:
                content.append(image_part)
        new_messages.insert(
            insert_pos,
            _tag_context_message(
                {"role": "user", "content": content}, "example_non_roi_tiles"
            ),
        )

    _logger.info(
        "[EXAMPLES] Injected %d ROI and %d non-ROI examples.",
        len(roi_paths),
        len(non_roi_paths),
    )

    state._example_rois_injected = True
    return new_messages


def _format_field_width_caption(field_width_um: Optional[float]) -> str:
    if field_width_um is None:
        return ""
    approx = int(round(field_width_um / 50.0) * 50)
    return f", field ~{approx} µm wide"


def _last_tool_info(
    messages: List[Dict[str, Any]],
) -> tuple[Optional[int], Optional[str]]:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "tool":
            return i, messages[i].get("name")
    return None, None


def _insert_context_message(
    messages: List[Dict[str, Any]],
    insert_pos: int,
    tag: str,
    content: List[Dict[str, Any]],
) -> int:
    messages.insert(
        insert_pos,
        _tag_context_message({"role": "user", "content": content}, tag),
    )
    return insert_pos + 1


def _aml_roi_counts() -> tuple[int, int, int]:
    kept_roi_count = len(state._roi_marks)
    max_accepted_rois = max(1, int(getattr(state, "MAX_ACCEPTED_ROIS", 10) or 10))
    target_roi_count = min(
        max_accepted_rois,
        max(1, int(getattr(state, "TARGET_ACCEPTED_ROIS", 5) or 5)),
    )
    return kept_roi_count, max_accepted_rois, target_roi_count


def _build_current_view_message(
    budget: Optional[_InlineImageBudget],
) -> Optional[Dict[str, Any]]:
    curr_path = state._current_view.get("debug_path") if state._current_view else None
    current_view_part = _make_image_part(curr_path or "", budget) if curr_path else None
    if not current_view_part:
        return None

    fw = state._current_view.get("field_width_um")
    extra = _format_field_width_caption(fw)
    kept_roi_count, max_accepted_rois, target_roi_count = _aml_roi_counts()
    _aml_at_target = _agent_type() == "aml" and kept_roi_count >= min(
        max_accepted_rois, target_roi_count
    )
    if _aml_at_target:
        text = (
            f"CURRENT VIEW{extra}. "
            "ROI target is reached — do NOT call any more navigation or marking tools. "
            "Write the final JSON output now based on the kept ROIs shown above.\n"
            + AML_FINAL_REVIEW_PROMPT
        )
    else:
        text = (
            f"CURRENT VIEW for navigation{extra}. "
            "All coordinates for NEXT tool call must be chosen relative to THIS image. "
            "PRIORITY: Look for regions with high cellularity (dense packed nucleated cells) and clear blast visibility. "
            "Once you find high-cellularity tissue with readable morphology, mark it with wsi_mark_roi_norm. "
            "Do NOT select boxes centered on blank/white background; always place boxes tightly around tissue and high-cellularity areas. "
            "Do not search indefinitely for marginal improvements. Mark if tissue is readable and diagnostic, even if not the single densest field. "
            "Avoid panning to empty areas."
        )

    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            current_view_part,
        ],
    }


def _aml_stop_lines(
    kept_roi_count: int,
    max_accepted_rois: int,
    target_roi_count: int,
) -> List[str]:
    if kept_roi_count < target_roi_count:
        return [
            f"- Current progress: {kept_roi_count}/{target_roi_count} ROIs marked.",
            f"Keep searching for additional distinct AML ROIs to reach the {target_roi_count} ROI target.",
        ]
    if kept_roi_count >= max_accepted_rois:
        return [
            f"- Hard cap reached: {kept_roi_count}/{max_accepted_rois} kept ROI(s). Provide final AML diagnosis.",
            AML_FINAL_REVIEW_PROMPT,
        ]
    return [
        f"- ROI target reached: {kept_roi_count}/{target_roi_count} kept ROI(s). Provide AML blast estimate and diagnosis.",
        AML_FINAL_REVIEW_PROMPT,
    ]


def _append_marked_roi_overview(
    content: List[Dict[str, Any]],
    budget: Optional[_InlineImageBudget],
) -> None:
    if not state._last_overview_with_box_path:
        return
    overview_img = _make_image_part(state._last_overview_with_box_path, budget)
    if overview_img:
        content.append(
            {
                "type": "text",
                "text": "Whole-slide overview (all marked ROI positions visible).",
            }
        )
        content.append(overview_img)


def _build_aml_guidance_content(
    budget: Optional[_InlineImageBudget],
) -> tuple[List[Dict[str, Any]], int, int, int]:
    kept_roi_count, max_accepted_rois, target_roi_count = _aml_roi_counts()
    aml_stop_lines = _aml_stop_lines(
        kept_roi_count,
        max_accepted_rois,
        target_roi_count,
    )
    state.CURRENT_AGENT_ACTION = "\n".join(aml_stop_lines)
    aml_guidance_content: List[Dict[str, Any]] = [
        {"type": "text", "text": "\n".join(aml_stop_lines)}
    ]
    if kept_roi_count >= target_roi_count and state._last_overview_with_box_path:
        _append_marked_roi_overview(aml_guidance_content, budget)
    return aml_guidance_content, kept_roi_count, max_accepted_rois, target_roi_count


def _build_aml_all_rois_review_content(
    budget: Optional[_InlineImageBudget],
) -> List[Dict[str, Any]]:
    all_roi_parts: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Target ROI count reached. Review ALL kept ROIs below and write the final JSON output. "
                "For EACH ROI, first classify the FIELD TYPE: HYPERCELLULAR PACKED MARROW / NORMOCELLULAR / HYPOCELLULAR-FATTY / HEMODILUTE-RBC-DOMINANT / SMEAR-EDGE-SERUM / ARTIFACT. "
                "Only HYPERCELLULAR PACKED MARROW with monotonous immature cells can support an AML call. "
                "Hypocellular/fatty, hemodilute (sea of pink-red donut RBCs), smear-edge/serum (tan-brown background with scattered cells), and artifact-dominated fields MUST report blast_range <5%. "
                "Do NOT label any non-hypercellular field as 20-50% or >50% blasts. "
                'If MOST kept ROIs are NOT hypercellular packed marrow, final_decision MUST be "Normal marrow" with explicit limitation noted. '
                'Default bias: when uncertain, choose "Normal marrow".'
            ),
        }
    ]
    _append_marked_roi_overview(all_roi_parts, budget)
    for roi in state._roi_marks:
        r_id = roi.get("roi_id", "?")
        r_label = roi.get("label", "")
        all_roi_parts.append({"type": "text", "text": f"ROI #{r_id}: {r_label}"})
        img_part = _make_image_part(roi.get("debug_path", ""), budget)
        if img_part:
            all_roi_parts.append(img_part)
    return all_roi_parts


def _candidate_injection_context(tool_name: Optional[str]) -> tuple[bool, bool]:
    latest_roi_debug_path = ""
    if state._roi_marks:
        latest_roi_debug_path = str(state._roi_marks[-1].get("debug_path") or "")
    current_view_debug_path = (
        str(state._current_view.get("debug_path") or "") if state._current_view else ""
    )
    current_view_nav_mode = (
        str(state._current_view.get("candidate_navigation_mode") or "")
        if state._current_view
        else ""
    )
    mark_tool_names = {"wsi_mark_roi_norm", "wsi_mark_candidate"}
    suppress_candidates_for_free_local_search = (
        current_view_nav_mode == "free_local_search"
    )
    kept_roi_count_for_candidates = len(state._roi_marks)
    target_accepted_rois_for_candidates = max(
        1, int(getattr(state, "TARGET_ACCEPTED_ROIS", 5) or 5)
    )
    at_or_past_target = (
        kept_roi_count_for_candidates >= target_accepted_rois_for_candidates
    )
    should_include_current_candidates = bool(state._last_roi_candidates) and (
        not suppress_candidates_for_free_local_search
        and not at_or_past_target
        and (
            tool_name not in mark_tool_names
            or current_view_debug_path != latest_roi_debug_path
        )
    )
    return suppress_candidates_for_free_local_search, should_include_current_candidates


def _free_local_search_content() -> List[Dict[str, Any]]:
    free_search_text = (
        "You are now inside a suggested marrow search region. "
        "Treat the CURRENT VIEW as a search area for readable morphology, not as proof of AML. "
        "Look for interpretable marrow with preserved single-cell detail, adequate nucleated cells, acceptable focus, and limited artifact. "
        "Prefer representative regions, including areas with maturation if present. "
        "Avoid empty/pale areas, severely RBC-dominant regions, heavy stain pooling, clot/crush artifact, and blurred or unreadable zones. "
        "Call wsi_mark_roi_norm only after identifying readable morphology; otherwise keep exploring within this field or skip to another region."
    )
    return [{"type": "text", "text": free_search_text}]


def _candidate_lines() -> List[str]:
    cand_lines = []
    for c in state._last_roi_candidates[: max(1, CONTEXT_ROI_CANDIDATE_LINES_MAX)]:
        rank = c.get("rank")
        score = c.get("score")
        score_txt = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
        quality_hint = c.get("quality_hint")
        hint_txt = (
            f" hint={quality_hint}"
            if isinstance(quality_hint, str) and quality_hint
            else ""
        )
        cand_lines.append(f"#{rank}: score={score_txt}{hint_txt}")
    return cand_lines


def _aml_candidate_meta_line() -> str:
    aml_meta_line = ""
    meta = state._roi_ranker_meta if isinstance(state._roi_ranker_meta, dict) else {}
    ref_stats = (
        meta.get("reference_stats")
        if isinstance(meta.get("reference_stats"), dict)
        else None
    )
    if ref_stats and _agent_type() == "aml":
        mode = ref_stats.get("reference_mode")
        bad_refs_enabled = bool(ref_stats.get("bad_references_enabled", True))
        bad_frac = ref_stats.get("wsi_bad_like_fraction")
        strong_bad_frac = ref_stats.get("wsi_bad_like_strong_fraction")
        ref_k = ref_stats.get("reference_neighbor_k")
        similarity = ref_stats.get("reference_similarity")
        parts = []
        if mode:
            parts.append(f"mode={mode}")
        if similarity:
            parts.append(f"similarity={similarity}")
        if isinstance(ref_k, int):
            parts.append(f"ref_k={ref_k}")
        if bad_refs_enabled and isinstance(bad_frac, (int, float)):
            parts.append(f"bad_like_fraction={float(bad_frac):.2f}")
        if bad_refs_enabled and isinstance(strong_bad_frac, (int, float)):
            parts.append(f"strong_bad_like_fraction={float(strong_bad_frac):.2f}")
        if parts:
            aml_meta_line = "\nAML quality summary: " + ", ".join(parts)
    return aml_meta_line


def _candidate_expected_source() -> tuple[str, str]:
    extractor_name = _selected_extractor_name()
    extractor_label = embedding_extractor_display_name(extractor_name)
    expected_source = (
        f"{extractor_label} tile embeddings + exact curated good-reference retrieval with morphology-aware ranking."
        if _agent_type() == "aml"
        else f"{extractor_label} tile embeddings + kNN ranking."
    )
    expected_source_name = (
        f"{extractor_name}_exact_retrieval"
        if _agent_type() == "aml"
        else f"{extractor_name}_knn"
    )
    return expected_source, expected_source_name


def _build_candidate_content(
    budget: Optional[_InlineImageBudget],
    include_candidate_overlay: bool,
) -> List[Dict[str, Any]]:
    source = state._last_roi_candidate_source or "unknown"
    cand_lines = _candidate_lines()
    aml_meta_line = _aml_candidate_meta_line()
    expected_source, expected_source_name = _candidate_expected_source()
    cand_text = (
        "Top ROI candidates for the CURRENT VIEW. "
        f"Candidate source: {source}. "
        f"Expected source: '{expected_source_name}' from {expected_source}. "
        "IMPORTANT: If the CURRENT VIEW shows high cellularity with good blast visibility or abundant nucleated cells, STAY IN THIS REGION and zoom to find the best local ROI—do not jump to other candidates. Only jump to a different candidate if the current region is clearly unsuitable (mostly empty, severe artifact, etc.). "
        "When zooming within the current high-cellularity field, look for areas with the clearest single-cell morphology and best blast visibility. "
        "To navigate to a different candidate, use wsi_open_candidate(rank) to jump into an approximately "
        + str(int(round(_selected_candidate_nav_field_um())))
        + " um field around that candidate. "
        "Treat quality_hint only as supportive reference evidence; it does not guarantee cellularity or interpretability. "
        "Prefer representative, interpretable marrow patches with readable single-cell morphology, abundant nucleated cells, acceptable focus, and limited artifact. "
        "High cellularity (dense packed nucleated cells) is a strong signal of good diagnostic potential—prioritize these regions. "
        "A partial but clearly usable cellular area is acceptable; broad full-field cellularity is not required, and high local blast concentration can be diagnostic. "
        "Avoid heavy stain pooling, dark blue clot-like material, stringy smear artifact, gray-black debris, and nearly acellular regions. "
        "Some empty/vacuolated space is acceptable if a nearby local ROI is still clearly usable for rough blast estimation. "
        "Use ranked candidates as region-level guidance only; choose the final ROI box based on the best local morphology:\n"
        + "\n".join(cand_lines)
        + aml_meta_line
    )
    candidate_content = [{"type": "text", "text": cand_text}]
    if include_candidate_overlay and _agent_type() != "aml":
        overlay_part = _make_image_part(
            state._last_roi_candidate_overlay_path or "", budget
        )
        if overlay_part:
            candidate_content.append(overlay_part)
    return candidate_content


def _build_overview_message(
    budget: Optional[_InlineImageBudget],
) -> Optional[Dict[str, Any]]:
    overview_part = _make_image_part(state._last_overview_with_box_path, budget)
    if not overview_part:
        return None
    fw = state._current_view.get("field_width_um") if state._current_view else None
    extra = _format_field_width_caption(fw)
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Whole-slide overview (red box = current view{extra}).",
            },
            overview_part,
        ],
    }


def _previous_view_messages(
    budget: Optional[_InlineImageBudget],
    prev_view_limit: int,
) -> List[Dict[str, Any]]:
    previous_messages = []
    for view in state._view_history[-prev_view_limit:]:
        image_part = _make_image_part(view["debug_path"], budget)
        if not image_part:
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
                image_part,
            ],
        }
        previous_messages.append(_tag_context_message(view_msg, "previous_view"))
    return previous_messages


def _inject_wsi_images(
    messages: List[Dict[str, Any]],
    *,
    budget: Optional[_InlineImageBudget] = None,
    include_candidate_overlay: bool = True,
    include_overview: bool = True,
    include_previous_views: bool = True,
) -> List[Dict[str, Any]]:
    last_tool_idx, tool_name = _last_tool_info(messages)
    if last_tool_idx is None:
        return messages

    new_messages = list(messages)
    insert_pos = last_tool_idx + 1

    current_view_msg = _build_current_view_message(budget)
    if current_view_msg:
        new_messages.insert(
            insert_pos, _tag_context_message(current_view_msg, "current_view")
        )
        insert_pos += 1

    if _agent_type() == "aml":
        (
            aml_guidance_content,
            kept_roi_count,
            _max_accepted_rois,
            target_roi_count,
        ) = _build_aml_guidance_content(budget)
        insert_pos = _insert_context_message(
            new_messages,
            insert_pos,
            "aml_guidance",
            aml_guidance_content,
        )

        # Inject all kept ROI images only on the turn the target is first reached
        # (i.e., when the last tool call was wsi_mark_roi_norm and we just hit the target).
        if (
            tool_name == "wsi_mark_roi_norm"
            and kept_roi_count >= target_roi_count
            and state._roi_marks
        ):
            all_roi_parts = _build_aml_all_rois_review_content(budget)
            if len(all_roi_parts) > 1:
                insert_pos = _insert_context_message(
                    new_messages,
                    insert_pos,
                    "aml_all_rois_review",
                    all_roi_parts,
                )

    (
        suppress_candidates_for_free_local_search,
        should_include_current_candidates,
    ) = _candidate_injection_context(tool_name)

    if suppress_candidates_for_free_local_search and _agent_type() == "aml":
        insert_pos = _insert_context_message(
            new_messages,
            insert_pos,
            "aml_free_local_search",
            _free_local_search_content(),
        )

    if should_include_current_candidates:
        insert_pos = _insert_context_message(
            new_messages,
            insert_pos,
            "roi_candidates",
            _build_candidate_content(budget, include_candidate_overlay),
        )

    if (
        include_overview
        and state._last_overview_with_box_path
        and not suppress_candidates_for_free_local_search
    ):
        overview_msg = _build_overview_message(budget)
        if overview_msg:
            new_messages.insert(
                insert_pos, _tag_context_message(overview_msg, "overview")
            )
            insert_pos += 1

    prev_view_limit = max(
        0, CONTEXT_PREVIOUS_VIEWS_MAX if include_previous_views else 0
    )
    if prev_view_limit:
        new_messages.extend(_previous_view_messages(budget, prev_view_limit))

    return new_messages


def _make_chat_request(create_fn, messages, model_name, exc_context):
    """Prepare messages, handle context length errors, and trace requests."""
    original_msgs = messages
    prepared_msgs = None
    if isinstance(original_msgs, list):
        prepared_msgs = _prepare_messages_for_request(original_msgs, minimal=False)
        exc_context["messages"] = prepared_msgs
        _append_trace(
            {
                "type": "request",
                "timestamp": datetime.utcnow().isoformat(),
                "model": model_name,
                "messages": _redact_messages_for_trace(prepared_msgs),
            }
        )
    return prepared_msgs, original_msgs


def _handle_context_length_error(
    exc: Exception,
    original_msgs: Optional[List[Dict[str, Any]]],
    model_name: str,
) -> bool:
    """Check if exception is a context length error and prepare fallback messages."""
    if not (isinstance(original_msgs, list) and _is_context_length_error(exc)):
        return False
    return True


def _trace_response(resp: Any, model_name: str) -> None:
    """Trace the response content and reasoning."""
    try:
        choice = resp.choices[0] if resp and resp.choices else None
        _append_trace(
            {
                "type": "response",
                "timestamp": datetime.utcnow().isoformat(),
                "model": model_name,
                "content": getattr(choice.message, "content", None) if choice else None,
                "reasoning": (
                    getattr(choice.message, "reasoning", None) if choice else None
                ),
            }
        )
    except Exception:
        pass


async def _patched_async_chat_create(*args, **kwargs):
    model_name = kwargs.get("model", MODEL_NAME)
    prepared_msgs, original_msgs = _make_chat_request(
        _real_async_chat_create,
        kwargs.get("messages"),
        model_name,
        {},
    )

    if prepared_msgs is not None:
        kwargs["messages"] = prepared_msgs

    try:
        resp = await _real_async_chat_create(*args, **kwargs)
    except Exception as exc:
        if not _handle_context_length_error(exc, original_msgs, model_name):
            raise
        fallback_msgs = _prepare_messages_for_request(original_msgs, minimal=True)
        kwargs["messages"] = fallback_msgs
        resp = await _real_async_chat_create(*args, **kwargs)

    _trace_response(resp, model_name)
    return resp


def _patched_sync_chat_create(*args, **kwargs):
    model_name = kwargs.get("model", MODEL_NAME)
    prepared_msgs, original_msgs = _make_chat_request(
        _real_sync_chat_create,
        kwargs.get("messages"),
        model_name,
        {},
    )

    if prepared_msgs is not None:
        kwargs["messages"] = prepared_msgs

    try:
        resp = _real_sync_chat_create(*args, **kwargs)
    except Exception as exc:
        if not _handle_context_length_error(exc, original_msgs, model_name):
            raise
        fallback_msgs = _prepare_messages_for_request(original_msgs, minimal=True)
        kwargs["messages"] = fallback_msgs
        resp = _real_sync_chat_create(*args, **kwargs)

    _trace_response(resp, model_name)
    return resp


def install_chat_patches() -> None:
    global _patch_installed
    if _patch_installed:
        return
    client_async.chat.completions.create = _patched_async_chat_create
    client_sync.chat.completions.create = _patched_sync_chat_create
    _patch_installed = True


AML_FINAL_REVIEW_PROMPT = """
Target ROI count reached. Review ALL kept ROIs below before writing the final JSON.
"""
