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

_real_async_chat_create = client_async.chat.completions.create
_real_sync_chat_create = client_sync.chat.completions.create
_patch_installed = False
_data_url_cache: OrderedDict[tuple[str, int, int], Optional[str]] = OrderedDict()

CONTEXT_IMAGE_MAX_DIM = int(os.getenv("CONTEXT_IMAGE_MAX_DIM", "256"))
CONTEXT_IMAGE_JPEG_QUALITY = int(os.getenv("CONTEXT_IMAGE_JPEG_QUALITY", "55"))
CONTEXT_MAX_INLINE_IMAGES = int(os.getenv("CONTEXT_MAX_INLINE_IMAGES", "6"))
CONTEXT_MAX_INLINE_IMAGE_URL_CHARS = int(os.getenv("CONTEXT_MAX_INLINE_IMAGE_URL_CHARS", "90000"))
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
                resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
                img.thumbnail((max_dim, max_dim), resampling)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=max(20, min(95, int(jpeg_quality))), optimize=True)
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
    return any(first_text.startswith(prefix) for prefix in _LEGACY_INJECTED_TEXT_PREFIXES)


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
                    clean_parts.append({k: v for k, v in part.items() if not str(k).startswith("_")})
                else:
                    clean_parts.append(part)
            msg["content"] = clean_parts
        clean_messages.append(msg)
    return clean_messages


def _tag_context_message(message: Dict[str, Any], tag: str) -> Dict[str, Any]:
    tagged = dict(message)
    tagged[_INJECTED_CONTEXT_TAG] = tag
    return tagged


def _make_image_part(path: str, budget: Optional[_InlineImageBudget]) -> Optional[Dict[str, Any]]:
    url = _encode_image_as_data_url(path)
    if budget is not None:
        url = budget.try_take(url)
    if not url:
        return None
    return {"type": "image_url", "image_url": {"url": url}}


def _prepare_messages_for_request(messages: List[Dict[str, Any]], *, minimal: bool = False) -> List[Dict[str, Any]]:
    msgs = _strip_injected_messages(messages)
    max_images = 1 if minimal else CONTEXT_MAX_INLINE_IMAGES
    max_url_chars = max(18000, CONTEXT_MAX_INLINE_IMAGE_URL_CHARS // 3) if minimal else CONTEXT_MAX_INLINE_IMAGE_URL_CHARS
    budget = _InlineImageBudget(max_images=max_images, max_url_chars=max_url_chars)
    msgs = _inject_wsi_images(
        msgs,
        budget=budget,
        include_candidate_overlay=not minimal,
        include_overview=not minimal,
        include_previous_views=not minimal,
    )
    if not minimal:
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

    good_paths = _collect_example_tiles(EXAMPLE_TILES_GOOD_DIR, EXAMPLE_TILES_MAX_PER_CLASS)
    bad_paths = _collect_example_tiles(EXAMPLE_TILES_BAD_DIR, EXAMPLE_TILES_MAX_PER_CLASS)

    new_messages = list(messages)
    insert_pos = 0

    if good_paths:
        content = [{
            "type": "text",
            "text": (
                "Example GOOD tiles (high-quality diagnostic ROI examples, not AML-vs-normal labels). "
                "Use them as ROI-quality references for tissue vs background, nucleated-cell richness, focus, and artifact rejection. "
                "Good AML/marrow tiles are hypercellular, deep blue-purple/basophilic, nucleated, in focus, low artifact, and morphologically informative."
            ),
        }]
        for p in good_paths:
            image_part = _make_image_part(p, budget)
            if image_part:
                content.append(image_part)
        new_messages.insert(insert_pos, _tag_context_message({"role": "user", "content": content}, "example_good_tiles"))
        insert_pos += 1

    if bad_paths:
        content = [{
            "type": "text",
            "text": (
                "Example BAD tiles (low-quality/non-diagnostic ROI examples, not AML-vs-normal labels). "
                "Use them to recognize empty/background-heavy, RBC/clot-dominant, gray-black low-chroma junk, artifact-dark, blurred, crushed, or non-representative edge/debris regions."
            ),
        }]
        for p in bad_paths:
            image_part = _make_image_part(p, budget)
            if image_part:
                content.append(image_part)
        new_messages.insert(insert_pos, _tag_context_message({"role": "user", "content": content}, "example_bad_tiles"))

    _logger.info("[EXAMPLES] Injected %d good and %d bad tiles.", len(good_paths), len(bad_paths))

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
    non_roi_paths = _collect_example_tiles(EXAMPLE_ROIS_NEG_DIR, EXAMPLE_ROIS_MAX_PER_CLASS)

    new_messages = list(messages)
    insert_pos = 0

    if roi_paths:
        content = [{
            "type": "text",
            "text": (
                "Example ROI images (diagnostic regions to keep). "
                "Good AML ROIs are hypercellular, deep blue-purple/basophilic, blast-suspected, in focus, low artifact, and representative. "
                "These examples help you find visually informative ROIs, not prove AML by themselves."
            ),
        }]
        for p in roi_paths:
            image_part = _make_image_part(p, budget)
            if image_part:
                content.append(image_part)
        new_messages.insert(insert_pos, _tag_context_message({"role": "user", "content": content}, "example_roi_tiles"))
        insert_pos += 1

    if non_roi_paths:
        content = [{
            "type": "text",
            "text": (
                "Example NON-ROI images (background/non-diagnostic regions to avoid). "
                "Avoid empty, RBC-heavy, gray-black low-chroma junk, artifact-dark, blurred, crushed, or edge/debris dominated regions."
            ),
        }]
        for p in non_roi_paths:
            image_part = _make_image_part(p, budget)
            if image_part:
                content.append(image_part)
        new_messages.insert(insert_pos, _tag_context_message({"role": "user", "content": content}, "example_non_roi_tiles"))

    _logger.info("[EXAMPLES] Injected %d ROI and %d non-ROI examples.", len(roi_paths), len(non_roi_paths))

    state._example_rois_injected = True
    return new_messages


def _format_field_width_caption(field_width_um: Optional[float]) -> str:
    if field_width_um is None:
        return ""
    approx = int(round(field_width_um / 50.0) * 50)
    return f", field ~{approx} µm wide"


def _inject_wsi_images(
    messages: List[Dict[str, Any]],
    *,
    budget: Optional[_InlineImageBudget] = None,
    include_candidate_overlay: bool = True,
    include_overview: bool = True,
    include_previous_views: bool = True,
) -> List[Dict[str, Any]]:
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

    if tool_name in {"wsi_mark_roi_norm", "wsi_mark_candidate"} and state._roi_marks:
        last_roi = state._roi_marks[-1]
        marked_roi_path = last_roi.get("debug_path")
        marked_roi_part = _make_image_part(marked_roi_path or "", budget) if marked_roi_path else None
        if marked_roi_part:
            roi_id = last_roi.get("roi_id")
            label = last_roi.get("label", "")
            roi_fw = last_roi.get("field_width_um")
            roi_extra = _format_field_width_caption(roi_fw)
            ref_evidence = last_roi.get("aml_reference_evidence") if isinstance(last_roi.get("aml_reference_evidence"), dict) else None
            ref_extra = ""
            if _agent_type() == "aml" and ref_evidence:
                summary = ref_evidence.get("summary")
                if isinstance(summary, str) and summary:
                    ref_extra = f" Retrieval evidence for this ROI: {summary}."
            next_rank_hint = last_roi.get("next_candidate_rank_hint")
            next_jump_text = (
                f" If you still need another ROI after that decision, continue with candidate #{int(next_rank_hint)} next."
                if isinstance(next_rank_hint, int)
                else " If you still need another ROI after that decision, jump directly to the next unvisited candidate."
            )
            text = (
                f"NEWLY MARKED ROI (ROI #{roi_id}: {label}{roi_extra}). "
                "This ROI has been kept as evidence. If it is actually mostly background or not diagnostic on review, "
                "your very next action should be to call wsi_discard_last_roi. "
                + next_jump_text
                + " The system may already have advanced CURRENT VIEW to the next candidate for navigation, "
                "so do not use this ROI image for coordinate selection."
                + ref_extra
            )
            marked_roi_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    marked_roi_part,
                ],
            }
            new_messages.insert(insert_pos, _tag_context_message(marked_roi_msg, "latest_marked_roi"))
            insert_pos += 1

    curr_path = state._current_view.get("debug_path") if state._current_view else None
    current_view_part = _make_image_part(curr_path or "", budget) if curr_path else None
    if current_view_part:
        fw = state._current_view.get("field_width_um")
        extra = _format_field_width_caption(fw)
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
                current_view_part,
            ],
        }
        new_messages.insert(insert_pos, _tag_context_message(current_view_msg, "current_view"))
        insert_pos += 1

    if _agent_type() == "aml":
        kept_roi_count = len(state._roi_marks)
        max_accepted_rois = max(1, int(getattr(state, "MAX_ACCEPTED_ROIS", 10) or 10))
        target_roi_count = min(
            max_accepted_rois,
            max(1, int(getattr(state, "TARGET_ACCEPTED_ROIS", 5) or 5)),
        )
        aml_stop_lines = [
            "AML efficiency reminder:",
            f"- Soft goal: target {target_roi_count} kept ROIs from distinct regions for AML if feasible.",
            f"- Hard stop: do not exceed {max_accepted_rois} kept ROIs.",
        ]
        if kept_roi_count >= max_accepted_rois:
            aml_stop_lines.append(
                f"- Hard cap reached: {kept_roi_count}/{max_accepted_rois} kept ROI(s). Stop calling ROI tools and give the final AML decision."
            )
        elif kept_roi_count < target_roi_count:
            aml_stop_lines.append(
                f"- Current progress: {kept_roi_count}/{target_roi_count} toward the soft goal. Keep searching for additional distinct AML ROIs."
            )
        else:
            aml_stop_lines.append(
                f"- Soft goal reached: {kept_roi_count}/{target_roi_count} kept ROI(s). Final AML decision is now allowed if the evidence is stable; only add more ROIs if they could materially change the decision before the hard cap."
            )
        if kept_roi_count:
            aml_stop_lines.append("- Kept ROI reference evidence:")
            for roi in state._roi_marks[-min(3, kept_roi_count):]:
                ref_evidence = roi.get("aml_reference_evidence") if isinstance(roi.get("aml_reference_evidence"), dict) else None
                summary = ref_evidence.get("summary") if ref_evidence else None
                if isinstance(summary, str) and summary:
                    aml_stop_lines.append(f"- ROI #{roi.get('roi_id')}: {summary}")
        new_messages.insert(
            insert_pos,
            _tag_context_message({"role": "user", "content": [{"type": "text", "text": "\n".join(aml_stop_lines)}]}, "aml_guidance"),
        )
        insert_pos += 1

    latest_roi_debug_path = ""
    if state._roi_marks:
        latest_roi_debug_path = str(state._roi_marks[-1].get("debug_path") or "")
    current_view_debug_path = str(state._current_view.get("debug_path") or "") if state._current_view else ""
    mark_tool_names = {"wsi_mark_roi_norm", "wsi_mark_candidate"}
    should_include_current_candidates = bool(state._last_roi_candidates) and (
        tool_name not in mark_tool_names or current_view_debug_path != latest_roi_debug_path
    )

    if should_include_current_candidates:
        source = state._last_roi_candidate_source or "unknown"
        cand_lines = []
        for c in state._last_roi_candidates[:max(1, CONTEXT_ROI_CANDIDATE_LINES_MAX)]:
            rank = c.get("rank")
            center = c.get("center_norm", [0, 0])
            score = c.get("score")
            score_txt = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
            reference_mode = str(c.get("reference_mode") or "")
            bad_refs_active = ("good_bad" in reference_mode) or ("bad_only" in reference_mode)
            quality_hint = c.get("quality_hint")
            bad_like = c.get("bad_likelihood")
            retrieval_score = c.get("retrieval_score")
            bad_top1 = c.get("bad_top1_similarity")
            good_top1 = c.get("good_top1_similarity")
            blast_top1 = c.get("blast_top1_similarity")
            bad_refs = c.get("retrieved_bad_refs")
            good_refs = c.get("retrieved_good_refs")
            blast_refs = c.get("retrieved_blast_refs")
            nav_bbox = c.get("navigation_bbox_norm")
            extras = []
            if isinstance(quality_hint, str) and quality_hint:
                extras.append(f"hint={quality_hint}")
            if isinstance(retrieval_score, (int, float)):
                extras.append(f"retrieval={float(retrieval_score):.2f}")
            if bad_refs_active and isinstance(bad_like, (int, float)):
                extras.append(f"bad_like={float(bad_like):.2f}")
            if bad_refs_active and isinstance(bad_top1, (int, float)):
                extras.append(f"bad_top1={float(bad_top1):.2f}")
            if isinstance(good_top1, (int, float)):
                extras.append(f"good_top1={float(good_top1):.2f}")
            if isinstance(blast_top1, (int, float)):
                extras.append(f"blast_top1={float(blast_top1):.2f}")
            if isinstance(nav_bbox, list) and len(nav_bbox) == 4:
                extras.append(
                    f"nav_box=({int(nav_bbox[0])},{int(nav_bbox[1])},{int(nav_bbox[2])},{int(nav_bbox[3])})"
                )
            if bad_refs_active and isinstance(bad_refs, list) and bad_refs:
                top_bad = bad_refs[0]
                sim = top_bad.get("similarity")
                name = top_bad.get("name") or os.path.basename(str(top_bad.get("path") or ""))
                if name and isinstance(sim, (int, float)):
                    extras.append(f"bad_nn={name}@{float(sim):.2f}")
            if isinstance(good_refs, list) and good_refs:
                top_good = good_refs[0]
                sim = top_good.get("similarity")
                name = top_good.get("name") or os.path.basename(str(top_good.get("path") or ""))
                if name and isinstance(sim, (int, float)):
                    extras.append(f"good_nn={name}@{float(sim):.2f}")
            if isinstance(blast_refs, list) and blast_refs:
                top_blast = blast_refs[0]
                sim = top_blast.get("similarity")
                name = top_blast.get("name") or os.path.basename(str(top_blast.get("path") or ""))
                if name and isinstance(sim, (int, float)):
                    extras.append(f"blast_nn={name}@{float(sim):.2f}")
            suffix = f", {', '.join(extras)}" if extras else ""
            cand_lines.append(f"#{rank}: center=({int(center[0])},{int(center[1])}), score={score_txt}{suffix}")
        aml_meta_line = ""
        meta = state._roi_ranker_meta if isinstance(state._roi_ranker_meta, dict) else {}
        ref_stats = meta.get("reference_stats") if isinstance(meta.get("reference_stats"), dict) else None
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
        extractor_name = _selected_extractor_name()
        extractor_label = embedding_extractor_display_name(extractor_name)
        expected_source = (
            f"{extractor_label} tile embeddings + exact curated good-reference retrieval with morphology-aware ranking."
            if _agent_type() == "aml"
            else f"{extractor_label} tile embeddings + kNN ranking."
        )
        expected_source_name = (
            f"{extractor_name}_exact_retrieval" if _agent_type() == "aml" else f"{extractor_name}_knn"
        )
        cand_text = (
            "Top ROI candidates for CURRENT VIEW (normalized 0-999 coordinates). "
            f"Candidate source: {source}. "
            f"Expected source is '{expected_source_name}' from {expected_source} "
            "Prefer wsi_open_candidate(rank) for the first jump into an approximately 1500 um field around a candidate, then choose the best local ROI region yourself with wsi_mark_roi_norm or skip. navigation_bbox_norm is available as a coordinate fallback. "
            "Interpret quality_hint as support from good-quality ROI references only, not as a diagnosis and not as a guarantee of cellularity by itself. "
            "Prefer good_like candidates first, but uncertain candidates are still acceptable when they look interpretable, reasonably cellular, and morphologically informative. "
            "Use blast_top1/blast_nn as separate blast-reference evidence, prioritize deep blue-purple cellular candidates first, treat dark red-pink as a fallback only when clearly cellular, prefer fields with many separate crisp round purple cells, and reject only clearly trash regions such as stringy, gray-black, acellular, or broad gray-clump ROIs. "
            "For wsi_mark_roi_norm, choose one of these candidate centers/bboxes; arbitrary ROI coords are rejected:\n"
            + "\n".join(cand_lines)
            + aml_meta_line
        )
        candidate_content = [{"type": "text", "text": cand_text}]
        if include_candidate_overlay:
            overlay_part = _make_image_part(state._last_roi_candidate_overlay_path or "", budget)
            if overlay_part:
                candidate_content.append(overlay_part)
        new_messages.insert(insert_pos, _tag_context_message({"role": "user", "content": candidate_content}, "roi_candidates"))
        insert_pos += 1

    if include_overview and state._last_overview_with_box_path:
        overview_part = _make_image_part(state._last_overview_with_box_path, budget)
        if overview_part:
            fw = state._current_view.get("field_width_um") if state._current_view else None
            extra = _format_field_width_caption(fw)
            overview_msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Whole-slide overview (red box = current view{extra}).",
                    },
                    overview_part,
                ],
            }
            new_messages.insert(insert_pos, _tag_context_message(overview_msg, "overview"))
            insert_pos += 1

    prev_view_limit = max(0, CONTEXT_PREVIOUS_VIEWS_MAX if include_previous_views else 0)
    if prev_view_limit:
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
            new_messages.append(_tag_context_message(view_msg, "previous_view"))

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
                "reasoning": getattr(choice.message, "reasoning", None) if choice else None,
            }
        )
    except Exception:
        pass


async def _patched_async_chat_create(*args, **kwargs):
    model_name = kwargs.get("model", MODEL_NAME)
    prepared_msgs, original_msgs = _make_chat_request(_real_async_chat_create, kwargs.get("messages"), model_name, {})

    try:
        resp = await _real_async_chat_create(*args, **kwargs)
    except Exception as exc:
        if not _handle_context_length_error(exc, original_msgs, model_name):
            raise
        fallback_msgs = _prepare_messages_for_request(original_msgs, minimal=True)
        kwargs["messages"] = fallback_msgs
        _append_trace(
            {
                "type": "context_retry",
                "timestamp": datetime.utcnow().isoformat(),
                "model": model_name,
                "reason": str(exc),
                "messages": _redact_messages_for_trace(fallback_msgs),
            }
        )
        resp = await _real_async_chat_create(*args, **kwargs)

    _trace_response(resp, model_name)
    return resp


def _patched_sync_chat_create(*args, **kwargs):
    model_name = kwargs.get("model", MODEL_NAME)
    prepared_msgs, original_msgs = _make_chat_request(_real_sync_chat_create, kwargs.get("messages"), model_name, {})

    try:
        resp = _real_sync_chat_create(*args, **kwargs)
    except Exception as exc:
        if not _handle_context_length_error(exc, original_msgs, model_name):
            raise
        fallback_msgs = _prepare_messages_for_request(original_msgs, minimal=True)
        kwargs["messages"] = fallback_msgs
        _append_trace(
            {
                "type": "context_retry",
                "timestamp": datetime.utcnow().isoformat(),
                "model": model_name,
                "reason": str(exc),
                "messages": _redact_messages_for_trace(fallback_msgs),
            }
        )
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
