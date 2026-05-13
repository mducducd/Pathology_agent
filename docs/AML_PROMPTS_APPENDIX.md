# Appendix: AML Prompt Text

This appendix summarizes the active AML-related prompts and runtime guidance from:

- `wsi_core_pkg/prompts.py`
- `wsi_core_pkg/tools.py`
- `wsi_core_pkg/context_injection.py`

Dynamic runtime values are shown with braces, for example `{kept_roi_count}` or `{target_roi_count}`.

## Prompt — Agent Assignment

| Prompt constant | Agent | Pipeline mode |
|---|---|---|
| `DEFAULT_AML_ROI_COLLECTION_PROMPT` | `WSIAmlRoiCollectorAgent` | `aml_roi`, `aml_auto` Stage 1 |
| `DEFAULT_AML_DIAGNOSIS_PROMPT` | `WSIAmlDiagnosisAgent` | `aml_diagnosis`, `aml_auto` Stage 2 |
| `DEFAULT_TILE_PROMPT` | `WSITileSelectorAgent` | `tile` |
| `DEFAULT_WSI_PROMPT` | `WSIPathologyAgent` | `wsi` |

## 1. Active AML Prompts

### 1.1 ROI Collection

`DEFAULT_AML_ROI_COLLECTION_PROMPT` is used by `WSIAmlRoiCollectorAgent` for Stage 1. It is navigation-only and stops after collecting accepted ROIs.

### 1.2 Diagnosis

`DEFAULT_AML_DIAGNOSIS_PROMPT` is used by `WSIAmlDiagnosisAgent` for Stage 2. It receives only ROI images and returns strict JSON.

### 1.3 Legacy Prompt Removal

The legacy single-stage AML detector prompt has been removed. The active AML prompt surface is the split `aml_roi` plus `aml_diagnosis` workflow.

## 2. Tool Schema and Runtime Guidance

### 2.1 `wsi_core_pkg/tools.py` - Function Tool Descriptions

These docstrings are exposed as tool descriptions and guide navigation, marking, saving, and discarding.

#### `wsi_get_overview_view`

```text
Get a thumbnail overview of the entire whole-slide image. Always call this first.
Returns roi_candidates ranked by quality — use wsi_open_candidate(rank) to navigate into them.
Response includes marked_roi_count and target_accepted_rois to track collection progress.
```

#### `wsi_zoom_current_norm`

```text
Zoom into a sub-region of the CURRENT VIEW using normalized [0–999] coordinates.
Use to inspect a specific area within the current field before marking or abandoning.
Limit: at most 2 zoom/pan actions per candidate before deciding to mark or move on.
```

#### `wsi_zoom_full_norm`

```text
Zoom into a region using normalized [0–999] coordinates relative to the FULL SLIDE.
Use when you need to jump to an absolute position rather than a sub-region of the current view.
```

#### `wsi_pan_current`

```text
Pan the current view by a relative offset in normalized [−999, +999] units.
Positive dx moves right, positive dy moves down. Keeps the same zoom level.
```

#### `wsi_get_view_info`

```text
Get metadata for the current view (magnification, field size in µm, tissue_fraction, roi_candidates).
Use sparingly — do NOT call this after target_accepted_rois is reached.
```

#### `wsi_mark_roi_norm`

```text
Mark a region of interest in the CURRENT VIEW using normalized [0–999] coordinates.
The system crops the ROI at native resolution for diagnosis. Response includes marked_roi_count
and target_accepted_rois — stop all tool calls once marked_roi_count >= target_accepted_rois.
```

#### `wsi_open_candidate`

```text
Navigate to a ranked ROI candidate from the current roi_candidates list (rank starts at 1).
Always inspect inside the candidate before opening another. Use wsi_mark_candidate or
wsi_mark_roi_norm to accept, or call wsi_open_candidate again to abandon and move on.
```

#### `wsi_mark_candidate`

```text
Mark a candidate from the roi_candidates list directly by rank without navigating to it first.
Faster than wsi_open_candidate + wsi_mark_roi_norm when the candidate location is already known.
```

#### `wsi_save_tile_norm`

```text
Save a tile from the current view for later analysis (tile agent only, not AML/WSI agents).
quality='good' for diagnostically useful tiles, 'bad' for negative examples.
Stop after 60 good tiles or when no more good regions are visible.
```

#### `wsi_discard_last_roi`

```text
Discard the most recently marked ROI if it is background, artifact, or not diagnostic.
Blocked when kept ROI count is near target to prevent over-discarding. Check the ROI image
immediately after marking and discard only if clearly unusable.
```

### 2.2 `wsi_core_pkg/tools.py` - AML Finalization and Stop Guards

#### Finalization required after target ROI count

```text
You already have {kept}/{target} kept ROI(s). No more tool calls are allowed. Output the final diagnosis JSON now.
```

#### Hard ROI cap reached

```text
You already have {kept} kept ROI(s), which reaches the configured cap of {cap}. Stop searching immediately and give the final answer from the ROIs already kept.
```

#### Near-target discard block

```text
Discard blocked: you already have {kept_roi_count} kept ROI(s) and target is {target_accepted_rois}. Near the target, keep borderline/interpretable AML ROIs. Reserve discard for clearly bad ROIs such as mostly background or empty views.
```

### 2.3 `wsi_core_pkg/tools.py` - AML Candidate Preparation Pipeline Text

These strings appear in tool output metadata to explain how candidate regions were ranked.

```text
Detect deep blue-purple basophilic regions -> dark-guided supertile coarse filter -> raw-tile hybrid quality filter -> {extractor_label} tile embeddings -> ROI-quality exemplar retrieval + nuclei/dark-region priors -> top-K candidate blast-suspected ROIs per view
```

```text
Detect deep blue-purple basophilic regions -> dark-guided supertile coarse filter -> raw-tile quality filter -> {extractor_label} tile embeddings -> ROI-quality exemplar retrieval + nuclei/dark-region priors -> top-K candidate blast-suspected ROIs per view
```

```text
Detect deep blue-purple basophilic regions -> dark-guided supertile coarse filter -> {extractor_label} tile embeddings -> ROI-quality exemplar retrieval + nuclei/dark-region priors -> top-K candidate blast-suspected ROIs per view
```

```text
No tile prefilter -> {extractor_label} tile embeddings -> ROI-quality exemplar retrieval + nuclei/dark-region heuristics -> top-K candidate blast-suspected ROIs per view
```

```text
No tile prefilter -> {extractor_label} tile embeddings -> ROI-quality exemplar retrieval + nuclei/dark-region heuristics -> top-K candidate blast-suspected ROIs per current view
```

### 2.4 `wsi_core_pkg/tools.py` - Recommended Actions and Navigation Warnings

#### Candidate jump recommendation

```text
Candidate #1 is pre-ranked by AML relevance. Jump first to an approximately {nav_field_um} um field around the candidate, then mark or skip.
```

#### Navigation budget exceeded

```text
Navigation budget exceeded with no ROI marked. Stop local wandering and jump to a different strong region now.
```

#### AML stop hints

```text
Accepted ROI cap reached ({kept_roi_count}/{max_accepted_rois}). Finalize now — do not call any more tools.
```

```text
ROI target reached ({kept_roi_count}/{target_accepted_rois}). Finalize now — do not call any more tools.
```

```text
Soft ROI target reached ({kept_roi_count}/{target_accepted_rois}; hard cap {max_accepted_rois}). Finalize unless an additional ROI would materially change the decision — do not call wsi_get_view_info to confirm.
```

```text
You already have {kept_roi_count} kept ROI(s). Soft target is {target_accepted_rois} kept AML ROIs, so inspect one more distinct informative ROI unless none can be found after reasonable search.
```

```text
One ROI is screening evidence only. Soft target is {target_accepted_rois} kept AML ROIs from distinct slide regions if feasible.
```

```text
You have {kept_roi_count} kept ROI(s). Continue toward the soft target of {target_accepted_rois} representative AML ROIs across distinct slide regions unless additional informative ROIs cannot be found.
```

#### Same-region warning

```text
You have taken {same_region_steps} consecutive steps in the same slide region. Do NOT keep searching here. Call wsi_open_candidate(rank={next_rank_hint}) now to move to a different candidate region.
```

```text
You have taken {same_region_steps} consecutive steps in the same slide region. Call wsi_get_overview_view or wsi_zoom_full_norm NOW to move to a completely different area.
```

#### Low-tissue warning

```text
ALERT: {low_tissue_steps} consecutive views have been mostly empty background (tissue_fraction < {LOW_TISSUE_THRESHOLD}). You are zoomed into empty glass. You MUST call wsi_open_candidate(rank={next_rank_hint}) RIGHT NOW to jump back to a candidate region with visible tissue.
```

```text
ALERT: {low_tissue_steps} consecutive views have been mostly empty background (tissue_fraction < {LOW_TISSUE_THRESHOLD}). You are zoomed into empty glass. You MUST call wsi_get_overview_view RIGHT NOW to reset to the full slide, then navigate to a region with visible tissue (pink/purple staining).
```

#### Exploration warnings

```text
CRITICAL: You have navigated {navigation_steps} times without marking any ROI. This is excessive exploration. Open candidate region #{next_rank_hint} next, then search inside that field with zoom/pan until you find a representative interpretable high-power ROI with adequate nucleated cells, readable morphology, and acceptable focus; mark only after that search, or skip the region. Do not mark the candidate center by default.
```

```text
CRITICAL: You have navigated {navigation_steps} times without marking any ROI. This is excessive exploration. Search within the current strong region with zoom/pan until you find a representative local interpretable ROI with adequate nucleated cells and readable morphology, or call wsi_get_overview_view and move to a different region. Do not mark the candidate center by default.
```

```text
WARNING: {navigation_steps} navigation steps with 0 ROIs marked. You are over-exploring. Use wsi_open_candidate(rank) to jump into a strong region, then search within that field by zooming to high power before you mark. Do not treat the candidate center tile as the ROI.
```

### 2.5 `wsi_core_pkg/tools.py` - Candidate Warnings and Guidance

#### Candidate source unavailable

```text
Primary candidate source unavailable for this view. Using fallback heuristic.
```

```text
Primary candidate source unavailable for this view. Fallback disabled.
```

#### Bad-like candidates hidden

```text
Hid {bad_like_hidden_count} bad_like ROI candidate(s). Do not mark bad_like candidates when good_like or uncertain alternatives exist.
```

#### Weak candidate support

```text
Only weak ROI support was available in this view, so one best-effort fallback candidate was kept to avoid an empty candidate list. Treat it as low-confidence and navigate to a better cellular region if feasible.
```

```text
Current view only produced bad_like ROI candidates. One best-effort fallback candidate is shown, but prefer moving to a different region or zoom level for better candidates if feasible.
```

```text
Current view only produced bad_like ROI candidates. Do NOT mark them. Navigate to a different region or zoom level and look for good_like/uncertain candidates.
```

#### Candidate guidance intro

```text
Treat roi_candidates as candidate blast-suspected ROIs selected from tissue, nucleated-cell, focus, RBC, and artifact heuristics across the current view. Prioritize deep dark blue-purple cellular fields; dark red-pink is only a rare fallback when clearly cellular, and gray-black low-chroma junk should be rejected.
```

#### Standard AML planner guidance

```text
Treat roi_candidates as candidate blast-suspected ROIs selected from tissue, nucleated-cell, focus, RBC, and artifact heuristics across the current view. Prioritize deep dark blue-purple cellular fields; dark red-pink is only a rare fallback when clearly cellular, and gray-black low-chroma junk should be rejected. Follow a standard practical hierarchy: tissue first, then nucleated-cell-rich interpretable marrow over RBC-rich/empty areas, then blast-suspected morphology. Prefer ROIs with adequate nucleated cells, readable single-cell detail, acceptable focus, and limited artifact. Moderate cellularity is acceptable if morphology is still assessable; do not reject a usable ROI only because it is not the single densest field in the region. A single ROI is screening evidence only. For AML, soft target is {target_accepted_rois} kept ROIs from representative distinct slide regions when feasible; hard cap is {max_accepted_rois}. Use roi_candidates only to jump into a promising region quickly. After opening a candidate region, search within that field by zooming/panning until you find a representative high-power ROI. Choose a nearby area with better readability or less artifact when available, but do not over-search indefinitely for a marginally denser patch. Then use wsi_mark_roi_norm.
```

#### Weak planner AML guidance

```text
Treat roi_candidates as candidate blast-suspected ROIs selected from tissue, nucleated-cell, focus, RBC, and artifact heuristics across the current view. Prioritize deep dark blue-purple cellular fields; dark red-pink is only a rare fallback when clearly cellular, and gray-black low-chroma junk should be rejected. Follow a standard practical hierarchy: tissue first, then nucleated-cell-rich interpretable marrow over RBC-rich/empty areas, then blast-suspected morphology. Collect as many accepted ROIs as feasible; stop as soon as you have enough evidence — even 1 ROI is acceptable if the slide yields no interpretable tissue. Hard cap is {max_accepted_rois}. Use roi_candidates to jump into a promising region. Once inside a candidate, mark it immediately with wsi_mark_roi_norm if tissue is interpretable — do NOT zoom/pan first. Accept borderline ROIs rather than skipping them. Only discard if tissue is clearly unusable (background-only, severe artifact, or stain pool).
```

#### Candidate search region guidance

```text
CRITICAL: This is a candidate-search region centered on a promising area. Systematically zoom into different sub-areas (corners, edges, quadrants) using wsi_zoom_current_norm to find ALL strong cellular, high-density subregions. Mark ROIs at EACH strong position you discover. If this candidate region contains multiple distinct high-quality areas, mark multiple ROIs here before moving to the next candidate. After marking each ROI, continue zooming to different areas of this region to search for additional strong ROIs. Do NOT mark ROI at this view's center - mark it where you find the best tissue via systematic zooming. Only call wsi_open_candidate(rank) to move to the next candidate when this region is exhausted.
```

### 2.6 `wsi_core_pkg/tools.py` - ROI Acceptance, Rejection, and Next-Action Text

#### Low-tissue ROI warning

```text
This high-power ROI is mostly background/empty glass (low tissue_fraction). This is one of the few cases where immediate discard is appropriate. Use wsi_discard_last_roi and select an ROI centered on diagnostic tissue.
```

#### Dark-blue flood rejection

```text
ROI auto-rejected: {flood_fraction_percent}% is saturated dark-blue flood (stain pool, no cell outlines). Move the box away from the flood region and try again.
```

#### Next-action hint after AML ROI marking

```text
You just requested a high-power ROI. In AML mode, discard only if this ROI is clearly bad on review, such as mostly background/empty glass. If it is borderline but interpretable, keep it and continue searching for additional ROIs. If you still need more evidence, use candidate #{next_rank_hint} next instead of continuing local search inside this ROI.
```

```text
You just requested a high-power ROI. In AML mode, discard only if this ROI is clearly bad on review, such as mostly background/empty glass. If it is borderline but interpretable, keep it and continue searching for additional ROIs. If you still need more evidence, jump directly to the next unvisited candidate instead of continuing local search inside this ROI.
```

#### Guidance for next action after ROI marking

```text
ROI #{roi_id} has been marked and saved. You can now:
1) Continue searching WITHIN THIS CANDIDATE REGION for additional strong ROIs by using wsi_zoom_current_norm to explore other sub-areas.
2) When this candidate region is exhausted (no more strong tissue), call wsi_open_candidate(rank={next_rank_hint}) to move to the next candidate.
If you just discarded an ROI, you can search for a different position in this region OR jump to the next candidate.
```

#### Zoom and coordinate warnings

```text
This zoomed field is mostly background/empty glass (low tissue_fraction). You should NOT mark ROIs here. Instead, zoom or pan toward visible tissue in this CURRENT VIEW before proceeding.
```

```text
Selected region is mostly background/empty glass (low tissue_fraction). You should pick coordinates over tissue areas in the overview and try again.
```

```text
Requested zoom was smaller than the minimum ROI inspection size, so the system expanded it back to ROI scale. Do not zoom smaller than the ROI crop; if the candidate is centered, use wsi_mark_candidate or wsi_mark_roi_norm instead.
```

```text
Requested zoom was smaller than the minimum ROI inspection size, so the system expanded it back to ROI scale. Do not zoom smaller than the ROI crop; if the candidate is centered, use wsi_mark_roi_norm instead.
```

## 3. Context-Injection Prompt Text

### 3.1 `wsi_core_pkg/context_injection.py` - Example Tile and ROI Primers

These messages are injected with example images when available.

#### Example GOOD tiles

```text
Example GOOD tiles (high-quality diagnostic ROI examples, not AML-vs-normal labels). Use them as ROI-quality references for tissue vs background, nucleated-cell richness, focus, and artifact rejection. Good AML/marrow tiles are hypercellular, deep blue-purple/basophilic, nucleated, in focus, low artifact, and morphologically informative.
```

#### Example BAD tiles

```text
Example BAD tiles (low-quality/non-diagnostic ROI examples, not AML-vs-normal labels). Use them to recognize empty/background-heavy, RBC/clot-dominant, gray-black low-chroma junk, artifact-dark, blurred, crushed, or non-representative edge/debris regions.
```

#### Example ROI images

```text
Example ROI images (diagnostic regions to keep). Good AML ROIs are hypercellular, deep blue-purple/basophilic, blast-suspected, in focus, low artifact, and representative. These examples help find visually informative ROIs, not prove AML by themselves.
```

#### Example NON-ROI images

```text
Example NON-ROI images (background/non-diagnostic regions to avoid). Avoid empty, RBC-heavy, gray-black low-chroma junk, artifact-dark, blurred, crushed, or edge/debris dominated regions.
```

### 3.2 `wsi_core_pkg/context_injection.py` - Current View Injection

#### Current view after ROI target is reached

```text
CURRENT VIEW{extra}. ROI target is reached — do NOT call any more navigation or marking tools. Write the final JSON output now based on the kept ROIs shown above.
```

#### Current view for navigation

```text
CURRENT VIEW for navigation{extra}. All coordinates for NEXT tool call must be chosen relative to THIS image. PRIORITY: Look for regions with high cellularity (dense packed nucleated cells) and clear blast visibility. Once you find high-cellularity tissue with readable morphology, mark it with wsi_mark_roi_norm. Do NOT select boxes centered on blank/white background; always place boxes tightly around tissue and high-cellularity areas. Do not search indefinitely for marginal improvements. Mark if tissue is readable and diagnostic, even if not the single densest field. Avoid panning to empty areas.
```

### 3.3 `wsi_core_pkg/context_injection.py` - AML Progress and Completion Reminders

#### Progress below target

```text
- Current progress: {kept_roi_count}/{target_roi_count} ROIs marked.
Keep searching for additional distinct AML ROIs to reach the {target_roi_count} ROI target.
```

#### Hard cap reached

```text
- Hard cap reached: {kept_roi_count}/{max_accepted_rois} kept ROI(s). Provide final AML diagnosis.
```

#### Target reached

```text
- ROI target reached: {kept_roi_count}/{target_roi_count} kept ROI(s). Provide AML blast estimate and diagnosis.
```

#### All kept ROIs review

```text
Target ROI count reached. Review ALL kept ROIs below and write the final JSON output. Assign a distinct blast_range per ROI based on each image.
```

```text
ROI #{r_id}: {r_label}
```

### 3.4 `wsi_core_pkg/context_injection.py` - AML Free Local Search Injection

```text
You are now inside a suggested AML search region. Treat the CURRENT VIEW as a search area. Look for areas with high cellularity (dense packed nucleated cells) and clear blast visibility—these are priority. Finding a visibly high-cellularity subregion: STAY and zoom into it to capture the clearest single-cell morphology. Avoid jumping to other regions unless the current area is clearly empty or severely artifact-affected. Prefer patches with readable single-cell detail, abundant nucleated cells, acceptable focus, and limited artifact; broad full-field cellularity is not required but high local concentration is a strong positive signal. Avoid empty/pale areas, severely RBC-dominant regions, heavy stain pooling, clot/crush artifact, and blurred or unreadable zones. Call wsi_mark_roi_norm only after identifying a high-cellularity local subregion with good blast visibility; otherwise keep exploring within this field or skip to another region.
```

### 3.5 `wsi_core_pkg/context_injection.py` - Current-View Candidate Injection

Candidate source summary:

```text
{extractor_label} tile embeddings + exact curated good-reference retrieval with morphology-aware ranking.
```

```text
{extractor_name}_exact_retrieval
```

Candidate list and selection guidance:

```text
Top ROI candidates for the CURRENT VIEW. Candidate source: {source}. Expected source: '{expected_source_name}' from {expected_source}. IMPORTANT: If the CURRENT VIEW shows high cellularity with good blast visibility or abundant nucleated cells, STAY IN THIS REGION and zoom to find the best local ROI—do not jump to other candidates. Only jump to a different candidate if the current region is clearly unsuitable (mostly empty, severe artifact, etc.). When zooming within the current high-cellularity field, look for areas with the clearest single-cell morphology and best blast visibility. To navigate to a different candidate, use wsi_open_candidate(rank) to jump into an approximately {candidate_nav_field_um} um field around that candidate. Treat quality_hint only as supportive reference evidence; it does not guarantee cellularity or interpretability. Prefer representative, interpretable marrow patches with readable single-cell morphology, abundant nucleated cells, acceptable focus, and limited artifact. High cellularity (dense packed nucleated cells) is a strong signal of good diagnostic potential—prioritize these regions. A partial but clearly usable cellular area is acceptable; broad full-field cellularity is not required, and high local blast concentration can be diagnostic. Avoid heavy stain pooling, dark blue clot-like material, stringy smear artifact, gray-black debris, and nearly acellular regions. Some empty/vacuolated space is acceptable if a nearby local ROI is still clearly usable for rough blast estimation. Use ranked candidates as region-level guidance only; choose the final ROI box based on the best local morphology:
{candidate_lines}
{aml_meta_line}
```

Candidate line template:

```text
#{rank}: score={score_txt}{hint_txt}
```

AML quality summary template:

```text
AML quality summary: mode={mode}, similarity={similarity}, ref_k={ref_k}, bad_like_fraction={bad_frac}, strong_bad_like_fraction={strong_bad_frac}
```

### 3.6 `wsi_core_pkg/context_injection.py` - Overview and Previous-View Context

These are not diagnostic AML prompts by themselves, but they are injected into AML runs as navigation context.

```text
Whole-slide overview (red box = current view{extra}).
```

```text
Previous view ({tag}{extra}).
```
