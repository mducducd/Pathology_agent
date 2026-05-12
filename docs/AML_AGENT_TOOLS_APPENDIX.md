# AML Agent Tool Appendix

## Purpose

This appendix describes the tool interface used by the AML whole-slide-image
(WSI) agent. The tools provide a bounded microscope-like workspace: the model
can open a slide overview, navigate to ranked candidate regions, inspect local
fields, mark diagnostic regions of interest (ROIs), and stop once sufficient
evidence has been collected.

The tool layer is intentionally separated from diagnosis. Navigation tools
change the visual field and expose candidate regions, but they do not determine
the final AML label. The final decision is made from accepted ROI evidence under
the morphology-only reporting rules.

## Tool — Agent Assignment

| Tool | `WSIAmlRoiCollectorAgent` | `WSIAmlDetectorAgent` | `WSITileSelectorAgent` | `WSIPathologyAgent` | `WSIAmlDiagnosisAgent` |
|---|:---:|:---:|:---:|:---:|:---:|
| `wsi_get_overview_view` | ✓ | ✓ | ✓ | ✓ | — |
| `wsi_zoom_current_norm` | ✓ | ✓ | ✓ | ✓ | — |
| `wsi_zoom_full_norm` | ✓ | ✓ | ✓ | ✓ | — |
| `wsi_pan_current` | ✓ | ✓ | ✓ | ✓ | — |
| `wsi_get_view_info` | ✓ | ✓ | ✓ | ✓ | — |
| `wsi_open_candidate` | ✓ | ✓ | — | ✓ | — |
| `wsi_mark_candidate` | ✓ | ✓ | — | ✓ | — |
| `wsi_mark_roi_norm` | ✓ | ✓ | — | ✓ | — |
| `wsi_discard_last_roi` | ✓ | ✓ | — | ✓ | — |
| `wsi_save_tile_norm` | — | ✓ | ✓ | ✓ | — |
| `wsi_rebuild_reference_index` | — | ✓ | — | ✓ | — |

`WSIAmlDiagnosisAgent` has no tools — it runs as a direct chat-completion call. It accepts either a `roi_collection.json` file directly or an output slide folder containing one.

## Tool Groups

| Group | Tools | Methodological role |
| --- | --- | --- |
| Slide initialization | `wsi_get_overview_view` | establishes the global slide context and initializes candidate search |
| Candidate navigation | `wsi_open_candidate`, `wsi_mark_candidate` | moves from the global view to ranked candidate fields |
| Local inspection | `wsi_zoom_current_norm`, `wsi_pan_current`, `wsi_zoom_full_norm` | supports limited field refinement before ROI selection |
| Evidence selection | `wsi_mark_roi_norm`, `wsi_discard_last_roi` | accepts or removes ROI evidence |
| View status | `wsi_get_view_info` | reports current field information when needed |
| Tile curation | `wsi_save_tile_norm`, `wsi_rebuild_reference_index` | supports reference-tile collection workflows outside the main AML diagnostic loop |

## Coordinate Convention

Most spatial tools use normalized coordinates on a 0 to 999 scale. For a current
view with level-0 bounding box $(x_v,y_v,w_v,h_v)$, a normalized point
$(u,v)$ maps to slide coordinates:

$$
x^{(0)} = x_v + \operatorname{round}\left(\frac{u}{999}w_v\right),
\qquad
y^{(0)} = y_v + \operatorname{round}\left(\frac{v}{999}h_v\right).
$$

This convention lets the VLM refer to visual regions without knowing the native
pixel dimensions of the WSI.

## Tool Contracts

### `wsi_get_overview_view`

Initializes the slide and returns a downsampled whole-slide overview. This is
the required first step in the diagnostic workflow. The response includes a
ranked candidate list, the current accepted ROI count, and the target ROI count.

Expected use: once at the beginning, or later as a reset if local navigation has
become unproductive.

### `wsi_open_candidate`

Opens a ranked candidate field from the current candidate list. Candidate rank
starts at 1. The candidate is a navigation prior, not a final ROI; after opening
the field, the VLM should inspect the local tissue and select an interpretable
subregion.

Expected use: move quickly between promising regions. Repeated calls without
inspection indicate candidate hopping rather than evidence acquisition.

### `wsi_zoom_current_norm`

Zooms into a normalized subregion of the current view. This is used to inspect
local tissue at higher detail before marking or rejecting a candidate field.

Expected use: at most one or two local refinements within a candidate field
before deciding whether to mark an ROI or move to another candidate.

### `wsi_pan_current`

Moves the current view laterally while preserving the same zoom scale. This is
useful when the opened candidate field contains nearby tissue that is more
readable than the initial center.

Expected use: short local adjustment, especially to avoid tissue edge,
background, crush, or stain artifact.

### `wsi_zoom_full_norm`

Jumps to a normalized region relative to the whole slide rather than the current
view. This is a global navigation tool and is less central in AML mode because
ranked candidates usually provide the preferred route to tissue.

Expected use: occasional manual recovery from the overview when candidate
navigation is not sufficient.

### `wsi_mark_roi_norm`

Marks a region of interest in the current view using normalized coordinates.
The marked ROI becomes part of the evidence set used for final diagnostic
synthesis. The tool response includes the updated accepted ROI count and target
count.

Expected use: accept a readable local field containing interpretable nucleated
marrow cells. The ROI should be chosen for morphology, not merely because it is
dark or highly stained.

### `wsi_mark_candidate`

Marks a ranked candidate directly without first opening it. This is a shortcut
for cases where the candidate center is already visually suitable.

Expected use: limited. In AML mode, opening the candidate and inspecting the
field is usually preferred because the best ROI may lie near, but not exactly
at, the candidate center.

### `wsi_discard_last_roi`

Removes the most recently accepted ROI when it is clearly unusable, such as
mostly background, severe artifact, or non-diagnostic tissue. Discarding is
conservative near the target ROI count to avoid endless replacement of
borderline but interpretable fields.

Expected use: immediately after marking, and only for clear failures.

### `wsi_get_view_info`

Reports current view properties such as field size, slide location, tissue
fraction, and candidate availability.

Expected use: rare during normal AML runs. The model should usually decide from
the visual field and candidate list rather than repeatedly requesting status.

### `wsi_save_tile_norm`

Saves a selected tile from the current view as a good or bad reference example.
This tool is primarily for tile-curation workflows and is not part of the
standard AML diagnostic loop.

Expected use: reference-bank enrichment, not final diagnosis.

### `wsi_rebuild_reference_index`

Refreshes the reference example set after new good or bad tiles have been
curated. This supports iterative construction of the ROI-quality memory used by
candidate retrieval.

Expected use: after tile-curation sessions, not during ordinary AML diagnostic
inference.

## AML Diagnostic Workflow

The preferred AML workflow is:

```text
1. wsi_get_overview_view
2. wsi_open_candidate(rank)
3. inspect the opened field with limited zoom or pan if needed
4. wsi_mark_roi_norm
5. repeat steps 2-4 until the target number of accepted ROIs is reached
6. produce the final morphology report
```

An ideal run is compact:

```text
overview
open candidate 1
local inspect
mark ROI 1

open candidate 2
local inspect
mark ROI 2

open candidate 3
local inspect
mark ROI 3

open candidate 4
local inspect
mark ROI 4

open candidate 5
local inspect
mark ROI 5

final morphology report
```

The expected loop is short and evidence-oriented. A candidate field should lead
to one of three decisions:

```text
readable cellular tissue -> mark ROI
borderline but locally improvable -> zoom or pan once, then decide
unreadable or non-diagnostic -> open the next candidate
```

The high-level state machine is:

```text
START
  -> OVERVIEW
  -> OPEN CANDIDATE
  -> LOCAL INSPECTION
  -> MARK, DISCARD, OR MOVE ON
  -> repeat until target ROI count
  -> FINAL REPORT
```

## Policy Constraints

The tools enforce a bounded evidence-acquisition policy:

1. The overview precedes local navigation.
2. Candidate regions are treated as search fields, not automatic diagnoses.
3. Local search is deliberately limited to prevent over-exploration.
4. Duplicate or near-duplicate ROI evidence is suppressed.
5. The run should stop once the target number of accepted ROIs is reached.
6. Final AML classification is based only on accepted ROI morphology.

## Failure Modes

Common tool-use failure modes include:

| Failure mode | Description | Preferred correction |
| --- | --- | --- |
| Candidate hopping | repeatedly opening new candidates without inspecting or marking | inspect each opened field and either mark or reject it |
| Over-zooming | performing long chains of zooms within one candidate | make a decision after limited local search |
| Status polling | repeatedly calling `wsi_get_view_info` | rely on the current image and candidate list |
| Center bias | marking the candidate center without checking readability | choose the best interpretable local subfield |
| Perfection search | skipping adequate ROIs while searching for an ideal one | accept good-enough morphology once readable tissue is present |

## Relation To The Methodology

The methodology describes how candidate fields are constructed and ranked. This
tool interface describes how the VLM interacts with those fields. Together they
implement retrieval-guided microscopy: the backend proposes promising regions,
and the VLM performs bounded local inspection before committing ROI evidence.
