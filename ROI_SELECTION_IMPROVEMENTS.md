# ROI Selection Diversity Improvements

## Problem Summary
The new agent was:
1. Always choosing the center tile from candidate subregions, instead of exploring and selecting from diverse positions
2. Auto-advancing to the next candidate after marking a single ROI, preventing discovery of multiple strong ROIs within the same candidate region

The old agent worked better because:
1. It freely searched the entire slide and marked ROIs at positions it intentionally zoomed to
2. It naturally found and marked multiple ROIs in strong regions before moving on

## Root Cause
When a candidate region is opened via `wsi_open_candidate(rank)`, two issues prevented multi-ROI discovery:
1. The guidance was not explicit enough about systematic exploration and non-center positioning
2. The code auto-advanced to the next candidate immediately after marking one ROI, preventing the agent from searching for additional ROIs in the same region

## Solution Implemented

### 1. Removed Auto-Advancement (tools.py - _mark_roi_from_candidate)

**Changed behavior (lines 1666-1684):**
- REMOVED the auto-advancement to next candidate after marking first ROI
- Added guidance explaining agent can mark multiple ROIs within same candidate region
- Changed to require explicit `wsi_open_candidate(rank)` call to move to next candidate
- Allows agent to continue searching within current region for additional strong ROIs

**Key change:**
```python
# OLD: Auto-advanced to next candidate immediately after marking ROI
# NEW: Agent stays in current region, explicitly decides when to move to next candidate
roi["guidance_for_next_action"] = (
    "1) Continue searching WITHIN THIS CANDIDATE REGION for additional strong ROIs...\n"
    "2) When this candidate region is exhausted, call wsi_open_candidate(rank=...)..."
)
```

### 2. Enhanced Agent Instructions (agents.py)

**Navigation Loop Improvements (lines 66-79):**
- Added "MULTIPLE ROIs PER CANDIDATE" section emphasizing all strong positions should be marked
- Emphasized systematic zooming to different areas within candidate for multiple ROI discovery
- Clarified that agent should only move to next candidate when current region is exhausted
- Added explicit instruction: "After marking each ROI, stay within the same candidate region to search for additional ROIs"
- Noted that strong regions may have 1, 2, or more distinct high-quality areas

**ROI Selection Guidance (lines 81-95):**
- Expanded guidance to prioritize exploring for multiple ROIs before moving on
- Added emphasis on exhausting each candidate before proceeding to next
- Clarified the agent controls when to advance, not automatic behavior

### 2. Enhanced AML Prompt (prompts.py)

**Navigation Section (line 84):**
- Changed from generic "search within that field" to "SYSTEMATICALLY SEARCH by zooming into different sub-areas (quadrants, corners, edges)"
- Added explicit instruction to "zoom to promising locations first, then mark ROI at the best position you discover via zoom"
- Emphasized using `wsi_zoom_current_norm` to examine multiple sub-regions before committing to ROI position

### 3. Enhanced Tool Feedback (tools.py)

**Candidate Opening Guidance (lines 1748-1750):**
- Added new `candidate_search_guidance` field to info returned by `_open_candidate_by_rank`
- This guidance is shown immediately when a candidate is opened, reinforcing the need to:
  - Systematically zoom into different sub-areas
  - Find the most cellular, high-density subregion
  - Mark ROI only after discovering best position via zoom
  - NEVER mark at the view center

## How the Improved Behavior Works

### Before (Center-Biased):
```
1. wsi_open_candidate(rank=5)
   → Shows a view centered on candidate 5
2. Agent may directly call wsi_mark_roi_norm(450, 450, 550, 550)
   → Always marks near center, missing off-center high-quality regions
```

### After (Exploration-Driven, Multiple ROIs per Candidate):
```
1. wsi_open_candidate(rank=5)
   → Shows a view centered on candidate 5
   → Displays: "Mark ALL strong ROIs. Continue zooming for additional ROIs..."

2. wsi_zoom_current_norm(0, 0, 300, 300)
   → Zoom to top-left quadrant, finds strong cellular region

3. wsi_mark_roi_norm(100, 150, 200, 250)
   → Marks ROI #1 at top-left position
   → Returns: "Continue searching WITHIN THIS CANDIDATE for additional ROIs..."

4. wsi_zoom_current_norm(700, 300, 999, 600)
   → Zoom to right-side quadrant, finds another distinct strong region

5. wsi_mark_roi_norm(850, 350, 950, 450)
   → Marks ROI #2 at right-side position
   → Agent can now: continue searching OR move to next candidate

6. wsi_zoom_current_norm(300, 700, 600, 999)
   → Zoom to bottom-center, finds lower quality tissue (RBCs)

7. wsi_open_candidate(rank=6)
   → Decides candidate #5 is exhausted, explicitly moves to next candidate
   
RESULT: 2 strong ROIs marked in candidate #5 before advancing
```

## Key Differences from Old Agent

The old agent worked well because:
- It had the entire slide available and could zoom anywhere
- No pre-computed candidates meant every zoom was an active decision
- Each ROI marking was a deliberate result of zooming to a specific location

The new agent now works similarly within candidate regions:
- Candidates provide region-level guidance (like overviews in old agent)
- Agent must zoom within candidate to find best position (like old agent's systematic search)
- ROI is marked at the zoomed position, not the candidate center (preserves old agent's freedom)

## Expected Outcomes

1. **Multiple ROIs per Strong Candidate**: Agent now marks ALL strong ROIs found within each candidate region (not limited to 1 per candidate)
2. **Increased ROI Diversity**: ROIs spread across different positions within each candidate region, not clustered at centers
3. **Better Quality Control**: Systematic zooming within candidates ensures evaluation of multiple sub-areas before marking each ROI
4. **Fewer Redundant ROIs**: Off-center exploration and multi-ROI marking naturally avoids clustering
5. **More Accurate Blast Estimation**: Multiple ROIs from strong regions provide better sampling of tissue morphology
6. **Improved Candidate Utilization**: Strong candidate regions are fully exploited rather than abandoned after first ROI

## Integration with Old Agent Patterns

The modifications preserve the best practices from the old agent:
- **Explicit Navigation**: Each move (zoom/pan) is intentional, not accidental
- **Quality-First Approach**: Search for good tissue before marking, not vice versa
- **Position Diversity**: ROIs at discovered positions, not defaults
- **Iterative Refinement**: Systematic evaluation before commitment

## Testing Recommendations

1. **Monitor ROI Positions**: Check that marked ROIs are at diverse positions, not clustered at centers
2. **Verify Zoom Patterns**: Logs should show multiple zoom calls within candidate regions before ROI marking
3. **Compare Blast Estimation**: With better position diversity, blast percentage estimates should be more reliable
4. **Check Candidate Coverage**: Ensure the agent explores unvisited candidates rather than refining the same one repeatedly

## Key Behavioral Changes Summary

### Before:
- Agent marked 1 ROI per candidate → auto-advanced to next candidate
- ROIs often defaulted to center positions
- Strong candidate regions were abandoned after first ROI
- VLM received limited guidance on systematic exploration

### After:
- Agent marks ALL strong ROIs found within each candidate
- ROIs marked at positions discovered via systematic zoom, not defaults
- Strong candidate regions are fully exploited
- VLM receives explicit guidance to zoom to corners/edges/quadrants before marking
- Agent explicitly controls when to advance to next candidate (via wsi_open_candidate call)

## Files Modified

- `wsi_core_pkg/agents.py`: Enhanced navigation instructions with multi-ROI emphasis
- `wsi_core_pkg/prompts.py`: Enhanced AML mode to emphasize finding ALL strong ROIs per candidate  
- `wsi_core_pkg/tools.py`: 
  - Removed auto-advancement to next candidate
  - Added guidance encouraging continued search within current candidate
  - Enhanced candidate_search_guidance with multi-ROI emphasis
