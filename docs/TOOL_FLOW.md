# Tool flow

The correct tool flow should be simple and bounded:

```
1. wsi_get_overview_view
2. wsi_open_candidate(rank)
3. inspect locally:
   - wsi_zoom_current_norm OR wsi_pan_current
   - at most 1–2 local actions
4. wsi_mark_roi_norm OR wsi_mark_candidate
5. if the same candidate region still contains another distinct strong pocket, mark one more ROI there
6. otherwise move to the next candidate
7. stop once target_accepted_rois is reached or max_accepted_rois is hit
8. final JSON, no more tools
```

A healthy run should look like this:

```
overview
open_candidate #1
zoom/pan once
mark ROI 1
zoom/pan to another strong sub-area
mark ROI 2

open_candidate #2
zoom/pan once
mark ROI 3

open_candidate #3
zoom/pan once
mark ROI 4

open_candidate #4
zoom/pan once
mark ROI 5

final JSON
```

`wsi_get_view_info` should be rare. In your AML agent, it should almost never be needed during normal navigation. It is mostly a debugging/status tool. If the VLM is behaving correctly, it should not keep calling it.

`wsi_open_candidate` should **not** be called many times in a row. This is bad:

```
open_candidate
open_candidate
open_candidate
open_candidate
```

Correct behavior after `wsi_open_candidate` is:

```
open_candidate
zoom/pan or mark
```

If the opened region is especially strong, the next good behavior is still to
stay inside that same candidate region briefly and mark another distinct local
ROI, not to reopen the same candidate or bounce elsewhere immediately.

Then either:

```
mark ROI
```

or, if bad:

```
open_candidate(next rank)
```

For each candidate region, the intended decision is:

```
candidate opened
↓
Is there readable cellular tissue?
    yes → mark a good-enough ROI quickly
           if another distinct strong pocket is nearby, one more local ROI is fine
    no  → maybe one local zoom/pan
           still bad → open next candidate
```

The VLM should not search for the “perfect” ROI. It should mark a **good-enough interpretable** ROI.

The ideal high-level state machine is:

```
START
  ↓
OVERVIEW
  ↓
OPEN CANDIDATE
  ↓
LOCAL INSPECTION
  ↓
MARK / DISCARD / MOVE ON
  ↓
ROI count < target? repeat
ROI count >= target or cap? final JSON
```

So the true flow is:

```
overview once
candidate region A → local inspect → mark
candidate region A or B → local inspect → mark
candidate region B or C → local inspect → mark
repeat until target or cap
finish
```

Anything like repeated candidate switching, repeated `wsi_get_view_info`, or long zoom chains means the VLM is not following the intended tool policy.
