# `weak_planner_model` — For Weaker VLMs

## Relationship to `AML_DISABLE_BAD_REFERENCES`

**Medium impact on GLM weakness.** When `false` (current default), bad references actively reject/penalize tiles before VLM — which is good in principle, but if the bad reference set is noisy or mismatched, it can incorrectly suppress valid ROI candidates before GLM ever sees them. Setting `AML_DISABLE_BAD_REFERENCES: true` switches to good-only mode, which makes the system more permissive — more candidates reach GLM, but GLM then has to do more discrimination work, which is exactly what it struggles with.

**Recommendation for weak VLMs:** keep `AML_DISABLE_BAD_REFERENCES: false` — let the upstream ranker do the rejecting so GLM only receives cleaner candidates.

## Relationship to `AML_BAD_TOP1_REJECT_THRESHOLD`

```yaml
# Hard reject threshold on top-1 bad-reference similarity. >1.0 means this check is effectively disabled.
AML_BAD_TOP1_REJECT_THRESHOLD: 2.0
```

Currently set to `2.0` — effectively **disabled** since cosine similarities are in `[0, 1]`. This means no tile is hard-rejected purely on bad-reference top-1 similarity alone. For a weak VLM, this is the right default: aggressive hard rejection upstream could eliminate borderline-but-valid candidates that a stronger ranker signal would have kept. Tightening this (e.g. to `0.80`) would pre-reject more tiles but risks false negatives if bad references are not well-calibrated.
