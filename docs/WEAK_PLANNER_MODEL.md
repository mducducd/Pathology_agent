# `weak_planner_model` — For Weaker VLMs

## Relationship to `AML_DISABLE_BAD_REFERENCES`

**Medium impact on GLM weakness.** In the current config,
`AML_DISABLE_BAD_REFERENCES: true`, so the live AML ranker runs in
good-reference-only mode by default. That is more permissive: more candidates
reach the VLM, but the VLM then has to do more discrimination work itself.
Setting `AML_DISABLE_BAD_REFERENCES: false` re-enables bad-reference penalties
and rejections upstream, which can make the candidate list cleaner if the bad
reference bank is well curated.

**Recommendation for weak VLMs:** consider switching to
`AML_DISABLE_BAD_REFERENCES: false` if your bad reference bank is reliable,
because it lets the upstream ranker reject more junk before the VLM sees it.
Keep the current `true` default if the bad bank is noisy or mismatched.

## Relationship to `AML_BAD_TOP1_REJECT_THRESHOLD`

```yaml
# Hard reject threshold on top-1 bad-reference similarity. >1.0 means this check is effectively disabled.
AML_BAD_TOP1_REJECT_THRESHOLD: 1.0
```

Currently set to `1.0`, which is still effectively near-disabled in practice
because cosine similarities almost never hit an exact `1.0` match. This means
top-1 bad-reference similarity alone rarely hard-rejects a tile; most rejection
pressure instead comes from the other AML bad-reference thresholds and penalties
when bad references are enabled. Tightening this (for example to `0.80`) would
pre-reject more tiles but risks false negatives if bad references are not well
calibrated.
