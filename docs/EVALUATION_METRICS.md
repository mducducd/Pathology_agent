# AML Agent Benchmark Metrics

This document defines the metrics reported for AML agent benchmark runs. The
benchmark keeps the headline table compact: one diagnostic accuracy metric, a
small number of ROI/tool-flow behavior metrics, one tool-count metric, runtime
summaries, AML/Normal confusion counts, and optional NPM1 accuracy.

## Expected AML Agent Workflow

The intended workflow is:

$$
\text{overview}
\rightarrow
(\text{open candidate}
\rightarrow
\text{local inspection}_{0\text{-}2}
\rightarrow
\text{mark ROI}) \times 5
\rightarrow
\text{final answer}
$$

A strong agent should finish with five accepted ROIs, avoid long exploratory
loops, and produce the correct final AML/Normal diagnosis.

## Notation

Let an experiment contain $N$ slide-level runs.

For slide $i$:

- $y_i$: ground-truth diagnostic label.
- $\hat{y}_i$: final diagnostic label predicted by the model, if parseable.
- $s_i \in \{0,1\}$: final completion indicator, where $s_i=1$ if the run completed successfully.
- $r_i \in \{0,1\}$: counted retry-failure indicator, where $r_i=1$ if the run needed an automatic retry caused by a non-LLM/API error.
- $t_i$: elapsed runtime in seconds.
- $A_i = (a_{i1}, a_{i2}, \ldots, a_{iT_i})$: ordered tool-action trajectory.
- $T_i = |A_i|$: number of tool calls in the trajectory.
- $m_i$: number of accepted or marked ROIs.
- $q_i$: bounded tool-flow score, with $q_i \in [0,1]$.
- $p_i \in \{0,1\}$: parseability indicator, where $p_i=1$ if $\hat{y}_i$ is parseable.

Let $\mathcal{G}$ be the set of slides with known ground truth and let
$N_g = |\mathcal{G}|$. In the standard AML benchmark, all slides are expected
to have ground truth, so typically $N_g=N$.

Use $[N] = \{1,2,\ldots,N\}$ for index ranges.

Let $\mathcal{P}$ be the set of slides with parseable final AML/Normal
prediction:

$$
\mathcal{P}
=
\{i \in \mathcal{G} : p_i = 1\}
$$

## Reported Headline Metrics

The headline benchmark table reports:

- `total_cases`
- `evaluated`
- `correct`, `accuracy_all_pct`
- `task_success_count`, `task_success_pct`
- `roi5_count`, `roi5_rate_pct`
- `flow_pass_count`, `flow_pass_pct`
- `avg_tool_calls`
- `stable_elapsed_n`, `avg_elapsed_stable_sec`
- `elapsed_ok_n`, `median_elapsed_all_ok_sec`
- `aml_tp`, `aml_fn`, `normal_tn`, `normal_fp`
- `npm1_scored`, `npm1_correct`, `npm1_accuracy_pct` when NPM1 ground truth is available

## Total Cases

Field: `total_cases`

`total_cases` is the number of slide-level runs evaluated in one experiment.

Formula:

$$
\text{total\_cases} = N
$$

Interpretation:

A model run with fewer total cases may not be directly comparable to a full
benchmark run. When comparing experiments, first check that `total_cases` is
the same or understand why it differs.

## Evaluated Cases

Field: `evaluated`

`evaluated` is the number of slides with known ground truth and a parseable
final AML/Normal prediction.

Formula:

$$
\text{evaluated}
= |\mathcal{P}|
$$

Interpretation:

This count is shown next to `total_cases` so readers can see whether the
diagnostic accuracy metric is based on the full benchmark set.

## Diagnostic Accuracy

Fields: `correct`, `accuracy_all_pct`

Diagnostic accuracy measures whether the model produced the correct final
AML/Normal diagnosis.

Slide-level definition:

$$
\mathrm{Correct}_i = \mathbb{1}[\hat{y}_i = y_i]
$$

Count:

$$
\text{correct}
= \sum_{i \in \mathcal{P}} \mathrm{Correct}_i
$$

Rate:

$$
\text{accuracy\_all\_pct}
=
100 \times \frac{\text{correct}}{|\mathcal{P}|}
$$

In the standard benchmark where all slides have ground truth and parseable
predictions are expected for every slide:

$$
\text{accuracy\_all\_pct}
=
100 \times \frac{\text{correct}}{\text{total\_cases}}
$$

Interpretation:

Higher is better. This is the main diagnostic metric and answers: among the
benchmark slides, what percentage were diagnosed correctly?

Important detail:

A failed, missing, or unparseable final prediction is not counted as correct.

## Task Success Rate

Fields: `task_success_count`, `task_success_pct`

Task success measures whether the slide-level run completed successfully at the
system level without any counted retry failure. It does not measure whether the
final diagnosis was correct.

A run is counted as task-successful if:

- the final run status is OK,
- no non-LLM/API retry failure occurred.

Retries caused by LLM API, model availability, rate limit, or service
availability errors are not counted against this metric.

Count:

$$
\text{task\_success\_count}
=
\sum_{i=1}^{N} \mathbb{1}[s_i=1 \land r_i=0]
$$

Rate:

$$
\text{task\_success\_pct}
=
100 \times \frac{\text{task\_success\_count}}{N}
$$

Interpretation:

Higher is better. This metric answers: how often did the agent finish the task
without a terminal error and without needing a non-LLM/API retry?

Relationship to error rate:

The run-level error rate is the complement of task success:

$$
\text{error\_rate\_pct}
=
100 - \text{task\_success\_pct}
$$

Important distinction:

`task_success_pct` is about task execution. `accuracy_all_pct` is about the
diagnostic decision. A run can finish successfully but still make the wrong
diagnosis.

## AML / Normal Confusion Counts

Fields: `aml_tp`, `aml_fn`, `normal_tn`, `normal_fp`

These counts summarize diagnostic outcomes by class. AML is treated as the
positive class and Normal as the negative class.

Definitions:

$$
\text{aml\_tp}
=
\sum_{i \in \mathcal{P}} \mathbb{1}[y_i=\mathrm{AML} \land \hat{y}_i=\mathrm{AML}]
$$

$$
\text{aml\_fn}
=
\sum_{i \in \mathcal{P}} \mathbb{1}[y_i=\mathrm{AML} \land \hat{y}_i=\mathrm{Normal}]
$$

$$
\text{normal\_tn}
=
\sum_{i \in \mathcal{P}} \mathbb{1}[y_i=\mathrm{Normal} \land \hat{y}_i=\mathrm{Normal}]
$$

$$
\text{normal\_fp}
=
\sum_{i \in \mathcal{P}} \mathbb{1}[y_i=\mathrm{Normal} \land \hat{y}_i=\mathrm{AML}]
$$

Interpretation:

- High `aml_tp` is desirable because AML cases are correctly detected.
- Low `aml_fn` is important because false-negative AML predictions are clinically concerning.
- High `normal_tn` is desirable because Normal cases are correctly ruled out.
- Low `normal_fp` is desirable because false-positive AML predictions indicate over-calling disease.

## ROI Completion Rate

Fields: `roi5_count`, `roi5_rate_pct`

The AML workflow requires the agent to select five accepted ROIs. ROI
completion measures whether the agent completed that requirement.

Slide-level indicator:

$$
\mathrm{ROI5}_i = \mathbb{1}[m_i \ge 5]
$$

Count:

$$
\text{roi5\_count}
=
\sum_{i=1}^{N} \mathrm{ROI5}_i
$$

Rate:

$$
\text{roi5\_rate\_pct}
=
100 \times \frac{\text{roi5\_count}}{N}
$$

Interpretation:

Higher is better. This metric answers: how often did the model collect the
required five ROIs?

## Flow Pass Rate

Fields: `flow_pass_count`, `flow_pass_pct`

Flow pass rate measures whether the agent completed the slide and followed the
expected ROI-selection behavior well enough.

A slide passes if all conditions are true:

- the run completed successfully,
- the final diagnosis is parseable,
- at least five ROIs were marked,
- the bounded tool-flow score is at least 0.75.

Slide-level definition:

$$
\mathrm{FlowPass}_i
=
\mathbb{1}[s_i=1 \land p_i=1 \land m_i \ge 5 \land q_i \ge 0.75]
$$

Count:

$$
\text{flow\_pass\_count}
=
\sum_{i=1}^{N} \mathrm{FlowPass}_i
$$

Rate:

$$
\text{flow\_pass\_pct}
=
100 \times \frac{\text{flow\_pass\_count}}{N}
$$

Interpretation:

Higher is better. This is stricter than ROI completion alone: the model must
both collect enough ROIs and keep its tool trajectory within the expected
bounded workflow.

## Average Tool Calls

Field: `avg_tool_calls`

`avg_tool_calls` is the average number of tool calls per slide among slides
with available action trajectories.

Let $\mathcal{S}$ be the set of slides with available trajectories and
$N_s = |\mathcal{S}|$.

Formula:

$$
\text{avg\_tool\_calls}
=
\frac{1}{N_s}
\sum_{i \in \mathcal{S}} T_i
$$

Interpretation:

Lower is generally better when diagnostic accuracy, ROI completion, and flow
pass rate are similar. Tool calls are a proxy for trajectory length, overhead,
latency, and how directly the model follows the intended workflow.

Important detail:

This is the benchmark's single action-count metric. We do not separately
report average step count because each recorded step corresponds to a tool call
in this workflow.

## Stable Average Runtime

Fields: `stable_elapsed_n`, `avg_elapsed_stable_sec`

Stable runtime estimates the average runtime under well-behaved conditions. It
excludes task-error cases and major trajectory-failure cases.

A slide is stable if all conditions are true:

- the run completed successfully,
- the final diagnosis is parseable,
- the slide did not require a non-LLM/API retry,
- the slide passed the bounded-flow criterion.

Slide-level indicator:

$$
\mathrm{Stable}_i
=
\mathbb{1}[s_i=1 \land r_i=0 \land p_i=1 \land q_i \ge 0.75]
$$

Stable-set definition:

$$
\mathcal{I}_{\mathrm{stable}}
=
\{i \in [N] : \mathrm{Stable}_i=1\}
$$

Number of stable slides:

$$
\text{stable\_elapsed\_n}
=
|\mathcal{I}_{\mathrm{stable}}|
$$

Stable average runtime:

$$
\text{avg\_elapsed\_stable\_sec}
=
\frac{1}{|\mathcal{I}_{\mathrm{stable}}|}
\sum_{i \in \mathcal{I}_{\mathrm{stable}}} t_i
$$

Interpretation:

Lower is better when diagnostic accuracy and flow quality are comparable. This
metric answers: how long does the agent take on non-error, flow-passing cases?

Caveat:

Interpret `avg_elapsed_stable_sec` together with `stable_elapsed_n`. If few
slides qualify as stable, the average may not represent the full benchmark.

## Task-Success Median Runtime

Fields: `elapsed_ok_n`, `median_elapsed_all_ok_sec`

This measures typical runtime among task-successful runs. A task-successful run
has final OK status and no non-LLM/API retry failure.

Successful-runtime set:

$$
\mathcal{I}_{\mathrm{ok}}
=
\{i \in [N] : s_i=1 \land r_i=0\}
$$

Number of successful runs:

$$
\text{elapsed\_ok\_n}
=
|\mathcal{I}_{\mathrm{ok}}|
$$

Median successful runtime:

$$
\text{median\_elapsed\_all\_ok\_sec}
=
\operatorname{median}(\{t_i : i \in \mathcal{I}_{\mathrm{ok}}\})
$$

Interpretation:

Lower is better when outcome and flow quality are similar. Median is used
because runtime can be skewed by a small number of very slow runs.

## NPM1 Accuracy

Fields: `npm1_scored`, `npm1_correct`, `npm1_accuracy_pct`

NPM1 accuracy evaluates whether the agent correctly predicts NPM1 status for
AML slides where NPM1 ground truth is available and the model's NPM1 prediction
is parseable.

For scored NPM1 case $j$:

$$
\mathrm{NPM1Correct}_j
=
\mathbb{1}[\widehat{z}_j=z_j]
$$

where $z_j$ is the ground-truth NPM1 label and $\widehat{z}_j$ is the predicted
NPM1 label.

Scored-case count:

$$
\text{npm1\_scored} = N_{\mathrm{npm1}}
$$

Count:

$$
\text{npm1\_correct}
=
\sum_{j=1}^{N_{\mathrm{npm1}}} \mathrm{NPM1Correct}_j
$$

Accuracy:

$$
\text{npm1\_accuracy\_pct}
=
100 \times \frac{\text{npm1\_correct}}{\text{npm1\_scored}}
$$

Interpretation:

Higher is better. This metric is only meaningful for AML cases with available
NPM1 labels, and it should always be interpreted with `npm1_scored`.

## Metrics Not Used In The Headline Table

The headline table does not report separate retry rate, raw average flow score,
average redundant calls, average step count, total
elapsed runtime, or successful-run P90 runtime.

Retry status is still useful for debugging individual runs, but the headline
benchmark focuses on final diagnostic accuracy, task success, ROI completion,
bounded workflow pass rate, tool-call count, runtime, confusion counts, and
optional NPM1 performance.

## Suggested Reporting Text

We evaluated each VLM-agent run using compact outcome-level and trajectory-level
metrics. Diagnostic performance was measured by accuracy and AML/Normal
confusion counts. Task reliability was measured by completion success rate. ROI
behavior was measured by the fraction of slides with five accepted ROIs and by
flow pass rate, which requires a completed, parseable, five-ROI trajectory with
bounded tool-flow quality. Efficiency was summarized with average tool calls
and runtime among successful or stable runs. NPM1 accuracy was reported for AML
slides with available NPM1 ground truth.
