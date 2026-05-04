# AML Agent Benchmark Metrics

This document defines the metrics used in the AML agent benchmark. The benchmark evaluates two complementary aspects of each VLM-agent run:

- Diagnostic outcome: whether the agent completed the slide-level task and produced the correct AML/Normal diagnosis.
- Agent behavior: whether the agent followed a concise, bounded ROI-selection workflow without excessive or redundant tool use.

The benchmark intentionally keeps the metric set compact. We report one action-count metric, `avg_tool_calls`, and do not separately report average step count because each recorded step corresponds to a tool call in this workflow.

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

A strong agent should usually finish with five accepted ROIs, few redundant calls, and no long exploratory loops.

## Notation

Let an experiment contain $N$ slide-level runs.

For slide $i$:

- $y_i$: ground-truth diagnostic label.
- $\hat{y}_i$: final diagnostic label predicted by the model, if parseable.
- $s_i \in \{0,1\}$: final completion indicator, where $s_i=1$ if the run completed successfully.
- $r_i \in \{0,1\}$: retry indicator, where $r_i=1$ if the first attempt failed and an automatic retry was triggered.
- $t_i$: elapsed runtime in seconds.
- $A_i = (a_{i1}, a_{i2}, \ldots, a_{iT_i})$: ordered tool-action trajectory.
- $T_i = |A_i|$: number of tool calls in the trajectory.
- $m_i$: number of accepted or marked ROIs.
- $d_i$: number of redundant tool calls.
- $q_i$: bounded tool-flow score, with $q_i \in [0,1]$.

Let $\mathcal{G}$ be the set of slides with known ground truth and let $N_g = |\mathcal{G}|$. In the standard benchmark, all slides are expected to have ground truth, so typically $N_g=N$.

Unless otherwise stated, rates use all benchmark slides as the denominator. Failed or unparseable runs are therefore penalized rather than excluded.

## Reported Benchmark Metrics

The benchmark reports the following fields:

- `total_cases`
- `correct`
- `accuracy_all_pct`
- `aml_tp`, `aml_fn`, `normal_tn`, `normal_fp`
- `unparsed`
- `status_ok`, `success_status_pct`
- `retry_count`, `first_attempt_fail_pct`
- `roi5_count`, `roi5_rate_pct`
- `flow_pass_count`, `flow_pass_pct`
- `avg_flow_score`
- `avg_tool_calls`
- `avg_redundant_calls`
- `stable_elapsed_n`, `avg_elapsed_stable_sec`
- `elapsed_ok_n`, `median_elapsed_all_ok_sec`
- `npm1_scored`, `npm1_correct`, `npm1_accuracy_pct`

## Total Cases

Field: `total_cases`

What it measures:

`total_cases` is the number of slide-level runs evaluated in one experiment. It is the denominator for most benchmark rates.

Formula:

$$
\text{total\_cases} = N
$$

Interpretation:

A model run with fewer total cases may not be directly comparable to a full benchmark run. When comparing experiments, first check that `total_cases` is the same or understand why it differs.

Why it matters:

This prevents partial runs from being mistaken for full benchmark results.

## Correct Diagnoses

Field: `correct`

What it measures:

`correct` counts the number of slides where the final diagnostic prediction is parseable and matches the ground truth.

Slide-level definition:

$$
\mathrm{Correct}_i = \mathbb{1}[\hat{y}_i = y_i]
$$

Experiment-level formula:

$$
\text{correct} = \sum_{i \in \mathcal{G}} \mathrm{Correct}_i
$$

Interpretation:

Higher is better. This is a raw count, so it should be interpreted together with `total_cases` or `accuracy_all_pct`.

Important detail:

A failed, missing, or unparseable final prediction is not correct. This means execution failures and invalid final answers reduce benchmark performance.

## Diagnostic Accuracy

Field: `accuracy_all_pct`

What it measures:

`accuracy_all_pct` is the all-case diagnostic accuracy. Failed, missing, or unparseable predictions are counted as incorrect.

Formula:

$$
\mathrm{Accuracy}_{\mathrm{all}}
=
\frac{\sum_{i \in \mathcal{G}} \mathrm{Correct}_i}{N_g}
$$

Reported percentage:

$$
\text{accuracy\_all\_pct}
=
100 \times \mathrm{Accuracy}_{\mathrm{all}}
$$

In the standard benchmark where all slides have ground truth:

$$
\text{accuracy\_all\_pct}
=
100 \times \frac{\text{correct}}{\text{total\_cases}}
$$

Interpretation:

Higher is better. This metric answers: among all evaluated slides, what percentage were diagnosed correctly?

Why it matters:

This is stricter than accuracy computed only on parseable successful outputs. It penalizes models that fail, omit the final answer, or produce an unparseable diagnosis.

Caveat:

A high diagnostic accuracy does not necessarily mean the agent behaved well. A model can be diagnostically accurate while still using too many tools or failing to collect the required ROIs. For that reason, accuracy should be interpreted together with `flow_pass_pct`, `avg_tool_calls`, and `avg_redundant_calls`.

## AML / Normal Confusion Counts

Fields: `aml_tp`, `aml_fn`, `normal_tn`, `normal_fp`

What they measure:

These counts summarize diagnostic errors by class. AML is treated as the positive class and Normal as the negative class.

Definitions:

$$
\text{aml\_tp}
=
\sum_i \mathbb{1}[y_i=\mathrm{AML} \land \hat{y}_i=\mathrm{AML}]
$$

$$
\text{aml\_fn}
=
\sum_i \mathbb{1}[y_i=\mathrm{AML} \land \hat{y}_i=\mathrm{Normal}]
$$

$$
\text{normal\_tn}
=
\sum_i \mathbb{1}[y_i=\mathrm{Normal} \land \hat{y}_i=\mathrm{Normal}]
$$

$$
\text{normal\_fp}
=
\sum_i \mathbb{1}[y_i=\mathrm{Normal} \land \hat{y}_i=\mathrm{AML}]
$$

Interpretation:

- High `aml_tp` is desirable because AML cases are correctly detected.
- Low `aml_fn` is important because false-negative AML predictions are clinically concerning.
- High `normal_tn` is desirable because Normal cases are correctly ruled out.
- Low `normal_fp` is desirable because false-positive AML predictions indicate over-calling disease.

Important detail:

Only slides with parseable diagnostic predictions are included in these confusion counts. Slides without parseable predictions are counted separately as `unparsed`.

## Other / Unparsed Cases

Field: `unparsed`

What it measures:

`unparsed` counts slides that cannot be included in the diagnostic confusion matrix because the prediction or ground truth is unavailable or unparseable.

Formula:

$$
\text{unparsed}
=
N - N_{\mathrm{scored}}
$$

where $N_{\mathrm{scored}}$ is the number of slides with both known ground truth and parseable prediction.

Interpretation:

Lower is better. A high `unparsed` count indicates that the model often failed to produce a usable final diagnosis, even if the run itself may have completed.

Why it matters:

Unparseable outputs are a practical failure mode for an autonomous agent. They should not be silently excluded from diagnostic evaluation.

## Final Completion Success

Fields: `status_ok`, `success_status_pct`

What it measures:

These metrics measure whether the run completed successfully at the system level, independent of diagnostic correctness.

Count:

$$
\text{status\_ok}
=
\sum_{i=1}^{N} s_i
$$

Rate:

$$
\mathrm{SuccessStatusRate}
=
\frac{\sum_{i=1}^{N} s_i}{N}
$$

Reported percentage:

$$
\text{success\_status\_pct}
=
100 \times \mathrm{SuccessStatusRate}
$$

Interpretation:

Higher is better. This metric answers: how often did the agent finish the run without terminal failure?

Why it matters:

A model may be accurate on completed slides but unreliable overall if many runs terminate early or fail. `success_status_pct` captures this reliability dimension.

Difference from diagnostic accuracy:

`success_status_pct` measures completion. `accuracy_all_pct` measures correctness. A completed run can still be diagnostically wrong, and a failed run is counted as incorrect in all-case accuracy.

## First-Attempt Failure Rate

Fields: `retry_count`, `first_attempt_fail_pct`

What it measures:

These metrics measure instability before automatic retry recovery.

Count:

$$
\text{retry\_count}
=
\sum_{i=1}^{N} r_i
$$

Rate:

$$
\mathrm{FirstAttemptFailRate}
=
\frac{\sum_{i=1}^{N} r_i}{N}
$$

Reported percentage:

$$
\text{first\_attempt\_fail\_pct}
=
100 \times \mathrm{FirstAttemptFailRate}
$$

Interpretation:

Lower is better. A high value means the model-agent system often fails on the first attempt, even if some failures are later recovered by retry.

Why it matters:

Final success alone can hide instability. If a model frequently needs retry, it is less reliable and may be more costly or operationally fragile.

Difference from final completion success:

`success_status_pct` asks whether the final attempt succeeded. `first_attempt_fail_pct` asks whether the original attempt failed. Both are useful because retry-recovered slides are still signs of instability.

## ROI Completion Rate

Fields: `roi5_count`, `roi5_rate_pct`

What it measures:

The AML workflow requires the agent to select five accepted ROIs. These metrics measure whether the agent completed that ROI collection requirement.

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
\mathrm{ROI5Rate}
=
\frac{\sum_{i=1}^{N} \mathrm{ROI5}_i}{N}
$$

Reported percentage:

$$
\text{roi5\_rate\_pct}
=
100 \times \mathrm{ROI5Rate}
$$

Interpretation:

Higher is better. This metric answers: how often did the model collect the required five ROIs?

Why it matters:

The final diagnosis should be supported by sufficient local evidence. A model that stops early, marks too few ROIs, or fails to complete the ROI protocol is not following the intended workflow.

Caveat:

`roi5_rate_pct` only measures quantity of accepted ROIs. It does not by itself measure whether the agent used tools efficiently or followed the correct sequence. For that, use `avg_flow_score`, `flow_pass_pct`, and `avg_redundant_calls`.

## Average Tool Calls

Field: `avg_tool_calls`

What it measures:

`avg_tool_calls` is the average number of tool calls per slide among slides with available action trajectories.

Let $\mathcal{S}$ be the set of slides with available trajectories and $N_s = |\mathcal{S}|$.

Formula:

$$
\text{avg\_tool\_calls}
=
\frac{1}{N_s}
\sum_{i \in \mathcal{S}} T_i
$$

Interpretation:

Lower is generally better when diagnostic and ROI performance are similar. A concise agent should collect five ROIs and finish without unnecessary exploration.

Why it matters:

Tool calls are a proxy for interaction length, computational overhead, latency, and the model's ability to follow the intended workflow.

Important detail:

This is the benchmark's single action-count metric. We do not separately report average step count because each recorded step corresponds to a tool call in this workflow.

Caveat:

Very low `avg_tool_calls` is not always good. It may indicate that the model stopped too early or failed to collect five ROIs. Interpret this metric together with `roi5_rate_pct` and `flow_pass_pct`.

## Average Redundant Tool Calls

Field: `avg_redundant_calls`

What it measures:

`avg_redundant_calls` estimates unnecessary or inefficient tool use. It is designed to capture tool-looping, repeated candidate inspection, over-exploration, and actions that occur outside the expected workflow.

A redundant action includes:

- repeated candidate openings without marking an ROI,
- more than two local inspection actions between candidate opening and ROI marking,
- extra overview calls beyond the initial overview,
- unexpected navigation or marking outside the expected workflow,
- tool calls after five ROIs have already been accepted.

Formula:

$$
\text{avg\_redundant\_calls}
=
\frac{1}{N_s}
\sum_{i \in \mathcal{S}} d_i
$$

Interpretation:

Lower is better. A high value suggests the model is confused, looping, over-inspecting, or failing to efficiently convert candidates into accepted ROIs.

Why it matters:

Weak VLMs may eventually finish but require many unnecessary actions. Redundant-call metrics expose this inefficiency even when final diagnostic accuracy appears acceptable.

Relationship to `avg_tool_calls`:

`avg_tool_calls` measures total trajectory length. `avg_redundant_calls` estimates the inefficient portion of that trajectory.

## Bounded Tool-Flow Score

Field: `avg_flow_score`

What it measures:

`avg_flow_score` measures how closely the model follows the expected AML ROI workflow. It combines ROI completion, sequence correctness, local inspection bounds, avoidance of late tools, and redundancy control.

Expected trajectory:

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

Slide-level score:

$$
q_i
=
0.20 O_i
+
0.25 R_i
+
0.20 L_i
+
0.15 C_i
+
0.10 F_i
+
0.10 D_i
$$

Component: overview score

$$
O_i
=
\mathbb{1}[\text{exactly one overview call}]
$$

This rewards using a single whole-slide overview at the start rather than repeatedly returning to overview.

Component: ROI completion score

$$
R_i
=
\frac{\min(m_i,5)}{5}
$$

This gives partial credit for collecting fewer than five ROIs and full credit for collecting at least five.

Component: bounded local inspection score

$$
L_i
=
\max\left(0, 1 - \frac{\mathrm{local\_overuse}_i}{\max(1, \mathrm{local\_action\_count}_i)}\right)
$$

This penalizes excessive local zooming or panning beyond the intended 0 to 2 local actions between candidate opening and ROI marking.

Component: candidate-to-mark score

$$
C_i
=
\frac{\mathrm{marked\_ROIs\_preceded\_by\_candidate\_opening}_i}{\max(1,m_i)}
$$

This rewards marking ROIs after opening candidate regions, rather than marking without a valid candidate-driven inspection sequence.

Component: no-late-tool score

$$
F_i
=
\mathbb{1}[\text{no tool calls after the fifth ROI mark}]
$$

This penalizes continued tool use after the required five ROIs have already been accepted.

Component: low-redundancy score

$$
D_i
=
\max\left(0, 1 - \frac{d_i}{10}\right)
$$

This gradually penalizes redundant actions. The score reaches zero for this component when redundancy is very high.

Experiment-level formula:

$$
\text{avg\_flow\_score}
=
\frac{1}{N_s}
\sum_{i \in \mathcal{S}} q_i
$$

Interpretation:

Higher is better. The score ranges from 0 to 1. A score near 1 indicates a concise, bounded, protocol-following trajectory. A lower score indicates missing ROIs, repeated tools, excessive local search, or incorrect ordering.

Why it matters:

This metric distinguishes models that merely produce an answer from models that actually behave like controlled ROI-selection agents.

## Flow Pass Rate

Fields: `flow_pass_count`, `flow_pass_pct`

What it measures:

Flow pass rate measures the fraction of slides where the agent both completed the run and followed the expected ROI-selection behavior well enough.

A slide passes if all conditions are true:

- the run completed successfully,
- the final diagnosis is parseable,
- at least five ROIs were marked,
- the bounded tool-flow score is at least 0.75.

Slide-level definition:

$$
\mathrm{FlowPass}_i
=
\mathbb{1}[s_i=1 \land \hat{y}_i \text{ is parseable} \land m_i \ge 5 \land q_i \ge 0.75]
$$

Count:

$$
\text{flow\_pass\_count}
=
\sum_{i=1}^{N} \mathrm{FlowPass}_i
$$

Rate:

$$
\mathrm{FlowPassRate}
=
\frac{\sum_{i=1}^{N} \mathrm{FlowPass}_i}{N}
$$

Reported percentage:

$$
\text{flow\_pass\_pct}
=
100 \times \mathrm{FlowPassRate}
$$

Interpretation:

Higher is better. This is a stricter behavioral pass metric than ROI completion alone.

Why it matters:

A model may mark five ROIs but do so through excessive looping or poor sequencing. `flow_pass_pct` requires both ROI completion and acceptable tool-flow quality.

Relationship to `avg_flow_score`:

`avg_flow_score` is a continuous average. `flow_pass_pct` is a thresholded pass rate using a score cutoff of 0.75.

## Stable Average Runtime

Fields: `stable_elapsed_n`, `avg_elapsed_stable_sec`

What it measures:

Stable runtime estimates the average runtime under well-behaved conditions. It excludes retry-driven and major trajectory-failure cases.

A slide is stable if all conditions are true:

- the run completed successfully,
- the final diagnosis is parseable,
- the slide did not require retry,
- the slide passed the bounded-flow criterion.

Slide-level definition:

$$
\mathrm{Stable}_i
=
\mathbb{1}[s_i=1 \land \hat{y}_i \text{ is parseable} \land r_i=0 \land \mathrm{FlowPass}_i=1]
$$

Stable-set definition:

$$
\mathcal{T}_{\mathrm{stable}}
=
\{t_i : \mathrm{Stable}_i=1\}
$$

Number of stable slides:

$$
\text{stable\_elapsed\_n}
=
|\mathcal{T}_{\mathrm{stable}}|
$$

Stable average runtime:

$$
\text{avg\_elapsed\_stable\_sec}
=
\frac{1}{|\mathcal{T}_{\mathrm{stable}}|}
\sum_{t_i \in \mathcal{T}_{\mathrm{stable}}} t_i
$$

Interpretation:

Lower is better when accuracy and flow quality are comparable. This metric answers: how long does the agent typically take when it behaves correctly and does not need retry?

Why it matters:

Weak models may have long runtime tails due to tool confusion or retries. Stable runtime isolates the normal operating regime from those failure modes.

Caveat:

Always interpret `avg_elapsed_stable_sec` together with `stable_elapsed_n`. If very few slides qualify as stable, the average may not represent the full benchmark.

## Successful-Run Median Runtime

Fields: `elapsed_ok_n`, `median_elapsed_all_ok_sec`

What it measures:

This measures typical runtime among final successful runs, including retry-recovered slides if the final result completed successfully.

Successful-runtime set:

$$
\mathcal{T}_{\mathrm{ok}}
=
\{t_i : s_i=1\}
$$

Number of successful runs:

$$
\text{elapsed\_ok\_n}
=
|\mathcal{T}_{\mathrm{ok}}|
$$

Median successful runtime:

$$
\text{median\_elapsed\_all\_ok\_sec}
=
\operatorname{median}(\mathcal{T}_{\mathrm{ok}})
$$

Interpretation:

Lower is better when outcome and flow quality are similar. Median is used because runtime can be skewed by a small number of very slow runs.

Why it matters:

This metric provides a robust estimate of typical runtime among completed cases. It complements `avg_elapsed_stable_sec`, which is stricter and excludes retry or flow-failure cases.

Difference from stable average runtime:

`median_elapsed_all_ok_sec` includes all final successful runs. `avg_elapsed_stable_sec` includes only successful, parseable, non-retry, flow-passing runs.

## NPM1 Accuracy

Fields: `npm1_scored`, `npm1_correct`, `npm1_accuracy_pct`

What it measures:

NPM1 accuracy evaluates whether the agent correctly predicts NPM1 status for AML slides where NPM1 ground truth is available and the model's NPM1 prediction is parseable.

For scored NPM1 case $j$:

$$
\mathrm{NPM1Correct}_j
=
\mathbb{1}[\widehat{z}_j=z_j]
$$

where $z_j$ is the ground-truth NPM1 label and $\widehat{z}_j$ is the predicted NPM1 label.

Count:

$$
\text{npm1\_correct}
=
\sum_{j=1}^{N_{\mathrm{npm1}}} \mathrm{NPM1Correct}_j
$$

Denominator:

$$
\text{npm1\_scored}
=
N_{\mathrm{npm1}}
$$

Accuracy:

$$
\text{npm1\_accuracy\_pct}
=
100 \times \frac{\text{npm1\_correct}}{\text{npm1\_scored}}
$$

Interpretation:

Higher is better. This metric is only meaningful for AML cases with available NPM1 labels.

Caveat:

If `npm1_scored` is small, `npm1_accuracy_pct` may be unstable. It should be interpreted with its denominator.

## Metrics Not Reported

The benchmark does not report the following metrics:

- average step count,
- total elapsed runtime,
- successful-run P90 runtime.

Average step count is redundant with average tool calls. Total elapsed runtime is dominated by the number of slides in a run rather than model behavior. Successful-run P90 runtime was removed to keep the benchmark focused on the most interpretable runtime summaries.

## Suggested Reporting Text

We evaluated each VLM run using outcome-level and trajectory-level metrics. Diagnostic performance was measured by all-case accuracy and AML/Normal confusion counts. Reliability was measured by final completion success and first-attempt failure rate. ROI behavior was assessed by the fraction of slides with five accepted ROIs. Tool-use quality was evaluated using average tool calls, redundant calls, and a bounded workflow score reflecting adherence to the expected overview-candidate-inspection-marking trajectory. Runtime was summarized using stable-slide average elapsed time and successful-run median elapsed time, separating well-behaved trajectories from retry or failure modes.
