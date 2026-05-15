# AML Agent Benchmark Metrics

This document describes the metrics currently exported by
`evaluate/export_batch_results_stats.py`.

The exporter reads per-slide `summary.json` and `report.json` files, normalizes
the final decision into a small label set, writes a per-slide CSV, and writes a
summary CSV of aggregate classification metrics.

## Scope

The current exporter uses the following rules:

- Ground truth is inferred from the slide directory name.
- A name starting with `aml` is treated as ground-truth `AML`.
- A name starting with `normaleskm` is treated as ground-truth `Normal marrow`.
- The model prediction is normalized from `final_decision`.
- `Acute leukemia` maps to predicted label `AML`.
- `Normal marrow` maps to predicted label `Normal marrow`.
- `Call for more diagnostics` stays `Call for more diagnostics`.
- Any other or missing decision becomes `Unknown`.
- If `summary_status == "ok"` but the decision is still missing, the exporter
  coerces the prediction to `Call for more diagnostics`.
- Slides with `summary_status == "error"` are excluded from the aggregate
  statistics. All other slides remain in the denominator.

## Exported Files

The exporter writes:

- `<prefix>_slides.csv`
- `<prefix>_summary.csv`

The per-slide CSV currently includes:

- `slide_name`
- `ground_truth`
- `summary_status`
- `tile_size_px`
- `predicted_label`
- `included_in_statistics`
- `is_correct`
- `error`
- `elapsed_sec`

The summary CSV currently includes:

- `total_slides_all`
- `slides_in_statistics`
- `aml_slides`
- `normal_slides`
- `ok_runs`
- `error_runs`
- `call_for_more_diagnostics_predictions`
- `correct_predictions`
- `accuracy_all_nonerror_slides_pct`
- `tp_aml_pred_aml`
- `tn_normal_pred_normal`
- `fp_normal_pred_aml`
- `fn_aml_pred_normal`
- `sensitivity_recall_aml_pct`
- `specificity_normal_pct`
- `precision_aml_pct`
- `npv_normal_pct`
- `f1_aml_pct`
- `balanced_accuracy_pct`

## Notation

Let the experiment contain $N$ slide directories.

For slide $i$:

- $y_i$ is the inferred ground-truth label.
- $\hat{y}_i$ is the normalized predicted label.
- $s_i \in \{\texttt{ok}, \texttt{error}, \texttt{other}\}$ is the exported
  run status.

Let the non-error set be:

$$
\mathcal{S} = \{ i \in [N] : s_i \neq \texttt{error} \}.
$$

These are exactly the slides marked `included_in_statistics = true`.

## Slide Counts

Fields:

- `total_slides_all`
- `slides_in_statistics`
- `aml_slides`
- `normal_slides`
- `ok_runs`
- `error_runs`

Definitions:

$$
N_{\mathrm{all}} = N
$$

$$
N_{\mathrm{stats}} = |\mathcal{S}|
$$

$$
N_{\mathrm{AML}} = \sum_{i \in \mathcal{S}} \mathbb{1}[y_i = \mathrm{AML}]
$$

$$
N_{\mathrm{Normal}} = \sum_{i \in \mathcal{S}} \mathbb{1}[y_i = \mathrm{NormalMarrow}]
$$

$$
N_{\mathrm{ok}} = \sum_{i=1}^{N} \mathbb{1}[s_i = \texttt{ok}]
$$

$$
N_{\mathrm{error}} = \sum_{i=1}^{N} \mathbb{1}[s_i = \texttt{error}]
$$

Interpretation:

- `slides_in_statistics` is the denominator for the main accuracy metric.
- `ok_runs` and `error_runs` are raw run-status counts and are not restricted to
  the non-error subset because they already summarize it directly.

## Call-For-More Count

Field: `call_for_more_diagnostics_predictions`

This counts how often the normalized prediction is `Call for more diagnostics`
within the non-error subset:

$$
C_{\mathrm{callmore}} =
\sum_{i \in \mathcal{S}}
\mathbb{1}[\hat{y}_i = \mathrm{CallForMoreDiagnostics}]
$$

Interpretation:

- Higher values indicate more non-binary or fallback outcomes among completed
  runs.
- Because these slides stay in `slides_in_statistics`, they lower the main
  accuracy unless they are excluded manually downstream.

## Correct Predictions And Accuracy

Fields:

- `correct_predictions`
- `accuracy_all_nonerror_slides_pct`

The exporter marks a slide as correct only when:

- `summary_status == "ok"`,
- the normalized prediction is either `AML` or `Normal marrow`,
- and that prediction matches the inferred ground truth.

Formally:

$$
\mathrm{Correct}_i =
\mathbb{1}\left[
  s_i = \texttt{ok}
  \land \hat{y}_i \in \{\mathrm{AML}, \mathrm{NormalMarrow}\}
  \land \hat{y}_i = y_i
\right]
$$

$$
C_{\mathrm{correct}} = \sum_{i \in \mathcal{S}} \mathrm{Correct}_i
$$

$$
\mathrm{Acc}_{\mathrm{nonerror}}(\%) =
100 \times \frac{C_{\mathrm{correct}}}{N_{\mathrm{stats}}}
$$

Important detail:

- The denominator is all non-error slides, not only binary-prediction slides.
- Therefore `Call for more diagnostics`, `Unknown`, and other non-binary
  completed outcomes reduce `accuracy_all_nonerror_slides_pct`.

## Confusion Counts

Fields:

- `tp_aml_pred_aml`
- `tn_normal_pred_normal`
- `fp_normal_pred_aml`
- `fn_aml_pred_normal`

Definitions over the non-error subset:

$$
\mathrm{TP} = \sum_{i \in \mathcal{S}} \mathbb{1}[y_i=\mathrm{AML} \land \hat{y}_i=\mathrm{AML}]
$$

$$
\mathrm{TN} = \sum_{i \in \mathcal{S}} \mathbb{1}[y_i=\mathrm{NormalMarrow} \land \hat{y}_i=\mathrm{NormalMarrow}]
$$

$$
\mathrm{FP} = \sum_{i \in \mathcal{S}} \mathbb{1}[y_i=\mathrm{NormalMarrow} \land \hat{y}_i=\mathrm{AML}]
$$

$$
\mathrm{FN} = \sum_{i \in \mathcal{S}} \mathbb{1}[y_i=\mathrm{AML} \land \hat{y}_i=\mathrm{NormalMarrow}]
$$

Important detail:

- `Call for more diagnostics` and `Unknown` are not folded into these four
  confusion counts.
- As a result, class-metric denominators can be smaller than
  `slides_in_statistics`.

## Derived Classification Metrics

Fields:

- `sensitivity_recall_aml_pct`
- `specificity_normal_pct`
- `precision_aml_pct`
- `npv_normal_pct`
- `f1_aml_pct`
- `balanced_accuracy_pct`

Formulas:

$$
\mathrm{Sensitivity}(\%) = 100 \times \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}
$$

$$
\mathrm{Specificity}(\%) = 100 \times \frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}}
$$

$$
\mathrm{Precision}(\%) = 100 \times \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}
$$

$$
\mathrm{NPV}(\%) = 100 \times \frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FN}}
$$

$$
\mathrm{F1}(\%) = 100 \times \frac{2\mathrm{TP}}{2\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}
$$

$$
\mathrm{BalancedAccuracy}(\%) =
\frac{\mathrm{Sensitivity}(\%) + \mathrm{Specificity}(\%)}{2}
$$

The exporter leaves a metric blank when its denominator is zero.

## Runtime And Other Fields

The per-slide CSV preserves `elapsed_sec`, but the current summary exporter does
not compute aggregate runtime metrics.

The current exporter also does not compute:

- ROI completion metrics
- tool-flow metrics
- average tool-call counts
- NPM1 accuracy summaries

Those metrics may still be derivable from raw artifacts such as `state.json`,
but they are not part of the current `export_batch_results_stats.py` output.
