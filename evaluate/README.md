# evaluate/ — CLI Runs

Headless CLI entrypoints for AML runs.

## Agents

| `--agent` value | Pipeline stage | Description |
|---|---|---|
| `aml_auto` (alias `aml`) | Stage 1 → 2 | Full two-stage pipeline: ROI collection then diagnosis (production default) |
| `aml_roi` | Stage 1 only | Navigate WSI and collect ROIs; outputs `roi_collection.json` + images |
| `aml_diagnosis` | Stage 2 only | Diagnose from an existing `roi_collection.json`; no slide file needed |
| `aml_detector` | single stage | Legacy single-pass AML agent (navigation + diagnosis in one turn) |
| `tile` | single stage | Tile selection and saving |
| `wsi` | single stage | General WSI pathology exploration |

## Single slide

Run one slide end-to-end (two-stage AML pipeline):

```bash
.venv/bin/python evaluate/run_single_slide.py \
    --slide /path/to/patient.mrxs \
    --output-dir ./batch_outputs \
    --model gemma-4-31B-it \
    --extractor reddino_large \
    --tile-filter hybrid \
    --tile-size-px 224 \
    --batch-size 512 \
    --agent aml_auto
```

Run Stage 1 only (ROI collection — no diagnosis):

```bash
.venv/bin/python evaluate/run_single_slide.py \
    --slide /path/to/patient.mrxs \
    --output-dir ./batch_outputs \
    --model gemma-4-31B-it \
    --extractor reddino_large \
    --tile-filter hybrid \
    --agent aml_roi
```

Run Stage 2 only (diagnosis from existing ROI collection):

```bash
.venv/bin/python evaluate/run_single_slide.py \
    --roi-input /path/to/case_dir/roi_collection.json \
    --output-dir ./batch_outputs \
    --model gemma-4-31B-it \
    --agent aml_diagnosis
```

Useful flags:

- `--model`: VLM name, for example `GLM-4.6V-FP8`, `gemma-4-31B-it`, `Qwen3.5-397B-A17B-FP8`
- `--extractor`: embedding extractor key such as `uni2`, `h_optimus_1`, `virchow2`, `dinobloom`, `dinobloom_giant`, `reddino`, `reddino_base`, `reddino_large`
- `--tile-filter`: one of `hybrid`, `quality`, `coarse`, `none`
- `--experiment-root`: shared cache/output root for repeated runs
- `--use-tile-cache`: reuse persisted tile cache across runs

Outputs are written under `--output-dir/<patient>/` and include:

- `summary.json`
- `final_output.txt`
- `report.json`
- copied ROI/debug images when available

## Batch AML run

Run the AML detector across a CSV of patients or slide stems:

```bash
bash evaluate/run_batch_aml.sh \
    --csv /path/to/patients.csv \
    --slides-root /mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs \
    --output-dir ./batch_result_qwen_reddino_large \
    --experiment-root ./batch_result_qwen_reddino_large \
    --model Qwen3.5-397B-A17B-FP8 \
    --extractor reddino_large \
    --tile-filter hybrid \
    --tile-size-px 224 \
    --batch-size 512 \
    --agent aml \
    --resume \
    --use-tile-cache
```

Notes:

- The CSV is read line-by-line after the header.
- Each row can be either a patient stem or a full `.mrxs` path.
- `--resume` skips patients whose `summary.json` has `status="ok"` and a non-empty `final_decision`.
- If a run fails during the current batch, the script automatically retries that slide once.

## Pre-extract shared cache

If you plan to run a large batch with `--use-tile-cache`, you can prewarm the shared AML tile cache first:

```bash
.venv/bin/python evaluate/preextract_hybrid_cache.py \
    --csv /path/to/patients.csv \
    --slides-root /mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs \
    --experiment-root ./aml_reddino_large_suite \
    --extractor reddino_large \
    --tile-filter hybrid \
    --agent aml \
    --tile-size-px 224 \
    --tile-size-um 256 \
    --batch-size 512 \
    --skip-existing-cache
```

Notes:

- This populates the shared cache under `<experiment-root>/_cache/tile_cache/<extractor>/`.
- It also prepares the AML reference cache under `<experiment-root>/_cache/reference_hnsw/<extractor>/`.
- `--skip-existing-cache` avoids recomputing slides that already have at least one cache zip for that extractor.
- `--limit N` is useful for a quick dry run on a subset of slides.
- This cache layout is the same one reused by `run_batch_aml.sh` and `run_batch_aml_suite.sh` when `--use-tile-cache` is enabled.

## Batch suite

`run_batch_aml_suite.sh` is a wrapper for launching multiple model/extractor combinations defined in the script's `RUNS` array. By default it runs strictly sequentially (one combo at a time); use `--parallel-models` to saturate multiple VLM endpoints simultaneously.

```bash
bash evaluate/run_batch_aml_suite.sh \
    --output-parent /mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin \
    --experiment-name aml_gemma4_embedding_suite \
    --cuda-device 0 \
    --extractors h_optimus_1 \
    --resume \
    --use-tile-cache
```

Notes:

- `--extractors` accepts a comma-separated list of extractor keys and filters the `RUNS` array to only those matching entries.
- `--models` accepts a comma-separated list of model names and filters the `RUNS` array to only those matching entries.
- `--parallel-models` launches one background worker per distinct model so all VLM endpoints are saturated; each worker still processes its slides sequentially. Ctrl-C kills all workers cleanly.
- `--slide-timeout N` kills a single slide run after N seconds (0 = disabled).
- `--auto-restart N` automatically re-invokes the suite with `--resume` up to N times if the run exits non-zero (default: 3). Pair with `--auto-restart-delay S` to set the wait between restarts (default: 30 s).
- The suite script forwards into `run_batch_aml.sh` for each selected run.
- Edit the `RUNS` array in [run_batch_aml_suite.sh](run_batch_aml_suite.sh) to choose which model/extractor combinations are launched.
- Edit `tools.slide` in [configs/config.yaml](../configs/config.yaml) to change the suite defaults for `TILE_FILTER`, `TILE_SIZE_PX`, `BATCH_SIZE`, `ROI_SIZE_PX`, and `AGENT`. These can also be overridden per-invocation with `--tile-filter`, `--roi-size-px`, and `--default-mpp-um`.
- Each `RUNS` entry has the form `"MODEL|EXTRACTOR|OUTPUT_DIR_NAME"`.
- `MODEL` is passed to `--model`, `EXTRACTOR` is passed to `--extractor`, and `OUTPUT_DIR_NAME` becomes the subdirectory created under `--output-parent/--experiment-name` or `--base-output-root`.

## Batch diagnosis

`run_batch_aml_diagnosis.sh` runs the `aml_diagnosis` agent on ROI images from a prior `aml_roi` / `aml_auto` experiment run — no slide files required.

```bash
# Run all model-extractor subfolders in parallel, results written in-place:
bash evaluate/run_batch_aml_diagnosis.sh \
    --exp-path /path/to/exp_290425_aml_suite \
    --parallel \
    --resume

# Mirror results to a separate directory:
bash evaluate/run_batch_aml_diagnosis.sh \
    --exp-path /path/to/exp_290425_aml_suite \
    --output-root /path/to/exp_diagnosis \
    --parallel \
    --resume

# Run a single subfolder sequentially:
bash evaluate/run_batch_aml_diagnosis.sh \
    --exp-path /path/to/exp_290425_aml_suite/GLM-4.6V-FP8_DinoBloom-G_224px
```

Notes:

- `--exp-path` accepts either an experiment root (containing model-extractor subdirs) or a single model-extractor subfolder. ROI images are discovered automatically under `<slide_id>/images/`.
- `--model` overrides the VLM used for diagnosis (default: `MODEL_NAME` from `configs/config.yaml`).
- `--output-root` mirrors the experiment structure into a new directory instead of writing in-place.
- `--subdir-filter` accepts a comma-separated list of subfolder names to restrict which model-extractor dirs are processed.
- `--parallel` launches one background process per model-extractor subfolder; logs are written to `<exp-path>/_diagnosis_logs/`.
- `--chunks-dir` / `--chunk-index` split work across part CSV files for multi-node runs. `--chunk-index` is 1-based.
- `--resume` skips slide dirs that already have a `diagnosis_done` marker.
- On success, ROI images from the original AML output are copied into the diagnosis output directory.

## Post-hoc ROI collection from prior runs

`create_roi_collections.py` (in the project root) reconstructs `roi_collection.json` files
for experiment runs that pre-date the two-stage pipeline or whose collection files were lost.
It walks an experiment directory tree (`exp_root/{model_extractor}/{slide_id}/`) and generates
a `roi_collection.json` per slide by parsing `final_output.txt` and `state.json`.

```bash
python create_roi_collections.py /path/to/experiment/root
```

Example:

```bash
python create_roi_collections.py \
    /mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/exp_290425_aml_suite
```

Notes:

- Requires `final_output.txt`, `state.json`, and an `images/` directory to exist for each slide.
- ROI metadata (bounding boxes, tissue fraction, field width) is recovered from `state.json`'s `roi_marks` list.
- The generated schema matches the `aml_roi` runtime schema and can be fed directly to `run_batch_aml_diagnosis.sh`.
- Slides missing any required file are skipped with a `[SKIP]` message; slides whose JSON cannot be parsed are logged as `[WARN]`.
