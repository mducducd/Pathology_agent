# Evaluate CLI Guide

This folder contains the headless CLI and batch tooling for running slide-agent evaluations outside the web UI.

## Prerequisites

From repo root:

```bash
uv sync
source .venv/bin/activate
```

Most scripts assume:

- slide data is available locally (for example `.mrxs` + companion folder)
- model backend env/config is set (for example `OPENAI_API_BASE`, `OPENAI_API_KEY`, and `configs/config.yaml`)

## Single Slide Runner

Use `run_single_slide.py` for one case at a time.

```bash
# Full AML pipeline (ROI collection + diagnosis)
python evaluate/run_single_slide.py \
  --slide /path/to/patient.mrxs \
  --output-dir ./batch_outputs \
  --agent aml_auto
```

```bash
# ROI collection only
python evaluate/run_single_slide.py \
  --slide /path/to/patient.mrxs \
  --output-dir ./batch_outputs \
  --agent aml_roi
```

```bash
# Diagnosis only from existing roi_collection.json
python evaluate/run_single_slide.py \
  --output-dir ./batch_outputs \
  --agent aml_diagnosis \
  --roi-input-path /path/to/roi_collection.json
```

Useful flags:

- `--model`: model name (default `GLM-4.6V-FP8`)
- `--extractor`: feature extractor (default `uni2`)
- `--tile-filter`: `none`, `coarse`, `quality`, `hybrid`
- `--experiment-root`: shared cache root for repeated runs
- `--use-tile-cache`: enable persistent tile cache
- `--cuda-device`: sets `CUDA_VISIBLE_DEVICES` for the run

## Batch AML Runner

Use `run_batch_aml.sh` to process a CSV list (one row per patient id/slide entry, header in first row).

```bash
bash evaluate/run_batch_aml.sh \
  --csv /path/to/patients.csv \
  --slides-root /path/to/ALL_WSIs \
  --output-dir ./batch_outputs \
  --model GLM-4.6V-FP8 \
  --extractor uni2 \
  --tile-filter hybrid \
  --resume
```

Notes:

- `--resume` skips already successful cases.
- each failed slide is retried once automatically in the script.

## Multi-Combo Suite Runner

Use `run_batch_aml_suite.sh` to run a matrix of model/extractor combinations.

```bash
bash evaluate/run_batch_aml_suite.sh \
  --csv /path/to/patients.csv \
  --output-parent /path/to/experiments \
  --experiment-name exp_aml_suite \
  --models "GLM-4.6V-FP8,gemma-4-31B-it" \
  --extractors "uni2,virchow2" \
  --resume
```

By default, it runs strictly sequentially. Add `--parallel-models` only if you have independent model endpoints and want one worker per model.

## Parallel Runs

There are two supported ways to parallelize safely:

- `run_batch_aml_suite.sh --parallel-models`: one worker per model, each worker remains sequential per slide.
- split a CSV with `split_csv.py` and run `run_batch_aml.sh` on separate machines or terminals per chunk.

Avoid stacking multiple parallel layers at once unless you have enough independent model endpoints and GPU capacity.

## Cache Pre-Extraction

Use this when you want to prewarm shared feature caches before large batch runs.

Single extractor:

```bash
python evaluate/preextract_hybrid_cache.py \
  --csv /path/to/patients.csv \
  --slides-root /path/to/ALL_WSIs \
  --experiment-root /path/to/exp_root \
  --extractor uni2 \
  --tile-filter hybrid \
  --skip-existing-cache
```

Extractor suite:

```bash
bash evaluate/preextract_hybrid_cache_suite.sh \
  --csv /path/to/patients.csv \
  --slides-root /path/to/ALL_WSIs \
  --experiment-root /path/to/exp_root \
  --extractors "uni2,virchow2,h-optimus-1"
```

## Reference Embeddings

AML retrieval uses curated reference tiles (`good` vs `bad`) to rank ROI candidates. Prebuilding embeddings avoids repeated extraction and speeds startup.

### Tile Layout

```text
Selected_Tiles/
├── Good_Tiles/
│   ├── tile_001.jpg
│   └── ...
└── Bad_Tiles/         # Non-diagnostic tile
    ├── tile_001.jpg
    └── ...
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`.

### Build Reference Cache

```bash
python -m wsi_core_pkg.embeddings.prebuild_reference_embeddings \
  --tiles-root ./Selected_Tiles \
  --output-dir ./outputs/cache/reference_hnsw \
  --extractor reddino
```

```bash
python -m wsi_core_pkg.embeddings.prebuild_reference_embeddings \
  --tiles-root ./Selected_Tiles \
  --output-dir ./outputs/cache/reference_hnsw \
  --extractor uni2
```

Key options:

- `--batch-size` (default `32`)
- `--device` (for example `cuda`, `cuda:0`, `cpu`)
- `--extractor` (`reddino`, `dinobloom`, `uni2`, plus repo aliases)

### Output Artifacts

The builder writes extractor-specific embedding and metadata files under the cache directory.

- `*_prebuilt_embeddings.npy`: normalized embedding matrix
- `*_prebuilt_meta.json`: metadata (tile paths, labels, dimension, fingerprint)
- fingerprinted files used by cache validation/loading

### Runtime Controls

```bash
export AML_REFERENCE_CACHE_DIR=./outputs/cache/reference_hnsw
export AML_REFERENCE_TOP_K=5
export AML_REFERENCE_AGGREGATION=mean
export AML_REFERENCE_LOGIT_SCALE=4.0
export AML_REFERENCE_USE_HNSW=true
export AML_REFERENCE_HNSW_M=32
export AML_REFERENCE_HNSW_EF_CONSTRUCTION=200
export AML_REFERENCE_HNSW_EF_SEARCH=100
```

### Troubleshooting

- No tiles found: verify `Selected_Tiles/Good_Tiles` and `Selected_Tiles/Bad_Tiles` exist with supported image files.
- CUDA OOM: reduce `--batch-size`, switch `--device cpu`, or use `reddino`.
- Cache mismatch/not loading: clear and rebuild `./outputs/cache/reference_hnsw`.

## Utility Scripts

- `split_csv.py`: split one patient CSV into `N` chunks for multi-machine runs.

```bash
python evaluate/split_csv.py --csv /path/to/patients.csv --parts 6 --out-dir /tmp/chunks
```

- `kill_all_runs.sh`: helper for terminating ongoing batch/suite jobs.
- `export_batch_results_stats.py`: aggregate metrics from batch outputs.
