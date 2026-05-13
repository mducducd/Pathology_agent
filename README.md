# Slide Agent

Open-source whole-slide pathology agent for AML ROI collection and diagnosis.


## Project Visuals

![Overview](static/assets/overview.png)
*System overview and WSI workflow.*

![AML Agent Illustration](static/assets/illustration.png)
*AML-focused ROI collection and diagnosis concept.*

![Workbench Demo](static/assets/demo.png)
*Web workbench run view and result flow.*

## Performance Benchmarks

Evaluated on a private dataset of 372 bone marrow WSIs. VLMs are offical FP8 quant

### VLM Performance (ROI Collection)

Results averaged across available feature extractors. `roi5 %` is the % of runs where the VLM successfully reached the 5 ROI target.


| Model | Success % | roi5 % | Tool calls |
|---|---:|---:|---:|
| [gemma-4-31B](https://huggingface.co/models?search=gemma-4-31B) | 100.00 | 71.30 | 21.24 |
| [Qwen3.5-397B](https://huggingface.co/models?search=Qwen3.5-397B) | 99.66 | 99.26 | 25.07 |
| [DeepSeek-V4](https://huggingface.co/models?search=DeepSeek-V4) | 100.00 | 99.73 | 27.10 |
| [GLM-4.6V](https://huggingface.co/models?search=GLM-4.6V) | 92.14 | 95.63 | 26.28 |
| [GPT-OSS-120B](https://huggingface.co/models?search=GPT-OSS-120B) | 99.63 | 23.99 | 37.36 |



### AML Detector Results

> We empirically chose gemma-4 and Qwen3.5 as the diagnosis models — gemma-4 demonstrates specific recognition of AML morphology, while Qwen3.5 is more general in blood cell image understanding.
>
> Results are not heavily affected by the AML diagnosis prompt, and are easily biased by minor changes in instruction wording.


| Model | Ext | Acc % | TP | FN | TN | FP | NPM1 % | HistSim |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| [gemma-4-31B](https://huggingface.co/models?search=gemma-4-31B) | Dino | 70.16 | 234 | 85 | 27 | 26 | 74.43 | 0.889 |
| [gemma-4-31B](https://huggingface.co/models?search=gemma-4-31B) | H-opt | 72.85 | 246 | 73 | 25 | 28 | 71.74 | 0.897 |
| [gemma-4-31B](https://huggingface.co/models?search=gemma-4-31B) | UNI2 | 78.23 | 269 | 50 | 22 | 31 | 73.02 | 0.919 |
| [gemma-4-31B](https://huggingface.co/models?search=gemma-4-31B) | Vir2 | 77.96 | 270 | 49 | 20 | 33 | 72.00 | 0.907 |
| [Qwen3.5-397B](https://huggingface.co/models?search=Qwen3.5-397B) | Dino | 60.48 | 177 | 142 | 48 | 5 | 65.67 | 0.885 |
| [Qwen3.5-397B](https://huggingface.co/models?search=Qwen3.5-397B) | H-opt | 61.02 | 182 | 137 | 45 | 8 | 65.96 | 0.896 |
| [Qwen3.5-397B](https://huggingface.co/models?search=Qwen3.5-397B) | UNI2 | 69.09 | 215 | 104 | 42 | 11 | 64.15 | 0.915 |
| [Qwen3.5-397B](https://huggingface.co/models?search=Qwen3.5-397B) | Vir2 | 66.13 | 207 | 112 | 39 | 14 | 67.88 | 0.906 |

> Note: `HistSim` computes histogram similarity between manual ROIs selected by clinicians and ROIs selected by the AML agent.


## Install

### Environment

```bash
uv sync

source .venv/bin/activate
```

### Configure `.env` to your needs

```bash
cp .env.example .env
```

### Configure model name

In `wsi_core.py` and `main.py`, set `MODEL_NAME` and `ALLOWED_MODEL_NAMES` to the models you want exposed in the UI.

### Configure server-side slide roots

The Explorer modal can browse server-local slide roots directly, so large HPC WSIs do not need to be uploaded through the browser.

By default it exposes:

```text
/mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs/
```

To add more roots, set `SERVER_SLIDE_ROOTS` as a colon-separated list before starting the app:

```bash
export SERVER_SLIDE_ROOTS="/mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs/:/some/other/root"
```

The Explorer supports:

- Standard slide files: `.svs`, `.tif`, `.tiff`, `.ndpi`
- MIRAX files: `.mrxs`, `.mrsx`
- MIRAX companion folders: select the folder whose sibling `.mrxs`/`.mrsx` has the same stem

## Run

```bash
python main.py
```

The web app starts on port `3008` by default. If that port is already in use, the server will fall back to the next available port.

## CLI Runs

See [evaluate/README.md](evaluate/README.md) for full documentation on headless CLI entrypoints: single-slide runs, batch AML, pre-extracting shared cache, the batch suite, and batch diagnosis.

## Workbench

The web workbench has three main panels:

- **Input & Run**: select slide source, configure the agent, model, embedding extractor, tile size, batch size, and tile filtering method, then start the run.
- **Slide Viewer**: shows the slide overview plus ROI snapshots collected during navigation.
- **Run Status**: shows live step updates, current model state, errors, and the final report link.

### Slide Sources

You can start a run from:

- uploaded slide files such as `.svs`, `.tif`, `.tiff`, `.ndpi`
- MIRAX folders or zip bundles
- the built-in server Explorer for server-local/HPC slide roots

### Run Controls

- **Agent**: choose between AML Auto (two-stage, recommended), AML ROI Collector (Stage 1 only), AML Diagnosis (Stage 2 only), AML Detector (legacy single-stage), Tile Selector, and General WSI Agent.
- **Model**: choose which VLM is exposed in the workbench.
- **Feature Extractor**: choose the embedding backbone used for ROI candidate preparation.
- **Tile size (px)**: controls the patch size used by the extractor path.
- **Batch size**: controls embedding throughput during tile feature extraction.
- **Tile filter**: controls how candidate tiles are reduced before expensive embedding.

Tile filter options:

- **Quality score**: ranks raw tiles by cheap focus, stain, texture, and artifact heuristics, then keeps the strongest subset plus a small safety reserve.
- **Coarse to fine**: uses a thumbnail-level region prefilter first, then embeds tiles only inside the selected regions.
- **Hybrid**: combines coarse region filtering with the raw-tile quality prefilter.
- **None**: disables the extra tile prefilter stage and keeps the baseline foreground/texture gating only.

### Typical Workbench Flow

1. Choose a slide source or browse the server Explorer.
2. Pick the agent, model, feature extractor, tile size, batch size, and tile filter.
3. Click **Start run**.
4. Follow live status updates in the right panel while reviewing the overview and ROI panes.

## Agents

The system provides five agents, each exposed via the `agent_type` parameter in the web UI, CLI, and API.

---

### WSIAmlRoiCollectorAgent — `aml_roi`

**Role:** Stage-1 of the two-stage AML pipeline. Navigates the WSI and marks exactly 5 high-quality high-power ROIs for downstream diagnosis.

**Tools:** `wsi_get_overview_view`, `wsi_zoom_current_norm`, `wsi_zoom_full_norm`, `wsi_pan_current`, `wsi_get_view_info`, `wsi_open_candidate`, `wsi_mark_candidate`, `wsi_mark_roi_norm`, `wsi_discard_last_roi`

**Inputs:** A WSI file path plus the shared candidate pipeline (tile extraction, embeddings, reference retrieval).

**Outputs:**
- `roi_collection.json` — manifest with ROI ids, image paths, bounding boxes, tissue fraction, and metadata.
- `images/roi_N.jpg` — high-resolution ROI crops at `ROI_SIZE_PX` (default 2048 px).
- `images/slide_overview.jpg` — overview with marked ROI positions.
- `images/roi_candidates.jpg` — candidate ranking overlay.
- A navigation summary report (`report.md`).

**Key behaviour:**
- Opens candidates with `wsi_open_candidate(rank)` and zooms systematically within each candidate field before marking.
- Marks exactly 5 accepted ROIs from reasonably distinct regions.
- Discards ROIs that are background-only, hemodilute, out-of-focus, or artifact-dominated.
- Stops immediately once 5 ROIs are accepted; does not perform diagnosis.

---

### WSIAmlDiagnosisAgent — `aml_diagnosis`

**Role:** Stage-2 of the two-stage AML pipeline. Receives only ROI images (no slide file, no navigation) and returns a strict morphology-only JSON diagnosis.

**Tools:** none — runs as a direct chat-completion call, not through the Runner loop.

**Inputs:** A `roi_collection.json` (or a directory containing one) produced by `aml_roi`, or a single ROI image file.

**Outputs:** A strict JSON object with the following fields:

```json
{
  "morphology_summary": "...",
  "accepted_rois": [
    { "roi_id": "1", "quality_reason": "...", "blast_range": "<5% | 5-9% | 10-19% | 20-50% | >50%", "key_features": ["..."] }
  ],
  "discard_summary": ["..."],
  "global_blast_range": "<5% | 5-9% | 10-19% | 20-50% | >50%",
  "triage_zone": "normal_like | borderline_suspicious | aml_like",
  "final_decision": "Normal marrow | Acute leukemia",
  "limitations_confidence": { "limitations": "...", "confidence": "low | medium | high" },
  "npm1_prediction": { "applicable": true, "classification": "NPM1_mutated | NPM1_wildtype", ... }
}
```

**Key behaviour:**
- Completely blind — receives only base64-encoded ROI images plus the system prompt. No filenames, no metadata.
- Re-evaluates each ROI for suitability before estimating blasts.
- NPM1 prediction is gated: only performed when `final_decision = "Acute leukemia"`.
- Temperature is fixed at 0.0 for deterministic output.

---

### aml_auto (two-stage pipeline) — `aml_auto` (alias: `aml`)

**Role:** Full end-to-end AML pipeline that chains `aml_roi` then `aml_diagnosis` automatically.

**Flow:**
```
WSI file
  → WSIAmlRoiCollectorAgent (aml_roi stage)
      → roi_collection.json + ROI images
  → WSIAmlDiagnosisAgent (aml_diagnosis stage)
      → final JSON diagnosis + report.md
```

**When to use:** Default for the web UI and batch scripts. Runs Stage 1 with the default ROI collection prompt (ignoring any custom prompt), then runs Stage 2 with the user-supplied or default diagnosis prompt.

---

### WSIAmlDetectorAgent — `aml_detector` (legacy single-stage)

**Role:** Single-stage AML agent that combines navigation and morphology diagnosis in one pass. Uses `DEFAULT_AML_PROMPT` which instructs the VLM to both collect ROIs and output the final JSON.

**Tools:** Same as `WSIAmlRoiCollectorAgent` plus `wsi_rebuild_reference_index`.

**When to use:** For single-turn debugging or when the two-stage split is not needed. The two-stage `aml_auto` pipeline is the production default.

---

### WSITileSelectorAgent — `tile`

**Role:** Systematic tile-selection agent. Navigates the WSI and saves individual tiles classified as `good` or `bad` for downstream model training or analysis.

**Tools:** `wsi_get_overview_view`, `wsi_zoom_current_norm`, `wsi_zoom_full_norm`, `wsi_pan_current`, `wsi_get_view_info`, `wsi_save_tile_norm`

**Outputs:** Tiles saved under `SELECTED_TILES_ROOT` with quality labels, up to `MAX_GOOD_TILES` (default 200) good tiles and `MAX_BAD_TILES` (default 50) bad tiles per run.

**Key behaviour:**
- Prioritizes deep blue-purple nucleated marrow over pink-red RBC-rich material and pale background.
- Uses example good/bad tiles provided at run start as visual guidance.
- Stops when the good-tile limit is reached or no further good tiles can be found.

---

### WSIPathologyAgent — `wsi`

**Role:** General-purpose WSI exploration agent for any pathology task (tumour description, MSI screening, inflammation assessment, etc.).

**Tools:** All navigation tools plus `wsi_mark_roi_norm`, `wsi_save_tile_norm`, `wsi_discard_last_roi`, `wsi_rebuild_reference_index`.

**When to use:** When the task is not AML-specific. The user prompt defines the clinical task; the agent adapts its navigation and reporting accordingly.

**MSI guidance:** When the prompt requests MSI screening, the agent samples at least three distinct tumour regions, marks representative ROIs, and provides a qualitative MSI-H/MSS assessment with the caveat that definitive status requires IHC or molecular testing.

---

### Agent — Pipeline Mode Summary

| `agent_type` | Python agent class | Stage | Slide required | ROI collection input |
|---|---|---|---|---|
| `wsi` | `WSIPathologyAgent` | single | yes | — |
| `tile` | `WSITileSelectorAgent` | single | yes | — |
| `aml_roi` | `WSIAmlRoiCollectorAgent` | Stage 1 | yes | — |
| `aml_diagnosis` | `WSIAmlDiagnosisAgent` | Stage 2 | no | `roi_collection.json` |
| `aml_auto` / `aml` | both (chained) | 1 → 2 | yes | auto |
| `aml_detector` | `WSIAmlDetectorAgent` | single | yes | — |

---

## Reference Embeddings (AML Mode)

For AML detection, the agent uses retrieval-based ranking against curated reference tiles. Pre-building embeddings significantly speeds up startup time.

**Quick start:**

```bash
# Pre-build embeddings with reddino (recommended - fast)
python -m wsi_core_pkg.embeddings.prebuild_reference_embeddings \
    --tiles-root ./Selected_Tiles \
    --output-dir ./outputs/cache/reference_hnsw \
    --extractor reddino
```

**Documentation:** See [REFERENCE_EMBEDDINGS.md](REFERENCE_EMBEDDINGS.md) for detailed instructions on:
- Organizing curated tiles
- Extractor options (uni2, h_optimus_1, virchow2, dinobloom, dinobloom_giant, reddino)
- Cache management and invalidation
- Environment variable configuration
- Dynamic prototype bank updates
