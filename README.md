# Slide Morphology Triage Agent

Open-source whole-slide pathology agent for mophology triage ROI collection, morphology-first diagnosis, and interactive WSI exploration. Support general slides and Acute myeloid leukemia (AML) slides

The project combines deterministic slide reduction, embedding-based candidate retrieval, and a vision-language model that navigates high-value regions instead of trying to reason over an entire gigapixel slide at once.

## Highlights

- Interactive FastAPI workbench for browser-based runs and live monitoring.
- Headless evaluation scripts for single-slide, batch, and cache-precomputation workflows.
- Support for standard WSI formats plus MIRAX files and server-local slide browsing.

## Project Visuals

![Overview](static/assets/overview.png)
*System overview and WSI workflow.*

![Workbench Demo](static/assets/demo.png)
*Web workbench run view and result flow.*

## Performance Benchmarks

Evaluated on a private dataset of 372 bone marrow WSIs. VLMs were run in official FP8 quantized variants where available.

### VLM Performance (ROI Collection)

Results are averaged across available feature extractors. `roi5 %` is the percentage of runs where the model successfully reached the 5-ROI target.

| Model | Success % | roi5 % | Avg. tool calls |
|---|---:|---:|---:|
| [Gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | 100.00 | 71.30 | 21.24 |
| [Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) | 99.66 | 99.26 | 25.07 |
| [DeepSeek-V4](https://huggingface.co/models?search=DeepSeek-V4) | 100.00 | 99.73 | 27.10 |
| [GLM-4.6V-FP8](https://huggingface.co/zai-org/GLM-4.6V-FP8) | 92.14 | 95.63 | 26.28 |
| [GPT-OSS-120B](https://huggingface.co/models?search=GPT-OSS-120B) | 99.63 | 23.99 | 37.36 |

### AML Diagnosis Results

Gemma-4-31B-it and Qwen3.5-397B-A17B-FP8 were the main diagnosis models explored here: Gemma-4-31B-it showed stronger AML-morphology specificity, while Qwen3.5-397B-A17B-FP8 was more general in blood-cell image understanding.

Prompt wording has some effect, but the results are not dominated by prompt changes alone.

<table>
<thead>
<tr>
<th><sub>Model</sub></th><th><sub>Extractor</sub></th><th><sub>Acc %</sub></th><th><sub>TP/FN</sub></th><th><sub>TN/FP</sub></th><th><sub>NPM1 %</sub></th><th><sub>HistSim</sub></th>
</tr>
</thead>
<tbody>
<tr><td><sub><a href="https://huggingface.co/google/gemma-4-31B-it">Gemma-4-31B-it</a></sub></td><td><sub>DinoBloom</sub></td><td align="right"><sub>70.16</sub></td><td align="right"><sub>234/85</sub></td><td align="right"><sub>27/26</sub></td><td align="right"><sub>74.43</sub></td><td align="right"><sub>0.889</sub></td></tr>
<tr><td><sub><a href="https://huggingface.co/google/gemma-4-31B-it">Gemma-4-31B-it</a></sub></td><td><sub>H-optimus-1</sub></td><td align="right"><sub>72.85</sub></td><td align="right"><sub>246/73</sub></td><td align="right"><sub>25/28</sub></td><td align="right"><sub>71.74</sub></td><td align="right"><sub>0.897</sub></td></tr>
<tr><td><sub><a href="https://huggingface.co/google/gemma-4-31B-it">Gemma-4-31B-it</a></sub></td><td><sub>UNI2</sub></td><td align="right"><sub>78.23</sub></td><td align="right"><sub>269/50</sub></td><td align="right"><sub>22/31</sub></td><td align="right"><sub>73.02</sub></td><td align="right"><sub>0.919</sub></td></tr>
<tr><td><sub><a href="https://huggingface.co/google/gemma-4-31B-it">Gemma-4-31B-it</a></sub></td><td><sub>Virchow2</sub></td><td align="right"><sub>77.96</sub></td><td align="right"><sub>270/49</sub></td><td align="right"><sub>20/33</sub></td><td align="right"><sub>72.00</sub></td><td align="right"><sub>0.907</sub></td></tr>
<tr><td><sub><a href="https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8">Qwen3.5-397B-A17B-FP8</a></sub></td><td><sub>DinoBloom</sub></td><td align="right"><sub>60.48</sub></td><td align="right"><sub>177/142</sub></td><td align="right"><sub>48/5</sub></td><td align="right"><sub>65.67</sub></td><td align="right"><sub>0.885</sub></td></tr>
<tr><td><sub><a href="https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8">Qwen3.5-397B-A17B-FP8</a></sub></td><td><sub>H-optimus-1</sub></td><td align="right"><sub>61.02</sub></td><td align="right"><sub>182/137</sub></td><td align="right"><sub>45/8</sub></td><td align="right"><sub>65.96</sub></td><td align="right"><sub>0.896</sub></td></tr>
<tr><td><sub><a href="https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8">Qwen3.5-397B-A17B-FP8</a></sub></td><td><sub>UNI2</sub></td><td align="right"><sub>69.09</sub></td><td align="right"><sub>215/104</sub></td><td align="right"><sub>42/11</sub></td><td align="right"><sub>64.15</sub></td><td align="right"><sub>0.915</sub></td></tr>
<tr><td><sub><a href="https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8">Qwen3.5-397B-A17B-FP8</a></sub></td><td><sub>Virchow2</sub></td><td align="right"><sub>66.13</sub></td><td align="right"><sub>207/112</sub></td><td align="right"><sub>39/14</sub></td><td align="right"><sub>67.88</sub></td><td align="right"><sub>0.906</sub></td></tr>
</tbody>
</table>

`HistSim` measures histogram similarity between clinician-selected ROIs and agent-selected ROIs.

## Quick Start

### Install

```bash
uv sync
source .venv/bin/activate
```

### Configure

If your model backend needs API keys or other runtime settings, create a `.env` file with the variables you use locally.

The OpenAI-compatible client settings live in [configs/config.yaml](configs/config.yaml) under the `agent` section:

```yaml
agent:
  OPENAI_API_BASE: "http://pluto/v1"
  OPENAI_API_KEY: "local"
```

Environment variables still override the YAML values, so you can also set them in your shell or `.env`:

```bash
export OPENAI_API_BASE="http://your-server/v1"
export OPENAI_API_KEY="your-key"
```

Update model exposure in [main.py](main.py) and [wsi_core.py](wsi_core.py) by setting `MODEL_NAME` and `ALLOWED_MODEL_NAMES`.

To expose server-local slide roots in the web Explorer, set `SERVER_SLIDE_ROOTS` before starting the app:

```bash
export SERVER_SLIDE_ROOTS="/some/other/root"
```

Supported sources:

- Standard slide files: `.svs`, `.tif`, `.tiff`, `.ndpi`
- MIRAX files: `.mrxs`, `.mrsx`
- MIRAX companion folders: select the folder whose sibling `.mrxs` or `.mrsx` has the same stem

### Run The Workbench

```bash
python main.py
```

The app starts on port `3008` by default and falls forward to the next free port if needed.

## How It Fits Together

```text
WSI
  -> tile extraction and quality filtering
  -> embedding-based candidate ranking
  -> agentic ROI navigation and selection
  -> morphology-only diagnosis from accepted ROIs
```

`aml_auto` is the default production path:

```text
WSI
  -> aml_roi
      -> roi_collection.json + ROI images
  -> aml_diagnosis
      -> final JSON diagnosis + report
```

## Workbench

The web UI is organized around three panels:

- **Input and Run**: slide source, agent, model, extractor, tile size, batch size, and tile filter.
- **Slide Viewer**: slide overview plus ROI snapshots collected during navigation.
- **Run Status**: live steps, current state, errors, and report link.

### Slide Sources

You can start a run from:

- uploaded slide files such as `.svs`, `.tif`, `.tiff`, `.ndpi`
- MIRAX folders or zip bundles
- the built-in server Explorer for server-local or HPC slide roots

### Main Controls

- **Agent**: `aml_auto`, `aml_roi`, `aml_diagnosis`, `tile`, or `wsi`
- **Model**: which VLM is exposed in the workbench
- **Feature extractor**: embedding backbone used for ROI candidate preparation
- **Tile size**: patch size for the extractor path
- **Batch size**: embedding throughput during tile feature extraction
- **Tile filter**: candidate prefilter before expensive embedding

Tile filter modes:

- **Quality score**: ranks tiles by focus, stain, texture, and artifact heuristics.
- **Coarse to fine**: thumbnail-level region prefilter before embedding.
- **Hybrid**: combines region screening with raw-tile quality prefiltering.
- **None**: keeps only the baseline foreground and texture gating.

### Typical Flow

1. Choose a slide source or browse the server Explorer.
2. Pick the agent, model, extractor, tile size, batch size, and tile filter.
3. Start the run.
4. Follow the live trace while reviewing the overview and ROI panes.

## Agents

| `agent_type` | Purpose | Slide input | Output |
|---|---|---|---|
| `aml_auto` | Full two-stage AML workflow | required | ROI bundle + diagnosis report |
| `aml_roi` | Stage 1 ROI collection only | required | `roi_collection.json` + ROI images |
| `aml_diagnosis` | Stage 2 diagnosis from existing ROIs | not required | strict AML JSON diagnosis |
| `tile` | Save good and bad tiles for curation or training | required | labeled tiles under `Selected_Tiles` |
| `wsi` | General-purpose pathology exploration agent | required | task-shaped report and saved ROIs |

### AML ROI Collector

`aml_roi` is the evidence-acquisition stage. It navigates the slide, opens ranked candidates, and marks acceptable high-power ROIs toward the configured target of 5 for downstream diagnosis.

Outputs include:

- `roi_collection.json`
- `images/roi_N.jpg`
- `images/slide_overview.jpg`
- `images/roi_candidates.jpg`
- navigation summary `report.md`

### AML Diagnosis

`aml_diagnosis` receives ROI images only. It does not navigate the slide and does not see slide-level metadata. The output is a strict JSON object with morphology summary, accepted ROI reasoning, blast range, triage zone, final decision, limitations, confidence, and conditional NPM1 prediction.

### General WSI Agent

`wsi` is the non-AML mode for broader pathology tasks such as tumour description, MSI-oriented inspection, inflammation review, or other prompt-defined workflows.

When prompted for MSI screening, the agent samples at least three distinct tumour regions, marks representative ROIs, and returns a qualitative MSI-H or MSS impression with the caveat that definitive status still requires IHC or molecular testing.

## CLI And Batch Runs

See the dedicated CLI guide: [evaluate/README.md](evaluate/README.md).

Main entrypoints:

- `evaluate/run_single_slide.py`: single-case headless run
- `evaluate/run_batch_aml.sh`: CSV-driven AML batch
- `evaluate/run_batch_aml_suite.sh`: model/extractor suite
- `evaluate/preextract_hybrid_cache.py`: feature-cache prewarm
- `evaluate/preextract_hybrid_cache_suite.sh`: multi-extractor prewarm

## Reference Embeddings

AML mode uses retrieval against curated reference tiles to improve ROI ranking. Prebuilding the reference index can reduce startup cost significantly.

Quick start:

```bash
python -m wsi_core_pkg.embeddings.prebuild_reference_embeddings \
    --tiles-root ./Selected_Tiles \
    --output-dir ./outputs/cache/reference_hnsw \
    --extractor reddino
```

Detailed notes live in [evaluate/README.md](evaluate/README.md) under `Reference Embeddings`.

## Documentation Map

- [evaluate/README.md](evaluate/README.md): CLI workflows and reference-embedding setup
- [docs/AML_AGENT_PIPELINE_METHODOLOGY.md](docs/AML_AGENT_PIPELINE_METHODOLOGY.md): methodology and algorithmic description
- [docs/AML_AGENT_PIPELINE_IMPLEMENTATION.md](/AML_AGENT_PIPELINE_IMPLEMENTATION.md): implementation details
- [docs/AML_AGENT_TOOLS_APPENDIX.md](docs/AML_AGENT_TOOLS_APPENDIX.md): tool and navigation appendix
- [docs/AML_PROMPTS_APPENDIX.md](docs/AML_PROMPTS_APPENDIX.md): prompt appendix
- [docs/EVALUATION_METRICS.md](docs/EVALUATION_METRICS.md): metric definitions
