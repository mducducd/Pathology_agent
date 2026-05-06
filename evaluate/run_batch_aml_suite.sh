#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_BATCH_SCRIPT="${SCRIPT_DIR}/run_batch_aml.sh"

# Auto-activate uv environment
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
        source "${REPO_ROOT}/.venv/bin/activate"
    fi
fi

CSV="/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/AML_HEALTHY_SLIDE_TEST.csv"
SLIDES_ROOT="/mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs"
OUTPUT_PARENT="/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin"
EXPERIMENT_NAME="exp_290425_aml_suite"
BASE_OUTPUT_ROOT=""
CUDA_DEVICE=""
eval "$(
    cd "${REPO_ROOT}" && python3 - <<'PY'
from pathlib import Path
import shlex
import yaml

CONFIG_PATH = Path("configs/config.yaml")

try:
    data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    if not isinstance(data, dict):
        data = {}
except Exception:
    data = {}

slide_cfg = data.get("tools", {}).get("slide", {})
if not isinstance(slide_cfg, dict):
    slide_cfg = {}

values = {
    "TILE_FILTER": str(slide_cfg.get("TILE_FILTER", "hybrid")),
    "TILE_SIZE_PX": str(slide_cfg.get("TILE_SIZE_PX", "224")),
    "BATCH_SIZE": str(slide_cfg.get("BATCH_SIZE", "512")),
    "ROI_SIZE_PX": str(slide_cfg.get("ROI_SIZE_PX", "2048")),
    "AGENT": str(slide_cfg.get("AGENT", "aml")),
    "DEFAULT_MPP_UM": str(slide_cfg.get("DEFAULT_MPP_UM", "0.159")),
    "RESUME": "true",
    "USE_TILE_CACHE": "true",
}

for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"
EXTRACTORS_FILTER=""
MODELS_FILTER=""
PARALLEL=false
CHUNKS_DIR=""
RUNS=(
    "GLM-4.6V-FP8|uni2|GLM-4.6V-FP8_UNI2_224px"
    "GLM-4.6V-FP8|virchow2|GLM-4.6V-FP8_Virchow2_224px"
    "GLM-4.6V-FP8|h_optimus_1|GLM-4.6V-FP8_H-optimus-1_224px"
    "GLM-4.6V-FP8|dinobloom_giant|GLM-4.6V-FP8_DinoBloom-G_224px"
    "GLM-4.6V-Flash|uni2|GLM-4.6V-Flash_UNI2_224px"
    "GLM-4.6V-Flash|virchow2|GLM-4.6V-Flash_Virchow2_224px"
    "GLM-4.6V-Flash|h_optimus_1|GLM-4.6V-Flash_H-optimus-1_224px"
    "GLM-4.6V-Flash|dinobloom_giant|GLM-4.6V-Flash_DinoBloom-G_224px"
    "gemma-4-31B-it|uni2|gemma-4-31B-it_UNI2_224px"
    "gemma-4-31B-it|virchow2|gemma-4-31B-it_Virchow2_224px"
    "gemma-4-31B-it|h_optimus_1|gemma-4-31B-it_H-optimus-1_224px"
    "gemma-4-31B-it|dinobloom_giant|gemma-4-31B-it_DinoBloom-G_224px"
    "medgemma-27b-it|uni2|medgemma-27b-it_UNI2_224px"
    "medgemma-27b-it|virchow2|medgemma-27b-it_Virchow2_224px"
    "medgemma-27b-it|h_optimus_1|medgemma-27b-it_H-optimus-1_224px"
    "medgemma-27b-it|dinobloom_giant|medgemma-27b-it_DinoBloom-G_224px"
    "DeepSeek-V4-Flash|uni2|DeepSeek-V4-Flash_UNI2_224px"
    "DeepSeek-V4-Flash|virchow2|DeepSeek-V4-Flash_Virchow2_224px"
    "DeepSeek-V4-Flash|h_optimus_1|DeepSeek-V4-Flash_H-optimus-1_224px"
    "DeepSeek-V4-Flash|dinobloom_giant|DeepSeek-V4-Flash_DinoBloom-G_224px"
    "Qwen3.5-397B-A17B-FP8|uni2|Qwen3.5-397B-A17B-FP8_UNI2_224px"
    "Qwen3.5-397B-A17B-FP8|virchow2|Qwen3.5-397B-A17B-FP8_Virchow2_224px"
    "Qwen3.5-397B-A17B-FP8|h_optimus_1|Qwen3.5-397B-A17B-FP8_H-optimus-1_224px"
    "Qwen3.5-397B-A17B-FP8|dinobloom_giant|Qwen3.5-397B-A17B-FP8_DinoBloom-G_224px"
)

format_elapsed() {
    local total_seconds="${1:-0}"
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf '%02dh:%02dm:%02ds' "$hours" "$minutes" "$seconds"
}

usage() {
    cat <<'EOF'
Usage:
  bash run_batch_aml_suite.sh [options]

Options:
  --csv PATH
  --slides-root PATH
  --output-parent PATH       Parent directory for experiment folders
  --experiment-name NAME     Experiment folder name
  --base-output-root PATH    Explicit full output directory; overrides parent/name
  --cuda-device ID           Set CUDA_VISIBLE_DEVICES, e.g. 1
  --models LIST              Comma-separated models to run, e.g. GLM-4.6V-FP8
  --extractors LIST          Comma-separated extractors to keep, e.g. uni2
  --parallel                 Launch all runs simultaneously in background
  --chunks-dir PATH          Directory of part_*.csv files (used with --parallel)
  --tile-filter NAME         Default from configs/config.yaml
  --roi-size-px N            AML ROI size in pixels, default from configs/config.yaml
  --default-mpp-um FLOAT     Preferred MPP override, default from configs/config.yaml
  --use-tile-cache
  --resume
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv) CSV="$2"; shift 2 ;;
        --slides-root) SLIDES_ROOT="$2"; shift 2 ;;
        --output-parent) OUTPUT_PARENT="$2"; shift 2 ;;
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --base-output-root) BASE_OUTPUT_ROOT="$2"; shift 2 ;;
        --cuda-device) CUDA_DEVICE="$2"; shift 2 ;;
        --extractors) EXTRACTORS_FILTER="$2"; shift 2 ;;
        --models) MODELS_FILTER="$2"; shift 2 ;;
        --parallel) PARALLEL=true; shift ;;
        --chunks-dir) CHUNKS_DIR="$2"; shift 2 ;;
        --tile-filter) TILE_FILTER="$2"; shift 2 ;;
        --roi-size-px) ROI_SIZE_PX="$2"; shift 2 ;;
        --default-mpp-um) DEFAULT_MPP_UM="$2"; shift 2 ;;
        --use-tile-cache) USE_TILE_CACHE=true; shift ;;
        --resume) RESUME=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$BASE_OUTPUT_ROOT" ]]; then
    BASE_OUTPUT_ROOT="${OUTPUT_PARENT}/${EXPERIMENT_NAME}"
fi

if [[ -n "$CUDA_DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

mkdir -p "$BASE_OUTPUT_ROOT"

FILTERED_RUNS=()
for spec in "${RUNS[@]}"; do
    IFS="|" read -r _MODEL _EXTRACTOR _OUTPUT_NAME <<<"$spec"
    _keep=true

    if [[ -n "$MODELS_FILTER" ]]; then
        _match=false
        IFS=',' read -r -a _requested_models <<<"$MODELS_FILTER"
        for _m in "${_requested_models[@]}"; do
            _m="${_m// /}"
            if [[ -n "$_m" && "$_MODEL" == "$_m" ]]; then
                _match=true; break
            fi
        done
        $_match || _keep=false
    fi

    if [[ -n "$EXTRACTORS_FILTER" ]] && $_keep; then
        _match=false
        IFS=',' read -r -a _requested_extractors <<<"$EXTRACTORS_FILTER"
        for _e in "${_requested_extractors[@]}"; do
            _e="${_e// /}"
            if [[ -n "$_e" && "$_EXTRACTOR" == "$_e" ]]; then
                _match=true; break
            fi
        done
        $_match || _keep=false
    fi

    $_keep && FILTERED_RUNS+=("$spec")
done

if [[ ${#FILTERED_RUNS[@]} -eq 0 ]]; then
    echo "No runs matched extractor filter: ${EXTRACTORS_FILTER}"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo " AML batch suite"
echo " CSV:              $CSV"
echo " Slides root:      $SLIDES_ROOT"
echo " Base output root: $BASE_OUTPUT_ROOT"
echo " Agent:            $AGENT"
echo " Tile filter:      $TILE_FILTER"
echo " Tile size:        ${TILE_SIZE_PX}px"
echo " Batch size:       $BATCH_SIZE"
echo " ROI size:         ${ROI_SIZE_PX}px"
echo " Default MPP:      ${DEFAULT_MPP_UM}"
echo " Tile cache:       $USE_TILE_CACHE"
echo " CUDA devices:     ${CUDA_VISIBLE_DEVICES:-all}"
echo " Cache root:       $BASE_OUTPUT_ROOT"
echo " Resume:           $RESUME"
echo " Models:           ${MODELS_FILTER:-all}"
echo " Extractors:       ${EXTRACTORS_FILTER:-all}"
echo " Runs:             ${#FILTERED_RUNS[@]}"
echo "═══════════════════════════════════════════════════════════════"


# Force BASE_OUTPUT_ROOT to absolute path
if [[ -n "$BASE_OUTPUT_ROOT" && "$BASE_OUTPUT_ROOT" != /* ]]; then
    BASE_OUTPUT_ROOT="$(realpath "$BASE_OUTPUT_ROOT")"
fi

SUITE_STARTED_EPOCH="$(date +%s)"

# Build the CSV list: either split chunks or single CSV
CSV_LIST=()
if $PARALLEL && [[ -n "$CHUNKS_DIR" ]]; then
    while IFS= read -r -d '' f; do
        CSV_LIST+=("$f")
    done < <(find "$CHUNKS_DIR" -maxdepth 1 -name 'part_*.csv' -print0 | sort -z)
    if [[ ${#CSV_LIST[@]} -eq 0 ]]; then
        echo "[ERROR] No part_*.csv found in $CHUNKS_DIR"
        exit 1
    fi
    echo " Parallel chunks:  ${#CSV_LIST[@]} (from $CHUNKS_DIR)"
else
    CSV_LIST=("$CSV")
fi

_build_cmd() {
    local _csv="$1" _model="$2" _extractor="$3" _output_dir="$4"
    local _cmd=(
        bash "$RUN_BATCH_SCRIPT"
        --csv "$_csv"
        --slides-root "$SLIDES_ROOT"
        --output-dir "$_output_dir"
        --experiment-root "$BASE_OUTPUT_ROOT"
        --model "$_model"
        --extractor "$_extractor"
        --tile-filter "$TILE_FILTER"
        --tile-size-px "$TILE_SIZE_PX"
        --batch-size "$BATCH_SIZE"
        --roi-size-px "$ROI_SIZE_PX"
        --default-mpp-um "$DEFAULT_MPP_UM"
        --agent "$AGENT"
    )
    [[ -n "$CUDA_DEVICE" ]] && _cmd+=(--cuda-device "$CUDA_DEVICE")
    $USE_TILE_CACHE       && _cmd+=(--use-tile-cache)
    $RESUME               && _cmd+=(--resume)
    printf '%s\n' "${_cmd[@]}"
}

LOG_DIR="${BASE_OUTPUT_ROOT}/_suite_logs"
mkdir -p "$LOG_DIR"

PIDS=()
LABELS=()

for spec in "${FILTERED_RUNS[@]}"; do
    IFS="|" read -r MODEL EXTRACTOR OUTPUT_NAME <<<"$spec"
    OUTPUT_DIR="${BASE_OUTPUT_ROOT}/${OUTPUT_NAME}"
    [[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

    for CHUNK_CSV in "${CSV_LIST[@]}"; do
        CHUNK_TAG="$(basename "$CHUNK_CSV" .csv)"
        LABEL="${MODEL}|${EXTRACTOR}|${CHUNK_TAG}"
        LOG_FILE="${LOG_DIR}/${MODEL//\//_}_${EXTRACTOR}_${CHUNK_TAG}.log"

        echo "───────────────────────────────────────────────────────────────"
        echo " Queuing: model=${MODEL} extractor=${EXTRACTOR} chunk=${CHUNK_TAG}"
        echo " Log:     ${LOG_FILE}"

        mapfile -t CMD < <(_build_cmd "$CHUNK_CSV" "$MODEL" "$EXTRACTOR" "$OUTPUT_DIR")

        if $PARALLEL; then
            "${CMD[@]}" >"$LOG_FILE" 2>&1 &
            PIDS+=($!)
            LABELS+=("$LABEL")
        else
            RUN_STARTED_EPOCH="$(date +%s)"
            if "${CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
                echo " Elapsed: $(format_elapsed "$(( $(date +%s) - RUN_STARTED_EPOCH ))")"
            else
                STATUS=$?
                echo " Failed after: $(format_elapsed "$(( $(date +%s) - RUN_STARTED_EPOCH ))")"
                exit "$STATUS"
            fi
        fi
    done
done

if $PARALLEL; then
    echo ""
    echo " Launched ${#PIDS[@]} parallel workers — waiting for all to finish..."
    echo " Logs in: $LOG_DIR"
    FAILED_LABELS=()
    for i in "${!PIDS[@]}"; do
        PID="${PIDS[$i]}"
        LABEL="${LABELS[$i]}"
        if wait "$PID"; then
            echo " [OK]   $LABEL"
        else
            echo " [FAIL] $LABEL"
            FAILED_LABELS+=("$LABEL")
        fi
    done
    if [[ ${#FAILED_LABELS[@]} -gt 0 ]]; then
        echo ""
        echo " ${#FAILED_LABELS[@]} worker(s) failed:"
        for l in "${FAILED_LABELS[@]}"; do echo "   $l"; done
        exit 1
    fi
fi

SUITE_ELAPSED_SECONDS=$(( $(date +%s) - SUITE_STARTED_EPOCH ))

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " AML batch suite complete"
echo " Suite elapsed:    $(format_elapsed "$SUITE_ELAPSED_SECONDS")"
echo "═══════════════════════════════════════════════════════════════"
