#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_BATCH_SCRIPT="${SCRIPT_DIR}/run_batch_aml.sh"

CSV="/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/AML_HEALTHY_SLIDE_TEST.csv"
SLIDES_ROOT="/mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs"
OUTPUT_PARENT="/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin"
EXPERIMENT_NAME="aml_gemma4_embedding_suite"
BASE_OUTPUT_ROOT=""
CUDA_DEVICE=""
TILE_FILTER="hybrid"
TILE_SIZE_PX="224"
BATCH_SIZE="512"
ROI_SIZE_PX="2048"
DEFAULT_MPP_UM="$(cd "${SCRIPT_DIR}/.." && python3 - <<'PY'
from pathlib import Path
import yaml
cfg = Path('configs/config.yaml')
default = '0.159'
try:
    data = yaml.safe_load(cfg.read_text()) or {}
    value = data.get('tools', {}).get('slide', {}).get('DEFAULT_MPP_UM', default)
    print(value)
except Exception:
    print(default)
PY
)"
AGENT="aml"
RESUME=true
USE_TILE_CACHE=true
EXTRACTORS_FILTER=""

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
  --extractors LIST          Comma-separated extractors to keep, e.g. reddino
  --tile-filter NAME         Tile prefilter method, e.g. hybrid or coarse
  --roi-size-px N            AML ROI size in pixels, default 2048
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

RUNS=(
    
    
    "GPT-OSS-120B|uni2|test_GPT-OSS-120B_UNI2_224px"
    "GPT-OSS-120B|dinobloom_giant|test_GPT-OSS-120B_DinoBloom-G_224px"
    # "GPT-OSS-120B|virchow2|GPT-OSS-120B_Virchow2_224px"
    # "GPT-OSS-120B|h_optimus_1|GPT-OSS-120B_H-optimus-1_224px"
    
    # "GPT-OSS-120B|dinobloom|GPT-OSS-120B_DinoBloom-S_224px"
)   

FILTERED_RUNS=()
if [[ -n "$EXTRACTORS_FILTER" ]]; then
    IFS=',' read -r -a REQUESTED_EXTRACTORS <<<"$EXTRACTORS_FILTER"
    for spec in "${RUNS[@]}"; do
        IFS="|" read -r MODEL EXTRACTOR OUTPUT_NAME <<<"$spec"
        for requested in "${REQUESTED_EXTRACTORS[@]}"; do
            requested="${requested// /}"
            if [[ -n "$requested" && "$EXTRACTOR" == "$requested" ]]; then
                FILTERED_RUNS+=("$spec")
                break
            fi
        done
    done
else
    FILTERED_RUNS=("${RUNS[@]}")
fi

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
echo " Extractors:       ${EXTRACTORS_FILTER:-all}"
echo " Runs:             ${#FILTERED_RUNS[@]}"
echo "═══════════════════════════════════════════════════════════════"


# Force BASE_OUTPUT_ROOT to absolute path
if [[ -n "$BASE_OUTPUT_ROOT" && "$BASE_OUTPUT_ROOT" != /* ]]; then
    BASE_OUTPUT_ROOT="$(realpath "$BASE_OUTPUT_ROOT")"
fi

SUITE_STARTED_EPOCH="$(date +%s)"

for spec in "${FILTERED_RUNS[@]}"; do
    IFS="|" read -r MODEL EXTRACTOR OUTPUT_NAME <<<"$spec"
    OUTPUT_DIR="${BASE_OUTPUT_ROOT}/${OUTPUT_NAME}"
    # Force OUTPUT_DIR to absolute path
    if [[ "$OUTPUT_DIR" != /* ]]; then
        OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
    fi

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo " Running: model=${MODEL} extractor=${EXTRACTOR}"
    echo " Output:  ${OUTPUT_DIR}"
    echo "───────────────────────────────────────────────────────────────"

    CMD=(
        bash "$RUN_BATCH_SCRIPT"
        --csv "$CSV"
        --slides-root "$SLIDES_ROOT"
        --output-dir "$OUTPUT_DIR"
        --experiment-root "$BASE_OUTPUT_ROOT"
        --model "$MODEL"
        --extractor "$EXTRACTOR"
        --tile-filter "$TILE_FILTER"
        --tile-size-px "$TILE_SIZE_PX"
        --batch-size "$BATCH_SIZE"
        --roi-size-px "$ROI_SIZE_PX"
        --default-mpp-um "$DEFAULT_MPP_UM"
        --agent "$AGENT"
    )

    if [[ -n "$CUDA_DEVICE" ]]; then
        CMD+=(--cuda-device "$CUDA_DEVICE")
    fi
    if $USE_TILE_CACHE; then
        CMD+=(--use-tile-cache)
    fi
    if $RESUME; then
        CMD+=(--resume)
    fi

    RUN_STARTED_EPOCH="$(date +%s)"
    if "${CMD[@]}"; then
        RUN_ELAPSED_SECONDS=$(( $(date +%s) - RUN_STARTED_EPOCH ))
        echo " Elapsed: $(format_elapsed "$RUN_ELAPSED_SECONDS")"
    else
        STATUS=$?
        RUN_ELAPSED_SECONDS=$(( $(date +%s) - RUN_STARTED_EPOCH ))
        echo " Failed after: $(format_elapsed "$RUN_ELAPSED_SECONDS")"
        exit "$STATUS"
    fi
done

SUITE_ELAPSED_SECONDS=$(( $(date +%s) - SUITE_STARTED_EPOCH ))

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " AML batch suite complete"
echo " Suite elapsed:    $(format_elapsed "$SUITE_ELAPSED_SECONDS")"
echo "═══════════════════════════════════════════════════════════════"
