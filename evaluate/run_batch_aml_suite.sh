#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_BATCH_SCRIPT="${SCRIPT_DIR}/run_batch_aml.sh"

CSV="/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/random_100_Normal_AML_Patients.csv"
SLIDES_ROOT="/mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs"
OUTPUT_PARENT="/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin"
EXPERIMENT_NAME="aml_reddino_hybrid_suite"
BASE_OUTPUT_ROOT=""
CUDA_DEVICE=""
TILE_FILTER="hybrid"
TILE_SIZE_PX="224"
BATCH_SIZE="512"
AGENT="aml"
RESUME=false
USE_TILE_CACHE=false
EXTRACTORS_FILTER=""

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
        --use-tile-cache) USE_TILE_CACHE=true; shift ;;
        --resume) RESUME=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$BASE_OUTPUT_ROOT" ]]; then
    BASE_OUTPUT_ROOT="${OUTPUT_PARENT}/${EXPERIMENT_NAME}"
fi

if ! $USE_TILE_CACHE && $RESUME && [[ -d "${BASE_OUTPUT_ROOT}/_cache/tile_cache" ]]; then
    USE_TILE_CACHE=true
    echo "[CACHE] Resume detected existing shared cache at ${BASE_OUTPUT_ROOT}/_cache/tile_cache; enabling cache reuse."
fi

if [[ -n "$CUDA_DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

mkdir -p "$BASE_OUTPUT_ROOT"

RUNS=(
    "gemma-4-31B-it|reddino|batch_result_gemma-4-31B-it_RedDino-Small_224px"
    # "GPT-OSS-120B|reddino|batch_result_GPT-OSS-120B_RedDino-Small_224px"
    # "GLM-4.6V-FP8|reddino_large|batch_result_GLM-4.6V-FP8_RedDino-Large_224px"
    # "Qwen3.5-122B-A10B-FP8|reddino|batch_result_Qwen3.5-122B-A10B-FP8_RedDino-Small_224px"
    # "Qwen3.5-397B-A17B-FP8|reddino|batch_result_Qwen3.5-397B-A17B-FP8_RedDino-Small_224px"
    # "Qwen3.5-397B-A17B-FP8|reddino|batch_result_Qwen3.5-397B-A17B-FP8_RedDino-Small_224px"
    # "GLM-4.6V-FP8|uni2|batch_result_GLM-4.6V-FP8_Uni2_224px"
    # "Qwen3.5-122B-A10B-FP8|reddino|batch_result_Qwen3.5-122B-A10B-FP8_RedDino-Samll_224px"
    
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
echo " Tile cache:       $USE_TILE_CACHE"
echo " CUDA devices:     ${CUDA_VISIBLE_DEVICES:-all}"
echo " Cache root:       $BASE_OUTPUT_ROOT"
echo " Resume:           $RESUME"
echo " Extractors:       ${EXTRACTORS_FILTER:-all}"
echo " Runs:             ${#FILTERED_RUNS[@]}"
echo "═══════════════════════════════════════════════════════════════"

for spec in "${FILTERED_RUNS[@]}"; do
    IFS="|" read -r MODEL EXTRACTOR OUTPUT_NAME <<<"$spec"
    OUTPUT_DIR="${BASE_OUTPUT_ROOT}/${OUTPUT_NAME}"

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

    "${CMD[@]}"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " AML batch suite complete"
echo "═══════════════════════════════════════════════════════════════"
