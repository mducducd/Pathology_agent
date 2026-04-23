#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREEXTRACT_SCRIPT="${SCRIPT_DIR}/preextract_hybrid_cache.py"

CSV="/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/AML_HEALTHY_SLIDE_TEST.csv"
SLIDES_ROOT="/mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs"
OUTPUT_PARENT="/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin"
EXPERIMENT_NAME="aml_gemma4_embedding_suite"
EXPERIMENT_ROOT=""
CUDA_DEVICE=""
TILE_FILTER="hybrid"
TILE_SIZE_PX="224"
TILE_SIZE_UM="256"
BATCH_SIZE="512"
AGENT="aml"
LIMIT="0"
SKIP_EXISTING_CACHE=true
EXTRACTORS_FILTER=""

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
    PYTHON_BIN="python"
fi

normalize_extractor_name() {
    local raw="${1:-}"
    raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
    raw="${raw//-/_}"
    raw="${raw// /_}"
    case "$raw" in
        uni2|uni_2) echo "uni2" ;;
        virchow2|virchow_2) echo "virchow2" ;;
        h_optimus_1) echo "h_optimus_1" ;;
        dinobloom|dino_bloom|dinobloom_s|dino_bloom_s|dinobloom_small|dino_bloom_small) echo "dinobloom" ;;
        dinobloom_b|dino_bloom_b|dinobloom_base|dino_bloom_base) echo "dinobloom_base" ;;
        dinobloom_l|dino_bloom_l|dinobloom_large|dino_bloom_large) echo "dinobloom_large" ;;
        dinobloom_g|dino_bloom_g|dinobloom_giant|dino_bloom_giant) echo "dinobloom_giant" ;;
        reddino|red_dino|reddino_small|red_dino_small) echo "reddino" ;;
        reddino_b|red_dino_b|reddino_base|red_dino_base) echo "reddino_base" ;;
        reddino_l|red_dino_l|reddino_large|red_dino_large) echo "reddino_large" ;;
        *) echo "$raw" ;;
    esac
}

normalize_tile_filter_name() {
    local raw="${1:-hybrid}"
    raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
    raw="${raw//-/_}"
    raw="${raw// /_}"
    case "$raw" in
        coarse_to_fine|coarse2fine) echo "hybrid" ;;
        none|coarse|quality|hybrid) echo "$raw" ;;
        *) echo "hybrid" ;;
    esac
}

usage() {
    cat <<'EOF'
Usage:
  bash preextract_hybrid_cache_suite.sh [options]

Options:
  --csv PATH
  --slides-root PATH
  --output-parent PATH       Parent directory for experiment folders
  --experiment-name NAME     Experiment folder name
  --experiment-root PATH     Explicit full experiment root; overrides parent/name
  --cuda-device ID           Set CUDA_VISIBLE_DEVICES, e.g. 1
  --extractors LIST          Comma-separated extractors to keep, e.g. virchow2,dino-bloom-g,reddino
  --tile-filter NAME
  --tile-size-px INT
  --tile-size-um FLOAT
  --batch-size INT
  --limit N
  --skip-existing-cache
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv) CSV="$2"; shift 2 ;;
        --slides-root) SLIDES_ROOT="$2"; shift 2 ;;
        --output-parent) OUTPUT_PARENT="$2"; shift 2 ;;
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --experiment-root) EXPERIMENT_ROOT="$2"; shift 2 ;;
        --cuda-device) CUDA_DEVICE="$2"; shift 2 ;;
        --extractors) EXTRACTORS_FILTER="$2"; shift 2 ;;
        --tile-filter) TILE_FILTER="$2"; shift 2 ;;
        --tile-size-px) TILE_SIZE_PX="$2"; shift 2 ;;
        --tile-size-um) TILE_SIZE_UM="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --skip-existing-cache) SKIP_EXISTING_CACHE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TILE_FILTER="$(normalize_tile_filter_name "$TILE_FILTER")"

if [[ -z "$EXPERIMENT_ROOT" ]]; then
    EXPERIMENT_ROOT="${OUTPUT_PARENT}/${EXPERIMENT_NAME}"
fi

if [[ -n "$CUDA_DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

mkdir -p "$EXPERIMENT_ROOT"

# Convenience multi-extractor wrapper over preextract_hybrid_cache.py.
# Keep this list aligned with the canonical PyTorch extractors in wsi_core_pkg.embeddings.
RUNS=(
    "dinobloom_giant|DinoBloom-G"

    # "dinobloom_base|DinoBloom-B"
    # "dinobloom_large|DinoBloom-L"
   
    "virchow2|Virchow2"
    "h_optimus_1|H-optimus-1"
    "dinobloom|DinoBloom-S"
    
    "uni2|UNI2-h"
    
    # "reddino|RedDino-Small"
    # "reddino_base|RedDino-base"
    # "reddino_large|RedDino-large"
)

FILTERED_RUNS=()
if [[ -n "$EXTRACTORS_FILTER" ]]; then
    IFS=',' read -r -a REQUESTED_EXTRACTORS <<<"$EXTRACTORS_FILTER"
    for spec in "${RUNS[@]}"; do
        IFS="|" read -r EXTRACTOR DISPLAY_NAME <<<"$spec"
        for requested in "${REQUESTED_EXTRACTORS[@]}"; do
            requested="$(normalize_extractor_name "${requested// /}")"
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
    echo "No preextract runs matched extractor filter: ${EXTRACTORS_FILTER}"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo " AML preextract cache suite"
echo " CSV:              $CSV"
echo " Slides root:      $SLIDES_ROOT"
echo " Experiment root:  $EXPERIMENT_ROOT"
echo " Agent:            $AGENT"
echo " Tile filter:      $TILE_FILTER"
echo " Tile size:        ${TILE_SIZE_PX}px / ${TILE_SIZE_UM}um"
echo " Batch size:       $BATCH_SIZE"
echo " Skip existing:    $SKIP_EXISTING_CACHE"
echo " CUDA devices:     ${CUDA_VISIBLE_DEVICES:-all}"
echo " Extractors:       ${EXTRACTORS_FILTER:-all}"
echo " Runs:             ${#FILTERED_RUNS[@]}"
echo "═══════════════════════════════════════════════════════════════"

if [[ "$EXPERIMENT_ROOT" != /* ]]; then
    EXPERIMENT_ROOT="$(realpath "$EXPERIMENT_ROOT")"
fi

for spec in "${FILTERED_RUNS[@]}"; do
    IFS="|" read -r EXTRACTOR DISPLAY_NAME <<<"$spec"
    CACHE_ROOT="${EXPERIMENT_ROOT}/_cache/feature_cache/${EXTRACTOR}/${TILE_FILTER}"
    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo " Preextracting: extractor=${EXTRACTOR} (${DISPLAY_NAME})"
    echo " Feature cache:  ${CACHE_ROOT}"
    echo "───────────────────────────────────────────────────────────────"

    if $SKIP_EXISTING_CACHE; then
        echo " Exact skip:     enabled"
        echo " Skip matching:  per-slide feature hash check in preextract_hybrid_cache.py"
    fi

    CMD=(
        "$PYTHON_BIN"
        "$PREEXTRACT_SCRIPT"
        --csv "$CSV"
        --slides-root "$SLIDES_ROOT"
        --experiment-root "$EXPERIMENT_ROOT"
        --extractor "$EXTRACTOR"
        --tile-filter "$TILE_FILTER"
        --agent "$AGENT"
        --tile-size-px "$TILE_SIZE_PX"
        --tile-size-um "$TILE_SIZE_UM"
        --batch-size "$BATCH_SIZE"
    )

    if [[ "$LIMIT" != "0" ]]; then
        CMD+=(--limit "$LIMIT")
    fi
    if [[ -n "$CUDA_DEVICE" ]]; then
        CMD+=(--cuda-device "$CUDA_DEVICE")
    fi
    if $SKIP_EXISTING_CACHE; then
        CMD+=(--skip-existing-cache)
    fi

    "${CMD[@]}"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " AML preextract cache suite complete"
echo "═══════════════════════════════════════════════════════════════"
