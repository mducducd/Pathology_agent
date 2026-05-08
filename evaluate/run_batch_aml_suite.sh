#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_batch_aml_suite.sh — Run AML detector across (model × extractor).
#
# STRICTLY SEQUENTIAL. No background workers, no `&`, no monitors.
# One combo at a time, one slide at a time. Ctrl-C stops everything
# immediately because there is only ever one foreground child.
#
# To run multiple combos in parallel, open multiple terminals and
# filter each one by --models (e.g. one terminal per VLM endpoint).
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

ORIGINAL_ARGS=("$@")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_BATCH_SCRIPT="${SCRIPT_DIR}/run_batch_aml.sh"

# Activate venv if present
if [[ -z "${VIRTUAL_ENV:-}" && -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# ── Defaults ─────────────────────────────────────────────────────────
CSV="/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/AML_HEALTHY_SLIDE_TEST.csv"
SLIDES_ROOT="/mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs"
OUTPUT_PARENT="/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin"
EXPERIMENT_NAME="exp_290425_aml_suite"
BASE_OUTPUT_ROOT=""
CUDA_DEVICE=""
EXTRACTORS_FILTER=""
MODELS_FILTER=""
RESUME=false
USE_TILE_CACHE=false
PARALLEL_MODELS=false
SLIDE_TIMEOUT=0
AUTO_RESTART=3
AUTO_RESTART_DELAY=30

# Pull tile/agent defaults from configs/config.yaml
eval "$(
    cd "${REPO_ROOT}" && python3 - <<'PY'
from pathlib import Path
import shlex
import yaml

try:
    data = yaml.safe_load(Path("configs/config.yaml").read_text()) or {}
except Exception:
    data = {}

slide_cfg = data.get("tools", {}).get("slide", {}) or {}
values = {
    "TILE_FILTER":    str(slide_cfg.get("TILE_FILTER", "hybrid")),
    "TILE_SIZE_PX":   str(slide_cfg.get("TILE_SIZE_PX", "224")),
    "BATCH_SIZE":     str(slide_cfg.get("BATCH_SIZE", "512")),
    "ROI_SIZE_PX":    str(slide_cfg.get("ROI_SIZE_PX", "2048")),
    "AGENT":          str(slide_cfg.get("AGENT", "aml")),
    "DEFAULT_MPP_UM": str(slide_cfg.get("DEFAULT_MPP_UM", "0.159")),
}
for k, v in values.items():
    print(f"{k}={shlex.quote(v)}")
PY
)"

# ── Run matrix ───────────────────────────────────────────────────────
RUNS=(
    "GLM-4.6V-FP8|uni2|GLM-4.6V-FP8_UNI2_224px"
    "GLM-4.6V-FP8|virchow2|GLM-4.6V-FP8_Virchow2_224px"
    "GLM-4.6V-FP8|h_optimus_1|GLM-4.6V-FP8_H-optimus-1_224px"
    "GLM-4.6V-FP8|dinobloom_giant|GLM-4.6V-FP8_DinoBloom-G_224px"
    "GLM-4.6V-Flash|uni2|GLM-4.6V-Flash_UNI2_224px"
    "GLM-4.6V-Flash|virchow2|GLM-4.6V-Flash_Virchow2_224px"
    "GLM-4.6V-Flash|h_optimus_1|GLM-4.6V-Flash_H-optimus-1_224px"
    "GLM-4.6V-Flash|dinobloom_giant|GLM-4.6V-Flash_DinoBloom-G_224px"
    "gemma-4-31B-it-h200|uni2|gemma-4-31B-it-h200_UNI2_224px"
    "gemma-4-31B-it-h200|virchow2|gemma-4-31B-it-h200_Virchow2_224px"
    "gemma-4-31B-it-h200|h_optimus_1|gemma-4-31B-it-h200_H-optimus-1_224px"
    "gemma-4-31B-it-h200|dinobloom_giant|gemma-4-31B-it-h200_DinoBloom-G_224px"
    "medgemma-27b-it|uni2|medgemma-27b-it_UNI2_224px"
    "medgemma-27b-it|virchow2|medgemma-27b-it_Virchow2_224px"
    "medgemma-27b-it|h_optimus_1|medgemma-27b-it_H-optimus-1_224px"
    "medgemma-27b-it|dinobloom_giant|medgemma-27b-it_DinoBloom-G_224px"
    "Qwen3.5-397B-A17B-FP8|uni2|Qwen3.5-397B-A17B-FP8_UNI2_224px"
    "Qwen3.5-397B-A17B-FP8|virchow2|Qwen3.5-397B-A17B-FP8_Virchow2_224px"
    "Qwen3.5-397B-A17B-FP8|h_optimus_1|Qwen3.5-397B-A17B-FP8_H-optimus-1_224px"
    "Qwen3.5-397B-A17B-FP8|dinobloom_giant|Qwen3.5-397B-A17B-FP8_DinoBloom-G_224px"
    "GPT-OSS-120B|uni2|GPT-OSS-120B_UNI2_224px"
    "GPT-OSS-120B|virchow2|GPT-OSS-120B_Virchow2_224px"
    "GPT-OSS-120B|h_optimus_1|GPT-OSS-120B_H-optimus-1_224px"
    "GPT-OSS-120B|dinobloom_giant|GPT-OSS-120B_DinoBloom-G_224px"
)

# ── Helpers ──────────────────────────────────────────────────────────
format_elapsed() {
    local t="${1:-0}"
    printf '%02dh:%02dm:%02ds' $((t/3600)) $(((t%3600)/60)) $((t%60))
}

usage() {
    cat <<EOF
Usage: bash run_batch_aml_suite.sh [options]

Filtering:
  --csv PATH               Patient CSV (default: ${CSV})
  --models LIST            Comma-separated models to keep (default: all)
  --extractors LIST        Comma-separated extractors to keep (default: all)

Output:
  --slides-root PATH       Slide root
  --output-parent PATH     Parent dir for the experiment folder
  --experiment-name NAME   Experiment folder name (default: ${EXPERIMENT_NAME})
  --base-output-root PATH  Override the full output dir
  --cuda-device ID         Set CUDA_VISIBLE_DEVICES

Behavior:
  --tile-filter NAME       Override config tile filter
  --roi-size-px N          Override config ROI size
  --default-mpp-um FLOAT   Override config MPP
  --use-tile-cache         Reuse cached tiles
  --resume                 Skip patients with status=ok summary.json
  --parallel-models        Run each model in its own background worker.
                           Concurrency = number of distinct models.
                           Each worker still runs its slides sequentially.
                           Ctrl-C kills all workers cleanly.
  -h, --help

Default is STRICTLY SEQUENTIAL (one combo at a time). Use
--parallel-models when you have N distinct VLM endpoints and want to
saturate all of them with 1 slide each at any moment.
EOF
}

_in_csv_list() {
    local needle="$1" haystack="$2"
    [[ -z "$haystack" ]] && return 0
    local IFS=','
    for item in $haystack; do
        item="${item// /}"
        [[ -n "$item" && "$item" == "$needle" ]] && return 0
    done
    return 1
}

# ── Parse args ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv) CSV="$2"; shift 2 ;;
        --slides-root) SLIDES_ROOT="$2"; shift 2 ;;
        --output-parent) OUTPUT_PARENT="$2"; shift 2 ;;
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --base-output-root) BASE_OUTPUT_ROOT="$2"; shift 2 ;;
        --cuda-device) CUDA_DEVICE="$2"; shift 2 ;;
        --models) MODELS_FILTER="$2"; shift 2 ;;
        --extractors) EXTRACTORS_FILTER="$2"; shift 2 ;;
        --tile-filter) TILE_FILTER="$2"; shift 2 ;;
        --roi-size-px) ROI_SIZE_PX="$2"; shift 2 ;;
        --default-mpp-um) DEFAULT_MPP_UM="$2"; shift 2 ;;
        --slide-timeout) SLIDE_TIMEOUT="$2"; shift 2 ;;
        --auto-restart) AUTO_RESTART="$2"; shift 2 ;;
        --auto-restart-delay) AUTO_RESTART_DELAY="$2"; shift 2 ;;
        --use-tile-cache) USE_TILE_CACHE=true; shift ;;
        --resume) RESUME=true; shift ;;
        --parallel-models) PARALLEL_MODELS=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1"; usage; exit 1 ;;
    esac
done

if [[ ! -f "$CSV" ]]; then
    echo "[ERROR] CSV not found: $CSV"
    exit 1
fi

[[ -z "$BASE_OUTPUT_ROOT" ]] && BASE_OUTPUT_ROOT="${OUTPUT_PARENT}/${EXPERIMENT_NAME}"
[[ "$BASE_OUTPUT_ROOT" != /* ]] && BASE_OUTPUT_ROOT="$(realpath "$BASE_OUTPUT_ROOT")"
mkdir -p "$BASE_OUTPUT_ROOT"

[[ -n "$CUDA_DEVICE" ]] && export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# ── Filter runs ──────────────────────────────────────────────────────
FILTERED_RUNS=()
for spec in "${RUNS[@]}"; do
    IFS="|" read -r _m _e _ <<<"$spec"
    _in_csv_list "$_m" "$MODELS_FILTER" || continue
    _in_csv_list "$_e" "$EXTRACTORS_FILTER" || continue
    FILTERED_RUNS+=("$spec")
done

if [[ ${#FILTERED_RUNS[@]} -eq 0 ]]; then
    echo "[ERROR] No runs matched models=${MODELS_FILTER:-*} extractors=${EXTRACTORS_FILTER:-*}"
    exit 1
fi

# ── Auto-restart on crash ────────────────────────────────────────────
_maybe_restart() {
    local rc="$1"
    if (( AUTO_RESTART <= 0 )); then
        echo " Auto-restart disabled, exiting (rc=${rc})"
        exit "$rc"
    fi
    echo ""
    echo " CRASH detected (rc=${rc}). Killing remaining children..."
    pkill -TERM -P $$ 2>/dev/null || true
    sleep 2
    pkill -KILL -P $$ 2>/dev/null || true
    echo " Restarting in ${AUTO_RESTART_DELAY}s... (restarts left: ${AUTO_RESTART})"
    sleep "$AUTO_RESTART_DELAY"
    # Strip --auto-restart from original args and re-exec with updated count
    NEW_ARGS=()
    skip_next=false
    for arg in "${ORIGINAL_ARGS[@]}"; do
        if $skip_next; then skip_next=false; continue; fi
        if [[ "$arg" == "--auto-restart" ]]; then skip_next=true; continue; fi
        NEW_ARGS+=("$arg")
    done
    exec "$0" "${NEW_ARGS[@]}" --resume --auto-restart $((AUTO_RESTART - 1))
}

# ── Banner ───────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
if $PARALLEL_MODELS; then
  echo " AML batch suite (one worker per model)"
else
  echo " AML batch suite (sequential)"
fi
echo " CSV:               $CSV"
echo " Output:            $BASE_OUTPUT_ROOT"
echo " Models filter:     ${MODELS_FILTER:-all}"
echo " Extractors filter: ${EXTRACTORS_FILTER:-all}"
echo " Total combos:      ${#FILTERED_RUNS[@]}"
echo " Resume:            $RESUME"
echo " Tile cache:        $USE_TILE_CACHE"
echo "═══════════════════════════════════════════════════════════════"

LOG_DIR="${BASE_OUTPUT_ROOT}/_suite_logs"
mkdir -p "$LOG_DIR"

# Build a single combo command (one model+extractor combo).
_build_cmd() {
    local model="$1" extractor="$2" out_dir="$3"
    local cmd=(
        bash "$RUN_BATCH_SCRIPT"
        --csv "$CSV"
        --slides-root "$SLIDES_ROOT"
        --output-dir "$out_dir"
        --experiment-root "$BASE_OUTPUT_ROOT"
        --model "$model"
        --extractor "$extractor"
        --tile-filter "$TILE_FILTER"
        --tile-size-px "$TILE_SIZE_PX"
        --batch-size "$BATCH_SIZE"
        --roi-size-px "$ROI_SIZE_PX"
        --default-mpp-um "$DEFAULT_MPP_UM"
        --agent "$AGENT"
    )
    [[ -n "$CUDA_DEVICE" ]]    && cmd+=(--cuda-device "$CUDA_DEVICE")
    (( SLIDE_TIMEOUT > 0 ))    && cmd+=(--slide-timeout "$SLIDE_TIMEOUT")
    $USE_TILE_CACHE            && cmd+=(--use-tile-cache)
    $RESUME                    && cmd+=(--resume)
    printf '%s\n' "${cmd[@]}"
}

SUITE_STARTED=$(date +%s)

if $PARALLEL_MODELS; then
    # ── One background worker per distinct model ─────────────────
    # Each worker iterates ITS OWN extractors strictly sequentially.
    # Total concurrent slides = number of distinct models.
    PIDS=()
    LABELS=()

    # Bulletproof cleanup: kill every descendant on Ctrl-C / EXIT.
    _cleanup() {
        local rc=$?
        trap - EXIT INT TERM
        echo ""
        echo " Cleaning up parallel workers (rc=${rc})..."
        for p in "${PIDS[@]:-}"; do
            kill -TERM "$p" 2>/dev/null || true
        done
        sleep 1
        for p in "${PIDS[@]:-}"; do
            kill -KILL "$p" 2>/dev/null || true
        done
        pkill -KILL -P $$ 2>/dev/null || true
        exit "$rc"
    }
    trap _cleanup EXIT INT TERM

    # Distinct models in the order they appear in FILTERED_RUNS
    DISTINCT_MODELS=()
    for spec in "${FILTERED_RUNS[@]}"; do
        IFS="|" read -r m _ _ <<<"$spec"
        seen=false
        for existing in "${DISTINCT_MODELS[@]:-}"; do
            [[ "$existing" == "$m" ]] && seen=true && break
        done
        $seen || DISTINCT_MODELS+=("$m")
    done

    # Count extractors per model (used for concurrency estimate)
    MAX_EXTRACTORS=0
    for model in "${DISTINCT_MODELS[@]}"; do
        n=0
        for spec in "${FILTERED_RUNS[@]}"; do
            IFS="|" read -r m _ _ <<<"$spec"
            [[ "$m" == "$model" ]] && n=$((n+1))
        done
        (( n > MAX_EXTRACTORS )) && MAX_EXTRACTORS=$n
    done
    TOTAL_CONCURRENT=$(( ${#FILTERED_RUNS[@]} ))

    echo " Distinct models:   ${#DISTINCT_MODELS[@]} → ${DISTINCT_MODELS[*]}"
    echo " Extractors/model:  up to ${MAX_EXTRACTORS} (run in parallel per model)"
    echo " Concurrent slides: ${TOTAL_CONCURRENT} (= models × extractors)"
    echo "═══════════════════════════════════════════════════════════════"

    for model in "${DISTINCT_MODELS[@]}"; do
        WORKER_LOG="${LOG_DIR}/${model//\//_}.worker.log"
        echo " Launching worker for ${model} → ${WORKER_LOG}"

        # Subshell: launch all of THIS model's extractors in parallel.
        # The subshell is its own process group, so its trap handles
        # cleanup of grandchildren when the parent suite is Ctrl-C'd.
        (
            set -m
            sub_pids=()
            _sub_cleanup() {
                for sp in "${sub_pids[@]:-}"; do
                    kill -TERM "$sp" 2>/dev/null || true
                done
                sleep 1
                pkill -KILL -P $$ 2>/dev/null || true
            }
            trap _sub_cleanup EXIT INT TERM

            for spec in "${FILTERED_RUNS[@]}"; do
                IFS="|" read -r m e out <<<"$spec"
                [[ "$m" == "$model" ]] || continue
                OUTPUT_DIR="${BASE_OUTPUT_ROOT}/${out}"
                echo "─── launching ${m} × ${e} ───────────────────"
                mapfile -t CMD < <(_build_cmd "$m" "$e" "$OUTPUT_DIR")
                "${CMD[@]}" &
                sub_pids+=($!)
            done

            rc=0
            for sp in "${sub_pids[@]}"; do
                if ! wait "$sp"; then rc=$?; fi
            done
            exit "$rc"
        ) >"$WORKER_LOG" 2>&1 &
        PIDS+=($!)
        LABELS+=("$model")
    done

    echo ""
    echo " ${#PIDS[@]} workers running. Tail logs in ${LOG_DIR}/."
    echo " Ctrl-C will kill all workers cleanly."
    echo ""

    FAILED=()
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo " [OK]   ${LABELS[$i]}"
        else
            echo " [FAIL] ${LABELS[$i]}"
            FAILED+=("${LABELS[$i]}")
        fi
    done

    trap - EXIT INT TERM

    if (( ${#FAILED[@]} > 0 )); then
        echo ""
        echo " ${#FAILED[@]} worker(s) failed: ${FAILED[*]}"
        exit 1
    fi
else
    # ── Strictly sequential ──────────────────────────────────────
    for spec in "${FILTERED_RUNS[@]}"; do
        IFS="|" read -r MODEL EXTRACTOR OUTPUT_NAME <<<"$spec"
        OUTPUT_DIR="${BASE_OUTPUT_ROOT}/${OUTPUT_NAME}"
        LOG_FILE="${LOG_DIR}/${MODEL//\//_}_${EXTRACTOR}.log"

        echo ""
        echo "─── ${MODEL} × ${EXTRACTOR} ─────────────────────────────"
        echo "    log: $LOG_FILE"

        mapfile -t CMD < <(_build_cmd "$MODEL" "$EXTRACTOR" "$OUTPUT_DIR")
        started=$(date +%s)
        if "${CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
            echo "    OK  elapsed=$(format_elapsed $(($(date +%s) - started)))"
        else
            STATUS=${PIPESTATUS[0]}
            echo "    FAIL status=${STATUS}"
            _maybe_restart "$STATUS"
        fi
    done
fi

ELAPSED=$(($(date +%s) - SUITE_STARTED))
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " AML batch suite complete — elapsed=$(format_elapsed $ELAPSED)"
echo "═══════════════════════════════════════════════════════════════"
