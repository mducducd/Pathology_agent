#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_batch_aml.sh — Run AML detector on every patient in a CSV list.
#
# Usage:
#   bash run_batch_aml.sh \
#       [--csv /path/to/patients.csv] \
#       [--slides-root /path/to/ALL_WSIs] \
#       [--output-dir ./batch_outputs] \
#       [--model GLM-4.6V-FP8] \
#       [--extractor uni2] \
#       [--tile-filter hybrid] \
#       [--tile-size-px 224] \
#       [--batch-size 512] \
#       [--experiment-root /path/to/experiment] \
#       [--use-tile-cache] \
#       [--resume]
#
# The --resume flag skips patients that already have a summary.json
# with status=ok and a non-empty final_decision in summary.json.
# If a slide run exits with an error during the current batch, the
# script reruns that slide once automatically.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_SINGLE_SLIDE="${SCRIPT_DIR}/run_single_slide.py"

# ── Defaults ─────────────────────────────────────────────────────────
CSV="/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/AML_HEALTHY_SLIDE_TEST.csv"
SLIDES_ROOT="/mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs"
BASE_OUTPUT_ROOT="/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/new_runs"
OUTPUT_DIR=""
EXPERIMENT_ROOT=""
CUDA_DEVICE=""
MODEL="GLM-4.6V-FP8"
EXTRACTOR="uni2"
TILE_FILTER="hybrid"
TILE_SIZE_PX="224"
BATCH_SIZE="512"
AGENT="aml"
RESUME=false
USE_TILE_CACHE=false
PYTHON_BIN="python"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi

cd "$REPO_ROOT"


# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv)          CSV="$2";         shift 2 ;;
        --slides-root)  SLIDES_ROOT="$2"; shift 2 ;;
        --base-output-root) BASE_OUTPUT_ROOT="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";  shift 2 ;;
        --experiment-root) EXPERIMENT_ROOT="$2"; shift 2 ;;
        --cuda-device)  CUDA_DEVICE="$2"; shift 2 ;;
        --model)        MODEL="$2";       shift 2 ;;
        --extractor)    EXTRACTOR="$2";   shift 2 ;;
        --tile-filter)  TILE_FILTER="$2"; shift 2 ;;
        --tile-size-px) TILE_SIZE_PX="$2"; shift 2 ;;
        --batch-size)   BATCH_SIZE="$2";  shift 2 ;;
        --agent)        AGENT="$2";       shift 2 ;;
        --use-tile-cache) USE_TILE_CACHE=true; shift ;;
        --resume)       RESUME=true;      shift   ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Force all output-related paths to absolute
if [[ -n "$BASE_OUTPUT_ROOT" && "$BASE_OUTPUT_ROOT" != /* ]]; then
    BASE_OUTPUT_ROOT="$(realpath "$BASE_OUTPUT_ROOT")"
fi
if [[ -n "$OUTPUT_DIR" && "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
fi
if [[ -n "$EXPERIMENT_ROOT" && "$EXPERIMENT_ROOT" != /* ]]; then
    EXPERIMENT_ROOT="$(realpath "$EXPERIMENT_ROOT")"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    case "$EXTRACTOR" in
        uni2) EXTRACTOR_TAG="UNI2" ;;
        h_optimus_1) EXTRACTOR_TAG="H-optimus-1" ;;
        virchow2) EXTRACTOR_TAG="Virchow2" ;;
        dinobloom) EXTRACTOR_TAG="DinoBloom-S" ;;
        dinobloom_base) EXTRACTOR_TAG="DinoBloom-B" ;;
        dinobloom_large) EXTRACTOR_TAG="DinoBloom-L" ;;
        dinobloom_giant) EXTRACTOR_TAG="DinoBloom-G" ;;
        reddino) EXTRACTOR_TAG="RedDino-Small" ;;
        reddino_base) EXTRACTOR_TAG="RedDino-base" ;;
        reddino_large) EXTRACTOR_TAG="RedDino-large" ;;
        *) EXTRACTOR_TAG="$EXTRACTOR" ;;
    esac
    OUTPUT_DIR="${BASE_OUTPUT_ROOT}/batch_result_${MODEL}_${EXTRACTOR_TAG}_${TILE_SIZE_PX}px"
    BASE_OUTPUT_ROOT="$(dirname "$OUTPUT_DIR")"
fi

if [[ -z "$EXPERIMENT_ROOT" ]]; then
    EXPERIMENT_ROOT="$OUTPUT_DIR"
fi
if [[ -n "$CUDA_DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

format_elapsed() {
    local total_seconds="${1:-0}"
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf '%02dh:%02dm:%02ds' "$hours" "$minutes" "$seconds"
}

print_failure_reason() {
    local summary_path="$1"
    local log_path="$2"
    local reason=""

    if [[ -f "$summary_path" ]]; then
        reason=$("$PYTHON_BIN" - "$summary_path" <<'PY'
import json
import sys

summary_path = sys.argv[1]

try:
    with open(summary_path) as handle:
        summary = json.load(handle)
except Exception as exc:
    print(f"Could not parse summary.json: {type(exc).__name__}: {exc}")
    raise SystemExit

error = str(summary.get("error") or "").strip()
status = str(summary.get("status") or "").strip()
if error:
    print(error)
elif status:
    print(f"Run ended with status={status!r} but no explicit error message was recorded")
PY
)
    fi

    if [[ -n "$reason" ]]; then
        echo "      reason: $reason"
        return
    fi

    if [[ -f "$log_path" ]]; then
        echo "      reason: summary.json had no error; showing tail of ${log_path}"
        tail -n 20 "$log_path" | sed 's/^/      | /'
        return
    fi

    echo "      reason: no summary.json error and no retry log found"
}

# ── Read patient list (skip header) ─────────────────────────────────
mapfile -t PATIENTS < <(tail -n +2 "$CSV" | sed 's/\r//g' | grep -v '^$')
TOTAL=${#PATIENTS[@]}

echo "═══════════════════════════════════════════════════════════════"
echo " Slide-Agent batch AML run"
echo " CSV:         $CSV"
echo " Slides root: $SLIDES_ROOT"
echo " Base root:   $BASE_OUTPUT_ROOT"
echo " Output dir:  $OUTPUT_DIR"
echo " Agent:       $AGENT"
echo " Model:       $MODEL   Extractor: $EXTRACTOR   Filter: $TILE_FILTER"
echo " Tile size:   ${TILE_SIZE_PX}px"
echo " Batch size:  $BATCH_SIZE"
echo " Tile cache:  $USE_TILE_CACHE"
echo " CUDA devices:${CUDA_VISIBLE_DEVICES:+ }${CUDA_VISIBLE_DEVICES:-all}"
echo " Cache root:  $EXPERIMENT_ROOT"
echo " Patients:    $TOTAL"
echo " Resume:      $RESUME"
echo "═══════════════════════════════════════════════════════════════"

LOG_DIR="${OUTPUT_DIR}/_logs"
mkdir -p "$LOG_DIR"

PASSED=0
FAILED=0
SKIPPED=0
RETRIED=0

for i in "${!PATIENTS[@]}"; do
    ENTRY="${PATIENTS[$i]%%,*}"
    ENTRY="${ENTRY%\"}"
    ENTRY="${ENTRY#\"}"
    IDX=$((i + 1))

    # ── Resolve slide path (.mrxs) ──────────────────────────────────
    if [[ "$ENTRY" == *.mrxs ]]; then
        SLIDE="$ENTRY"
        PATIENT="$(basename "${SLIDE%.mrxs}")"
    else
        PATIENT="$ENTRY"
        SLIDE="${SLIDES_ROOT}/${PATIENT}.mrxs"
    fi
    if [[ ! -f "$SLIDE" ]]; then
        echo "[$IDX/$TOTAL] SKIP  $PATIENT — .mrxs not found"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # ── Resume: skip if already completed ───────────────────────────
    SUMMARY="${OUTPUT_DIR}/${PATIENT}/summary.json"
    if $RESUME && [[ -f "$SUMMARY" ]]; then
        RESUME_STATE=$("$PYTHON_BIN" - "$SUMMARY" <<'PY'
import json
import sys

summary_path = sys.argv[1]

try:
    with open(summary_path) as handle:
        summary = json.load(handle)
except Exception:
    print("rerun:invalid summary.json")
    raise SystemExit

if str(summary.get("status") or "").strip() != "ok":
    print("rerun")
    raise SystemExit

summary_final_decision = str(summary.get("final_decision") or "").strip()
if summary_final_decision:
    print("skip")
    raise SystemExit

print("rerun:status=ok but final_decision is missing")
PY
)
        if [[ "$RESUME_STATE" == "skip" ]]; then
            echo "[$IDX/$TOTAL] DONE  $PATIENT (already completed, skipping)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        if [[ "$RESUME_STATE" == rerun:* ]]; then
            echo "[$IDX/$TOTAL] RERUN $PATIENT — ${RESUME_STATE#rerun:}"
        fi
    fi

    # ── Run (suppress all python output) ───────────────────────────
    LOG="${LOG_DIR}/${PATIENT}.log"
    SLIDE_STARTED_EPOCH="$(date +%s)"

    RUN_CMD=(
        "$PYTHON_BIN"
        "$RUN_SINGLE_SLIDE"
        --slide "$SLIDE"
        --output-dir "$OUTPUT_DIR"
        --model "$MODEL"
        --extractor "$EXTRACTOR"
        --tile-filter "$TILE_FILTER"
        --tile-size-px "$TILE_SIZE_PX"
        --batch-size "$BATCH_SIZE"
        --agent "$AGENT"
    )
    if [[ -n "$CUDA_DEVICE" ]]; then
        RUN_CMD+=(--cuda-device "$CUDA_DEVICE")
    fi
    RUN_CMD+=(--experiment-root "$EXPERIMENT_ROOT")
    if $USE_TILE_CACHE; then
        RUN_CMD+=(--use-tile-cache)
    fi

    if "${RUN_CMD[@]}" \
        >"$LOG" 2>&1; then
        PASSED=$((PASSED + 1))
        SLIDE_ELAPSED_SECONDS=$(( $(date +%s) - SLIDE_STARTED_EPOCH ))
        echo "[$IDX/$TOTAL] OK    $PATIENT elapsed=$(format_elapsed "$SLIDE_ELAPSED_SECONDS")"
    else
        RETRY_LOG="${LOG_DIR}/${PATIENT}.retry.log"
        echo "[$IDX/$TOTAL] RETRY $PATIENT — previous attempt returned error"
        if "${RUN_CMD[@]}" \
            >"$RETRY_LOG" 2>&1; then
            PASSED=$((PASSED + 1))
            RETRIED=$((RETRIED + 1))
            SLIDE_ELAPSED_SECONDS=$(( $(date +%s) - SLIDE_STARTED_EPOCH ))
            echo "[$IDX/$TOTAL] OK    $PATIENT retry_succeeded elapsed=$(format_elapsed "$SLIDE_ELAPSED_SECONDS")"
        else
            FAILED=$((FAILED + 1))
            SLIDE_ELAPSED_SECONDS=$(( $(date +%s) - SLIDE_STARTED_EPOCH ))
            echo "[$IDX/$TOTAL] FAIL  $PATIENT elapsed=$(format_elapsed "$SLIDE_ELAPSED_SECONDS")"
            print_failure_reason "$SUMMARY" "$RETRY_LOG"
        fi
    fi

    echo "[$IDX/$TOTAL] Progress: $PASSED ok / $FAILED fail / $SKIPPED skip / $RETRIED retried"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " BATCH COMPLETE"
echo " Total: $TOTAL  |  OK: $PASSED  |  FAIL: $FAILED  |  SKIP: $SKIPPED  |  RETRIED: $RETRIED"
echo "═══════════════════════════════════════════════════════════════"
