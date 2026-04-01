#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_batch_aml.sh — Run AML detector on every patient in a CSV list.
#
# Usage:
#   bash run_batch_aml.sh \
#       [--csv /path/to/patients.csv] \
#       [--slides-root /path/to/ALL_WSIs] \
#       [--output-dir ./batch_outputs] \
#       [--model GPT-OSS-120B] \
#       [--extractor uni2] \
#       [--tile-filter hybrid] \
#       [--resume]
#
# The --resume flag skips patients that already have a summary.json
# with status=ok and a non-empty final_decision in artifacts/wsi_reports/report.json.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────
CSV="/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/random_100_Normal_AML_Patients.csv"
SLIDES_ROOT="/mnt/copernicus3/PATHOLOGY/others/private/haemadata/ALL_WSIs"
OUTPUT_DIR="./batch_outputs"
MODEL="GPT-OSS-120B"
EXTRACTOR="uni2"
TILE_FILTER="hybrid"
RESUME=false

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv)          CSV="$2";         shift 2 ;;
        --slides-root)  SLIDES_ROOT="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";  shift 2 ;;
        --model)        MODEL="$2";       shift 2 ;;
        --extractor)    EXTRACTOR="$2";   shift 2 ;;
        --tile-filter)  TILE_FILTER="$2"; shift 2 ;;
        --resume)       RESUME=true;      shift   ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Read patient list (skip header) ─────────────────────────────────
mapfile -t PATIENTS < <(tail -n +2 "$CSV" | sed 's/\r//g' | grep -v '^$')
TOTAL=${#PATIENTS[@]}

echo "═══════════════════════════════════════════════════════════════"
echo " Slide-Agent batch AML run"
echo " CSV:         $CSV"
echo " Slides root: $SLIDES_ROOT"
echo " Output dir:  $OUTPUT_DIR"
echo " Model:       $MODEL   Extractor: $EXTRACTOR   Filter: $TILE_FILTER"
echo " Patients:    $TOTAL"
echo " Resume:      $RESUME"
echo "═══════════════════════════════════════════════════════════════"

PASSED=0
FAILED=0
SKIPPED=0

for i in "${!PATIENTS[@]}"; do
    PATIENT="${PATIENTS[$i]}"
    IDX=$((i + 1))

    # ── Resolve slide path (.mrxs) ──────────────────────────────────
    SLIDE="${SLIDES_ROOT}/${PATIENT}.mrxs"
    if [[ ! -f "$SLIDE" ]]; then
        echo "[$IDX/$TOTAL] SKIP  $PATIENT — .mrxs not found"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # ── Resume: skip if already completed ───────────────────────────
    SUMMARY="${OUTPUT_DIR}/${PATIENT}/summary.json"
    REPORT_JSON="${OUTPUT_DIR}/${PATIENT}/artifacts/wsi_reports/report.json"
    if $RESUME && [[ -f "$SUMMARY" ]]; then
        RESUME_STATE=$(python3 - "$SUMMARY" "$REPORT_JSON" <<'PY'
import json
import sys

summary_path, report_json_path = sys.argv[1], sys.argv[2]

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

try:
    with open(report_json_path) as handle:
        report = json.load(handle)
except Exception:
    print("rerun:status=ok but report.json is missing or unreadable")
    raise SystemExit

final_decision = str(report.get("final_decision") or "").strip()
if final_decision:
    print("skip")
else:
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
    LOG="${OUTPUT_DIR}/${PATIENT}/run.log"
    mkdir -p "${OUTPUT_DIR}/${PATIENT}"

    if python evaluate/run_single_slide.py \
        --slide "$SLIDE" \
        --output-dir "$OUTPUT_DIR" \
        --model "$MODEL" \
        --extractor "$EXTRACTOR" \
        --tile-filter "$TILE_FILTER" >"$LOG" 2>&1; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
    fi

    echo "[$IDX/$TOTAL] Progress: $PASSED ok / $FAILED fail / $SKIPPED skip"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " BATCH COMPLETE"
echo " Total: $TOTAL  |  OK: $PASSED  |  FAIL: $FAILED  |  SKIP: $SKIPPED"
echo "═══════════════════════════════════════════════════════════════"
