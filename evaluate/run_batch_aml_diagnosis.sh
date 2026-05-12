#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_batch_aml_diagnosis.sh — Run aml_diagnosis on existing ROI
# collections from a prior aml_roi / aml_auto experiment run.
#
# INPUT STRUCTURE (two accepted forms):
#   Exp root:        /path/to/exp_290425_aml_suite/
#     └── {model_extractor_dir}/   (e.g. GLM-4.6V-FP8_DinoBloom-G_224px)
#           └── {slide_id}/
#                 └── images/roi.png (or roi_1.jpg / roi_*.jpg)
#   Single subfolder: /path/to/exp_290425_aml_suite/GLM-4.6V-FP8_DinoBloom-G_224px
#         └── {slide_id}/images/roi.png (or roi_1.jpg / roi_*.jpg)
#
# PARALLELISM MODES (mutually exclusive):
#   --parallel     Run each model-extractor subfolder as its own
#                  background worker (up to N concurrent processes where
#                  N = number of matched subfolders).
#   --chunks-dir   A directory containing part_N.csv files. Each CSV
#                  must have a column whose values match slide dir names
#                  (case_id). One chunk → one terminal / screen session.
#                  Use with --chunk-index to pick which chunk to run.
#
# OUTPUT: diagnosis results (final_output.txt, report.md) are written
# INSIDE the same slide dir. A
# "diagnosis_done" marker is created on success to support --resume.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate venv if present
if [[ -z "${VIRTUAL_ENV:-}" && -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# ── Pull diagnosis model from config.yaml ────────────────────────────
DIAG_MODEL="$(
    cd "${REPO_ROOT}" && python3 - <<'PY'
from pathlib import Path
import yaml, shlex
try:
    data = yaml.safe_load(Path("configs/config.yaml").read_text()) or {}
except Exception:
    data = {}
model = data.get("tools", {}).get("slide", {}).get("MODEL_NAME") \
     or data.get("agent", {}).get("MODEL_NAME", "GLM-4.6V-FP8")
print(shlex.quote(str(model)))
PY
)"
# strip surrounding quotes from shlex.quote
DIAG_MODEL="${DIAG_MODEL//\'/}"

# ── Defaults ─────────────────────────────────────────────────────────
EXP_PATH=""
OUTPUT_ROOT=""         # optional: write results here instead of in-place
CHUNKS_DIR=""
CHUNK_INDEX=""
SUBDIR_FILTER=""       # comma-sep model-extractor subfolder names to include
RESUME=false
PARALLEL=false
CUDA_DEVICE=""

# ── Usage ─────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: bash run_batch_aml_diagnosis.sh --exp-path PATH [options]

Required:
  --exp-path PATH         Experiment root OR a single model-extractor subfolder.
                          Examples:
                            /path/exp_290425_aml_suite
                            /path/exp_290425_aml_suite/GLM-4.6V-FP8_DinoBloom-G_224px

Options:
  --model NAME            VLM model to use for diagnosis (default: from config)
  --output-root PATH      Write results to this directory mirroring the exp structure.
                          Default: overwrite in-place in each slide directory.
  --subdir-filter LIST    Comma-separated subfolder names to include (default: all)

Parallelism:
  --parallel              Launch one background process per model-extractor subfolder.
                          Each process runs its slides sequentially.
  --chunks-dir PATH       Directory with part_1.csv .. part_N.csv files.
                          Use with --chunk-index to select one chunk.
  --chunk-index N         Which chunk (1-based) to run when using --chunks-dir.

Behavior:
  --resume                Skip slide dirs that already have a diagnosis_done marker.
  --cuda-device ID        Set CUDA_VISIBLE_DEVICES.
  -h, --help

Examples:
  # Overwrite in-place, all subfolders in parallel:
  bash run_batch_aml_diagnosis.sh --exp-path /path/exp --parallel

  # Save to a separate dir with same structure:
  bash run_batch_aml_diagnosis.sh --exp-path /path/exp \\
    --output-root /path/exp_diagnosis --parallel

  # Gemma only, chunk 1, separate output dir:
  bash run_batch_aml_diagnosis.sh --exp-path /path/exp \\
    --output-root /path/exp_diagnosis \\
    --subdir-filter gemma-4-31B-it_DinoBloom-G_224px,gemma-4-31B-it_UNI2_224px \\
    --chunks-dir /path/chunks_gemma --chunk-index 1

  # Run a specific subfolder sequentially:
  bash run_batch_aml_diagnosis.sh \\
    --exp-path /path/exp/GLM-4.6V-FP8_DinoBloom-G_224px
EOF
}

# ── Parse args ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-path)       EXP_PATH="$2"; shift 2 ;;
        --output-root)    OUTPUT_ROOT="$2"; shift 2 ;;
        --model)          DIAG_MODEL="$2"; shift 2 ;;
        --subdir-filter)  SUBDIR_FILTER="$2"; shift 2 ;;
        --chunks-dir)     CHUNKS_DIR="$2"; shift 2 ;;
        --chunk-index)    CHUNK_INDEX="$2"; shift 2 ;;
        --resume)         RESUME=true; shift ;;
        --parallel)       PARALLEL=true; shift ;;
        --cuda-device)    CUDA_DEVICE="$2"; shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "Unknown arg: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "$EXP_PATH" ]]; then
    echo "[ERROR] --exp-path is required"
    usage; exit 1
fi
EXP_PATH="$(realpath "$EXP_PATH")"
if [[ ! -d "$EXP_PATH" ]]; then
    echo "[ERROR] Not a directory: $EXP_PATH"
    exit 1
fi
[[ -n "$CUDA_DEVICE" ]] && export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# ── Validate chunk CSV if requested ────────────────────────────────────
CHUNK_CSV=""
if [[ -n "$CHUNKS_DIR" ]]; then
    if [[ -z "$CHUNK_INDEX" ]]; then
        echo "[ERROR] --chunk-index is required when using --chunks-dir"
        exit 1
    fi
    CHUNK_CSV="${CHUNKS_DIR}/part_${CHUNK_INDEX}.csv"
    if [[ ! -f "$CHUNK_CSV" ]]; then
        echo "[ERROR] Chunk CSV not found: $CHUNK_CSV"
        exit 1
    fi
    CHUNK_COUNT="$(grep -v '^\(#\|$\|[Cc]ase\|[Ss]lide\|[Pp]atient\|[Ii]d\|[Nn]ame\)' "$CHUNK_CSV" | wc -l)"
    echo "[INFO] Chunk ${CHUNK_INDEX}: ${CHUNK_COUNT} slides from ${CHUNK_CSV}"
fi

_in_chunk() {
    local case_id="$1"
    [[ -z "$CHUNK_CSV" ]] && return 0  # No chunk filter, always include
    grep -q "^${case_id}[,\"]" "$CHUNK_CSV" 2>/dev/null && return 0
    grep -q "^${case_id}$" "$CHUNK_CSV" 2>/dev/null && return 0
    return 1
}

# ── Detect whether exp-path is a root or a single subfolder ──────────
# A single-subfolder input has a primary ROI image in child images/ dirs
# A root input has model-extractor subdirs as children

SUBDIRS=()
# Robustly detect whether EXP_PATH is:
#   1) a single model-extractor dir: at least one direct child has ROI image
#   2) an experiment root: one or more direct children have slide dirs with ROI image
# This intentionally ignores helper dirs such as _logs, _cache, and _suite_logs.
if find "$EXP_PATH" -mindepth 3 -maxdepth 3 -type f \( \
        -path "*/images/roi.png" -o \
        -path "*/images/roi_1.jpg" -o \
        -path "*/images/roi_1.jpeg" -o \
        -path "*/images/roi_1.png" -o \
        -path "*/images/roi_*.jpg" -o \
        -path "*/images/roi_*.jpeg" -o \
        -path "*/images/roi_*.png" \
    \) -print -quit | grep -q .; then
    SUBDIRS+=("$EXP_PATH")
else
    while IFS= read -r d; do
        if find "$d" -mindepth 3 -maxdepth 3 -type f \( \
                -path "*/images/roi.png" -o \
                -path "*/images/roi_1.jpg" -o \
                -path "*/images/roi_1.jpeg" -o \
                -path "*/images/roi_1.png" -o \
                -path "*/images/roi_*.jpg" -o \
                -path "*/images/roi_*.jpeg" -o \
                -path "*/images/roi_*.png" \
            \) -print -quit | grep -q .; then
            SUBDIRS+=("$d")
        fi
    done < <(find "$EXP_PATH" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort)
fi

if [[ ${#SUBDIRS[@]} -eq 0 ]]; then
    echo "[ERROR] No model-extractor subdirs with ROI images found under: $EXP_PATH"
    exit 1
fi

# ── Apply subdir filter ───────────────────────────────────────────────
_in_list() {
    local needle="$1" haystack="$2"
    [[ -z "$haystack" ]] && return 0
    local IFS=','; for item in $haystack; do
        [[ "${item// /}" == "$needle" ]] && return 0
    done
    return 1
}

FILTERED_SUBDIRS=()
for d in "${SUBDIRS[@]}"; do
    name="$(basename "$d")"
    _in_list "$name" "$SUBDIR_FILTER" && FILTERED_SUBDIRS+=("$d")
done
SUBDIRS=("${FILTERED_SUBDIRS[@]}")

if [[ ${#SUBDIRS[@]} -eq 0 ]]; then
    echo "[ERROR] No subdirs matched subdir-filter='${SUBDIR_FILTER}'"
    exit 1
fi

# ── Core: run diagnosis for one slide dir ────────────────────────────
run_slide_diagnosis() {
    local slide_dir="$1"
    local case_id; case_id="$(basename "$slide_dir")"
    local roi_input=""
    if [[ -f "${slide_dir}/images/roi.png" ]]; then
        roi_input="${slide_dir}/images/roi.png"
    elif [[ -f "${slide_dir}/images/roi_1.jpg" ]]; then
        roi_input="${slide_dir}/images/roi_1.jpg"
    elif [[ -f "${slide_dir}/images/roi_1.jpeg" ]]; then
        roi_input="${slide_dir}/images/roi_1.jpeg"
    elif [[ -f "${slide_dir}/images/roi_1.png" ]]; then
        roi_input="${slide_dir}/images/roi_1.png"
    else
        roi_input="$(find "${slide_dir}/images" -maxdepth 1 -type f \( -name 'roi_*.jpg' -o -name 'roi_*.jpeg' -o -name 'roi_*.png' \) | sort | head -n 1 || true)"
    fi
    local done_marker="${slide_dir}/diagnosis_done"

    [[ -z "$roi_input" || ! -f "$roi_input" ]] && return 0

    if $RESUME && [[ -f "$done_marker" ]]; then
        echo "  [SKIP] ${case_id} (already done)"
        return 0
    fi

    local out_parent; out_parent="$(dirname "$slide_dir")"
    echo "  [RUN]  ${case_id} -> ${slide_dir}"
    if python3 "${SCRIPT_DIR}/run_single_slide.py" \
            --output-dir "$out_parent" \
            --agent aml_diagnosis \
            --roi-input-path "$roi_input" \
            --model "$DIAG_MODEL" 2>&1; then
        touch "$done_marker"
        echo "  [OK]   ${case_id}"
    else
        echo "  [FAIL] ${case_id}"
    fi
}

# ── Resolve output dir for a slide (in-place or mirrored) ────────────
# Args: subdir (model-extractor dir), case_id
_resolve_out_dir() {
    local subdir="$1" case_id="$2"
    if [[ -n "$OUTPUT_ROOT" ]]; then
        local subdir_name; subdir_name="$(basename "$subdir")"
        echo "${OUTPUT_ROOT}/${subdir_name}/${case_id}"
    else
        echo "${subdir}/${case_id}"
    fi
}

# ── Core: run one model-extractor subfolder sequentially ─────────────
run_subdir() {
    local subdir="$1"
    local name; name="$(basename "$subdir")"
    echo ""
    echo "─── ${name} ─────────────────────────────────────────"

    local count=0 done=0 skip=0 fail=0
    while IFS= read -r slide_dir; do
        local case_id; case_id="$(basename "$slide_dir")"

        # Chunk filter
        if ! _in_chunk "$case_id"; then
            continue
        fi

        count=$((count + 1))
        local roi_input=""
        if [[ -f "${slide_dir}/images/roi.png" ]]; then
            roi_input="${slide_dir}/images/roi.png"
        elif [[ -f "${slide_dir}/images/roi_1.jpg" ]]; then
            roi_input="${slide_dir}/images/roi_1.jpg"
        elif [[ -f "${slide_dir}/images/roi_1.jpeg" ]]; then
            roi_input="${slide_dir}/images/roi_1.jpeg"
        elif [[ -f "${slide_dir}/images/roi_1.png" ]]; then
            roi_input="${slide_dir}/images/roi_1.png"
        else
            roi_input="$(find "${slide_dir}/images" -maxdepth 1 -type f \( -name 'roi_*.jpg' -o -name 'roi_*.jpeg' -o -name 'roi_*.png' \) | sort | head -n 1 || true)"
        fi

        if [[ -z "$roi_input" || ! -f "$roi_input" ]]; then
            echo "  [SKIP] ${case_id} (no ROI image in images/)"
            skip=$((skip + 1))
            continue
        fi

        # Resolve where to write results
        local out_dir; out_dir="$(_resolve_out_dir "$subdir" "$case_id")"
        local done_marker="${out_dir}/diagnosis_done"

        if $RESUME && [[ -f "$done_marker" ]]; then
            echo "  [SKIP] ${case_id} (already done)"
            skip=$((skip + 1))
            continue
        fi

        mkdir -p "$out_dir"
        local out_parent; out_parent="$(dirname "$out_dir")"
        echo "  [RUN]  ${case_id} -> ${out_dir}"
        if python3 "${SCRIPT_DIR}/run_single_slide.py" \
                --output-dir "$out_parent" \
                --agent aml_diagnosis \
                --roi-input-path "$roi_input" \
                --model "$DIAG_MODEL" 2>&1; then

            # When writing to a mirrored output root, carry over the source ROI images.
            # In-place diagnosis already points at the same images directory.
            if [[ -d "${slide_dir}/images" && "$(realpath "$slide_dir")" != "$(realpath "$out_dir")" ]]; then
                rm -rf "${out_dir}/images"
                mkdir -p "${out_dir}/images"
                cp -a "${slide_dir}/images/." "${out_dir}/images/"
            fi

            touch "$done_marker"
            done=$((done + 1))
        else
            echo "  [FAIL] ${case_id}"
            fail=$((fail + 1))
        fi
    done < <(find "$subdir" -maxdepth 1 -mindepth 1 -type d | sort)

    echo "  ── ${name}: total=${count} done=${done} skip=${skip} fail=${fail}"
}

# ── Banner ────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo " AML Diagnosis batch"
echo " Exp path:     $EXP_PATH"
if [[ -n "$OUTPUT_ROOT" ]]; then
    echo " Output root:  $OUTPUT_ROOT  (mirrored structure)"
else
    echo " Output root:  (in-place, overwrites existing)"
fi
echo " Subfolders:   ${#SUBDIRS[@]}"
echo " Diag model:   $DIAG_MODEL"
echo " Resume:       $RESUME"
echo " Parallel:     $PARALLEL"
[[ -n "$CHUNKS_DIR" ]] && echo " Chunk:        ${CHUNKS_DIR}/part_${CHUNK_INDEX}.csv (${CHUNK_COUNT:-?} slides)"
echo "═══════════════════════════════════════════════════════════════"

SUITE_START=$(date +%s)

if $PARALLEL; then
    # ── One background process per model-extractor subfolder ─────
    PIDS=(); LABELS=()
    _cleanup() {
        trap - EXIT INT TERM
        echo ""; echo " Cleaning up..."
        for p in "${PIDS[@]:-}"; do kill -TERM "$p" 2>/dev/null || true; done
        sleep 1
        pkill -KILL -P $$ 2>/dev/null || true
    }
    trap _cleanup EXIT INT TERM

    LOG_DIR="${EXP_PATH}/_diagnosis_logs"
    mkdir -p "$LOG_DIR"

    for subdir in "${SUBDIRS[@]}"; do
        name="$(basename "$subdir")"
        log="${LOG_DIR}/${name}.log"
        echo " Launching: ${name} → ${log}"
        ( run_subdir "$subdir" ) >"$log" 2>&1 &
        PIDS+=($!); LABELS+=("$name")
    done

    echo ""
    echo " ${#PIDS[@]} workers running. Logs: ${LOG_DIR}/"
    echo " Ctrl-C kills all cleanly."
    echo ""

    FAILED=()
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then echo " [OK]   ${LABELS[$i]}"
        else echo " [FAIL] ${LABELS[$i]}"; FAILED+=("${LABELS[$i]}"); fi
    done
    trap - EXIT INT TERM
    (( ${#FAILED[@]} > 0 )) && { echo " ${#FAILED[@]} failed: ${FAILED[*]}"; exit 1; }
else
    # ── Sequential ────────────────────────────────────────────────
    for subdir in "${SUBDIRS[@]}"; do
        run_subdir "$subdir"
    done
fi

ELAPSED=$(($(date +%s) - SUITE_START))
printf '\n%s\n AML Diagnosis complete — elapsed=%02dh:%02dm:%02ds\n%s\n' \
    "═══════════════════════════════════════════════════════════════" \
    $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) \
    "═══════════════════════════════════════════════════════════════"
