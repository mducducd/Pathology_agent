#!/usr/bin/env bash
# Launch one tmux pane per VLM. Each pane runs the suite filtered to
# that single model. Ctrl-C in a pane kills only that pane.
#
# Attach: tmux attach -t aml
# Detach (without killing): Ctrl-b d
# Switch panes:              Ctrl-b o   (or arrow keys: Ctrl-b ←/→)
# Kill the whole session:    tmux kill-session -t aml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SESSION="aml"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-exp_rerun_clean}"

MODELS=(
    "GLM-4.6V-FP8"
    "Qwen3.5-397B-A17B-FP8"
    "gemma-4-31B-it-h200"
)

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[ERROR] tmux session '$SESSION' already exists."
    echo "  Attach:  tmux attach -t $SESSION"
    echo "  Or kill: tmux kill-session -t $SESSION"
    exit 1
fi

# Create session with first model
first="${MODELS[0]}"
tmux new-session -d -s "$SESSION" -n "$first" \
    "cd '$REPO_ROOT' && bash evaluate/run_batch_aml_suite.sh \
        --models '$first' --experiment-name '$EXPERIMENT_NAME' \
        --resume --use-tile-cache; \
     echo; echo '[done] press any key to close'; read -n1"

# Add a window per remaining model
for m in "${MODELS[@]:1}"; do
    tmux new-window -t "$SESSION" -n "$m" \
        "cd '$REPO_ROOT' && bash evaluate/run_batch_aml_suite.sh \
            --models '$m' --experiment-name '$EXPERIMENT_NAME' \
            --resume --use-tile-cache; \
         echo; echo '[done] press any key to close'; read -n1"
done

echo "═══════════════════════════════════════════════════════════════"
echo " tmux session '$SESSION' started with ${#MODELS[@]} windows."
echo " One window per VLM, each running independently."
echo ""
echo "  Attach:        tmux attach -t $SESSION"
echo "  Switch window: Ctrl-b n (next) / Ctrl-b p (prev) / Ctrl-b <num>"
echo "  Detach:        Ctrl-b d  (leaves everything running)"
echo "  Kill all:      tmux kill-session -t $SESSION"
echo "═══════════════════════════════════════════════════════════════"
