#!/usr/bin/env bash
# Kill every batch/agent process this user has running on the current host.
# Run on each machine (saturn, mars, neptune, etc.) where workers may have leaked.

set +e
USER_NAME="${USER:-$(whoami)}"

echo "═══════════════════════════════════════════════════════════"
echo " Cleanup on $(hostname) for user ${USER_NAME}"
echo "═══════════════════════════════════════════════════════════"

echo ""
echo "Before kill:"
ps -u "${USER_NAME}" -o pid,ppid,cmd | grep -E "run_single_slide|run_batch_aml|run_batch_aml_suite|python.*Pathology_agent|python.*evaluate/run" | grep -v grep | tee /tmp/_pre_kill.txt
echo "Total: $(wc -l < /tmp/_pre_kill.txt)"

echo ""
echo "Killing..."
pkill -9 -u "${USER_NAME}" -f run_single_slide.py 2>/dev/null
pkill -9 -u "${USER_NAME}" -f run_batch_aml_suite.sh 2>/dev/null
pkill -9 -u "${USER_NAME}" -f run_batch_aml.sh 2>/dev/null
pkill -9 -u "${USER_NAME}" -f "python.*evaluate/run_single_slide" 2>/dev/null
pkill -9 -u "${USER_NAME}" -f "Pathology_agent.*evaluate" 2>/dev/null

sleep 3

echo ""
echo "After kill:"
ps -u "${USER_NAME}" -o pid,ppid,cmd | grep -E "run_single_slide|run_batch_aml|python.*Pathology_agent|python.*evaluate/run" | grep -v grep | tee /tmp/_post_kill.txt
remaining=$(wc -l < /tmp/_post_kill.txt)
echo "Total remaining: ${remaining}"

if [[ "${remaining}" -gt 0 ]]; then
    echo ""
    echo "Some survived. Try TERM signal on them:"
    awk '{print $1}' /tmp/_post_kill.txt | xargs -r kill -9 2>/dev/null
    sleep 2
    echo "Final count: $(ps -u "${USER_NAME}" -o pid,cmd | grep -E "run_single_slide|run_batch_aml|python.*Pathology_agent" | grep -v grep | wc -l)"
fi

rm -f /tmp/_pre_kill.txt /tmp/_post_kill.txt
echo ""
echo " Done."
