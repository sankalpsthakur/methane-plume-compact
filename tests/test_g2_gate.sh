#!/bin/bash
set -u; cd "$(dirname "$0")/.." || exit 1
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
echo '{"merged":{"per_candidate":{"d1":{"seed_job_s":1338,"control_job_s":2776}}}}' > "$T/budget.json"
echo '{"deadline_s":6900,"runs":[{"name":"gf_d1_s0"},{"name":"ctrl_mnv2"}]}' > "$T/kcfg.json"
echo '{"deadline_s":6900,"runs":[{"name":"gf_d1_s0"}]}' > "$T/kcfg_nc.json"
export BUDGET="$T/budget.json" KCFG="$T/kcfg.json" RUN_STATUS="$T/rs.json"
echo '{"runs":{"g_d1":{"secs":1333}}}' > "$T/rs.json"; out=$(scripts/g2_timing_gate.sh d1); rc=$?; [ $rc -eq 0 ] && echo "$out" | grep -q "projected wall 6052 s" || { echo "plan-equal case wrong: $out"; exit 1; }
echo '{"runs":{"g_d1":{"secs":2000}}}' > "$T/rs.json"; out=$(scripts/g2_timing_gate.sh d1); rc=$?; [ $rc -eq 1 ] && echo "$out" | grep -q "projected wall 8769 s" || { echo "slow case wrong: $out"; exit 1; }
echo '{"runs":{"g_d1":{"secs":900}}}' > "$T/rs.json"; out=$(scripts/g2_timing_gate.sh d1); rc=$?; [ $rc -eq 0 ] || { echo "fast case wrong: $out"; exit 1; }
KCFG="$T/kcfg_nc.json" out=$(scripts/g2_timing_gate.sh d1); rc=$?; [ $rc -eq 0 ] && echo "$out" | grep -q "G2 projected" || { echo "no-control case wrong: $out"; exit 1; }
out=$(scripts/g2_timing_gate.sh d9); rc=$?; [ $rc -eq 1 ] || { echo "missing run must fail: $out"; exit 1; }
echo "g2_timing_gate: 5 checks ok"
