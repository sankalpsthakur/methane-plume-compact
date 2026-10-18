#!/bin/bash
# G2 timing gate (Amendment 4 item 3, in its merged G2+G3 form, Amendment 5 note 1). Timing only, never metrics; read-only.
#   scripts/g2_timing_gate.sh <cfg_id>      e.g. d1
# Takes the measured wall seconds of the chosen config's G1 run (RUN_STATUS.json: same steps, same two-worker layout, includes init and one validation pass), turns them into the cost of one G2 seed job
# (+5 s: the 75 s validation pass becomes the 80 s test pass), scales the planned control job by the same measured slowdown (never below its plan), schedules the queue [5 seeds, control] on two workers
# in queue order, adds 300 s kernel overhead + 300 s edge stage, and compares with the soft deadline of kernels/gpu-compact/cfg.json. Exit 0 = push may proceed; 1 = do NOT push G2, ask the owner.
# Test hooks: RUN_STATUS, BUDGET, KCFG.
set -u; cd "$(dirname "$0")/.." || exit 1
CFG=${1:?usage: g2_timing_gate.sh <cfg_id>}; RS=${RUN_STATUS:-kernel_outputs/gpu-wide/RUN_STATUS.json}; BJ=${BUDGET:-protocol/gpu_budget.json}; KC=${KCFG:-kernels/gpu-compact/cfg.json}
S=$(jq -r --arg n "g_$CFG" '.runs[$n].secs // empty' "$RS" 2>/dev/null); [ -n "$S" ] || { echo "FAIL: no measured run g_$CFG in $RS"; exit 1; }
D=$(jq -r '.deadline_s // empty' "$KC"); WITHC=$(jq -r '((.runs | map(.name) | index("ctrl_mnv2")) != null)' "$KC")
SJ=$(jq -r --arg c "$CFG" '.merged.per_candidate[$c].seed_job_s // empty' "$BJ"); CJ=$(jq -r --arg c "$CFG" '.merged.per_candidate[$c].control_job_s // empty' "$BJ")
[ -n "$D" ] && [ -n "$SJ" ] && [ -n "$CJ" ] || { echo "FAIL: deadline_s in $KC or per-candidate planned jobs for $CFG in $BJ missing (regenerate G2 with --with-control and keep the plan JSON)"; exit 1; }
awk -v s="$S" -v sj="$SJ" -v cj="$CJ" -v d="$D" -v withc="$WITHC" 'BEGIN{
  seed = s + 5; k = seed / sj; if (k < 1) k = 1; ctrl = cj * k; a = 0; b = 0
  for (i = 0; i < 5; i++) { if (a <= b) a += seed; else b += seed }
  if (withc == "true") { if (a <= b) a += ctrl; else b += ctrl }
  t = 300 + 300 + (a > b ? a : b)
  printf "G2%s projected wall %d s (seed job %d s, control job %d s, slowdown vs plan x%.2f) against the soft deadline %d s: %s\n", (withc == "true" ? "+G3" : ""), t, seed, (withc == "true" ? ctrl : 0), k, d, (t <= d ? "PASS" : "FAIL, do NOT push G2, ask the owner")
  exit !(t <= d) }'
