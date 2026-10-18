#!/bin/bash
# Tests of scripts/gpu_gate.sh with a fake `kaggle quota`, a temp lock dir and a temp ledger (no network, no Kaggle call).
set -u; cd "$(dirname "$0")/.." || exit 1
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
cat > "$T/kaggle" <<'EOK'
#!/bin/bash
printf 'resource  used   remaining  total   refreshAt\n--------  -----  ---------  ------  -------------------\nGPU       %sh  %sh  30.00h  2026-10-17T00:00:00\nTPU       0.00h  20.00h  20.00h  2026-10-17T00:00:00\n' "${FAKE_USED:-0.00}" "${FAKE_REM:-30.00}"
EOK
chmod +x "$T/kaggle"; mkdir -p "$T/locks"
export KAGGLE_BIN="$T/kaggle" LOCKS_DIR="$T/locks" LEDGER="$T/ledger.csv" GATE_NOW=2026-10-10T01:00:00Z
ok=0; run() { out=$(scripts/gpu_gate.sh "$@" 2>&1); rc=$?; }
expect() { if [ "$rc" -eq "$1" ]; then ok=$((ok + 1)); else echo "FAILED TEST: $2 (rc $rc, wanted $1)"; echo "$out"; exit 1; fi; }
run kernels/gpu-wide;                                              expect 0 "all gates pass";                         echo "$out" | grep -q -- "-t 3900" || { echo "wall cap missing"; exit 1; }
mkdir "$T/locks/kaggle-gpu"; run kernels/gpu-wide;                  expect 1 "held lock refuses"; rmdir "$T/locks/kaggle-gpu"
FAKE_REM=1.00 run kernels/gpu-wide;                                 expect 1 "remaining quota below the billed cap refuses"
echo "utc,stage,ref,used_before_h,used_after_h,account_delta_h,billed_h,source" > "$T/ledger.csv"; echo "t,compact,x,0,5.5,5.50,5.50,quota delta" >> "$T/ledger.csv"
run kernels/gpu-wide;                                              expect 1 "ledger 5.5 h + 2.17 h exceeds 7.0 h"
run kernels/gpu-wide --retry;                                      expect 0 "the same request fits under 8.0 h with --retry"
GATE_NOW=2026-10-09T23:59:00Z run kernels/gpu-control;              expect 1 "before the quota refresh refuses"
mkdir -p "$T/k"; cp kernels/gpu-wide/kernel-metadata.json "$T/k/"; jq 'del(.billing_b)' kernels/gpu-wide/cfg.json > "$T/k/cfg.json"; run "$T/k"; expect 1 "cfg without billing_b refuses"
jq '.billing_b=4' kernels/gpu-wide/cfg.json > "$T/k/cfg.json"; run "$T/k"; expect 1 "b x wall above the billed cap refuses"
rm -f "$T/ledger.csv"; scripts/gpu_gate.sh record wide sankalpsthakur/bench-gas-gpu-wide 0.63 --after 2.70 > /dev/null; grep -q ",wide,sankalpsthakur/bench-gas-gpu-wide,0.63,2.70,2.07,2.07,quota delta" "$T/ledger.csv" || { echo "record row wrong"; cat "$T/ledger.csv"; exit 1; }
scripts/gpu_gate.sh record wide2 sankalpsthakur/bench-gas-gpu-wide 0.63 --after 2.70 --gpus 2 --wall-s 1800 > /dev/null; grep -q ",wide2,.*,0.63,2.70,2.07,1.00,gpus x wall (2 x 1800 s)" "$T/ledger.csv" || { echo "wall x gpus row wrong"; cat "$T/ledger.csv"; exit 1; }
printf '[{"stream_name":"stdout","time":1.5,"data":"x"}\n,{"stream_name":"stdout","time":3000.5,"data":"y"}\n]' > "$T/k.log"; scripts/gpu_gate.sh record wide3 sankalpsthakur/bench-gas-gpu-wide 0.63 --after 2.70 --gpus 2 --log "$T/k.log" > /dev/null; grep -q ",wide3,.*,0.63,2.70,2.07,1.68,gpus x wall (2 x 3030 s)" "$T/ledger.csv" || { echo "log-derived row wrong"; cat "$T/ledger.csv"; exit 1; }
rm -f "$T/ledger.csv"; for k in probe wide compact control; do run kernels/gpu-$k; expect 0 "gpu-$k passes the gate on a fresh account"; done
echo "gpu_gate: $((ok + 4)) checks ok"
