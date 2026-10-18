#!/bin/bash
# After `bench-gas-final` is COMPLETE (the watcher already fetched log/json/jsonl/csv/txt into kernel_outputs/final):
#   1. fetch the small npz summaries (about 20 MB) and the ONNX of seed 0 (second fetch, under 50 MB, never the cache)
#   2. tables (finalize.py), TEST-LOG rows (testlog_rows.py), M4 latency (only when the 1-min load is under 4)
# If the reference kernel `bench-gas-final-ref` has also finished, put its small outputs in kernel_outputs/final-ref (REF=1) and its released-model rows are added.
# usage: scripts/after_final.sh [prefix, default f_d1_s]   (from the gas repo root; local python steps go through ../bin/guard.sh, which needs >= 30 GiB free disk)
set -u; cd "$(dirname "$0")/.." || exit 1
PREFIX=${1:-f_d1_s}; D=kernel_outputs/final; R=kernel_outputs/final-ref; G=../bin/guard.sh; K=~/.local/bin/kaggle
df -h / | tail -1
$K kernels output sankalpsthakur/bench-gas-final -p $D --file-pattern '\.npz$' -q 2>&1 | tail -2
$K kernels output sankalpsthakur/bench-gas-final -p $D --file-pattern "edge/${PREFIX}0_(fp32|int8)\.onnx$" -q 2>&1 | tail -2
if [ -d $R ]; then
  $K kernels output sankalpsthakur/bench-gas-final-ref -p $R --file-pattern '\.npz$' -q 2>&1 | tail -2
  $K kernels output sankalpsthakur/bench-gas-final-ref -p $R --file-pattern 'edge/released_mag1c_rgb_(fp32|int8)\.onnx$' -q 2>&1 | tail -2
fi
ls -la $D $D/edge 2>/dev/null | head -50
$G -t 60 -m 512 -- python3 scripts/finalize.py $D --prefix $PREFIX --out results/final
if [ -d $R ]; then $G -t 30 -m 256 -- python3 scripts/testlog_rows.py $D --ref-dir $R > results/testlog_rows.md; else $G -t 30 -m 256 -- python3 scripts/testlog_rows.py $D > results/testlog_rows.md; fi
cat results/testlog_rows.md
L=$(uptime | sed 's/.*load averages*: //' | awk '{print $1}' | tr -d ,)
echo "1-min load: $L"
if awk "BEGIN{exit !($L < 4.0)}"; then
  for f in $D/edge/${PREFIX}0_fp32.onnx $D/edge/${PREFIX}0_int8.onnx $R/edge/released_mag1c_rgb_fp32.onnx $R/edge/released_mag1c_rgb_int8.onnx; do [ -f "$f" ] && $G -t 60 -m 512 -- python3 scripts/latency_m4.py "$f"; done
else echo "latency deferred (machine busy)"; fi
