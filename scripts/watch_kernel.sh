#!/bin/bash
# usage: watch_kernel.sh <owner/slug> <cpu-slot-number|gpu|-> <outdir> [max_minutes]   (gpu: removes locks/kaggle-gpu at the end and prints `kaggle quota`; `-` releases nothing)
# Every 5 min: `kaggle kernels status`. At a terminal state: fetch the SMALL outputs only (log/json/jsonl/csv/txt, never .npy/.onnx/.npz/.pt),
# remove the CPU slot lock (if a slot number was given), print the final status line and exit (the caller is notified by the exit).
K=$1; SLOT=$2; OUT=$3; MAXM=${4:-300}; END=$((SECONDS + MAXM*60)); S=""; LOCKS=${LOCKS_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/locks}
while [ $SECONDS -lt $END ]; do
  S=$(~/.local/bin/kaggle kernels status "$K" 2>&1 | tail -1)
  case "$S" in *COMPLETE*|*ERROR*|*CANCEL*)
    mkdir -p "$OUT"; ~/.local/bin/kaggle kernels output "$K" -p "$OUT" --file-pattern '\.(log|json|jsonl|csv|txt)$' -q 2>&1 | tail -3
    if [ "$SLOT" = "gpu" ]; then rm -rf "$LOCKS/kaggle-gpu" && echo "released locks/kaggle-gpu"; ~/.local/bin/kaggle quota 2>&1 | tail -3
    elif [ "$SLOT" != "-" ]; then rm -rf "$LOCKS/kaggle-cpu-$SLOT" && echo "released slot $SLOT"; fi
    echo "$(date -u +%H:%MZ) $K :: $S"; exit 0;;
  esac
  sleep 300
done
echo "TIMEOUT watching $K (last: $S)"
