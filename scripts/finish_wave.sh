#!/bin/bash
# usage: finish_wave.sh <owner/slug> <outdir>
# Use when a kernel has ended (COMPLETE/ERROR) and you do not want to wait for the 5-min watcher: stops that kernel's watcher (so it cannot remove somebody else's lock later),
# fetches the SMALL outputs, and removes the CPU slot lock whose owner file names this kernel.
REF=$1; OUT=$2; LOCKS=${LOCKS_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/locks}; K=~/.local/bin/kaggle
S=$($K kernels status "$REF" 2>&1 | tail -1); echo "$S"
case "$S" in *COMPLETE*|*ERROR*|*CANCEL*) ;; *) echo "kernel still running; nothing done"; exit 1;; esac
pkill -f "watch_kernel.sh $REF " ; pkill -f "run_wave.sh .* $REF "
mkdir -p "$OUT"; $K kernels output "$REF" -p "$OUT" --file-pattern '\.(log|json|jsonl|csv|txt)$' -q 2>&1 | tail -3
for d in $LOCKS/kaggle-cpu-*/; do
  if grep -q "$REF " "$d/owner" 2>/dev/null; then rm -rf "$d" && echo "released $(basename $d)"; fi
done
find "$OUT" -type f | wc -l
