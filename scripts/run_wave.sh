#!/bin/bash
# usage: run_wave.sh <kernel_dir> <owner/slug> <outdir> [max_watch_minutes]
# Takes a free Kaggle CPU slot (waits if all four are taken), pushes the kernel (retrying while the account is at its session cap),
# then watches it (5-min status checks), fetches the SMALL outputs when it ends and releases the slot. Run it as ONE background command.
cd "$(dirname "$0")/.." || exit 1
KD=$1; REF=$2; OUT=$3; MAXM=${4:-300}
N=$(scripts/slot.sh take "$REF"); echo "$(date -u +%H:%MZ) slot $N taken for $REF"
if scripts/push_when_free.sh "$KD" 30; then
  scripts/watch_kernel.sh "$REF" "$N" "$OUT" "$MAXM"
else
  rm -rf "${LOCKS_DIR:-../locks}/kaggle-cpu-$N"; echo "$(date -u +%H:%MZ) push failed; slot $N released"; exit 1
fi
