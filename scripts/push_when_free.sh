#!/bin/bash
# usage: push_when_free.sh <kernel_dir> [max_minutes]: retries `kaggle kernels push` while the account is at the 5-session CPU cap
D=$1; MAXM=${2:-30}; END=$((SECONDS + MAXM*60))
while [ $SECONDS -lt $END ]; do
  OUT=$(~/.local/bin/kaggle kernels push -p "$D" 2>&1 | tail -1)
  case "$OUT" in *"successfully pushed"*) echo "$(date -u +%H:%M) pushed $D :: $OUT"; exit 0;; esac
  case "$OUT" in *"Maximum batch"*) ;; *) echo "$(date -u +%H:%M) unexpected: $OUT"; exit 1;; esac
  sleep 25
done
echo "gave up $D"; exit 2
