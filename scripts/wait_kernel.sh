#!/bin/bash
# usage: wait_kernel.sh <owner/slug> [max_minutes]; exits when the kernel reaches a terminal state; prints status
K=$1; MAXM=${2:-120}; END=$((SECONDS + MAXM*60))
while [ $SECONDS -lt $END ]; do
  S=$(~/.local/bin/kaggle kernels status "$K" 2>&1 | tail -1)
  case "$S" in *COMPLETE*|*ERROR*|*CANCEL*) echo "$K :: $S"; exit 0;; esac
  sleep 20
done
echo "$K :: TIMEOUT waiting ($S)"
