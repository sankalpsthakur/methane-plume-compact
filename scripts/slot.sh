#!/bin/bash
# Kaggle CPU slot helper (PLAN.md): slot.sh take "<kernel ref>"  -> prints the slot number taken (waits until one is free); owner file = "gas <kernel ref> <UTC start>"
#                                    slot.sh release <N>
LOCKS=${LOCKS_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/locks}
case "$1" in
  take)
    while true; do
      for n in 1 2 3 4; do
        if mkdir "$LOCKS/kaggle-cpu-$n" 2>/dev/null; then
          printf 'gas %s %s\n' "$2" "$(date -u +%Y-%m-%dT%H:%MZ)" > "$LOCKS/kaggle-cpu-$n/owner"; echo "$n"; exit 0
        fi
      done
      sleep 20
    done;;
  release) rm -rf "$LOCKS/kaggle-cpu-$2" && echo "released $2";;
esac
