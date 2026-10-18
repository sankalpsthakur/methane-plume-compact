#!/bin/bash
# Launch one GPU kernel of the gas wave: wait for locks/kaggle-gpu, run the Amendment 5 gate, take the lock, push with -t (wall seconds), start ONE background waiter, log RESOURCES.md.
#   scripts/gpu_launch.sh <kernel_dir> [--retry] [--max-wait-min N]
# It pushes only if (1) `git status` is clean under <kernel_dir>, protocol/ and scripts/ (the amendment text and the plan are committed BEFORE the push: pre-registration order is enforced by git),
# (2) scripts/gpu_gate.sh passes, (3) `mkdir locks/kaggle-gpu` succeeds (atomic; if another track wins the race it waits again). Any gate failure other than "the lock is held" stops it without pushing.
# Afterwards: scripts/gpu_gate.sh record <stage> <ref> <used_before>  (the used_before value is also in kernel_outputs/<stage>.used_before).
# The lock must have been continuously free for MIN_FREE_S seconds (default 180) before it is taken, so a runner that releases and re-takes it between push attempts keeps its priority.
# Test hooks: KAGGLE_BIN, LOCKS_DIR, RESOURCES, WATCHER, GATE_NOW, NO_GIT_CHECK=1, MIN_FREE_S, POLL_S, SESSION_POLL_S.
set -u; cd "$(dirname "$0")/.." || exit 1
KD=${1:?usage: gpu_launch.sh <kernel_dir> [--retry] [--max-wait-min N]}; shift; RETRY=""; MAXW=1440
while [ $# -gt 0 ]; do case "$1" in --retry) RETRY=--retry;; --max-wait-min) MAXW=$2; shift;; esac; shift; done
K=${KAGGLE_BIN:-$HOME/.local/bin/kaggle}; LOCKS=${LOCKS_DIR:-../locks}; RES=${RESOURCES:-../RESOURCES.md}; WATCHER=${WATCHER:-scripts/watch_kernel.sh}
if [ -z "${NO_GIT_CHECK:-}" ]; then dirty=$(git status --porcelain -- "$KD" protocol scripts | head -5); [ -z "$dirty" ] || { echo "REFUSED: uncommitted changes (commit the amendment / plan / kernel first):"; echo "$dirty"; exit 1; }; fi
END=$((SECONDS + MAXW * 60)); WAITED=0
FREE_SINCE=""
while true; do
  [ $SECONDS -lt $END ] || { echo "gave up waiting for locks/kaggle-gpu after $MAXW minutes; nothing pushed"; exit 2; }
  out=$(scripts/gpu_gate.sh "$KD" $RETRY 2>&1); rc=$?
  if [ $rc -eq 0 ]; then
    # take the lock only after it has been continuously free for MIN_FREE_S seconds: another track's runner that releases and re-takes the lock between push attempts (small-lm, which has priority) is not preempted
    [ -n "$FREE_SINCE" ] || FREE_SINCE=$SECONDS
    if [ $((SECONDS - FREE_SINCE)) -lt ${MIN_FREE_S:-180} ]; then sleep ${POLL_S:-15}; continue; fi
    REF=$(jq -r .id "$KD/kernel-metadata.json"); NAME=${REF#*/bench-gas-}; WALL=$(jq -r .wall_cap_s "$KD/cfg.json"); CAP=$(jq -r .billed_cap_s "$KD/cfg.json"); SHAPE=$(jq -r .machine_shape "$KD/kernel-metadata.json")
    if mkdir "$LOCKS/kaggle-gpu" 2>/dev/null; then
      T0=$(date -u +%FT%TZ); T1=$(date -u -v+$((WALL + 300))S +%FT%TZ 2>/dev/null || echo "+$((WALL + 300))s"); BEFORE=$($K quota 2>&1 | awk '/^GPU/{v=$2; gsub("h","",v); print v}')
      printf 'gas %s %s expected end %s (GPU; -t %s s wall, billed cap %s s, shape %s; quota used before: %s h)\n' "$REF" "$T0" "$T1" "$WALL" "$CAP" "$SHAPE" "$BEFORE" > "$LOCKS/kaggle-gpu/owner"
      mkdir -p kernel_outputs; echo "$BEFORE" > "kernel_outputs/$NAME.used_before"
      pout=$($K kernels push -p "$KD" -t "$WALL" 2>&1); echo "$pout"
      if echo "$pout" | grep -q "successfully pushed"; then
        printf '| %s | gas | Kaggle GPU session `%s` (private, machine_shape %s, push -t %s s wall = billed cap %s s = %.2f billed h at b=%s; Amendment 5 gate passed; one background waiter; nothing downloaded except small json/log) | up to %.2f billed GPU-h, GPU used before %s h | running, lock locks/kaggle-gpu held by gas |\n' "$(date -u '+%F %H:%M')" "$REF" "$SHAPE" "$WALL" "$CAP" "$(awk -v c=$CAP 'BEGIN{print c/3600}')" "$(jq -r .billing_b "$KD/cfg.json")" "$(awk -v c=$CAP 'BEGIN{print c/3600}')" "$BEFORE" >> "$RES"
        mkdir -p "kernel_outputs/$NAME"; nohup "$WATCHER" "$REF" gpu "kernel_outputs/$NAME" $((WALL / 60 + 25)) > "/tmp/watch-gas-$NAME.log" 2>&1 &
        echo "LAUNCHED $REF at $T0 (used before $BEFORE h; waiter log /tmp/watch-gas-$NAME.log; record with: scripts/gpu_gate.sh record $NAME $REF $BEFORE --gpus $(jq -r .billing_b "$KD/cfg.json") --log kernel_outputs/$NAME/bench-gas-$NAME.log)"; exit 0
      fi
      rm -rf "$LOCKS/kaggle-gpu"; rm -f "kernel_outputs/$NAME.used_before"
      if echo "$pout" | grep -q "Maximum batch"; then   # the account allows 2 concurrent GPU sessions (the owner'"'"'s own kernels count): release the lock, wait, try again
        [ -n "${REFUSED_LOGGED:-}" ] || printf '| %s | gas | push of `%s` refused by Kaggle (%s); lock released, no quota used; the launcher keeps waiting and retries every %s s | 0 | waiting |\n' "$(date -u '+%F %H:%M')" "$REF" "$(echo "$pout" | tail -1 | cut -c1-90)" "${SESSION_POLL_S:-60}" >> "$RES"; REFUSED_LOGGED=1
        echo "push refused (session limit): lock released, retrying"; sleep ${SESSION_POLL_S:-60}
      else
        printf '| %s | gas | PUSH FAILED for `%s` (%s); lock released, no quota used | 0 | stopped |\n' "$(date -u '+%F %H:%M')" "$REF" "$(echo "$pout" | tail -1 | cut -c1-120)" >> "$RES"; echo "PUSH FAILED; lock released"; exit 1
      fi
    else
      echo "lost the race for locks/kaggle-gpu, waiting again"
    fi
  else
    FREE_SINCE=""
    fails=$(echo "$out" | grep -c '^FAIL'); held=$(echo "$out" | grep -c 'kaggle-gpu is held')
    if [ "$fails" -ne "$held" ] || [ "$held" -ne 1 ]; then echo "GATE FAILED (not only the lock); nothing pushed:"; echo "$out"; exit 1; fi
  fi
  sleep ${POLL_S:-15}
done
