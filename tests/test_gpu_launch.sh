#!/bin/bash
# Tests of scripts/gpu_launch.sh with a fake kaggle (quota + push), a fake waiter, a temp lock dir / ledger / RESOURCES file. Nothing is pushed to Kaggle.
set -u; cd "$(dirname "$0")/.." || exit 1
T=$(mktemp -d); trap 'rm -rf "$T" kernel_outputs/gpu-wide.used_before protocol/.test_dirty' EXIT
cat > "$T/kaggle" <<'EOK'
#!/bin/bash
if [ "$1" = quota ]; then printf 'resource  used   remaining  total   refreshAt\n--------  -----  ---------  ------  -------------------\nGPU       %sh  %sh  30.00h  2026-10-17T00:00:00\nTPU       0.00h  20.00h  20.00h  x\n' "${FAKE_USED:-0.63}" "${FAKE_REM:-29.37}"; exit 0; fi
if [ "$1" = kernels ] && [ "$2" = push ]; then echo "PUSH $*" >> "$FAKE_LOG"; n=$(grep -c PUSH "$FAKE_LOG")   # like the real CLI: errors are printed, the exit status stays 0
  [ "${FAKE_PUSH_MODE:-ok}" = other ] && { echo "Kernel push error: boom"; exit 0; }
  [ "${FAKE_PUSH_MODE:-ok}" = session ] && [ "$n" -le "${FAKE_SESSION_FAILS:-2}" ] && { echo "Kernel push error: Maximum batch GPU session count of 2 reached."; exit 0; }
  echo "Kernel version 1 successfully pushed."; exit 0; fi
exit 0
EOK
printf '#!/bin/bash\necho "WATCH $*" >> "$FAKE_LOG"\n' > "$T/watcher"; chmod +x "$T/kaggle" "$T/watcher"
export MIN_FREE_S=0 KAGGLE_BIN="$T/kaggle" LOCKS_DIR="$T/locks" LEDGER="$T/ledger.csv" RESOURCES="$T/res.md" WATCHER="$T/watcher" GATE_NOW=2026-10-10T01:00:00Z NO_GIT_CHECK=1 POLL_S=1 FAKE_LOG="$T/log"
: > "$T/log"; : > "$T/res.md"; mkdir -p "$T/locks/kaggle-gpu"; echo "small-lm test hold" > "$T/locks/kaggle-gpu/owner"
# A: the lock is held, then released after 3 s -> the launcher waits, then takes the lock, pushes with the WALL cap, starts one waiter
( sleep 3; rm -rf "$T/locks/kaggle-gpu" ) & out=$(scripts/gpu_launch.sh kernels/gpu-wide --max-wait-min 1 2>&1); rc=$?; wait
[ $rc -eq 0 ] && echo "$out" | grep -q LAUNCHED || { echo "A: $out"; exit 1; }
grep -q "PUSH kernels push -p kernels/gpu-wide -t 3900" "$T/log" || { echo "A: push line wrong:"; cat "$T/log"; exit 1; }
sleep 1; grep -q "WATCH sankalpsthakur/bench-gas-gpu-wide gpu kernel_outputs/gpu-wide 90" "$T/log" || { echo "A: waiter line wrong:"; cat "$T/log"; exit 1; }
grep -q "gas sankalpsthakur/bench-gas-gpu-wide" "$T/locks/kaggle-gpu/owner" && grep -q "billed cap 7800 s" "$T/locks/kaggle-gpu/owner" || { echo "A: lock owner wrong"; cat "$T/locks/kaggle-gpu/owner"; exit 1; }
grep -q "running, lock locks/kaggle-gpu held by gas" "$T/res.md" && [ "$(cat kernel_outputs/gpu-wide.used_before)" = "0.63" ] || { echo "A: ledger/resources/used_before wrong"; exit 1; }
[ "$(grep -c PUSH "$T/log")" = 1 ] || { echo "A: pushed more than once"; exit 1; }
# B: a held lock is NOT overridden by a second launch while the first holds it, and another gate failure stops without pushing (quota too low)
rm -rf "$T/locks/kaggle-gpu"; : > "$T/log"; out=$(FAKE_REM=1.00 scripts/gpu_launch.sh kernels/gpu-wide --max-wait-min 1 2>&1); rc=$?
[ $rc -eq 1 ] && echo "$out" | grep -q "GATE FAILED" && [ ! -s "$T/log" ] && [ ! -e "$T/locks/kaggle-gpu" ] || { echo "B: rc=$rc out=$out"; exit 1; }
# C: a failing push releases the lock and logs it
: > "$T/log"; out=$(FAKE_PUSH_MODE=other scripts/gpu_launch.sh kernels/gpu-control --max-wait-min 1 2>&1); rc=$?
[ $rc -eq 1 ] && echo "$out" | grep -q "PUSH FAILED" && [ ! -e "$T/locks/kaggle-gpu" ] && grep -q "PUSH FAILED" "$T/res.md" || { echo "C: rc=$rc out=$out"; exit 1; }
# D: uncommitted change under protocol/ -> refused before anything else
echo x > protocol/.test_dirty; out=$(NO_GIT_CHECK= scripts/gpu_launch.sh kernels/gpu-wide 2>&1); rc=$?; rm -f protocol/.test_dirty
[ $rc -eq 1 ] && echo "$out" | grep -q "REFUSED: uncommitted" || { echo "D: rc=$rc out=$out"; exit 1; }
# E: gave up waiting
mkdir -p "$T/locks/kaggle-gpu"; out=$(POLL_S=1 scripts/gpu_launch.sh kernels/gpu-wide --max-wait-min 0 2>&1); rc=$?; [ $rc -eq 2 ] && echo "$out" | grep -q "gave up" || { echo "E: rc=$rc out=$out"; exit 1; }
# F: the account's GPU session limit (the owner's own kernels count): the CLI prints an error and exits 0 -> lock released each time, launcher waits and retries, then launches once
rm -rf "$T/locks/kaggle-gpu"; : > "$T/log"; : > "$T/res.md"; out=$(FAKE_PUSH_MODE=session FAKE_SESSION_FAILS=2 SESSION_POLL_S=1 scripts/gpu_launch.sh kernels/gpu-wide --max-wait-min 1 2>&1); rc=$?
[ $rc -eq 0 ] && echo "$out" | grep -q LAUNCHED && [ "$(grep -c PUSH "$T/log")" = 3 ] && [ "$(grep -c 'refused by Kaggle' "$T/res.md")" = 1 ] && [ "$(grep -c 'running, lock locks/kaggle-gpu held by gas' "$T/res.md")" = 1 ] && [ -e "$T/locks/kaggle-gpu" ] || { echo "F: rc=$rc out=$out"; cat "$T/log" "$T/res.md"; exit 1; }
# G: a lock that flaps (free for 1 s, then held again) never lets the launcher through when MIN_FREE_S is 4 s; a lock that stays free does
rm -rf "$T/locks/kaggle-gpu"; : > "$T/log"; ( for i in 1 2 3 4 5; do sleep 2; mkdir "$T/locks/kaggle-gpu" 2>/dev/null; sleep 0.2; rmdir "$T/locks/kaggle-gpu"; done ) & FL=$!
out=$(MIN_FREE_S=4 POLL_S=1 scripts/gpu_launch.sh kernels/gpu-wide --max-wait-min 0 2>&1); rc=$?; wait $FL
[ ! -s "$T/log" ] || { echo "G: pushed while the lock was flapping"; cat "$T/log"; exit 1; }
: > "$T/log"; rm -rf "$T/locks/kaggle-gpu"; t0=$SECONDS; out=$(MIN_FREE_S=3 POLL_S=1 scripts/gpu_launch.sh kernels/gpu-wide --max-wait-min 1 2>&1); rc=$?
[ $rc -eq 0 ] && [ $((SECONDS - t0)) -ge 3 ] && echo "$out" | grep -q LAUNCHED || { echo "G2: rc=$rc out=$out"; exit 1; }
echo "gpu_launch: 7 scenarios ok"
