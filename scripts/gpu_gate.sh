#!/bin/bash
# Amendment 5 push gate (read-only: it pushes nothing and creates no lock). Budgets are in BILLED GPU-hours (GPUs held x wall; the unit of `kaggle quota`).
#   scripts/gpu_gate.sh <kernel_dir> [--retry]      checks, then prints the exact lock / push (-t wall seconds) / watch commands
#   scripts/gpu_gate.sh record <stage> <kaggle_ref> <quota_used_before_h> [--after H] [--gpus N --log FILE | --gpus N --wall-s S]    appends a finished kernel to results/gpu_ledger.csv
#       billed_h = quota delta (after - before) by default; with --gpus and --log/--wall-s it is GPUs x wall hours instead (the account counter also bills the owner's own concurrent GPU kernels, so a raw delta can overcount;
#       --log takes the fetched Kaggle log: wall = its last "time" stamp + 30 s teardown). Both numbers are kept in the row; the gate sums billed_h.
# Gate: (1) after the quota refresh 2026-10-10T00:00Z; (2) metadata is_private, enable_gpu, machine_shape set; (3) cfg has billed_cap_s, wall_cap_s, billing_b and billing_b x wall_cap_s <= billed_cap_s;
# (4) no locks/kaggle-gpu; (5) `kaggle quota` remaining >= the billed cap; (6) gas ledger + billed cap <= 7.0 h (the 8 h gas cap minus the 1 h retry reserve), or <= 8.0 h with --retry
# (a retry needs an infrastructure cause announced first in STATUS.md, Amendment 5 item 7; the flag only opens the reserve, it does not grant the retry).
# Test hooks: KAGGLE_BIN, LOCKS_DIR, LEDGER, GATE_NOW.
set -u; cd "$(dirname "$0")/.." || exit 1
K=${KAGGLE_BIN:-$HOME/.local/bin/kaggle}; LOCKS=${LOCKS_DIR:-../locks}; LEDGER=${LEDGER:-results/gpu_ledger.csv}; NOW=${GATE_NOW:-$(date -u +%FT%TZ)}
GAS_CAP_H=8.0; RESERVE_H=1.0
quota() { $K quota 2>&1 | awk -v col="$1" '/^GPU/{v=$col; gsub("h","",v); print v}'; }
if [ "${1:-}" = "record" ]; then
  [ $# -ge 4 ] || { echo "usage: gpu_gate.sh record <stage> <ref> <used_before_h> [--after H] [--gpus N --log FILE | --gpus N --wall-s S]"; exit 2; }
  STAGE=$2; RREF=$3; BEFORE=$4; shift 4; AFTER=""; GPUS=""; LOG=""; WALLS=""
  while [ $# -gt 0 ]; do case "$1" in --after) AFTER=$2; shift;; --gpus) GPUS=$2; shift;; --log) LOG=$2; shift;; --wall-s) WALLS=$2; shift;; esac; shift; done
  [ -n "$AFTER" ] || AFTER=$(quota 2); [ -f "$LEDGER" ] || echo "utc,stage,ref,used_before_h,used_after_h,account_delta_h,billed_h,source" > "$LEDGER"
  [ -z "$WALLS" ] && [ -n "$LOG" ] && [ -f "$LOG" ] && WALLS=$(grep -o '"time":[0-9.]*' "$LOG" | tail -1 | cut -d: -f2 | awk '{printf "%d", $1 + 30}')
  awk -v b="$BEFORE" -v a="$AFTER" -v t="$NOW" -v s="$STAGE" -v r="$RREF" -v g="$GPUS" -v w="$WALLS" 'BEGIN{d = a - b; if (g != "" && w != "") {printf "%s,%s,%s,%s,%s,%.2f,%.2f,gpus x wall (%s x %s s)\n", t, s, r, b, a, d, g * w / 3600, g, w} else printf "%s,%s,%s,%s,%s,%.2f,%.2f,quota delta\n", t, s, r, b, a, d, d}' >> "$LEDGER"; tail -1 "$LEDGER"; exit 0
fi
KD=${1:?usage: gpu_gate.sh <kernel_dir> [--retry]}; RETRY=${2:-}; meta="$KD/kernel-metadata.json"; cfg="$KD/cfg.json"; fail=0
chk() { if [ "$1" = ok ]; then echo "PASS  $2"; else echo "FAIL  $2"; fail=1; fi; }
[ -f "$meta" ] && [ -f "$cfg" ] || { echo "FAIL  $meta or $cfg missing"; exit 1; }
REF=$(jq -r .id "$meta"); SHAPE=$(jq -r '.machine_shape // ""' "$meta"); CAP=$(jq -r '.billed_cap_s // empty' "$cfg"); WALL=$(jq -r '.wall_cap_s // empty' "$cfg"); B=$(jq -r '.billing_b // empty' "$cfg")
[[ "$NOW" > "2026-10-10T00:00:00Z" || "$NOW" == "2026-10-10T00:00:00Z" ]] && chk ok "date $NOW is after the quota refresh (2026-10-10T00:00Z)" || chk bad "date $NOW is before the quota refresh (2026-10-10T00:00Z)"
[ "$(jq -r '.is_private' "$meta")" = "true" ] && [ "$(jq -r '.enable_gpu' "$meta")" = "true" ] && [ -n "$SHAPE" ] && chk ok "metadata: private, enable_gpu true, machine_shape $SHAPE" || chk bad "metadata: need is_private true, enable_gpu true, machine_shape set"
if [ -n "$CAP" ] && [ -n "$WALL" ] && [ -n "$B" ] && awk -v b="$B" -v w="$WALL" -v c="$CAP" 'BEGIN{exit !(b*w <= c && w > 0)}'; then chk ok "cfg: billed cap $CAP s, wall cap $WALL s, billing factor b=$B (b x wall = $((B * WALL)) s <= cap)"; else chk bad "cfg.json lacks billed_cap_s / wall_cap_s / billing_b, or b x wall exceeds the billed cap (regenerate with scripts/mk_gpu_wave.py)"; fi
[ ! -e "$LOCKS/kaggle-gpu" ] && chk ok "no $LOCKS/kaggle-gpu lock" || chk bad "$LOCKS/kaggle-gpu is held: $(cat "$LOCKS/kaggle-gpu/owner" 2>/dev/null)"
REM=$(quota 3); USED=$(quota 2); CAPH=$(awk -v c="${CAP:-0}" 'BEGIN{printf "%.3f", c/3600}')
[ -n "$REM" ] && awk -v r="$REM" -v c="$CAPH" 'BEGIN{exit !(r >= c)}' && chk ok "kaggle quota: ${REM} h remaining >= billed cap ${CAPH} h (used now ${USED} h)" || chk bad "kaggle quota: remaining '${REM}' h is below the billed cap ${CAPH} h (or quota unreadable)"
GAS=$([ -f "$LEDGER" ] && awk -F, 'NR>1{s+=$7} END{printf "%.3f", s+0}' "$LEDGER" || echo 0.000); LIMIT=$(awk -v g="$GAS_CAP_H" -v r="$RESERVE_H" -v f="$RETRY" 'BEGIN{print (f=="--retry") ? g : g-r}')
awk -v u="$GAS" -v c="$CAPH" -v l="$LIMIT" 'BEGIN{exit !(u + c <= l + 1e-9)}' && chk ok "gas ledger: ${GAS} h billed so far + cap ${CAPH} h <= ${LIMIT} h ($([ "$RETRY" = "--retry" ] && echo "retry reserve opened" || echo "8 h gas cap minus the 1 h retry reserve"))" || chk bad "gas ledger: ${GAS} h billed so far + cap ${CAPH} h > ${LIMIT} h"
if [ $fail -ne 0 ]; then echo "GATE: FAIL, do not push"; exit 1; fi
NAME=${REF#*/bench-gas-}; END=$(date -u -v+$((WALL + 300))S +%FT%TZ 2>/dev/null || echo "now+$((WALL + 300))s")
echo "GATE: PASS. Note 'used now' ${USED} h as <used_before>; then (one GPU session at a time):"
echo "  mkdir $LOCKS/kaggle-gpu && printf 'gas %s %s end<=%s\n' $REF \"\$(date -u +%FT%TZ)\" $END > $LOCKS/kaggle-gpu/owner"
echo "  $K kernels push -p $KD -t $WALL        # wall seconds; billed cap ${CAP} s = ${CAPH} h at b=$B"
echo "  nohup scripts/watch_kernel.sh $REF gpu kernel_outputs/$NAME $((WALL / 60 + 25)) > /tmp/watch-$NAME.log 2>&1 &"
echo "  after it ends: scripts/gpu_gate.sh record $NAME $REF $USED --gpus $B --log kernel_outputs/$NAME/bench-gas-$NAME.log      # appends the billed hours to $LEDGER"
exit 0
