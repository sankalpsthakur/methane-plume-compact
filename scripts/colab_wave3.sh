#!/bin/bash
# Wave 3 (Amendment 6) on a free Colab T4: one session, route A setup, then the given kernel jobs one at a time; small outputs come back;
# the session is stopped on exit whatever happens.
# usage: scripts/colab_wave3.sh <session> <out_dir> <kernel_dir> [<kernel_dir> ...]
# The Kaggle credential sent to the VM is the short-lived OAuth access token of ~/.kaggle/credentials.json (never the refresh token),
# written to a 0600 temp file, uploaded, and the temp file removed at once; vm_setup.py deletes the VM copy after the download.
# Polling every POLL_S seconds keeps the Colab kernel active; JOB_MAX_S caps one job (Amendment 6 caps: 1.5 h per job).
set -u
cd "$(dirname "$0")/.." || exit 1
C=${COLAB_BIN:-$HOME/.local/bin/colab}; S=$1; OUT=$2; shift 2
POLL_S=${POLL_S:-300}; JOB_MAX_S=${JOB_MAX_S:-5700}; JOB_CAP_S=${JOB_CAP_S:-5400}; STEPS=${STEPS:-78675}; KEEP=0
mkdir -p "$OUT"; LOG="$OUT/runner.log"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
cleanup() {
  if [ "$KEEP" = 1 ]; then log "KEEPING session $S: an output download failed; fetch it by hand, then: $C stop -s $S"; return; fi
  log "stopping session $S"; $C stop -s "$S" >> "$LOG" 2>&1; $C sessions >> "$LOG" 2>&1; }
trap cleanup EXIT

JOBS=$(for k in "$@"; do basename "$k"; done | paste -sd, -)
log "new session $S (T4) for jobs $JOBS"
$C new -s "$S" --gpu T4 >> "$LOG" 2>&1 || { log "FAILED: no T4 session"; exit 2; }
$C exec -s "$S" -f scripts/colab/vm_mkdirs.py --env JOBS="$JOBS" --timeout 60 >> "$LOG" 2>&1
for k in "$@"; do
  $C upload -s "$S" "$k/script.py" "/content/w3/$(basename "$k")/script.py" >> "$LOG" 2>&1 || { log "FAILED: upload $k"; exit 3; }
done
chk=$($C exec -s "$S" -f scripts/colab/vm_check_uploads.py --env JOBS="$JOBS" --timeout 60 2>&1); echo "$chk" >> "$LOG"
echo "$chk" | grep -q "UPLOADS True" || { log "FAILED: uploaded scripts not where expected; the token is NOT sent"; exit 3; }
TOK=$(mktemp "${TMPDIR:-/tmp}/kt.XXXXXX"); chmod 600 "$TOK"
python3 -c 'import json, os, sys; c = json.load(open(os.path.expanduser("~/.kaggle/credentials.json"))); open(sys.argv[1], "w").write(c["access_token"])' "$TOK"
$C upload -s "$S" "$TOK" /root/.kaggle/access_token >> "$LOG" 2>&1; rc=$?; rm -f "$TOK"
[ $rc -eq 0 ] || { log "FAILED: token upload"; exit 3; }
log "setup (hardware gate, cache download, sha256)"
so=$($C exec -s "$S" -f scripts/colab/vm_setup.py --timeout 3300 2>&1); echo "$so" >> "$LOG"
echo "$so" | grep -q "token file on VM removed: True" || log "WARNING: token removal not confirmed"
echo "$so" | grep -q "SETUP OK" || { log "FAILED: setup (see $LOG)"; exit 4; }
$C download -s "$S" /content/w3/hw.json "$OUT/hw.json" >> "$LOG" 2>&1 && python3 -c 'import json, sys; json.load(open(sys.argv[1]))' "$OUT/hw.json" || { log "FAILED: download round trip"; exit 4; }
log "download round trip ok: $(cat "$OUT/hw.json" | tr -d '\n' | cut -c1-200)"

for k in "$@"; do
  n=$(basename "$k"); log "start $n"
  $C exec -s "$S" -f scripts/colab/vm_start.py --env JOB="$n" --timeout 60 >> "$LOG" 2>&1
  t0=$SECONDS; st=running; fails=0; prev_step=""; prev_t=""
  while [ "$st" = running ] && [ $((SECONDS - t0)) -lt "$JOB_MAX_S" ]; do
    sleep "$POLL_S"
    o=$($C exec -s "$S" -f scripts/colab/vm_poll.py --env JOB="$n" --timeout 90 2>&1)
    echo "$o" >> "$LOG"
    s=$(echo "$o" | grep -o 'JOBSTATE=[a-z]*' | tail -1 | cut -d= -f2)
    if [ -z "$s" ]; then fails=$((fails + 1)); [ $fails -ge 3 ] && { st=lost; log "$n: 3 failed polls, session presumed lost"; }; continue; fi
    fails=0; st=$s
    log "$n: $(echo "$o" | grep -E '^PROGRESS' | tail -1) state=$st"
    step=$(echo "$o" | grep -E '^PROGRESS' | tail -1 | awk '{print $4}')
    if [ "$st" = running ] && [ -n "$step" ] && [ -n "$prev_step" ] && [ "$step" -gt "$prev_step" ] 2>/dev/null; then
      proj=$(awk -v a="$step" -v b="$prev_step" -v dt=$((SECONDS - prev_t)) -v e=$((SECONDS - t0)) -v tot="$STEPS" 'BEGIN{r=(a-b)/dt; printf "%d %.1f", e + (tot-a)/r + 150, r}')
      log "$n: $(echo "$proj" | cut -d' ' -f2) steps/s, projected job total $(echo "$proj" | cut -d' ' -f1) s (cap $JOB_CAP_S s)"
      if [ "$(echo "$proj" | cut -d' ' -f1)" -gt "$JOB_CAP_S" ]; then
        log "$n: projection above the cap: infrastructure stop on throughput alone (Amendment 6 note 1 b)"
        $C exec -s "$S" -f scripts/colab/vm_stop_job.py --env JOB="$n" --timeout 60 >> "$LOG" 2>&1; st=stopped; sleep 30
      fi
    fi
    [ -n "$step" ] && { prev_step=$step; prev_t=$SECONDS; }
  done
  [ "$st" = running ] && log "$n: JOB_MAX_S reached while running (the driver's soft deadline should have ended it)"
  if [ "$st" != lost ]; then
    pk=$($C exec -s "$S" -f scripts/colab/vm_pack.py --env JOB="$n" --timeout 180 2>&1); echo "$pk" >> "$LOG"
    want=$(echo "$pk" | grep -o 'sha256 [0-9a-f]*' | tail -1 | cut -d' ' -f2)
    if $C download -s "$S" "/content/w3/out_$n.tgz" "$OUT/out_$n.tgz" >> "$LOG" 2>&1 && [ -n "$want" ] && [ "$(shasum -a 256 "$OUT/out_$n.tgz" | cut -d' ' -f1)" = "$want" ]; then
      mkdir -p "$OUT/$n" && tar xzf "$OUT/out_$n.tgz" -C "$OUT/$n" && log "$n: outputs in $OUT/$n (sha256 ok)"
    else
      KEEP=1; log "$n: output download failed or hash differs; the session is kept for a manual retry"
    fi
  fi
  log "$n finished: state=$st"
  [ "$st" = lost ] && exit 5
done
log "all jobs done"
