#!/bin/bash
# RTWALK114 (10 Nov): release the user hold on the first job of the option-A chain (JOBID) once the slot caps allow it:
# all.q has >= MINFREE available slots, our running/pending non-held slots + 97 <= OURMAX, and fewer than MAXJ of our
# non-held rtw114_* / rtw114B_* production jobs exist (MAXJ while s104v3* / s114* jobs exist, else MAXJ_FREE).
# 1-slot SGE job; polls every POLL s; exits after the qrls (or if JOBID is gone). Never qdel. Log: $R/logs/releaseA.log
R=/data/dsge_data_dir/estimation-AI/work/rtwalk114
JOBID=${JOBID:?}; MAXJ=${MAXJ:-2}; MAXJ_FREE=${MAXJ_FREE:-4}; MINFREE=${MINFREE:-150}; OURMAX=${OURMAX:-400}; POLL=${POLL:-300}
unset CLAUDE_CODE_MESSAGING_TOKEN
echo "releaseA start $(date -Is) job=$JOB_ID target=$JOBID"
while true; do
  st=$(qstat -u "$USER" 2>/dev/null | awk -v j=$JOBID 'NR>2 && $1 == j {print $5}')
  [ -z "$st" ] && { echo "$(date -Is) job $JOBID no longer queued: exit"; exit 0; }
  case "$st" in *h*) ;; *) echo "$(date -Is) job $JOBID not held (state $st): exit"; exit 0;; esac
  free=$(qstat -g c 2>/dev/null | awk '$1 == "all.q" {print $5}')
  ours=$(qstat -u "$USER" 2>/dev/null | awk 'NR>2 && $5 !~ /h/ {s += ($5 ~ /r/ ? $9 : $8)} END {print s+0}')
  nrun=$(qstat -u "$USER" 2>/dev/null | awk 'NR>2 && $5 !~ /h/ && ($3 ~ /^rtw114_/ || $3 ~ /^rtw114B_/) {n++} END {print n+0}')
  big=$(qstat -u "$USER" 2>/dev/null | awk 'NR>2 && ($3 ~ /^s104v3/ || $3 ~ /^s114/) {n++} END {print n+0}')
  mj=$MAXJ; [ "$big" -eq 0 ] && mj=$MAXJ_FREE
  if [ "$nrun" -lt "$mj" ] && [ "${free:-0}" -ge "$MINFREE" ] && [ $((ours + 97)) -le "$OURMAX" ]; then
    qrls $JOBID; echo "$(date -Is) qrls $JOBID (running $nrun of max $mj, free $free, ours $ours)"; exit 0
  fi
  sleep $POLL
done
