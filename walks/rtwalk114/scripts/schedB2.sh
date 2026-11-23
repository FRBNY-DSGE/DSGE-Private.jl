#!/bin/bash
# schedB2.sh (10 Nov) = schedB.sh, except that held jobs (state h*) are not counted as running/pending production jobs and it resubmits itself as schedB2.sh.
# RTWALK114 slot-capped scheduler for option-A/B production jobs (Marco, 9 Nov 2026). Runs as a 1-slot SGE job.
# Every POLL seconds:
#   - running/pending production jobs = my jobs named rtw114_* or rtw114B_* (qstat). If fewer than MAXJ (2) AND all.q has
#     >= MINFREE (150) available slots (qstat -g c), submit the next spec whose prerequisite file exists and that was not
#     submitted before (marker sched/submitted_<name>). At most one submission per poll.
#   - spec file (SPECS, default $R/sched/specs.txt): one line per job:  <name>|<prerequisite file>|<qsub arguments after -N name>
#     The prerequisite is the ss104 walk's HEALTH_PASS at t (bridgeB.jl also checks it at start and exits 3 if missing).
#   - exits when every spec is submitted; before its h_rt it resubmits itself (same SPECS) and exits.
# Read-only on everything else; never qdel. Log: $R/logs/schedB.log
R=/data/dsge_data_dir/estimation-AI/work/rtwalk114
SPECS=${SPECS:-$R/sched/specs.txt}
MAXJ=${MAXJ:-2}; MAXJ_FREE=${MAXJ_FREE:-$MAXJ}; MINFREE=${MINFREE:-150}; OURMAX=${OURMAX:-400}; JOBSLOTS=${JOBSLOTS:-97}; POLL=${POLL:-600}; LIFE=${LIFE:-133200}
# (10 Nov, Marco) MAXJ applies while a big run (job names s104v3*, s114*: ss104v3, the ss114 direct estimation and their
# health checks) is running or queued; otherwise MAXJ_FREE. A new job is submitted only if all our slots (every job of
# ours, running or pending, workers included) + JOBSLOTS <= OURMAX.
mkdir -p $R/sched
unset CLAUDE_CODE_MESSAGING_TOKEN
echo "schedB start $(date -Is) host=$(hostname) job=$JOB_ID specs=$SPECS maxj=$MAXJ minfree=$MINFREE"
SECONDS=0
while true; do
  left=0; nrun=$(qstat -u "$USER" 2>/dev/null | awk 'NR>2 && $5 !~ /h/ && ($3 ~ /^rtw114_/ || $3 ~ /^rtw114B_/) {n++} END {print n+0}')   # held jobs (the A chain's -hold_jid lines, A line 1 before release) do not count
  free=$(qstat -g c 2>/dev/null | awk '$1 == "all.q" {print $5}')
  ours=$(qstat -u "$USER" 2>/dev/null | awk 'NR>2 && $5 !~ /h/ {s += ($5 ~ /r/ ? $9 : $8)} END {print s+0}')
  big=$(qstat -u "$USER" 2>/dev/null | awk 'NR>2 && ($3 ~ /^s104v3/ || $3 ~ /^s114/) {n++} END {print n+0}')
  mj=$MAXJ; [ "$big" -eq 0 ] && mj=$MAXJ_FREE
  done1=0
  while IFS='|' read -r name prereq args; do
    [ -z "$name" ] && continue; case "$name" in \#*) continue;; esac
    [ -f "$R/sched/submitted_$name" ] && continue
    left=$((left + 1))
    [ $done1 -eq 1 ] && continue
    [ -f "$prereq" ] || continue
    if [ "$nrun" -lt "$mj" ] && [ "${free:-0}" -ge "$MINFREE" ] && [ $((ours + JOBSLOTS)) -le "$OURMAX" ]; then
      out=$(qsub -N "$name" $args 2>&1); echo "$(date -Is) submit $name (running $nrun of max $mj, free $free, ours $ours): $out"
      echo "$out" > "$R/sched/submitted_$name"; done1=1; left=$((left - 1))
    fi
  done < "$SPECS"
  [ $left -eq 0 ] && { echo "$(date -Is) all specs submitted: exit"; exit 0; }
  if [ $SECONDS -ge $LIFE ]; then
    qsub -b y -q all.q -wd $R -j y -N rtw114sched -o $R/logs/schedB.log -l h_vmem=1G,h_rt=38:00:00 -v SPECS=$SPECS,MAXJ=$MAXJ,MAXJ_FREE=$MAXJ_FREE,MINFREE=$MINFREE,OURMAX=$OURMAX,POLL=$POLL /bin/bash $R/scripts/schedB2.sh
    echo "$(date -Is) lifetime reached, resubmitted myself: exit"; exit 0
  fi
  sleep $POLL
done
