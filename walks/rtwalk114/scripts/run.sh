#!/bin/bash
# RTWALK114 job wrapper (= rtwalk104 run.sh with R -> work/rtwalk114; precompile line adds ClusterManagers) (pattern of work/rtvint104/scripts/run.sh). Private depot, --compiled-modules=existing.
# usage (head node, qsub only):
#   R=/data/dsge_data_dir/estimation-AI/work/rtwalk114
#   qsub -b y -q all.q -wd $R -j y -r y -N NAME -o $R/logs/NAME.log -l h_vmem=8G,h_rt=HH:MM:SS [-pe smp NW+1] \
#        -v NW=<workers, 0 = none>,WALL=<seconds, < h_rt>,PRECOMP=1,<other env vars> /bin/bash $R/scripts/run.sh <script.jl>
# PRECOMP=1 runs a precompile step first (into work/rtwalk114/depot only). Workers: `julia -p NW` (single node).
#
# Exit codes (RTWALK104-REVIEW §6; SGE semantics, tested on this site by tests/test_exitcodes.sh):
#   julia runs under `timeout WALL` (default 133200 s = 37 h; WALL must be < the job's h_rt and < the worker cap 38.27 h).
#   wall limit reached (timeout: 124, or 137 after the 10-min KILL grace) -> exit 99 = SGE requeues the job (same job
#       id, same -v env; step.jl resumes from its newest loadable stage file). At most MAX_REQUEUE (default 6) times per
#       job id (counter logs/requeue_<jobid>.count), then exit 100.
#   any other non-zero exit (error = 1, HEALTH_FAIL = 2, ...) -> exit 100 = SGE puts the job in error state (Eqw), so
#       jobs held on it with -hold_jid STAY held (a plain non-zero exit would release them). Clear with qdel, or fix and
#       `qmod -cj <jobid>` to rerun it. After a HEALTH_FAIL, first `qalter -adds v HEALTH_REDO 1 -adds v H_POLISH <k>
#       <jobid>` (not `qalter -v`, which may redefine the whole -v list); see the chain_qsub.txt header (FIXVERIFY N1).
#   on any non-zero julia exit with FRBNY_NW > 0, pending FRBNY worker jobs (julia-*, cwd $R/workers) are listed as
#       possible orphans of the killed master ("N4 WARNING"); they are never deleted here (FIXVERIFY N4).
#   0 -> 0
# JL=<path> replaces the julia binary (tests/test_exitcodes.sh only).
SCRIPT=$1; NW=${NW:-0}
R=/data/dsge_data_dir/estimation-AI/work/rtwalk114
ENVDIR=${WALK_ENV_DIR:-$R/env}
JL=${JL:-/apps/julia-1.12.6/bin/julia}
WALL=${WALL:-133200}; MAX_REQUEUE=${MAX_REQUEUE:-6}
export JULIA_DEPOT_PATH="${WALK_DEPOT_DIR:-$R/depot}:/home/rcemxd02/.julia:/apps/julia-1.12.6/local/share/julia:/apps/julia-1.12.6/share/julia"
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 GKSwstype=100 JULIA_PROJECT=$ENVDIR
unset XDG_RUNTIME_DIR LD_PRELOAD DISPLAY
echo "host=$(hostname) start=$(date -Is) jobid=$JOB_ID slots=$NSLOTS script=$SCRIPT NW=$NW WALL=${WALL}s restarted=${RESTARTED:-0}"
echo "Julia project: $ENVDIR; script root: $(dirname "$0")"
cd $R
SECONDS=0
if [ "${PRECOMP:-0}" = "1" ]; then
  $JL --project=$ENVDIR --startup-file=no -e "using DSGE, SMC, ModelConstructors, JLD2, CSV, DataFrames, SHA, StatsBase, ClusterManagers, frbny; println(\"precompiled into \", DEPOT_PATH[1])" || { echo "precompile failed: exit 100"; exit 100; }
  echo "precompile done after ${SECONDS}s"
fi
P=""; [ "$NW" -gt 0 ] && P="-p $NW"
TJ=$SECONDS
timeout -s TERM -k 600 ${WALL} $JL --project=$ENVDIR --startup-file=no --compiled-modules=existing $P $R/scripts/$SCRIPT
rc=$?
echo "end=$(date -Is) julia_exit=$rc elapsed=${SECONDS}s"
# N4 (RTWALK104-FIXVERIFY): the julia master is gone. A FRBNY worker job it submitted that is still PENDING (name
# julia-<pid>, cwd $R/workers) is now an orphan. List such jobs in the log; read-only, never qdel, never changes rc.
stray_workers() {
  { [ "${FRBNY_NW:-0}" -gt 0 ] && command -v qstat >/dev/null 2>&1; } || return 0
  for id in $(qstat -u "${USER:-$(id -un)}" -s p 2>/dev/null | awk '$3 ~ /^julia-/ {print $1}'); do
    qstat -j "$id" 2>/dev/null | grep -Eq "^cwd: +$R/workers\$" && \
      echo "N4 WARNING: pending FRBNY worker job $id (cwd $R/workers) is probably an orphan of this killed master: check it, then qdel $id by hand"
  done
  return 0
}
[ $rc -ne 0 ] && stray_workers
if [ $rc -eq 124 ] || { [ $rc -eq 137 ] && [ $(( SECONDS - TJ )) -ge $WALL ]; }; then
  CNT=$R/logs/requeue_${JOB_ID:-nojob}.count
  n=$(( $(cat $CNT 2>/dev/null || echo 0) + 1 )); echo $n > $CNT
  if [ $n -gt $MAX_REQUEUE ]; then
    echo "wall limit reached for the ${n}th time (> MAX_REQUEUE=$MAX_REQUEUE): exit 100 (error state, dependants stay held)"; exit 100
  fi
  echo "wall limit ${WALL}s reached (requeue $n of at most $MAX_REQUEUE): exit 99 = requeue; the step resumes from its stage files"
  exit 99
fi
[ $rc -ne 0 ] && { echo "julia exit $rc: exit 100 (error state; jobs held on this one stay held)"; exit 100; }
exit 0
