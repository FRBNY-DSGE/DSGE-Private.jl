#!/bin/bash
# RTWALK114 gate for the all-t option-B runs (Marco, 9 Nov; 10 Nov: cost criterion dropped, phase-2 cap 2 -> 4 jobs once the
# big runs end, all our slots <= 400). Runs as a 1-slot SGE job (16G), polls every POLL s.
#   Waits until each of the three phase-1 runs has a verdict marker:
#     A  = steps/2020Q3_to_2020Q4/HEALTH_{PASS,FAIL}       B4 = steps_modelbridge/2020Q4/HEALTH_{PASS,FAIL}
#     B10 = steps_modelbridge/2022Q2/HEALTH_{PASS,FAIL}
#   Any HEALTH_FAIL -> sched/GATE_FAIL, exit (nothing further is submitted).
#   All PASS -> runs compareAB.jl (thresholds in its last block) -> out/compare_AB_2020Q4_gate.txt.
#     GATE PASS -> sched/GATE_PASS and submits the phase-2 scheduler (schedB.sh with sched/specs_phase2.txt).
#     GATE FAIL -> sched/GATE_FAIL with the reasons; nothing further is submitted.
#   Before its h_rt it resubmits itself. Never qdel. Log: $R/logs/gateAB.log
R=/data/dsge_data_dir/estimation-AI/work/rtwalk114
POLL=${POLL:-600}; LIFE=${LIFE:-133200}
unset CLAUDE_CODE_MESSAGING_TOKEN
echo "gateAB start $(date -Is) host=$(hostname) job=$JOB_ID"
SECONDS=0
mk() { [ -f "$1/HEALTH_PASS" ] && echo PASS && return; [ -f "$1/HEALTH_FAIL" ] && echo FAIL && return; echo none; }
while true; do
  a=$(mk $R/steps/2020Q3_to_2020Q4); b4=$(mk $R/steps_modelbridge/2020Q4); b10=$(mk $R/steps_modelbridge/2022Q2)
  if [ $a != none ] && [ $b4 != none ] && [ $b10 != none ]; then
    echo "$(date -Is) markers: A $a, B 2020Q4 $b4, B 2022Q2 $b10"
    if [ $a = PASS ] && [ $b4 = PASS ] && [ $b10 = PASS ]; then
      WALL=3000 /bin/bash $R/scripts/run.sh compareAB.jl > $R/logs/compareAB.log 2>&1
      g=$R/out/compare_AB_2020Q4_gate.txt
      if [ -f $g ] && head -1 $g | grep -q "GATE PASS"; then
        cp $g $R/sched/GATE_PASS
        out=$(qsub -b y -q all.q -wd $R -j y -N rtw114sched2 -o $R/logs/schedB2.log -l h_vmem=1G,h_rt=38:00:00 -v SPECS=$R/sched/specs_phase2.txt,MAXJ=2,MAXJ_FREE=4,MINFREE=150,OURMAX=400,POLL=300 /bin/bash $R/scripts/schedB.sh 2>&1)
        echo "$(date -Is) GATE PASS; phase-2 scheduler: $out"; echo "$out" >> $R/sched/GATE_PASS
      else
        { echo "GATE FAIL $(date -Is)"; [ -f $g ] && cat $g || echo "compareAB.jl produced no gate file (see logs/compareAB.log)"; } > $R/sched/GATE_FAIL
        echo "$(date -Is) GATE FAIL"; cat $R/sched/GATE_FAIL
      fi
    else
      echo "GATE FAIL $(date -Is): health A $a, B 2020Q4 $b4, B 2022Q2 $b10" > $R/sched/GATE_FAIL; cat $R/sched/GATE_FAIL
    fi
    exit 0
  fi
  if [ $SECONDS -ge $LIFE ]; then
    qsub -b y -q all.q -wd $R -j y -N rtw114gate -o $R/logs/gateAB.log -l h_vmem=16G,h_rt=38:00:00 -v POLL=$POLL /bin/bash $R/scripts/gateAB.sh
    echo "$(date -Is) lifetime reached, resubmitted myself: exit"; exit 0
  fi
  sleep $POLL
done
