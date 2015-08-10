#!/bin/bash

# Submit executable to all 10 nodes for timing test.
# Snake submission to nodes to hopefully eliminate some bias in how jobs are
# submitted. Make sure to modify paths below to actual file names.
#
# Created 2015-08-07 MJS

cd /path/to/testing/code

ALL_QUEUES="n01 n02 n03 n04 n05 n11 n12 n13 n14 15"

SUBMIT_FORWARD=1
for i in $ALL_QUEUES;
do
  echo -n "node: $i, seconds: " >> my_results_jl.txt
  echo -n "node: $i, seconds: " >> my_results_mat.txt
  if ((SUBMIT_FORWARD));
  then 
    matlab14a-custom-queue -q $i -r my_test_script.m >> my_results_mat.txt
    julia-0.3.9-custom-queue -q $i my_test_script.jl >> my_results_jl.txt
  else
    julia-0.3.9-custom-queue -q $i my_test_script.jl >> my_results_jl.txt
    matlab14a-custom-queue -q $i -r my_test_script.m >> my_results_mat.txt
  fi
  # Flip option.
  SUBMIT_FORWARD=$((1-SUBMIT_FORWARD))
done
