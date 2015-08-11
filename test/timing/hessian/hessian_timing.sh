#!/bin/bash

# Submit executable to all 10 nodes for timing test.
# Snake submission to nodes to hopefully eliminate some bias in how jobs are
# submitted. Make sure to modify paths below to actual file names.
#
# Created 2015-08-07 MJS

#cd /path/to/testing/code

ALL_NODES="n01 n02 n03 n04 n05 n11 n12 n13 n14 n15"

SUBMIT_FORWARD=1
for i in $ALL_NODES;
do
  #echo -n "node: $i, seconds: " >> hessian_results_jl.txt
  #echo -n "node: $i, seconds:" >> hessian_results_mat.txt
  if ((SUBMIT_FORWARD));
  then 
    echo "../matlab14a-custom-node -n $i -r \"hessian_speed $i\" >> hessian_results_mat.txt" | batch
    echo "../julia-0.3.9-custom-node -n $i hessian_speed.jl $i >> hessian_results_jl.txt" | batch
  else
    echo "../julia-0.3.9-custom-node -n $i hessian_speed.jl $i >> hessian_results_jl.txt" | batch
    echo "../matlab14a-custom-node -n $i -r \"hessian_speed $i\" >> hessian_results_mat.txt" | batch
  fi
  # Flip option.
  SUBMIT_FORWARD=$((1-SUBMIT_FORWARD))
done
