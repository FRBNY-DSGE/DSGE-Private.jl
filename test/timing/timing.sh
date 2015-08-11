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
  if ((SUBMIT_FORWARD));
  then 
    nohup submit_julia.sh $i 2>/dev/null &
    nohup submit_matlab.sh $i 2>/dev/null &
  else
    nohup submit_matlab.sh $i 2>/dev/null &
    nohup submit_julia.sh $i 2>/dev/null &
  fi
  # Flip option.
  SUBMIT_FORWARD=$((1-SUBMIT_FORWARD))
done
