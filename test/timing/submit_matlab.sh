#!/bin/bash

NODE=$1
source /home/$USER/.bashrc
echo -n "node: $i, seconds: " >> my_results_mat.txt
matlab14a-custom-node -n $i -r gensys_speed \
  | tail -n 2 >> my_results_mat.txt
