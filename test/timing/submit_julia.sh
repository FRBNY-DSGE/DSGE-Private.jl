#!/bin/bash

NODE=$1
source /home/$USER/.bashrc
echo -n "node: ${NODE}, seconds: " >> my_results_jl.txt
./julia-0.3.9-custom-node -n ${NODE} gensys_speed.jl >> my_results_jl.txt
