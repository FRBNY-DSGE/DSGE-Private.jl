#!/bin/bash

echo "Timing Results for Julia:"
grep -e "node" metropolis_hastings_results_jl.txt | ./summarize_results.awk
echo ""
echo "Timing Results for MATLAB:"
grep -e "node" metropolis_hastings_results_mat.txt | ./summarize_results.awk


