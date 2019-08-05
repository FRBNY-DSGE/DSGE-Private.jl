using DSGE, DSGEModels, FileIO, CSV, StatsBase, Plots, Dates, StateSpaceRoutines, Random, MAT, Distributions
# This script estimates in real time

filepath = "$(filepath)/../../../"
pm <= PoolModel()
pm <= Setting(:saveroot, "$(filepath)/save/")
pm <= Setting(:dataroot, "$(filepath)/save/input_data")

# Construct real time estimation of lambda (evolution over time)
h = 4
data = df_to_matrix(load_data(pm))
print("Starting to run SMC\n")
Random.seed!(1793)
for t in 1:size(data,2)
    # run smc estimation
    DSGE.estimate(pm, data[:,1:t]; filestring_addl = ["period=$(t)", "preddens=right805current904"])
end


