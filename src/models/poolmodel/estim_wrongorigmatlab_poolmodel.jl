using DSGE, Dates, Random
# This script estimates in real time

current_directory = dirname(@__FILE__)
filepath = "$(current_directory)/../../../"
pm = PoolModel()
pm <= Setting(:saveroot, "$(filepath)save/")
pm <= Setting(:dataroot, "$(filepath)save/input_data/")

# Construct real time estimation of lambda (evolution over time)
h = 4
data = df_to_matrix(pm,load_data(pm))
print("Starting to run SMC\n")
Random.seed!(1793)
pm <= Setting(:sampling_method, :SMC)

@everywhere using DSGE, OrderedCollections
for t in 1:size(data,2)
    # run smc estimation
    DSGE.estimate(pm, data[:,1:t]; filestring_addl = ["period=$(t)", "preddens=wrongorigmatlab"])
end
