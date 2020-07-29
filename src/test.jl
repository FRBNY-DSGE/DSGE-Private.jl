using DSGE, DelimitedFiles, JLD2

#m = GHLS()
@load "model.jld2" m
m <= Setting(:sampling_method, :SMC)
m <= Setting(:use_parallel_workers, true)
m <= Setting(:n_particles, 1000)
m <= Setting(:date_presample_start, quartertodate("1983-Q1"))
m <= Setting(:date_mainsample_start, quartertodate("1983-Q1"))
m <= Setting(:date_mainsample_end, quartertodate("2014-Q1"))

m <= Setting(:optimization_attempts, 25)
m <= Setting(:optimization_xtol, .0001)
m <= Setting(:optimization_ftol, .0001)
m <= Setting(:optimization_gtol, .0001)

# Reads in data and keeps only series we need
#data_raw = readdlm("glss_data.txt")
#data_matrix = data_raw
#data_matrix = data_raw[:, [1, 2, 3, 6, 7]]
#data_matrix = data_raw[:, [1, 2, 3, 6, 7]]
#t_data_matrix = convert(Matrix, data_matrix')
#we want the data with periods as columns
#estimate(m,t_data_matrix,verbose=:none)
estimate(m,verbose=:high)
