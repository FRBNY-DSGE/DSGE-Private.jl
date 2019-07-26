using DSGE

m = GHLS()
m <= Setting(:sampling_method, :SMC)
m <= Setting(:use_parallel_workers, false)
m <= Setting(:n_particles, 100)
m <= Setting(:date_presample_start, quartertodate("1983-Q1"))
m <= Setting(:date_mainsample_start, quartertodate("1983-Q1"))
m <= Setting(:date_mainsample_end, quartertodate("2014-Q1"))

# Reads in data and keeps only series we need
data_raw = readdlm("glss_data.txt")
data_matrix = data_raw[:, [1, 2, 3, 6, 7]]
data_matrix_transpose = zeros(Float64, 5, 125) #we want the data with periods as columns

estimate(m, transpose!(data_matrix_transpose, data_matrix))
