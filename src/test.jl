using DSGE

m = GHLS()
m <= Setting(:sampling_method, :SMC)
m <= Setting(:use_parallel_workers, false)
m <= Setting(:n_particles, 100)
estimate(m)
