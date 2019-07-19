using DSGE

m = GHLS()
m <= Setting(:sampling_method, :SMC)
m <= Setting(:use_parallel_workers, true)
estimate(m)
