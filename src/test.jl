using DSGE

m = GHLS()
m <= Setting(:sampling_method, :SMC)
m <= Setting(:use_parallel_workers, false)
m <= Setting(:n_particles, 100)
m <= Setting(:date_presample_start, quartertodate("1983-Q1"))
m <= Setting(:date_mainsample_start, quartertodate("1983-Q1"))
m <= Setting(:date_mainsample_end, quartertodate("2014-Q1"))
estimate(m)
