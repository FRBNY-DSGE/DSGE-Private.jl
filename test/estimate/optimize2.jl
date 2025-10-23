using DSGE
using ModelConstructors
using StateSpaceRoutines
using HDF5
path = dirname(@__FILE__)
writing_output = false
 

#optimizer_config = :csminwel
#optimizer_config = :lbfgs
#optimizer_config = :trust_region_newton
optimizer_config = :pso
println(optimizer_config)

custom_settings = [Setting(:date_forecast_start, quartertodate("2015-Q4"))]
m = AnSchorfheide(custom_settings = custom_settings, testing = true)


# Load data
file = "$path/../reference/optimize_in.h5"
x0   = h5read(file, "params")
data = h5read(file, "data")'
  
# For regenerating test file
params_test = deepcopy(x0)
data_test   = Matrix{Float64}(data)
 
file = "$path/../reference/optimize_out.h5"
minimizer  = h5read(file, "minimizer")
minimum    = h5read(file, "minimum")
H_expected = h5read(file, "H")
 
# See src/estimate/estimate.jl
DSGE.update!(m, x0)
n_iterations = 600

x0 = Float64[p.value for p in m.parameters]

#start timer
start_time = time()
out, H = optimize!(m, data; method = optimizer_config, iterations = n_iterations)
seconds = time() - start_time

println(out)
println(seconds)
