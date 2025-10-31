using DSGE
using ModelConstructors
using StateSpaceRoutines
using HDF5
using Random
path = dirname(@__FILE__)
writing_output = false
 

optimizer_config = :csminwel
#optimizer_config = :lbfgs
#optimizer_config = :trust_region_newton
#optimizer_config = :pso
println(optimizer_config)

custom_settings = [Setting(:date_forecast_start, quartertodate("2015-Q4"))]
m = AnSchorfheide(custom_settings = custom_settings, testing = true)
#m = DSSW()

#load data
file = "$path/../reference/optimize_in.h5"
x0   = h5read(file, "params")
data = h5read(file, "data")'

#test file
params_test = deepcopy(x0)
data_test   = Matrix{Float64}(data)

data = construct_data()

file = "$path/../reference/optimize_out.h5"
minimizer  = h5read(file, "minimizer")
minimum    = h5read(file, "minimum")
H_expected = h5read(file, "H")
 
# See src/estimate/estimate.jl
DSGE.update!(m, x0)
n_iterations = 50


# Set seed for reproducibility
Random.seed!(42)

#=
x0 = Float64[]
for p in m.parameters
    lower = p.valuebounds[1]
    upper = p.valuebounds[2]
    push!(x0, lower + rand() * (upper - lower))
end

=#
# Update x0
#DSGE.update!(m, x0)

#start timer
start_time = time()
out, H, callback_data = optimize!(m, data; method = optimizer_config, iterations = n_iterations, show_trace = true)
seconds = time() - start_time

println(out)
println(seconds)
