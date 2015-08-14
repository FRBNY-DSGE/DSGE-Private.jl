# Add paths

using Base: Test
using MATLAB 
using HDF5
using Debug
using DSGE
using DSGE: DistributionsExt
include("../../util.jl")

path = dirname(@__FILE__)
savepath = joinpath(path, "save")

# Load variables

mf = MatFile("$path/pre_mh_state.mat")
mode = get_variable(mf, "params")
hessian = get_variable(mf, "hessian")
YY0 = get_variable(mf, "YY0")
YY = get_variable(mf, "YY")
data = [YY0; YY]
σ = get_variable(mf, "sigscale")
close(mf)

# Load random vectors
mf3 = MatFile("$path/metropolis_hastings.mat")
randvecs = get_variable(mf3, "randvecs")
randvals = get_variable(mf3, "randvals")
close(mf3)

model = Model990()
cc0 = 0.01
cc = 0.09

propdist = DegenerateMvNormal(mode, σ)

# Call metropolis_hastings
tic()
metropolis_hastings(propdist, model, data, cc0, cc, randvecs, randvals)
time_elapsed = toq()

for node in ARGS
    println("node: $node, seconds: $time_elapsed")
end
