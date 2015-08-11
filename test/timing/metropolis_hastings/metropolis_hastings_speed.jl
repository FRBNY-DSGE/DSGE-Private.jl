# Add paths

using Base: Test
using MATLAB 
using HDF5

using DSGE
using DSGE: DistributionsExt
# include("../util.jl")

path = dirname(@__FILE__)

# Load variables
mf = MatFile("$path/metropolis_hastings.mat")
mode = get_variable(mf, "params")
hessian = get_variable(mf, "hessian")
YY = get_variable(mf, "YYall")

randvecs = get_variable(mf, "randvecs")
randvals = get_variable(mf, "randvals")
close(mf)

mf2 = MatFile("$path/sigscale.mat")
σ = get_variable(mf2, "sigscale")
close(mf2)

model = Model990()
cc0 = 0.01
cc = 0.09

propdist = DegenerateMvNormal(mode, σ)

# The first time tic and toc are called, the function gets compiled, so we should ignore the returned time
tic()
for i = 1:1
    metropolis_hastings(propdist, model, YY, cc0, cc, randvecs, randvals)
end
toq()

# Call metropolis_hastings
iterations = 1
tic()
for i = 1:iterations
    metropolis_hastings(propdist, model, YY, cc0, cc, randvecs, randvals)
end
time_elapsed = toq()

println(time_elapsed)
