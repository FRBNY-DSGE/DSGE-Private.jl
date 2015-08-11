# Add paths
using DSGE, MATLAB
path = dirname(@__FILE__)

# Load variables
mf = MatFile("$path/csminwel.mat")
x0 = get_variable(mf, "x0")
H0 = get_variable(mf, "H0")
nit = int(get_variable(mf, "nit"))
crit = get_variable(mf, "crit")
YY0 = get_variable(mf, "YY0")
YY = get_variable(mf, "YY")
randvecs = get_variable(mf, "randvecs")
close(mf)

model = Model990()
data = [YY0; YY]

function posterior_min!{T<:FloatingPoint}(x::Vector{T})
    tomodel!(x, model.Θ)
    return -posterior(model, data)
end

# Call csminwel
tic()
csminwel(posterior_min!, x0, H0; ftol=crit, iterations=nit, randvecs=randvecs)
time_elapsed = toq()

# Print time elapsed
for node in ARGS
    println("node: $node, seconds: $time_elapsed")
end
