# Add paths
using DSGE, MATLAB
path = dirname(@__FILE__)

# Load variables
mf = MatFile("$path/hessian.mat")
params = get_variable(mf, "params")
YY0 = get_variable(mf, "YY0")
YY = get_variable(mf, "YY")
close(mf)
model = Model990()
data = [YY0; YY]

# Call hessian
tic()
hessizero!(params, model, data; noisy=false)
time_elapsed = toq()

for node in ARGS
    println("node: $node, seconds: $time_elapsed")
end
