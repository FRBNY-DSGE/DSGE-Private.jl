# Add paths
using DSGE, MATLAB
path = dirname(@__FILE__)

# Load variables
model = Model990()
mf = MatFile("$path/posterior.mat")
YY0 = get_variable(mf, "YY0")
YY = get_variable(mf, "YY")
close(mf)
data = [YY0; YY]

# Call posterior
iterations = int(1e5)
tic()
for i = 1:iterations
    posterior(model, data)
end
time_elapsed = toq()

println("$iterations calls to posterior executed in $time_elapsed seconds")
