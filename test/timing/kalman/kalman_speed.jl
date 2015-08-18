# Add paths
using DSGE, MATLAB
path = dirname(@__FILE__)

# Load variables
mf = MatFile("$path/kalman.mat")
data = get_variable(mf, "data")
lead = 1
a = reshape(get_variable(mf, "a"), 72, 1)
F = get_variable(mf, "F")
b = reshape(get_variable(mf, "b"), 12, 1)
H = get_variable(mf, "H")
variance = get_variable(mf, "variance")
z0 = reshape(get_variable(mf, "z0"), 72, 1)
vz0 = get_variable(mf, "vz0")
close(mf)

# The first time tic and toc are called, the function gets compiled, so we should ignore the returned time
tic()
for i = 1:1
    L, zend, Pend = kalcvf2NaN(data, lead, a, F, b, H, variance, z0, vz0)
end
toq()

# Call Kalman filter
iterations = 1000
tic()
for i = 1:iterations
    L, zend, Pend = kalcvf2NaN(data, lead, a, F, b, H, variance, z0, vz0)
end
time_elapsed = toq()

for node in ARGS
    println("node: $node, seconds: $time_elapsed")
end
