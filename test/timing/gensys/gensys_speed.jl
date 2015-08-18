# Add paths
using DSGE, MATLAB
path = dirname(@__FILE__)

# Load variables
mf = MatFile("$path/gensys.mat")
G0 = get_variable(mf, "G0")
G1 = get_variable(mf, "G1")
C = get_variable(mf, "C")
PSI = get_variable(mf, "PSI")
PIE = get_variable(mf, "PIE")
close(mf)

# The first time tic and toc are called, the function gets compiled, so we should ignore the returned time
tic()
for i = 1:1
    gensys(G0, G1, C, PSI, PIE, 1+1e-6)
end
toq()

# Call gensys
iterations = 1000
tic()
for i = 1:iterations
    gensys(G0, G1, C, PSI, PIE, 1+1e-6)
end
time_elapsed = toq()

println(time_elapsed)
