using DSGE, NLsolve, Test, BenchmarkTools, TimerOutputs

#m=GHLS()
#@show "anderson first"
#@btime alpha_and = solve($m; parallel = false, anderson = true)

#const to = TimerOutput()
function speeds()
m = GHLS()
@show "Iteration next"
#@timeit to "solve" solve(m; parallel = false, anderson = false)
@btime alpha_iter = solve($m; parallel = false, anderson = false)
end
speeds()
#=
m=GHLS()
alpha_and = solve(m; parallel = false, anderson = true)

m=GHLS()
alpha_iter = solve(m; parallel = false, anderson = false)

#@show alpha_and
#@show alpha_iter

@test alpha_and ≈ alpha_iter
@show alpha_and ≈ alpha_iter
=#
