using DSGE, NLsolve, Test, BenchmarkTools

#m=GHLS()
#@show "anderson first"
#@btime alpha_and = solve($m; parallel = false, anderson = true)

#const to = TimerOutput()
function speeds()
m = GHLS()
@btime alpha_iter = solve($m; parallel = true, anderson = false)
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
