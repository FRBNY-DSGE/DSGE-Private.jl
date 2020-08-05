using DSGE, Test, Plots, BenchmarkTools

m = HetDSGEGovDebt(; ref_dir = joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
m11 = HetDSGEGovDebt("ss11"; ref_dir = joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))

println("Old endogenous grid method")
@btime begin
    DSGE.steadystate!(m, tol = 3e-4, verbose = :none, doplots = false)
end

println("New endogenous grid method, interpolation")
@btime begin
    DSGE.method3_steadystate!(m; maxit = 20, tol = 3e-4, use_quadrature = false, doplots = false, verbose = :none)
end

println("New endogenous grid method, quadrature")
@btime begin
    DSGE.method3_steadystate!(m; maxit = 20, tol = 3e-4, use_quadrature = true, doplots = false, verbose = :none)
end

println("New endogenous grid method, interpolation")
@btime begin
    DSGE.method3_steadystate!(m; maxit = 20, tol = 3e-4, use_quadrature = false, doplots = false, verbose = :none)
end

println("Old endogenous grid method, lower tol")
@btime begin
    DSGE.steadystate!(m, tol = 1e-4, verbose = :none, doplots = false)
end

println("New endogenous grid method, interpolation, lower tol")
@btime begin
    DSGE.method3_steadystate!(m; tol = 1e-4, use_quadrature = false, doplots = false, verbose = :none)
end


println("New endogenous grid method, quadrature, lower tol")
@btime begin
    DSGE.method3_steadystate!(m; tol = 1e-4, use_quadrature = true, doplots = false, verbose = :none)
end

DSGE.method3_steadystate!(m11; tol = 1e-4, use_quadrature = false, doplots = false, verbose = :high)
