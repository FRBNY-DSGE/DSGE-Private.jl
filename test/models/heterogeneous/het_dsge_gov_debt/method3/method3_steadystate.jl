using DSGE, Test, Plots, BenchmarkTools, Roots

# m = HetDSGEGovDebt(; ref_dir =
#                    joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
# DSGE.steadystate!(m, tol = 5e-4, verbose = :high, doplots = true)
# m = HetDSGEGovDebt(; ref_dir =
#                    joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
# DSGE.method3_steadystate!(m, tol = 5e-4, verbose = :high, doplots = true, use_quadrature = false)
# DSGE.method3_steadystate!(m, tol = 5e-4, verbose = :high, doplots = false, use_quadrature = false,
                                   # βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue),
                                   # βhi = .9)

#=
m = HetDSGEGovDebt(; ref_dir =
                   joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
DSGE.method3_steadystate!(m, tol = 5e-4, verbose = :high, doplots = true, use_quadrature = true)
=#

println("Old endogenous grid method")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.steadystate!(m, tol = 5e-4, verbose = :none, doplots = false)
end

#=
println("New endogenous grid method, interpolation")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4 use_quadrature = false, doplots = false, verbose = :none,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue),
                              βhi = .9)
end

println("New endogenous grid method, interpolation, Bisection from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, use_quadrature = false, doplots = false, verbose = :none,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Bisection(),
                              βhi = .9)
end
=#
#=
println("New endogenous grid method, interpolation, Secant method from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, use_quadrature = false, doplots = false, verbose = :none,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Order1(),
                              βhi = .9)
end

println("New endogenous grid method, interpolation, Steffenson method from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, use_quadrature = false, doplots = false, verbose = :none,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Order2(),
                              βhi = .8)
end
=#

println("New endogenous grid method, interpolation, Anderson acceleration for Euler iteration")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, use_quadrature = false, doplots = false, verbose = :none,
                              euler_anderson = true)
end

#=
println("New endogenous grid method, interpolation, Anderson acceleration for Kolmogorov equation")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, use_quadrature = false, doplots = false, verbose = :none,
                              kf_anderson = true)
end
=#
#=
println("New endogenous grid method, quadrature")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, use_quadrature = true, doplots = false, verbose = :none)
end

m11 = HetDSGEGovDebt("ss11"; ref_dir =
                     joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
DSGE.method3_steadystate!(m11; tol = 5e-4, use_quadrature = false, doplots = false, verbose = :high)
=#
