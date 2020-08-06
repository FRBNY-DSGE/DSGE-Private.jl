using DSGE, Test, Plots, BenchmarkTools, Roots

# Uncomment following block to plot the consumption policy function and distribution
#=
m = HetDSGEGovDebt(; ref_dir =
                   joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
DSGE.steadystate!(m, tol = 5e-4, verbose = :high, doplots = true)
m = HetDSGEGovDebt(; ref_dir =
                   joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
DSGE.method3_steadystate!(m, tol = 5e-4, verbose = :high, doplots = true)
=#

#=
println("Old endogenous grid method")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.steadystate!(m, tol = 5e-4, verbose = :none, doplots = false)
end
=#

println("New endogenous grid method, interpolation")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none)
end

println("New endogenous grid method, interpolation, Anderson acceleration for Euler iteration")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none,
                              euler_anderson = true)
end

println("New endogenous grid method, interpolation, Anderson acceleration for Kolmogorov equation")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none,
                              kf_anderson = true)
end

println("New endogenous grid method, interpolation, Anderson acceleration for Euler iteration and Kolmogorov equation")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none,
                              euler_anderson = true, kf_anderson = true)

end

m11 = HetDSGEGovDebt("ss11"; ref_dir =
                     joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
DSGE.method3_steadystate!(m11; tol = 5e-4, doplots = false, verbose = :high)


## Using bracket methods from Roots. Generally slower than a direct implementation of a simple bisection search
#=
println("New endogenous grid method, interpolation, Bisection from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Bisection(),
                              βhi = .9)
end
=#
#=
println("New endogenous grid method, interpolation, A42 from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Roots.A42(),
                              βhi = .9)
end

println("New endogenous grid method, interpolation, AlefeldPotraShi from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Roots.AlefeldPotraShi(),
                              βhi = .9)
end

println("New endogenous grid method, interpolation, FalsePosition from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Roots.FalsePosition(),
                              βhi = .89)
end

println("New endogenous grid method, interpolation, Brent from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Roots.Brent(),
                              βhi = .89)
end
=#

## Using derivative-free methods from Roots. These generally don't work b/c they choose β which don't yield inner-loop convergence
#=
println("New endogenous grid method, interpolation, Secant method from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Order1(),
                              βhi = .9)
end

println("New endogenous grid method, interpolation, Steffenson method from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Order2(),
                              βhi = .8)
end
=#
