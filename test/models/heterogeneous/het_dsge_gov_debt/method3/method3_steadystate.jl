using DSGE, Test, Plots, BenchmarkTools, Roots, ModelConstructors

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

println("New endogenous grid method, interpolation, Anderson acceleration for Euler iteration and Kolmogorov equation, AlefeldPotraShi")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none, roots_algorithm = Roots.AlefeldPotraShi(),
                              euler_anderson = true, kf_anderson = true, βhi = .89)

end

println("New endogenous grid method, interpolation, Anderson acceleration for Euler iteration and Kolmogorov equation, Brent")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none, roots_algorithm = Roots.Brent(),
                              euler_anderson = true, kf_anderson = true, βhi = .89)

end

m11 = HetDSGEGovDebt("ss11"; ref_dir =
                     joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
DSGE.method3_steadystate!(m11; tol = 5e-4, doplots = false, verbose = :high)

# Uncomment following block to plot the consumption policy function and distribution
#=
m = HetDSGEGovDebt(; ref_dir =
                   joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
DSGE.steadystate!(m, tol = 5e-4, verbose = :high, doplots = true)
m = HetDSGEGovDebt(; ref_dir =
                   joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
DSGE.method3_steadystate!(m, tol = 5e-4, verbose = :high, doplots = true)
=#

## Using bracket methods from Roots. Have to specify lower βlo and βhi that work, or the methods won't work, but if these
## values are specified, then AlefeldPotraShi, FalsePosition, and Brent are faster
## (with AlefeldPotraShi fastest, Brent slowest of these 3). However, in combination with Anderson acceleration, still slower.
#=
println("New endogenous grid method, interpolation, tighter bisection")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), βhi = .9)
end

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
                              βhi = .9)
end

println("New endogenous grid method, interpolation, Brent from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method3_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Roots.Brent(),
                              βhi = .9)
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
