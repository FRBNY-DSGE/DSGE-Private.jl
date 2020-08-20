using DSGE, Test, Plots, BenchmarkTools, Roots, ModelConstructors

m = HetDSGEGovDebt(; ref_dir =
                   joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))

#=
println("Old endogenous grid method")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.steadystate!(m, tol = 5e-4, verbose = :none, doplots = false)
end

println("New endogenous grid method")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :high,
                              euler_anderson = false, kf_anderson = false)
end

println("New endogenous grid method, Anderson acceleration for Euler iteration")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none,
                              kf_anderson = false, kf_eigen = false)
end

println("New endogenous grid method, Anderson acceleration for Kolmogorov equation")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none,
                              euler_anderson = false, kf_eigen = false)
end

println("New endogenous grid method, Anderson acceleration for Kolmogorov equation, m = 3")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    m <= Setting(:m_anderson, 3)
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none,
                              euler_anderson = false, kf_eigen = false)
end

println("New endogenous grid method, Anderson acceleration for Euler iteration and Kolmogorov equation")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none, kf_eigen = false)
end

println("New endogenous grid method, Anderson acceleration for Euler iteration and Kolmogorov equation, m = 3")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    m <= Setting(:m_anderson, 3)
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none, kf_eigen = false)
end
=#

println("New endogenous grid method, Anderson acceleration for Euler iteration and eigenvector approach for Kolmogorov equation")
massert = HetDSGEGovDebt(; ref_dir =
                   joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
@test_throws AssertionError DSGE.method4_steadystate!(massert; kf_eigen = true, kf_anderson = true)
@btime begin
    m[:βstar] = NaN
    m <= Setting(:m_anderson, 1)
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none, kf_eigen = true, kf_anderson = false)
end

println("New endogenous grid method, Anderson acceleration for Euler iteration with damping = 1.2, and eigenvector approach for Kolmogorov equation")
@btime begin
    m[:βstar] = NaN
    m <= Setting(:m_anderson, 1)
    m <= Setting(:β_anderson, 1.2)
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none, kf_eigen = true, kf_anderson = false)
end

println("New endogenous grid method, Anderson acceleration for Euler iteration with damping = .8, and eigenvector approach for Kolmogorov equation")
@btime begin
    m[:βstar] = NaN
    m <= Setting(:m_anderson, 1)
    m <= Setting(:β_anderson, .8)
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none, kf_eigen = true, kf_anderson = false)
end

println("New endogenous grid method, eigenvector approach for Kolmogorov equation")
@btime begin
    m[:βstar] = NaN
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none, kf_eigen = true, kf_anderson = false, euler_anderson = false)
end

m11 = HetDSGEGovDebt("ss11"; ref_dir =
                     joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
# DSGE.method4_steadystate!(m11; tol = 5e-4, doplots = false, verbose = :high, kf_eigen = true, kf_anderson = false)

# Uncomment following block to plot the consumption policy function and distribution
#=
m = HetDSGEGovDebt(; ref_dir =
                   joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
DSGE.steadystate!(m, tol = 5e-4, verbose = :high, doplots = true)
m = HetDSGEGovDebt(; ref_dir =
                   joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
DSGE.method4_steadystate!(m, tol = 5e-4, verbose = :high, doplots = true)
=#

## Using bracket methods from Roots. Have to specify lower βlo and βhi that work, but seems to be non-robust. Often land on βs yielding
## poor Euler iteration convergence.
#=
println("New endogenous grid method, tighter bisection")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none, kf_eigen = true, kf_anderson = false,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), βhi = .91)
end

println("New endogenous grid method, Bisection from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none, kf_eigen = true, kf_anderson = false,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Bisection(),
                              βhi = .91)
end

println("New endogenous grid method, A42 from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none, kf_eigen = true, kf_anderson = false,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Roots.A42(),
                              βhi = .91)
end

println("New endogenous grid method, AlefeldPotraShi from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none, kf_eigen = true, kf_anderson = false,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Roots.AlefeldPotraShi(),
                              βhi = .91)
end

println("New endogenous grid method, FalsePosition from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none, kf_eigen = true, kf_anderson = false,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Roots.FalsePosition(),
                              βhi = .91)
end

println("New endogenous grid method, Brent from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none, kf_eigen = true, kf_anderson = false,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Roots.Brent(),
                              βhi = .91)
end
=#
## Using derivative-free methods from Roots. These generally don't work b/c they choose β which don't yield inner-loop convergence
#=
println("New endogenous grid method, Secant method from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Order1(),
                              βhi = .9)
end

println("New endogenous grid method, Steffenson method from Roots")
@btime begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none,
                              βlo = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), roots_algorithm = Order2(),
                              βhi = .8)
end
=#
