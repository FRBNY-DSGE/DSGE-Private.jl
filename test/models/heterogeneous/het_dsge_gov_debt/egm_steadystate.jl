using DSGE, Test, JLD2
using BenchmarkTools, ModelConstructors

generate_output = false
time_methods = false
doplots = false

for ss in ["ss11", "ss15"]

    m = HetDSGEGovDebt(ss; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))

    @testset "EGM Steady State Solution for HetDSGEGovDebt for $(ss)" begin
        @test_throws AssertionError DSGE.steadystate!(m; kf_eigen = true, kf_anderson = true)
        DSGE.steadystate!(m)

        if generate_output
            JLD2.jldopen(joinpath(dirname(@__FILE__), "../../../reference/het_dsge_gov_debt_$(ss)_egm_steadystate.jld2"),
                         true, true, true, IOStream) do file
                             write(file, "lstar", m[:lstar].value)
                             write(file, "Dstar", m[:Dstar].value)
                             write(file, "cstar", m[:cstar].value)
                             write(file, "beta", m[:βstar].value)
                         end
        else
            output = JLD2.jldopen(joinpath(dirname(@__FILE__), "../../../reference/het_dsge_gov_debt_$(ss)_egm_steadystate.jld2"), "r")
            @test all(m[:lstar].value .≈ output["lstar"])
            @test all(m[:Dstar].value .≈ output["Dstar"])
            @test all(m[:cstar].value .≈ output["cstar"])
            @test m[:βstar].value ≈ output["beta"]
        end
    end
end

if time_methods

    println("New endogenous grid method, Anderson acceleration for Euler iteration")
    @btime begin
        m[:βstar] = NaN
        DSGE.steadystate!(m; tol = 1e-4, doplots = false, verbose = :none,
                          kf_eigen = true)
    end

    println("New endogenous grid method, Anderson acceleration for Kolmogorov equation")
    @btime begin
        m[:βstar] = NaN
        DSGE.steadystate!(m; tol = 1e-4, doplots = false, verbose = :none,
                          euler_anderson = false, kf_eigen = false)
    end

    println("New endogenous grid method, Anderson acceleration for Kolmogorov equation, m = 3")
    @btime begin
        m[:βstar] = NaN
        m <= Setting(:m_anderson, 3)
        DSGE.steadystate!(m; tol = 1e-4, doplots = false, verbose = :none,
                          euler_anderson = false, kf_eigen = false)
    end

    println("New endogenous grid method, Anderson acceleration for Euler iteration and Kolmogorov equation")
    @btime begin
        m[:βstar] = NaN
        DSGE.steadystate!(m; tol = 1e-4, doplots = false, verbose = :none, kf_eigen = false)
    end

    println("New endogenous grid method, Anderson acceleration for Euler iteration and Kolmogorov equation, m = 3")
    @btime begin
        m[:βstar] = NaN
        m <= Setting(:m_anderson, 3)
        DSGE.steadystate!(m; tol = 1e-4, doplots = false, verbose = :none, kf_eigen = false)
    end

    println("New endogenous grid method, Anderson acceleration for Euler iteration and eigenvector approach for Kolmogorov equation")
    @btime begin
        m[:βstar] = NaN
        m <= Setting(:m_anderson, 1)
        DSGE.steadystate!(m; tol = 1e-4, doplots = false, verbose = :none, kf_eigen = true, kf_anderson = false)
    end

    println("New endogenous grid method, Anderson acceleration for Euler iteration with damping = 1.2, and eigenvector approach for Kolmogorov equation")
    @btime begin
        m[:βstar] = NaN
        m <= Setting(:m_anderson, 1)
        m <= Setting(:β_anderson, 1.2)
        DSGE.steadystate!(m; tol = 1e-4, doplots = false, verbose = :none, kf_eigen = true, kf_anderson = false)
    end


    println("New endogenous grid method, Anderson acceleration for Euler iteration with damping = .8, and eigenvector approach for Kolmogorov equation")
    @btime begin
        m[:βstar] = NaN
        m <= Setting(:m_anderson, 1)
        m <= Setting(:β_anderson, .8)
        DSGE.steadystate!(m; tol = 1e-4, doplots = false, verbose = :none, kf_eigen = true, kf_anderson = false)
    end

    println("New endogenous grid method, eigenvector approach for Kolmogorov equation")
    @btime begin
        m[:βstar] = NaN
        DSGE.steadystate!(m; tol = 1e-4, doplots = false, verbose = :none, kf_eigen = true, kf_anderson = false, euler_anderson = false)
    end

    println("New endogenous grid method, Anderson acceleration for Euler iteration, second fixed point for consumption policy")
    @btime begin
        m1[:βstar] = NaN
        m1 <= Setting(:cas_fixedpoint, true)
        DSGE.steadystate!(m1; tol = 1e-4, doplots = false, verbose = :none,
                          kf_eigen = true)
    end
end

if doplots
    DSGE.steadystate!(m; tol = 1e-4, doplots = true, verbose = :high, kf_eigen = true, kf_anderson = false)
end
