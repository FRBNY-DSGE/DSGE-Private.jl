using Test, Revise, NLsolve
using DSGE, ModelConstructors
include("../helpers/genweight.jl")
include("../helpers/sub2ind.jl")
include("../helpers/q_cons2.jl")
include("../helpers/gradient.jl")

include("../steadystate/steadystate.jl")

# Build model without triggering the full SS solve in the constructor
m = mBBQ()
m <= Setting(:compute_full_steadystate, false)

@testset "mBBQ ss_ump_new" begin
    results = ss_ump_new(m)

    @test results isa Dict{String, Any}

    for key in ["Output", "R_fc", "H_fc", "W_fc", "r_k", "mc", "v", "u",
                "V", "M", "f", "Profits_fc", "THETA", "NWb", "Profit_FI",
                "Ab", "Bb", "Cb", "n", "c_n_guess", "b_n_star", "c_a_guess",
                "b_a_star", "a_a_star", "psi_guess", "mu_dist", "AProb",
                "Value", "mutil_c", "Vb", "Va", "SS_stats", "grid"]
        @test haskey(results, key)
    end

    @test results["u"] > 0
    @test results["Output"] > 0
    @test sum(results["mu_dist"]) ≈ 1.0 atol = 1e-6
end
@testset "mBBQ _store_ss_results!" begin
    results = ss_ump_new(m)
    _store_ss_results!(m, results)

    # scalar SS parameters — non-NaN and sensible
    for sym in [:Output_star, :u_star, :w_bar_star, :R_a_star, :r_k_star,
                :mc_star, :n_star, :v_star, :f_star, :M_star, :V_star,
                :Lambda_star, :THETA_star, :NWb_star, :Profit_FI_star,
                :Profit_star, :Ab_star, :Bb_star, :Cb_star]
        @test !isnan(Float64(m[sym].value))
        @test m[sym].value > 0
    end

    # distributions sum to 1
    @test sum(m[:marginal_pdf_b_star].value)  ≈ 1.0 atol = 1e-6
    @test sum(m[:marginal_pdf_a_star].value)  ≈ 1.0 atol = 1e-6
    @test sum(m[:marginal_pdf_se_star].value) ≈ 1.0 atol = 1e-6
    @test sum(m[:mu_dist_star].value)         ≈ 1.0 atol = 1e-6

    # CDFs end at 1
    @test m[:marginal_cdf_b_star].value[end]  ≈ 1.0 atol = 1e-6
    @test m[:marginal_cdf_a_star].value[end]  ≈ 1.0 atol = 1e-6
    @test m[:marginal_cdf_se_star].value[end] ≈ 1.0 atol = 1e-6

    # value functions have correct size
    nb  = get_setting(m, :nb)
    na  = get_setting(m, :na)
    nse = get_setting(m, :nse)
    @test length(m[:Value_star].value)   == nb * na * nse
    @test length(m[:Vb_star].value)      == nb * na * nse
    @test length(m[:Va_star].value)      == nb * na * nse
    @test length(m[:mutil_c_star].value) == nb * na * nse

    # settings populated
    @test haskey(m.settings, :SS_stats)
    @test haskey(m.settings, :copula)
    @test haskey(m.settings, :anal_stats)
    @test haskey(m.settings, :Q_dists)

    # grids updated from SS solve
    @test haskey(m.grids, :K) && !isnan(m.grids[:K])
    @test haskey(m.grids, :L) && !isnan(m.grids[:L])
    @test haskey(m.grids, :N) && !isnan(m.grids[:N])
end
