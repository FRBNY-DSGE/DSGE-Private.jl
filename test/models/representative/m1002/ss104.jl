using DSGE, ModelConstructors, Test, Dates

# Model1002 ss104, configured as the production forecast builds it (2026Q3 forecast,
# two anticipated mark-up shocks switched on). No data or saved output is needed.
function build_ss104(; fcast_date = Date(2026, 9, 30), extra = Setting[])
    Model1002("ss104", custom_settings = vcat([Setting(:date_forecast_start, fcast_date),
                                               Setting(:date_conditional_end, fcast_date),
                                               Setting(:add_ant_markup_shocks_ind, 2),
                                               Setting(:fix_ant_markup_shocks_0, false),
                                               Setting(:add_avgshortinfl, true)], extra))
end

# Value of regime-switching parameter `k` in model regime `r`
function regime_value(m, k::Symbol, r::Int)
    p = m[k]
    haskey(p.regimes, :value) || return p.value
    m2p = get_setting(m, :model2para_regime)
    preg = haskey(m2p, k) ? m2p[k][r] : r
    return p.regimes[:value][preg]
end

m = build_ss104()
sys = compute_system(m)
exo = m.exogenous_shocks
n_regimes = get_setting(m, :n_regimes)

@testset "ss104 (bugs doc §1.1): QQ holds variances for the anticipated mark-up shocks" begin
    for r in 1:n_regimes, i in 1:get_setting(m, :add_ant_markup_shocks_ind)
        sh = exo[Symbol("λ_f_ant_sh$i")]
        @test sys[r, :QQ][sh, sh] ≈ regime_value(m, Symbol("σ_λ_f$i"), r)^2 atol = 1e-15
    end
    # Same convention as the contemporaneous mark-up shock
    for r in 1:n_regimes
        @test sys[r, :QQ][exo[:λ_f_sh], exo[:λ_f_sh]] ≈ regime_value(m, :σ_λ_f, r)^2
    end
end
