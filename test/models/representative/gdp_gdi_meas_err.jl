using DSGE, ModelConstructors, Test, Dates, LinearAlgebra

# The GDP and GDI measurement errors e_gdp, e_gdi are AR(1) states whose innovations load on gdp_sh and
# gdi_sh (QQ holds σ_gdp², σ_gdi²). ρ_gdpvar is documented and given a prior as their correlation, so
# the innovation covariance RRR·QQ·RRR' must have correlation ρ_gdpvar and GDP variance QQ[gdp_sh].
# The loading used to be ρ_gdpvar·σ_gdp² (estimation-AI/NUMERICAL-INVARIANTS.md §1.4 C), which gives a
# correlation of about 0.01·ρ_gdpvar at the prior.
function gdp_gdi_innovation_moments(m, sys, r)
    R, Q = sys isa System ? (sys[:RRR], sys[:QQ]) : (sys[r, :RRR], sys[r, :QQ])
    Σ = R * Q * R'
    g, i = m.endogenous_states_augmented[:e_gdp_t], m.endogenous_states_augmented[:e_gdi_t]
    return (corr = Σ[g, i] / sqrt(Σ[g, g] * Σ[i, i]), var_gdp = Σ[g, g], var_gdi = Σ[i, i],
            qq_gdp = Q[m.exogenous_shocks[:gdp_sh], m.exogenous_shocks[:gdp_sh]],
            qq_gdi = Q[m.exogenous_shocks[:gdi_sh], m.exogenous_shocks[:gdi_sh]])
end

set_value!(m, k, v) = (m.parameters[m.keys[k]] = parameter(m[k], v))

@testset "GDP/GDI measurement errors: ρ_gdpvar is their correlation" begin
    @test DSGE.gdp_gdi_meas_err_loadings(0.5, 0.2, 0.0) == (1.0, 0.0)          # σ_gdi = 0: no correlation possible
    @test all(DSGE.gdp_gdi_meas_err_loadings(0.6, 0.2, 0.4) .≈ (0.8, 0.3))

    fd = Date(2026, 9, 30)
    m104 = Model1002("ss104", custom_settings = [Setting(:date_forecast_start, fd), Setting(:date_conditional_end, fd),
                                                 Setting(:add_ant_markup_shocks_ind, 2), Setting(:fix_ant_markup_shocks_0, false),
                                                 Setting(:add_avgshortinfl, true)])
    for (m, regs) in ((m104, 1:get_setting(m104, :n_regimes)), (Model1010("ss20"), 1:1))
        set_value!(m, :ρ_gdpvar, -0.75)       # about the ss104 modal value
        sys = compute_system(m)
        for r in regs
            mo = gdp_gdi_innovation_moments(m, sys, r)
            mo.qq_gdp > 0 && mo.qq_gdi > 0 || continue
            @test mo.corr ≈ -0.75 rtol = 1e-10
            @test mo.var_gdp ≈ mo.qq_gdp rtol = 1e-10
            @test mo.var_gdi ≈ mo.qq_gdi rtol = 1e-10
        end
    end
end
