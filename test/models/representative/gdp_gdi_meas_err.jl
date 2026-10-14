module TestGDPGDIMeasErr   # isolated from other test files' globals

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
    return (corr = Σ[g, i] / sqrt(Σ[g, g] * Σ[i, i]), cov = Σ[g, i],
            var_gdp = Σ[g, g], var_gdi = Σ[i, i],
            qq_gdp = Q[m.exogenous_shocks[:gdp_sh], m.exogenous_shocks[:gdp_sh]],
            qq_gdi = Q[m.exogenous_shocks[:gdi_sh], m.exogenous_shocks[:gdi_sh]])
end

set_value!(m, k, v) = (m.parameters[m.keys[k]] = parameter(m[k], v))

@testset "GDP/GDI measurement errors: ρ_gdpvar is their correlation" begin
    @test DSGE.gdp_gdi_meas_err_loadings(0.5, 0.2, 0.0) == (1.0, 0.0)          # σ_gdi = 0: no correlation possible
    @test all(DSGE.gdp_gdi_meas_err_loadings(0.6, 0.2, 0.4) .≈ (0.8, 0.3))
    @test all(DSGE.gdp_gdi_meas_err_loadings(-0.75, 0.2, 0.4; legacy = true) .≈
              (1.0, -0.75 * 0.2^2))
    # Holding σ_gdp and σ_gdi fixed, converting ρ matches covariance but not GDP variance.
    ρ_legacy, σ_gdp, σ_gdi = -0.75, 0.2, 0.4
    ρ_converted = ρ_legacy * σ_gdp * σ_gdi
    old_a, old_b = DSGE.gdp_gdi_meas_err_loadings(ρ_legacy, σ_gdp, σ_gdi; legacy = true)
    new_a, new_b = DSGE.gdp_gdi_meas_err_loadings(ρ_converted, σ_gdp, σ_gdi)
    @test old_b * σ_gdi^2 ≈ new_b * σ_gdi^2
    @test old_a^2 * σ_gdp^2 + old_b^2 * σ_gdi^2 >
          new_a^2 * σ_gdp^2 + new_b^2 * σ_gdi^2

    fd = Date(2026, 9, 30)
    m104 = Model1002("ss104", custom_settings = [Setting(:date_forecast_start, fd), Setting(:date_conditional_end, fd),
                                                 Setting(:add_ant_markup_shocks_ind, 2), Setting(:fix_ant_markup_shocks_0, false),
                                                 Setting(:add_avgshortinfl, true)])
    for (m, regs) in ((m104, 1:get_setting(m104, :n_regimes)), (Model1010("ss20"), 1:1))
        set_value!(m, :ρ_gdpvar, -0.75)       # about the ss104 modal value
        @test !DSGE.use_legacy_gdp_gdi_meas_err_loadings(m)
        sys = compute_system(m)
        for r in regs
            mo = gdp_gdi_innovation_moments(m, sys, r)
            mo.qq_gdp > 0 && mo.qq_gdi > 0 || continue
            @test mo.corr ≈ -0.75 rtol = 1e-10
            @test mo.var_gdp ≈ mo.qq_gdp rtol = 1e-10
            @test mo.var_gdi ≈ mo.qq_gdi rtol = 1e-10
        end

        m <= Setting(:legacy_gdp_gdi_meas_err_loadings, true)
        @test DSGE.use_legacy_gdp_gdi_meas_err_loadings(m)
        sys_legacy = compute_system(m)
        for r in regs
            R = sys_legacy isa System ? sys_legacy[:RRR] : sys_legacy[r, :RRR]
            g = m.endogenous_states_augmented[:e_gdp_t]
            old_loading = R[g, m.exogenous_shocks[:gdi_sh]]
            mo = gdp_gdi_innovation_moments(m, sys_legacy, r)
            @test R[g, m.exogenous_shocks[:gdp_sh]] == 1.0
            if m isa Model1010
                @test old_loading ≈ -0.75 * m[:σ_gdp]^2
            end
            @test mo.cov ≈ old_loading * mo.qq_gdi
            @test mo.var_gdp ≈ mo.qq_gdp + old_loading^2 * mo.qq_gdi
        end
    end
end

end # module TestGDPGDIMeasErr
