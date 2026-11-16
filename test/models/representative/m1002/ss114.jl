module TestSS114

using DSGE, ModelConstructors, Dates, Test

const m = Model1002("ss114", custom_settings = [
    Setting(:date_forecast_start, Date(2026, 9, 30)),
    Setting(:date_conditional_end, Date(2026, 9, 30)),
])

function anticipated_entries(m, reg)
    _, Γ1, _, Ψ, _ = DSGE.zlb_rule_eqcond(m, reg)
    eq, endo, exo = m.equilibrium_conditions, m.endogenous_states, m.exogenous_shocks
    [(Γ1[eq[i == 1 ? :eq_rm : Symbol("eq_rml$(i-1)")], endo[Symbol("rm_tl$i")]],
      Ψ[eq[Symbol("eq_rml$i")], exo[Symbol("rm_shl$i")]]) for i in 1:n_mon_anticipated_shocks(m)]
end

@testset "ss114 keeps anticipated Taylor shocks only in the 2021Q4 belief ZLB" begin
    @test get_setting(m, :zlb_rule_keep_mon_anticipated_shocks_regimes) == [9, 10]
    @test get_setting(m, :regime_eqcond_info)[9].alternative_policy.key == :zlb_rule
    @test get_setting(m, :regime_eqcond_info)[10].alternative_policy.key == :default_policy
    @test get_setting(m, :n_regimes) > 10

    for reg in (5, 8, 9, 10, 11)
        entries = anticipated_entries(m, reg)
        @test length(entries) == n_mon_anticipated_shocks(m)
        @test all(x == (reg in (9, 10) ? (1.0, 1.0) : (0.0, 0.0)) for x in entries)
    end

    # The shared setting, rather than an ss114-only code path, controls the ZLB rule.
    other = deepcopy(m)
    other <= Setting(:zlb_rule_keep_mon_anticipated_shocks_regimes, [8])
    @test all(==((1.0, 1.0)), anticipated_entries(other, 8))
    @test all(==((0.0, 0.0)), anticipated_entries(other, 9))
end

end # module
