using DSGE, ModelConstructors, Test, Dates

# Model1002 ss205 (no AIT), built from the subspec alone and with the settings the 2026 ss205
# forecast driver passes (2025Q4 forecast). No data or saved output is needed.
ss205_driver_settings(fd) = Setting[Setting(:date_forecast_start, fd), Setting(:date_conditional_end, fd),
                                    Setting(:forecast_horizons, 40),
                                    Setting(:add_pseudo_gdp, true), Setting(:add_pseudo_corepce, true),
                                    Setting(:standard_shocks_mode_adjust, 1, false, "modeadj", ""),
                                    Setting(:standard_shocks_spread_adjust, 2, false, "spreadadj", ""),
                                    Setting(:add_altpolicy_pgap, true), Setting(:add_altpolicy_ygap, true),
                                    Setting(:add_shadow_taylor, true),
                                    Setting(:expected_ffr, [1, 2, 3, 4, 5, 6]), Setting(:add_taylor_rm, true)]

m205  = Model1002("ss205")
m205d = Model1002("ss205", custom_settings = ss205_driver_settings(Date(2025, 12, 31)))

@testset "ss205: the model solves as built (pgap/ygap settings)" begin
    for mm in (m205, m205d)
        # Used to throw KeyError(:pgap_type) from eqcond
        sys = compute_system(mm)
        @test length(sys.transitions) == get_setting(mm, :n_regimes)
        @test get_setting(mm, :pgap_type) == :flexible_ait
        @test get_setting(mm, :ygap_type) == :flexible_ait
    end
    # A value passed by the caller is kept
    m_own = Model1002("ss205", custom_settings = [Setting(:pgap_value, 0.5)])
    @test get_setting(m_own, :pgap_value) == 0.5
    # ss207 builds on ss205, so it solves now too
    @test compute_system(Model1002("ss207")) isa RegimeSwitchingSystem
end

@testset "ss205: parameter_groupings works without anticipated AIT shocks" begin
    # mon_anticipated_ait_shocks returned `false` when unset; looping over it built :σ_ait_r_mfalse
    @test isempty(DSGE.mon_anticipated_ait_shocks(m205))
    for mm in (m205, m205d, Model1002("ss10"))
        g = DSGE.parameter_groupings(mm)
        keys_in_groups = vcat([[p.key for p in v] for v in values(g)]...)
        @test !any(k -> occursin("false", string(k)), keys_in_groups)
        @test :σ_r_m in keys_in_groups
    end
    # Models that do have them keep the list
    m_ait = deepcopy(m205)
    m_ait <= Setting(:mon_anticipated_ait_shocks, [1, 2])
    @test DSGE.mon_anticipated_ait_shocks(m_ait) == [1, 2]
end

@testset "ss205: Taylor-rule policy shocks stay on after the 2022Q1 liftoff (no AIT)" begin
    for mm in (m205, m205d)
        m2p = get_setting(mm, :model2para_regime)
        nreg = get_setting(mm, :n_regimes)
        taylor = [:σ_r_m; [Symbol("σ_r_m$i") for i in 1:n_mon_anticipated_shocks(mm) if !mm[Symbol("σ_r_m$i")].fixed]]
        for k in taylor
            # Used to map regimes >= 10 (2022Q1 on) to a parameter regime fixed at 0
            @test all(m2p[k][r] == 1 for r in 1:nreg)
        end
        sys = compute_system(mm)
        rm = mm.exogenous_shocks[:rm_sh]
        @test all(sys[r, :QQ][rm, rm] ≈ mm[:σ_r_m].regimes[:value][1]^2 for r in 10:nreg)
        rm1 = mm.exogenous_shocks[:rm_shl1]
        @test all(sys[r, :QQ][rm1, rm1] > 0 for r in 10:nreg)
    end
    # The parameter vector layout is unchanged (the 2026 ss205 estimate loads)
    @test ModelConstructors.n_parameters_regime_switching(m205d.parameters) == 158
    # AIT subspecs keep switching the Taylor shocks off at liftoff
    m104 = Model1002("ss104", custom_settings = [Setting(:date_forecast_start, Date(2026, 9, 30)),
                                                 Setting(:date_conditional_end, Date(2026, 9, 30)),
                                                 Setting(:add_ant_markup_shocks_ind, 2),
                                                 Setting(:fix_ant_markup_shocks_0, false),
                                                 Setting(:add_avgshortinfl, true)])
    @test all(get_setting(m104, :model2para_regime)[:σ_r_m][r] == 2 for r in 10:get_setting(m104, :cur_model_regime))
end
