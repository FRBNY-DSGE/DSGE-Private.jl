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
