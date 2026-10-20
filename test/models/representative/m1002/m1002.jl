using DSGE, ModelConstructors, HDF5, Test, Dates
path = dirname(@__FILE__)

@testset "Model1002 parameter groupings across subspecs" begin
    formerly_ungrouped = ("ss62", "ss67", "ss68", "ss69", "ss70", "ss71", "ss72",
        "ss73", "ss74", "ss75", "ss76", "ss77", "ss78", "ss80", "ss82", "ss83",
        "ss86", "ss88", "ss89", "ss90", "ss91", "ss92", "ss94", "ss95",
        "ss96", "ss98", "ss99")
    for ss in formerly_ungrouped
        m = Model1002(ss)
        groups = DSGE.parameter_groupings(m)
        keys = [p.key for group in Base.values(groups) for p in group]
        @test length(keys) == length(unique(keys))
        @test !isempty(keys)
    end

    m207 = Model1002("ss207")
    @test m207[:κ_pce].fixed
    @test all(p.key != :κ_pce for group in Base.values(DSGE.parameter_groupings(m207)) for p in group)

    m207_history = Model1002("ss207"; custom_settings=[
        Setting(:date_forecast_start, Date(2026, 9, 30)),
        Setting(:date_conditional_end, Date(2026, 9, 30))])
    sample_end = Date(2026, 6, 30)
    periods = DSGE.subtract_quarters(sample_end, date_presample_start(m207_history)) + 1
    indices, _, splice = DSGE.zlb_plus_regime_indices(m207_history, zeros(1, periods))
    @test splice
    @test all(!isempty, indices)
    @test last(indices[end]) == periods
    @test length(indices) == DSGE.likelihood_regime_count(m207_history, sample_end,
        get_setting(m207_history, :n_regimes)) + 1
end

### Model
model = Model1002("ss10")

### Parameters

@testset "Test parameter bounds checking" begin
    for θ in model.parameters
        @test isa(θ, AbstractParameter)
        if !θ.fixed
            (left, right) = θ.valuebounds
            @test left < θ.value < right
        end
    end
end


### Model indices

@testset "Checking model indices" begin
    # Endogenous states
    endo = model.endogenous_states
    @test length(endo) == 68
    @test endo[:Ez_t] == 60

    # Exogenous shocks
    exo = model.exogenous_shocks
    @test length(exo) == 24
    @test exo[:corepce_sh] == 16

    # Expectation shocks
    ex = model.expected_shocks
    @test length(ex) == 13
    @test ex[:EL_f_sh] == 12

    # Equations
    eq = model.equilibrium_conditions
    @test length(eq) == 68
    @test eq[:eq_Ez] == 59

    # Additional states
    endo_new = model.endogenous_states_augmented
    @test length(endo_new) == 16
    @test endo_new[:y_t1] == 69

    # Observables
    obs = model.observables
    @test length(obs) == 19
    @test obs[:obs_tfp] == 12
end

### Equilibrium conditions
Γ0, Γ1, C, Ψ, Π = eqcond(model)

# Transition and measurement equations
TTT, RRR, CCC = solve(model)
meas = measurement(model, TTT, RRR, CCC)

### Pseudo-measurement equation
pseudo_meas = pseudo_measurement(model, TTT, RRR, CCC)

@testset "Check eqcond and state-space dimensions" begin
    # Matrices are of expected dimensions
    @test size(Γ0) == (68, 68)
    @test size(Γ1) == (68, 68)
    @test size(C) == (68,)
    @test size(Ψ) == (68, 24)
    @test size(Π) == (68, 13)

    @test size(meas[:ZZ]) == (19,84)
    @test size(meas[:DD]) == (19,)
    @test size(meas[:QQ]) == (24,24)
    @test size(meas[:EE]) == (19,19)

    @test size(TTT) == (84,84)
    @test size(RRR) == (84,24)
    @test size(CCC) == (84,)
end

G0_exp, G1_exp, C_exp, PSI_exp, PIE_exp = h5open("$path/eqcond.h5") do file
    read(file, "G0"), read(file, "G1"), read(file, "C"),
    read(file, "PSI"), read(file, "PIE")
end
TTT_exp, RRR_exp, CCC_exp = h5open("$path/transition.h5") do file
    read(file, "TTT"), read(file, "RRR"), read(file, "CCC")
end
Z_exp, D_exp, Q_exp, E_exp = h5open("$path/measurement.h5") do file
    read(file, "ZZ"), read(file, "DD"), read(file, "QQ"), read(file, "EE")
end
Z_pseudo_exp, D_pseudo_exp = h5open("$path/pseudo_measurement.h5") do file
    read(file, "ZZ_pseudo"), read(file, "DD_pseudo")
end

@testset "Compare eqcond, transition, measurement, and pseudo-measurement matrices against reference matrices" begin
    @test @test_matrix_approx_eq Γ0 G0_exp
    @test @test_matrix_approx_eq Γ1 G1_exp
    @test @test_matrix_approx_eq C C_exp
    @test @test_matrix_approx_eq Ψ PSI_exp
    @test @test_matrix_approx_eq Π PIE_exp

    @test @test_matrix_approx_eq TTT TTT_exp
    @test @test_matrix_approx_eq RRR RRR_exp
    @test @test_matrix_approx_eq CCC CCC_exp

    @test @test_matrix_approx_eq meas[:ZZ] Z_exp
    @test @test_matrix_approx_eq meas[:DD] D_exp
    @test @test_matrix_approx_eq meas[:QQ] Q_exp
    @test @test_matrix_approx_eq meas[:EE] E_exp

    @test @test_matrix_approx_eq pseudo_meas[:ZZ_pseudo] Z_pseudo_exp
    @test @test_matrix_approx_eq pseudo_meas[:DD_pseudo] D_pseudo_exp
end

@testset "Check ss62 is properly implemented" begin
    m = Model1002("ss62")
    for (i, d) in enumerate([date_presample_start(m), Date(2020, 3, 31), Date(2020, 6, 30), Date(2020, 9, 30),
                             Date(2020, 12, 31), Date(2021, 3, 31)])
        @test get_setting(m, :regime_dates)[i] == d
    end
    sys = compute_system(m)
    exo = m.exogenous_shocks
    inds1 = 1:24 # standard shock indices
    inds2 = vcat(1:14, 17:24) # standard shock indices excluding measurement error
    inds3 = 25:30 # covid shock indices
    for k in [:σ_biidc, :σ_φ, :σ_ziid]
        @test haskey(m[k].regimes, :prior)
        @test get(regime_prior(m[k], 1)).ν == get(regime_prior(m[k], 2)).ν
        @test get(regime_prior(m[k], 1)).ν == get(regime_prior(m[k], 3)).ν
        @test get(regime_prior(m[k], 1)).τ == get(regime_prior(m[k], 2)).τ
        @test get(regime_prior(m[k], 1)).τ != get(regime_prior(m[k], 3)).τ
    end

    for i in 2:get_setting(m, :n_regimes)
        @test sys[1, :TTT] ≈ sys[i, :TTT]
        @test sys[1, :CCC] ≈ sys[i, :CCC]
        @test sys[1, :ZZ]  ≈ sys[i, :ZZ]
        @test sys[1, :DD]  ≈ sys[i, :DD]

        if i >= 5
            @test sys[1, :QQ][inds1, inds1] ≈ sys[i, :QQ][inds1, inds1]
        end
        if i >= 4
            @test sys[1, :QQ][inds2, inds2] ≈ sys[i, :QQ][inds2, inds2]
        end
        if 2 <= i <= 4
            @test sys[i, :QQ][exo[:biidc_sh], exo[:biidc_sh]] == 16
            @test sys[i, :QQ][exo[:ziid_sh], exo[:ziid_sh]] == 25.
            @test sys[i, :QQ][exo[:φ_sh], exo[:φ_sh]] == 160000.
            @test (sys[1, :QQ][exo[:corepce_sh], exo[:corepce_sh]] * 10^2) == sys[i, :QQ][exo[:corepce_sh], exo[:corepce_sh]]
            @test (sys[1, :QQ][exo[:gdpdef_sh], exo[:gdpdef_sh]] * 10^2) == sys[i, :QQ][exo[:gdpdef_sh], exo[:gdpdef_sh]]
        elseif i == 5
            @test sys[i, :QQ][exo[:biidc_sh], exo[:biidc_sh]] == .04^2
            @test sys[i, :QQ][exo[:ziid_sh], exo[:ziid_sh]] == .05^2
            @test sys[i, :QQ][exo[:φ_sh], exo[:φ_sh]] == 16.
        end

        if 2 <= i <= 3
            @test (sys[1, :QQ][exo[:rm_sh], exo[:rm_sh]] * 10^2) == sys[i, :QQ][exo[:rm_sh], exo[:rm_sh]]
            @test (sys[1, :QQ][exo[:b_sh], exo[:b_sh]] * .25^2) == sys[i, :QQ][exo[:b_sh], exo[:b_sh]]
            @test (sys[1, :QQ][exo[:g_sh], exo[:g_sh]] * .25^2) == sys[i, :QQ][exo[:g_sh], exo[:g_sh]]
        end

        @test sys[i, :QQ][exo[:ziid_shl1], exo[:ziid_shl1]] == 0.
        @test sys[i, :QQ][exo[:φ_shl1], exo[:φ_shl1]] == 0.
        if 2 <= i <= 3 || i >= 6
            @test sys[i, :QQ][exo[:biidc_shl1], exo[:biidc_shl1]] == 0.
        else
            @test sys[i, :QQ][exo[:biidc_shl1], exo[:biidc_shl1]] == 16.
        end
    end
end

@testset "Check ss63 is properly implemented" begin
    m = Model1002("ss63")
    for (i, d) in enumerate([date_presample_start(m), Date(2020, 3, 31), Date(2020, 6, 30), Date(2020, 9, 30),
                             Date(2020, 12, 31), Date(2021, 3, 31)])
        @test get_setting(m, :regime_dates)[i] == d
    end
    sys = compute_system(m)
    exo = m.exogenous_shocks
    inds1 = 1:24 # standard shock indices
    inds2 = vcat(1:14, 17:24) # standard shock indices excluding measurement error
    inds3 = 25:30 # covid shock indices
    for k in [:σ_biidc, :σ_φ, :σ_ziid]
        @test haskey(m[k].regimes, :prior)
        @test get(regime_prior(m[k], 1)).ν == get(regime_prior(m[k], 2)).ν
        @test get(regime_prior(m[k], 1)).ν == get(regime_prior(m[k], 3)).ν
        @test get(regime_prior(m[k], 1)).τ == get(regime_prior(m[k], 2)).τ
        @test get(regime_prior(m[k], 1)).τ != get(regime_prior(m[k], 3)).τ
    end

    for i in 2:get_setting(m, :n_regimes)
        @test sys[1, :TTT] ≈ sys[i, :TTT]
        @test sys[1, :CCC] ≈ sys[i, :CCC]
        @test sys[1, :ZZ]  ≈ sys[i, :ZZ]
        @test sys[1, :DD]  ≈ sys[i, :DD]

        if i >= 5
            @test sys[1, :QQ][inds1, inds1] ≈ sys[i, :QQ][inds1, inds1]
        end
        if i >= 4
            @test sys[1, :QQ][inds2, inds2] ≈ sys[i, :QQ][inds2, inds2]
        end
        if 2 <= i <= 4
            @test sys[i, :QQ][exo[:biidc_sh], exo[:biidc_sh]] == 16
            @test sys[i, :QQ][exo[:ziid_sh], exo[:ziid_sh]] == 25.
            @test sys[i, :QQ][exo[:φ_sh], exo[:φ_sh]] == 160000.
            @test sys[1, :QQ][exo[:corepce_sh], exo[:corepce_sh]] * 10^2 ≈ sys[i, :QQ][exo[:corepce_sh], exo[:corepce_sh]]
            @test sys[1, :QQ][exo[:gdpdef_sh], exo[:gdpdef_sh]] * 10^2 ≈ sys[i, :QQ][exo[:gdpdef_sh], exo[:gdpdef_sh]]
        elseif i == 5
            @test sys[i, :QQ][exo[:biidc_sh], exo[:biidc_sh]] == .04^2
            @test sys[i, :QQ][exo[:ziid_sh], exo[:ziid_sh]] == .05^2
            @test sys[i, :QQ][exo[:φ_sh], exo[:φ_sh]] == 16.
        end

        if 2 <= i <= 3
            @test sys[1, :QQ][exo[:rm_sh], exo[:rm_sh]] * 10^2 == sys[i, :QQ][exo[:rm_sh], exo[:rm_sh]]
            @test sys[1, :QQ][exo[:g_sh], exo[:g_sh]] == sys[i, :QQ][exo[:g_sh], exo[:g_sh]]
            @test sys[1, :QQ][exo[:b_sh], exo[:b_sh]] == sys[i, :QQ][exo[:b_sh], exo[:b_sh]]
        end

        @test sys[i, :QQ][exo[:ziid_shl1], exo[:ziid_shl1]] == 0.
        @test sys[i, :QQ][exo[:φ_shl1], exo[:φ_shl1]] == 0.
        if 2 <= i <= 3 || i >= 6
            @test sys[i, :QQ][exo[:biidc_shl1], exo[:biidc_shl1]] == 0.
        else
            @test sys[i, :QQ][exo[:biidc_shl1], exo[:biidc_shl1]] == 16.
        end
    end
end
