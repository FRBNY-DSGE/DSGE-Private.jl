m = AnSchorfheide()
system = compute_system(m)
Φ, Ψ, F_ϵ, F_u = DSGE.compute_system_function(system)
zero_sys = DSGE.zero_system_constants(system)

@testset "Check System, Transition, Measurement, and PseduoMeasurement access functions" begin
    @test typeof(system) == System{Float64}
    @test typeof(system[:transition]) == Transition{Float64}
    @test typeof(system[:measurement]) == Measurement{Float64}
    @test typeof(system[:pseudo_measurement]) == PseudoMeasurement{Float64}
    @test typeof(system[:transition][:TTT]) == Matrix{Float64}
    @test typeof(system[:transition][:RRR]) == Matrix{Float64}
    @test typeof(system[:transition][:CCC]) == Vector{Float64}
    @test typeof(system[:measurement][:ZZ]) == Matrix{Float64}
    @test typeof(system[:measurement][:DD]) == Vector{Float64}
    @test typeof(system[:measurement][:QQ]) == Matrix{Float64}
    @test typeof(system[:measurement][:EE]) == Matrix{Float64}
    @test typeof(system[:pseudo_measurement][:ZZ_pseudo]) == Matrix{Float64}
    @test typeof(system[:pseudo_measurement][:DD_pseudo]) == Vector{Float64}
end

@testset "Check miscellaneous functions acting on System types" begin
    @test sum(zero_sys[:CCC]) == 0.
    @test sum(zero_sys[:DD]) == 0.
    @test sum(zero_sys[:DD_pseudo]) == 0.
    @test Φ(ones(size(system[:transition][:TTT],2)), zeros(size(system[:transition][:RRR],2))) ==
        system[:transition][:TTT] * ones(size(system[:transition][:TTT],2))
    @test Ψ(ones(size(system[:transition][:TTT],2))) ==
        system[:measurement][:ZZ] * ones(size(system[:transition][:TTT],2)) + system[:measurement][:DD]
    @test F_ϵ.μ == zeros(3)
    @test F_ϵ.Σ.mat == system[:measurement][:QQ]
    @test F_u.μ == zeros(3)
    @test F_u.Σ.mat == system[:measurement][:EE]
end

@testset "Using compute_system to update an existing system" begin

    # Check updating
    system1 = compute_system(m, system)
    system2 = compute_system(m, system; observables = collect(keys(m.observables))[1:end - 1],
                                  pseudo_observables = collect(keys(m.pseudo_observables))[1:end - 1],
                                  shocks = collect(keys(m.exogenous_shocks))[1:end - 1],
                                  states = collect(keys(m.endogenous_states))[1:end - 1])
    system3 = compute_system(m, system; zero_DD = true)
    system4 = compute_system(m, system; zero_DD_pseudo = true)

    @test system1[:TTT] ≈ system[:TTT]
    @test system1[:RRR] ≈ system[:RRR]
    @test system1[:CCC] ≈ system[:CCC]
    @test system1[:ZZ] ≈ system[:ZZ]
    @test system1[:DD] ≈ system[:DD]
    @test system1[:QQ] ≈ system[:QQ]
    @test system1[:EE] ≈ zeros(size(system[:ZZ], 1), size(system[:ZZ], 1))
    @test system1[:ZZ_pseudo] ≈ system[:ZZ_pseudo]
    @test system1[:DD_pseudo] ≈ system[:DD_pseudo]

    @test system2[:TTT] ≈ system[:TTT][1:end - 1, 1:end - 1]
    @test system2[:RRR] ≈ system[:RRR][1:end - 1, 1:end - 1]
    @test sum(abs.(system2[:CCC])) ≈ sum(abs.(system[:CCC][1:end - 1])) ≈ 0.
    @test system2[:ZZ] ≈ system[:ZZ][1:end - 1, 1:end - 1]
    @test system2[:DD] ≈ system[:DD][1:end - 1]
    @test system2[:QQ] ≈ system[:QQ][1:end - 1, 1:end - 1]
    @test system2[:EE] ≈ zeros(size(system[:ZZ], 1) - 1, size(system[:ZZ], 1) - 1)
    @test system2[:ZZ_pseudo] ≈ system[:ZZ_pseudo][1:end - 1, 1:end - 1]
    @test system2[:DD_pseudo] ≈ system[:DD_pseudo][1:end - 1]

    @test system3[:TTT] ≈ system[:TTT]
    @test system3[:RRR] ≈ system[:RRR]
    @test system3[:CCC] ≈ system[:CCC]
    @test system3[:ZZ] ≈ system[:ZZ]
    @test system3[:DD] ≈ zeros(length(system[:DD]))
    @test system3[:QQ] ≈ system[:QQ]
    @test system3[:EE] ≈ zeros(size(system[:ZZ], 1), size(system[:ZZ], 1))
    @test system3[:ZZ_pseudo] ≈ system[:ZZ_pseudo]
    @test system3[:DD_pseudo] ≈ system[:DD_pseudo]

    @test system4[:TTT] ≈ system[:TTT]
    @test system4[:RRR] ≈ system[:RRR]
    @test system4[:CCC] ≈ system[:CCC]
    @test system4[:ZZ] ≈ system[:ZZ]
    @test system4[:DD] ≈ system[:DD]
    @test system4[:QQ] ≈ system[:QQ]
    @test system4[:EE] ≈ zeros(size(system[:ZZ], 1), size(system[:ZZ], 1))
    @test system4[:ZZ_pseudo] ≈ system[:ZZ_pseudo]
    @test system4[:DD_pseudo] ≈ zeros(length(system[:DD_pseudo]))

    # Check errors
    @test_throws ErrorException compute_system(m, system; observables = [:blah])
    @test_throws ErrorException compute_system(m, system; states = [:blah])
    @test_throws KeyError compute_system(m, system; shocks = [:blah])
    @test_throws ErrorException compute_system(m, system; pseudo_observables = [:blah])
end

@testset "VAR approximation of state space" begin
    m = Model1002("ss10"; custom_settings =
                  Dict{Symbol,Setting}(:add_laborshare_measurement =>
                                       Setting(:add_laborshare_measurement, true),
                                       :add_NominalWageGrowth =>
                                       Setting(:add_NominalWageGrowth, true)))
    system = compute_system(m)
    system = compute_system(m, system; observables = [:obs_hours, :obs_gdpdeflator,
                                                      :laborshare_t, :NominalWageGrowth],
                            shocks = collect(keys(m.exogenous_shocks)))
    yyyyd, xxyyd, xxxxd = DSGE.var_approx_state_space(system[:TTT], system[:RRR], system[:QQ],
                                                 system[:DD], system[:ZZ], system[:EE],
                                                 zeros(size(system[:ZZ], 1),
                                                       DSGE.n_shocks_exogenous(m)),
                                                 4; get_population_moments = true)
    yyyydc, xxyydc, xxxxdc = DSGE.var_approx_state_space(system[:TTT], system[:RRR], system[:QQ],
                                                         system[:DD], system[:ZZ], system[:EE],
                                                         zeros(size(system[:ZZ], 1),
                                                               DSGE.n_shocks_exogenous(m)),
                                                         4; get_population_moments = true,
                                                         use_intercept = true)
    β, Σ = DSGE.var_approx_state_space(system[:TTT], system[:RRR], system[:QQ], system[:DD],
                                  system[:ZZ], system[:EE],
                                  zeros(size(system[:ZZ], 1), DSGE.n_shocks_exogenous(m)),
                                  4; get_population_moments = false)
    βc, Σc = DSGE.var_approx_state_space(system[:TTT], system[:RRR], system[:QQ], system[:DD],
                                  system[:ZZ], system[:EE],
                                  zeros(size(system[:ZZ], 1), DSGE.n_shocks_exogenous(m)),
                                  4; get_population_moments = false, use_intercept = true)

    expmat = load("reference/exp_var_approx_state_space.jld2")
    @test @test_matrix_approx_eq yyyyd expmat["yyyyd"]
    @test @test_matrix_approx_eq xxyyd expmat["xxyyd"][2:end, :]
    @test @test_matrix_approx_eq xxxxd expmat["xxxxd"][2:end, 2:end]
    @test @test_matrix_approx_eq yyyydc expmat["yyyyd"]
    @test @test_matrix_approx_eq xxyydc expmat["xxyyd"]
    @test @test_matrix_approx_eq xxxxdc expmat["xxxxd"]

    expβ = \(expmat["xxxxd"][2:end, 2:end], expmat["xxyyd"][2:end, :])
    expΣ = expmat["yyyyd"] - expmat["xxyyd"][2:end, :]' * expβ
    expΣ += expΣ'
    expΣ ./= 2
    @test @test_matrix_approx_eq β expβ
    @test @test_matrix_approx_eq Σ expΣ

    expβc = \(expmat["xxxxd"], expmat["xxyyd"])
    expΣc = expmat["yyyyd"] - expmat["xxyyd"]' * expβc
    expΣc += expΣc'
    expΣc ./= 2
    @test @test_matrix_approx_eq βc expβc
    @test @test_matrix_approx_eq Σc expΣc
end

@testset "RegimeSwitchingSystem" begin
    transitions         = Vector{Transition{Float64}}(undef, 3)
    measurements        = Vector{Measurement{Float64}}(undef, 3)
    pseudo_measurements = Vector{PseudoMeasurement{Float64}}(undef, 3)
    for i in 1:3
        transitions[i]         = system.transition
        measurements[i]        = system.measurement
        pseudo_measurements[i] = system.pseudo_measurement
    end

    regswitch_sys1 = RegimeSwitchingSystem(transitions, measurements, pseudo_measurements)
    regswitch_sys2 = RegimeSwitchingSystem(transitions, measurements)
    regswitch_sys3 = RegimeSwitchingSystem([system, system, system])
    regswitch_sys4 = copy(regswitch_sys1)
    regswitch_sys5 = deepcopy(regswitch_sys1)

    # Test access/utility functions first
    @test regswitch_sys1[:regimes]             == 1:3
    @test regswitch_sys1[:transitions]         == transitions
    @test regswitch_sys1[:measurements]        == measurements
    @test regswitch_sys1[:pseudo_measurements] == pseudo_measurements
    @test n_regimes(regswitch_sys1)            == 3
    for i in 1:3
        # Creating a System for a specific regime
        for tmpsys in [System(regswitch_sys1, i), System(regswitch_sys3, i),
                       System(regswitch_sys4, i), System(regswitch_sys5, i)]
            @test isa(tmpsys, System)
            @test @test_matrix_approx_eq tmpsys[:TTT] system[:TTT]
            @test @test_matrix_approx_eq tmpsys[:RRR] system[:RRR]
            @test @test_matrix_approx_eq tmpsys[:CCC] system[:CCC]
            @test @test_matrix_approx_eq tmpsys[:ZZ]  system[:ZZ]
            @test @test_matrix_approx_eq tmpsys[:DD]  system[:DD]
            @test @test_matrix_approx_eq tmpsys[:QQ]  system[:QQ]
            @test @test_matrix_approx_eq tmpsys[:EE]  system[:EE]
            @test @test_matrix_approx_eq tmpsys[:ZZ_pseudo]  system[:ZZ_pseudo]
            @test @test_matrix_approx_eq tmpsys[:DD_pseudo]  system[:DD_pseudo]
        end

        tmpsys2 = regswitch_sys2[i] # this creates a System underneath the hood
        @test isa(tmpsys2, System)
        @test @test_matrix_approx_eq tmpsys2[:TTT] system[:TTT]
        @test @test_matrix_approx_eq tmpsys2[:RRR] system[:RRR]
        @test @test_matrix_approx_eq tmpsys2[:CCC] system[:CCC]
        @test @test_matrix_approx_eq tmpsys2[:ZZ]  system[:ZZ]
        @test @test_matrix_approx_eq tmpsys2[:DD]  system[:DD]
        @test @test_matrix_approx_eq tmpsys2[:QQ]  system[:QQ]
        @test @test_matrix_approx_eq tmpsys2[:EE]  system[:EE]
        @test all(tmpsys2[:ZZ_pseudo] .== 0.)
        @test all(tmpsys2[:DD_pseudo] .== 0.)

        # Accessing specific data types or regime matrices
        @test @test_matrix_approx_eq regswitch_sys1[i, :transition][:TTT] transitions[i][:TTT]
        @test @test_matrix_approx_eq regswitch_sys1[i, :transition][:RRR] transitions[i][:RRR]
        @test @test_matrix_approx_eq regswitch_sys1[i, :transition][:CCC] transitions[i][:CCC]
        @test @test_matrix_approx_eq regswitch_sys1[i, :TTT]              transitions[i][:TTT]
        @test @test_matrix_approx_eq regswitch_sys1[i, :RRR]              transitions[i][:RRR]
        @test @test_matrix_approx_eq regswitch_sys1[i, :CCC]              transitions[i][:CCC]

        @test @test_matrix_approx_eq regswitch_sys1[i, :measurement][:ZZ] measurements[i][:ZZ]
        @test @test_matrix_approx_eq regswitch_sys1[i, :measurement][:DD] measurements[i][:DD]
        @test @test_matrix_approx_eq regswitch_sys1[i, :measurement][:QQ] measurements[i][:QQ]
        @test @test_matrix_approx_eq regswitch_sys1[i, :measurement][:EE] measurements[i][:EE]
        @test @test_matrix_approx_eq regswitch_sys1[i, :ZZ]               measurements[i][:ZZ]
        @test @test_matrix_approx_eq regswitch_sys1[i, :DD]               measurements[i][:DD]
        @test @test_matrix_approx_eq regswitch_sys1[i, :QQ]               measurements[i][:QQ]
        @test @test_matrix_approx_eq regswitch_sys1[i, :EE]               measurements[i][:EE]

        @test @test_matrix_approx_eq regswitch_sys1[i, :pseudo_measurement][:ZZ_pseudo] pseudo_measurements[i][:ZZ_pseudo]
        @test @test_matrix_approx_eq regswitch_sys1[i, :pseudo_measurement][:DD_pseudo] pseudo_measurements[i][:DD_pseudo]
        @test @test_matrix_approx_eq regswitch_sys1[i, :ZZ_pseudo]                      pseudo_measurements[i][:ZZ_pseudo]
        @test @test_matrix_approx_eq regswitch_sys1[i, :DD_pseudo]                      pseudo_measurements[i][:DD_pseudo]

        # Check errors when trying to access values
        @test_throws KeyError regswitch_sys1[1, :a]
        @test_throws KeyError regswitch_sys1[:a]
        @test_throws BoundsError regswitch_sys1[4, :transition]
        @test_throws BoundsError regswitch_sys1[4]
        @test_throws BoundsError System(regswitch_sys1, 4)

        # Check copying and deepcopying
        oldval = copy(regswitch_sys1[i, :TTT][1, 1])
        regswitch_sys1[i, :TTT][1, 1] = oldval + 1.
        @test @test_matrix_approx_eq regswitch_sys1[i, :TTT] regswitch_sys4[i, :TTT]
        @test !(regswitch_sys1[i, :TTT] ≈ regswitch_sys5[i, :TTT])
        regswitch_sys1[i, :TTT][1, 1] = oldval
    end
end

nothing
