using Test, DSGE, ModelConstructors

@testset "Model1002 estimation reuse" begin
    m = Model1002("ss10")
    σ_g_index = findfirst(parameter -> parameter.key == :σ_g, m.parameters)
    σ_c_index = findfirst(parameter -> parameter.key == :σ_c, m.parameters)

    proposed = [parameter.value for parameter in m.parameters]
    proposed[σ_g_index] *= 0.9
    @test DSGE.skip_steadystate_update(m, proposed)

    proposed_parameters = deepcopy(m.parameters)
    proposed_parameters[σ_g_index].value = proposed[σ_g_index]
    @test DSGE.skip_steadystate_update(m, proposed_parameters)

    structural_proposal = copy(proposed)
    structural_proposal[σ_c_index] *= 0.99
    @test !DSGE.skip_steadystate_update(m, structural_proposal)

    original = compute_system(m)
    @test length(DSGE.MODEL1002_GENSYS_CACHE[m]) == 1
    DSGE.update!(m, proposed)
    revised = compute_system(m)
    @test revised[:TTT] == original[:TTT]
    @test revised[:RRR] == original[:RRR]
    @test revised[:CCC] == original[:CCC]
    @test revised[:QQ] != original[:QQ]
    @test length(DSGE.MODEL1002_GENSYS_CACHE[m]) == 1

    proposed_parameters[σ_g_index].value *= 0.9
    @test DSGE.skip_steadystate_update(m, proposed_parameters)
    DSGE.update!(m, proposed_parameters)
    revised_again = compute_system(m)
    @test revised_again[:TTT] == revised[:TTT]
    @test revised_again[:QQ] != revised[:QQ]
    @test length(DSGE.MODEL1002_GENSYS_CACHE[m]) == 1

    Γ0, Γ1, C, Ψ, Π = eqcond(m)
    Ψ_changed = copy(Ψ)
    first_loading = findfirst(x -> !iszero(x), Ψ_changed)
    Ψ_changed[first_loading] += 0.1
    cached_solution = DSGE.gensys_for_model(m, Γ0, Γ1, C, Ψ_changed, Π, 1 + 1e-6)
    direct_solution = DSGE.gensys(Γ0, Γ1, C, Ψ_changed, Π, 1 + 1e-6)
    @test cached_solution[1] ≈ direct_solution[1]
    @test cached_solution[2] ≈ direct_solution[2]
    @test cached_solution[3] ≈ direct_solution[3]
    @test length(DSGE.MODEL1002_GENSYS_CACHE[m]) == 2

    regime_model = Model1002("ss10")
    set_regime_val!(regime_model[:σ_g], 1, regime_model[:σ_g].value)
    set_regime_val!(regime_model[:σ_g], 2, 0.9 * regime_model[:σ_g].value)
    first_regime = DSGE.solve_one_regime(regime_model; regime = 1)
    second_regime = DSGE.solve_one_regime(regime_model; regime = 2)
    @test first_regime == second_regime
    @test length(DSGE.MODEL1002_GENSYS_CACHE[regime_model]) == 1
    first_measurement = measurement(regime_model, first_regime...; reg = 1)
    second_measurement = measurement(regime_model, second_regime...; reg = 2)
    @test first_measurement[:QQ] != second_measurement[:QQ]
end
