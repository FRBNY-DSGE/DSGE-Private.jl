function compute_system!(m::OnionModel)
    solution_method = get_setting(m, :solution_method)

    if solution_method == :klein
        A, B, C = eqcond!(m)
        TTT_jump, TTT_state, eu = klein_BP(m, A, B)

        if eu==-1
            throw(KleinError())
        end

        TTT, RRR = klein_transition_matrices(m, real(TTT_state), real(TTT_jump))
        CCC = zeros(n_model_states(m))
        @show size(TTT), size(RRR), size(CCC)

        TTT, RRR, CCC        = augment_states(m, TTT, RRR, CCC)

        @show size(TTT), size(RRR), size(CCC)

        measurement_equation = measurement(m, TTT, RRR, CCC)
    elseif solution_method == :gensys

        TTT, RRR, CCC = solve!(m)
        measurement_equation = measurement(m, TTT, RRR, CCC)

    end

    transition_equation = Transition(TTT, RRR, CCC)
    return System(transition_equation, measurement_equation)
end


function solve!(m::OnionModel)
    Γ0, Γ1, Const, Ψ, Π, C, norm_mat = eqcond!(m)
    TTT_gensys, CCC_gensys, RRR_gensys, eu = gensys(Γ0, Γ1, Const, Ψ, Π)

    if eu[1] != 1 || eu[2] != 1
        println("eu(1): $(eu[1])")
        println("eu(2): $(eu[2])")
        throw(GensysError())
    end

    TTT_gensys = real(TTT_gensys)
    RRR_gensys = real(RRR_gensys)
    CCC_gensys = real(CCC_gensys)

    TTT, RRR, CCC = augment_states(m, TTT_gensys, RRR_gensys, CCC_gensys)

    return TTT, RRR, CCC
end
