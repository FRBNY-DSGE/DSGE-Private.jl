function augment_states(m::OnionModel, TTT::Matrix{T}, RRR::Matrix{T}, CCC::Vector{T}) where {T<:AbstractFloat}
    endo     = m.endogenous_states
    endo_new = m.endogenous_states_augmented
    exo      = m.exogenous_shocks

    n_endo = n_states(m)
    n_exo  = n_shocks_exogenous(m)

    @assert (n_endo, n_endo) == size(TTT)
    @assert (n_endo, n_exo)  == size(RRR)
    @assert (n_endo,)        == size(CCC)

    n_states_add = length(endo_new)

    TTT_aug = zeros(n_endo + n_states_add, n_endo + n_states_add)
    TTT_aug[1:n_endo, 1:n_endo] = TTT

    RRR_aug = [RRR; zeros(n_states_add, n_exo)]
    CCC_aug = [CCC; zeros(n_states_add)]

    # Lagged
    TTT_aug[endo_new[:i_1], endo[:li]] = - m[:ρ_i]
    TTT_aug[endo_new[:i_1], endo[:πc]] = - (1. - m[:ρ_i])*m[:mp_cpi_infl]
    TTT_aug[endo_new[:i_1], endo_new[:i_1]] = 1.


    TTT_aug[endo_new[:w_1], endo[:lw]] = -1.
    TTT_aug[endo_new[:w_1], endo[:πw]] = 1.
    TTT_aug[endo_new[:w_1], endo[:πc]] = -1.
    TTT_aug[endo_new[:w_1], endo_new[:w_1]] = 1.

    return TTT_aug, RRR_aug, CCC_aug
end
