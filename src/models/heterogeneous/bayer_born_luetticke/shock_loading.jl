function shock_loading(m::BayerBornLuetticke{T}, TTT_jump::Matrix{T}) where {T <: Real}

    # Set up
    exo      = m.exogenous_shocks
    endo     = m.endogenous_states
    n_states = n_backward_looking_states(m)::Int
    n_exo_sh = n_shocks_exogenous(m)::Int
    _RRR     = zeros(n_states, n_exo_sh)
    RRR      = Matrix{T}(undef, n_model_states(m)::Int, n_exo_sh)

    # Populate _RRR for the shocks
    _RRR[endo[:A′_t], exo[:A_sh]] .= 1.
    _RRR[endo[:Z′_t], exo[:Z_sh]] .= 1.
    _RRR[endo[:Ψ′_t], exo[:Ψ_sh]] .= 1.
    _RRR[endo[:μ_p′_t], exo[:μ_p_sh]] .= 1.
    _RRR[endo[:μ_w′_t], exo[:μ_w_sh]] .= 1.
    _RRR[endo[:G_sh′_t], exo[:G_sh]] .= 1.
    _RRR[endo[:R_sh′_t], exo[:R_sh]] .= 1.
    _RRR[endo[:S_sh′_t], exo[:S_sh]] .= 1.
    _RRR[endo[:P_sh′_t], exo[:P_sh]] .= 1.

    # Loading on states and jumps
    RRR[1:n_states, :]     = _RRR
    RRR[n_states+1:end, :] = TTT_jump * _RRR

    return RRR
end
