function augment_states(m::OnionModel, TTT::Matrix{T}, RRR::Matrix{T}, CCC::Vector{T}) where {T<:AbstractFloat}
    subspec_int = parse(Int, subspec(m)[3:end])

    endo     = m.endogenous_states
    endo_new = m.endogenous_states_augmented
    exo      = m.exogenous_shocks

    n_endo = n_states(m)
    n_exo  = n_shocks_exogenous(m)

    #@show size(TTT), n_endo
    @assert (n_endo, n_endo) == size(TTT)
    @assert (n_endo, n_exo)  == size(RRR)
    @assert (n_endo,)        == size(CCC)

    n_states_add = length(endo_new)


    TTT_aug = zeros(n_endo + n_states_add, n_endo + n_states_add)
    TTT_aug[1:n_endo, 1:n_endo] = TTT
    RRR_aug = [RRR; zeros(n_states_add, n_exo)]


    ### Track lags

    TTT_aug[endo_new[:w_t1], endo[:w_t]] = 1.
    TTT_aug[endo_new[:c_t1], endo[:c_t]] = 1.
    TTT_aug[endo_new[:r_t1], endo[:r_t]] = 1.
    TTT_aug[endo_new[:πc_t1], endo[:πc_t]] = 1.

    ### Add cpi measurement error
    if subspec_int ∈ [13, 113, 114, 115, 116, 117, 118]
        TTT_aug[endo_new[:e_meas_cpi_t], endo_new[:e_meas_cpi_t]] = m[:ρ_meas_cpi]
    end

    # Add FFR measurement error
    if subspec_int ∈ [118]
        TTT_aug[endo_new[:e_meas_ffr_t], endo_new[:e_meas_ffr_t]] = m[:ρ_meas_cpi]
        RRR_aug[endo_new[:e_meas_ffr_t], exo[:meas_ffr_sh]] = 1.0
    end
   

    #Add measurement error:
    #TTT_aug[endo_new[:e_meas_πc_t], endo_new[:e_meas_πc_t]] = m[:ρ_meas_πc]

#=
    #Measurement errors:
    RRR_aug[endo_new[Symbol("e_cpi_1")]:endo_new[Symbol("e_cpi_$(get_setting(m, :n_sectors))")], exo[Symbol("μ_trend_1_sh")]: exo[Symbol("μ_trend_$(get_setting(m, :n_sectors))_sh")]] = 1.
    RRR_aug[endo_new[:e_μw], exo[:μw_sh]] = 1.
    RRR_aug[endo_new[:e_πstar], exo[:
=#

    if subspec_int ∈ [13, 113, 114, 115, 116, 117, 118]
        RRR_aug[endo_new[:e_meas_cpi_t], exo[:meas_cpi_sh]] = 1.0
    end


    CCC_aug = [CCC; zeros(n_states_add)]

    return TTT_aug, RRR_aug, CCC_aug
end
