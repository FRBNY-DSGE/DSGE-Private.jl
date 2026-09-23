"""
    parameters_agg_EST_v3!(m, param_update)

Update aggregate parameters in `m` (mBBQ model) with values from `param_update`.
Modifies `m` in place.
"""
function parameters_agg_EST_v3!(m::mBBQ, param_update::Dict)
    # TF

    m[:κ]          = param_update["kappa"]
    # m[:Calvo]    = param_update["Calvo"]
    m[:γ]          = param_update["GAMMA"]
    m[:ϕ]          = param_update["phi"]
    m[:ϕ_x]        = param_update["phi_x"]
    m[:ρ_w]        = param_update["rho_w"]
    m[:d]          = param_update["d"]
    m[:ρ_B]        = param_update["rho_B"]
    # m[:γ_π]      = param_update["gamma_pi"]
    # m[:γ_T]      = param_update["gamma_T"]
    m[:ρ_R]        = param_update["rho_R"]
    m[:ϕ_π]        = param_update["phi_pi"]
    m[:ϕ_u]        = param_update["phi_u"]
    m[:ρ_passive_QE] = 0.95
    m[:ρ_X_QE]     = param_update["rho_X_QE"]
    m[:ϕ_π_QE]     = param_update["phi_pi_QE"]
    m[:ϕ_u_QE]     = param_update["phi_u_QE"]
    m[:ρ_Z]        = param_update["rho_Z"]
    m[:ρ_ψ_rp]    = param_update["rho_PSI_RP"]
    m[:ρ_η]        = param_update["rho_eta"]
    m[:ρ_BB]       = param_update["rho_BB"]
    m[:ρ_D]        = param_update["rho_D"]
    m[:ρ_G]        = param_update["rho_G"]
    m[:ρ_ψ_rp]    = param_update["rho_PSI_RP"]
    # m[:ρ_LT]     = param_update["rho_LT"]      # TODO: SS/transition matrix object — handle separately
    # m[:ρ_TAU]    = param_update["rho_TAU"]      # TODO: SS/transition matrix object — handle separately
    # m[:ρ_varphi] = param_update["rho_varphi"]   # TODO: SS/transition matrix object — handle separately
    m[:ρ_ι]        = param_update["rho_iota"]
    m[:σ_D]        = param_update["sig_D"]
    m[:σ_G]        = param_update["sig_G"]
    m[:σ_R]        = param_update["sig_R"]
    # m[:σ_QE]     = param_update["sig_QE"]       # TODO: SS/transition matrix object — handle separately
    m[:σ_Z]        = param_update["sig_Z"]
    # m[:σ_Q]      = param_update["sig_Q"]        # TODO: SS/transition matrix object — handle separately
    m[:σ_RP]       = param_update["sig_RP"]
    m[:σ_η]        = param_update["sig_eta"]
    m[:σ_ι]        = param_update["sig_iota"]
    m[:σ_w]        = param_update["sig_w"]
    m[:σ_BB]       = param_update["sig_BB"]
    m[:σ_B_F]      = param_update["sig_B_F"]
    m[:ρ_B_F]      = param_update["rho_B_F"]
    m[:ρ_ψ_w]     = param_update["rho_PSI_W"]
    m[:ρ_mp]       = 0
    m[:α_lk]       = param_update["alpha_lk"]

    m[:ρ_A_g]      = 0.95

    m[:γ_B]        = param_update["gamma_B"]
    m[:γ_π]        = param_update["gamma_pi"]
    m[:γ_Y]        = param_update["gamma_Y"]

    # m[:γ_B]      = 0.144
    # m[:γ_π]     = -1.134
    # m[:γ_Y]      = -0.176

    m[:γ_π]        = param_update["gamma_pi"]
    m[:γ_T]        = param_update["gamma_T"]
    m[:ε_w]        = param_update["eps_w"]
    #= TODO: SS/transition matrix object — handle separately
    m["R_ZLB"]          = 1
    =#
    m[:ϕ_b]        = 0
    #= TODO: SS/transition matrix object — handle separately
    m["rho_MRS_ent"]    = 0
    =#
    m[:σ_2]        = 1.5
    # m[:κ]        = (1-m[:Calvo]*m[:β])*(1-m[:Calvo])/m[:Calvo]

    return m
end
