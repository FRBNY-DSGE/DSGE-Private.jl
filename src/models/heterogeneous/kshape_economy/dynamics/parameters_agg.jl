"""
    parameters_agg!(m, SS_stats)

Set aggregate parameters in `m` (mBBQ model) using steady state statistics.
Modifies `m` in place.

Note: In MATLAB, QE_param.mat is loaded but its values are not actually used (commented out).
"""
function parameters_agg!(m::mBBQ, SS_stats::Dict)
    # TFP

    # m[:Calvo]   = 0.80
    # m[:Calvo]   = 0.8
    # m[:Calvo]   = 0.7
    # m[:κ]   = (1-m[:Calvo]*m[:β])*(1-m[:Calvo])/m[:Calvo]
    m[:κ]   = 0.0275

    m[:γ]   = 0.1454                               # degree of indexation to previous period's inflation
    # m[:γ]   = 0.574

    # m[:ϕ]     = 54.302
    # m[:ϕ]     = 61.678
    m[:ϕ]     = 47.847
    # m[:ϕ_x]   = 0.05       # sales-to-stock ratio adjustment costs
    # m[:ϕ_x]   = 0.02       # sales-to-stock ratio adjustment costs
    # m[:ϕ_x]   = 0.001      # sales-to-stock ratio adjustment costs
    m[:ϕ_x]   = 0.01        # sales-to-stock ratio adjustment costs
    # m[:ϕ_x]   = 0.02

    # m[:ρ_w]   = 0
    m[:ρ_w]   = 0.8116
    # m[:ρ_w]   = 0.512
    m[:ε_w]   = 1.0045      # real wage rigidity
    m[:d]       = 1 - 0.0806  # degree of partial indexation of the nominal wage


    # m[:ρ_B]    = 0.5
    m[:ρ_B]    = 0.6323
    m[:γ_π] = 0
    m[:γ_T]  = 0

    # m[:ρ_R]   = 0.726                              # smoothing parameter
    m[:ρ_R]   = 0.8307
    m[:σ_R] = 0.000625                            # standard deviation of MP shock
    m[:ϕ_π]  = 1.9853                              # response to inflation
    m[:ϕ_u]   = 0.4511                              # response to output gap
    # m[:ϕ_π]  = 1.8                               # response to inflation
    # m[:ϕ_u]   = 0.1                               # response to output gap
    # m[:ϕ_u]   = 0.05                              # response to output gap
    m[:ρ_mp]   = 0                                   # persistence of MP shock

    #= TODO: SS/transition matrix object — handle separately
    m["R_ZLB"]   = 1
    =#

    m[:α_lk] = 0
    # m[:α_lk] = -0.067

    #  9) Capital quality shock

    #= TODO: SS/transition matrix object — handle separately
    m["nu_cp"]    = 1.5    # GK (2011): moderate intervention
    =#
    # m[:τ_cp]   = 0.01     # Unit efficiency cost of credit policy
    # m[:ρ_A_g]  = m[:ρ_R]

    #= TODO: SS/transition matrix object — handle separately
    m["phi_A_1"] = 0
    m["phi_A_2"] = 0
    =#
    # m["phi_A_2"] = 10
    m[:ϕ_b]   = 0
    # m[:ϕ_b]   = 0.01

    # m["phi_A_g"] = 0.6

    #= TODO: SS/transition matrix object — handle separately
    m["tau_FG"] = 0
    =#

    #= TODO: SS/transition matrix object — handle separately
    m["rho_MRS_ent"] = 0
    =#

    ## QE

    # m[:ϕ_π_QE] = m[:ϕ_π]
    # m[:ϕ_u_QE]  = m[:ϕ_u]

    # m[:ρ_ψ_rp] = m[:ρ_R]

    # m[:ϕ_π_QE] = 0.015
    # m[:ϕ_u_QE]  = 0.001

    m[:ρ_passive_QE] = 0.95
    m[:ρ_X_QE]       = 0.7

    # In MATLAB: load('QE_param.mat')
    # QE_param is passed as argument

    # m[:ϕ_π_QE] = m[:ϕ_π]
    # m[:ϕ_u_QE]  = m[:ϕ_u]
    # m[:ρ_A_g]   = m[:ρ_R]

    m[:ϕ_π_QE] = 10 * m[:ϕ_π]
    m[:ϕ_u_QE]  = 10 * m[:ϕ_u]
    # m[:ϕ_π_QE] = 1
    # m[:ϕ_u_QE]  = 4
    # m[:ϕ_π_QE] = 1.8
    # m[:ϕ_u_QE]  = 0.1
    m[:ρ_A_g]   = 0.7

    # m[:ϕ_π_QE] = QE_param["phi_pi_QE"]
    # m[:ϕ_u_QE]  = QE_param["phi_u_QE"]
    # m[:ρ_A_g]   = QE_param["rho_A_g"]

    #= TODO: SS/transition matrix object — handle separately
    m["rho_psi_cp"] = 0
    =#

    m[:σ_2] = 1.5
    # m[:σ_2] = 20
    # m[:σ_2] = 48

    m[:ρ_Z]       = 0.9
    #= TODO: SS/transition matrix object — handle separately
    m["rho_gamma_Q"] = 0.9
    =#
    m[:ρ_ψ_rp]  = 0.9
    m[:ρ_η]     = 0.9
    m[:ρ_D]       = 0.9
    m[:ρ_G]       = 0.9
    #= TODO: SS/transition matrix object — handle separately
    m["rho_LT"]      = 0.9
    m["rho_TAU"]     = 0.9
    m["rho_varphi"]  = 0.9
    =#
    m[:ρ_ι]    = 0.9
    m[:ρ_mp]      = 0.8
    m[:ρ_BB]      = 0.99995
    m[:ρ_ψ_w]   = 0.8
    m[:ρ_mp]      = 0

    m[:ρ_B_F]     = 0.9

    # m["delta_2"]    = 5*m[:δ_1]

    # TODO: SS/transition matrix object — handle separately
    # m[:δ_1] = SS_stats["r_k"] / param["delta_ss"] * param["v"]
    # m[:δ_0] = param["delta_ss"] / param["v"]^m[:δ_1]

    # m[:γ_B]  = 0
    # m[:γ_π] = 0
    # m[:γ_Y]  = 0

    # m[:α_lk] = 0.1
    # m[:α_lk] = 0
    m[:α_ll] = 0
    m[:α_kk] = 0

    return m
end
