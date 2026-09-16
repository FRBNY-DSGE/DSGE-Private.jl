"""
    parameters_agg!(param, SS_stats)

Set aggregate parameters in `param` using steady state statistics.
Modifies `param` in place.

Note: In MATLAB, QE_param.mat is loaded but its values are not actually used (commented out).
"""
function parameters_agg_CSC(param::Dict, SS_stats::Dict)
    # TFP

    # param["Calvo"]   = 0.80
    # param["Calvo"]   = 0.8
    # param["Calvo"]   = 0.7
    # param["kappa"]   = (1-param["Calvo"]*param["beta"])*(1-param["Calvo"])/param["Calvo"]
    param["kappa"]   = 0.0275

    param["GAMMA"]   = 0.1454                               # degree of indexation to previous period's inflation
    # param["GAMMA"]   = 0.574

    # param["phi"]     = 54.302
    # param["phi"]     = 61.678
    param["phi"]     = 47.847
    # param["phi_x"]   = 0.05       # sales-to-stock ratio adjustment costs
    # param["phi_x"]   = 0.02       # sales-to-stock ratio adjustment costs
    # param["phi_x"]   = 0.001      # sales-to-stock ratio adjustment costs
    param["phi_x"]   = 0.01        # sales-to-stock ratio adjustment costs
    # param["phi_x"]   = 0.02

    # param["rho_w"]   = 0
    param["rho_w"]   = 0.8116
    # param["rho_w"]   = 0.512
    param["eps_w"]   = 1.0045      # real wage rigidity
    param["d"]       = 1 - 0.0806  # degree of partial indexation of the nominal wage


    # param["rho_B"]    = 0.5
    param["rho_B"]    = 0.6323
    param["gamma_pi"] = 0
    param["gamma_T"]  = 0

    # param["rho_R"]   = 0.726                              # smoothing parameter
    param["rho_R"]   = 0.8307
    param["sigma_R"] = 0.000625                            # standard deviation of MP shock
    param["phi_pi"]  = 1.9853                              # response to inflation
    param["phi_y"]   = 0.4511                              # response to output gap
    param["phi_u"]   = 0.4511                              # response to output gap
    # param["phi_pi"]  = 1.8                               # response to inflation
    # param["phi_y"]   = 0.1                               # response to output gap
    # param["phi_u"]   = 0.1                               # response to output gap
    # param["phi_y"]   = 0.05                              # response to output gap
    # param["phi_u"]   = 0.05                              # response to output gap
    param["rho_m"]   = 0                                   # persistence of MP shock

    param["R_ZLB"]   = 1

    param["alpha_lk"] = 0
    # param["alpha_lk"] = -0.067

    #  9) Capital quality shock


    param["nu_cp"]    = 1.5    # GK (2011): moderate intervention
    # param["tau_cp"]   = 0.01     # Unit efficiency cost of credit policy
    # param["rho_A_g"]  = param["rho_R"]

    param["phi_A_1"] = 0
    param["phi_A_2"] = 0
    # param["phi_A_2"] = 10
    param["phi_B"]   = 0
    # param["phi_B"]   = 0.01

    # param["phi_A_g"] = 0.6

    param["tau_FG"] = 0

    param["rho_MRS_ent"] = 0

    ## QE

    # param["phi_pi_QE"] = param["phi_pi"]
    # param["phi_u_QE"]  = param["phi_y"]

    # param["rho_psi_cp"] = param["rho_R"]

    # param["phi_pi_QE"] = 0.015
    # param["phi_u_QE"]  = 0.001

    param["rho_passive_QE"] = 0.95
    param["rho_X_QE"]       = 0.7

    # In MATLAB: load('QE_param.mat')
    # QE_param is passed as argument

    # param["phi_pi_QE"] = param["phi_pi"]
    # param["phi_u_QE"]  = param["phi_y"]
    # param["rho_A_g"]   = param["rho_R"]

    param["phi_pi_QE"] = 10 * param["phi_pi"]
    param["phi_u_QE"]  = 10 * param["phi_y"]
    # param["phi_pi_QE"] = 1
    # param["phi_u_QE"]  = 4
    # param["phi_pi_QE"] = 1.8
    # param["phi_u_QE"]  = 0.1
    param["rho_A_g"]   = 0.7

    # param["phi_pi_QE"] = QE_param["phi_pi_QE"]
    # param["phi_u_QE"]  = QE_param["phi_u_QE"]
    # param["rho_A_g"]   = QE_param["rho_A_g"]

    param["rho_psi_cp"] = 0

    param["sigma2"] = 1.5
    # param["sigma2"] = 20
    # param["sigma2"] = 48

    param["rho_Z"]       = 0.9
    param["rho_gamma_Q"] = 0.9
    param["rho_PSI_RP"]  = 0.9
    param["rho_eta"]     = 0.9
    param["rho_D"]       = 0.9
    param["rho_G"]       = 0.9
    param["rho_LT"]      = 0.9
    param["rho_TAU"]     = 0.9
    param["rho_varphi"]  = 0.9
    param["rho_iota"]    = 0.9
    param["rho_MP"]      = 0.8
    param["rho_BB"]      = 0.99995
    param["rho_PSI_W"]   = 0.8
    param["rho_MP"]      = 0

    param["rho_B_F"]     = 0.9

    # param["delta_2"]    = 5*param["delta_1"]


    param["delta_1"] = SS_stats["r_k"] / param["delta_ss"] * param["v"]
    param["delta_0"] = param["delta_ss"] / param["v"]^param["delta_1"]

    # param["gamma_B"]  = 0
    # param["gamma_pi"] = 0
    # param["gamma_Y"]  = 0

    # param["alpha_lk"] = 0.1
    # param["alpha_lk"] = 0
    param["alpha_ll"] = 0
    param["alpha_kk"] = 0

    return param
end

"""
    sync_kshape_parameters!(m, param)

Copy the scalar MATLAB/steady-state parameter dictionary into the `kCSC9`
model parameters used by `Fsys_agg`. The two representations use different
names for Greek symbols, so the handoff must be explicit.
"""
function sync_kshape_parameters!(m::kCSC9, param::Dict)
    mapping = (
        :Eratio => "Eratio", :GHH => "GHH", :Z => "Z", :a => "a",
        :b_a_aux2 => "b_a_aux2", :b_share => "b_share", :d_1 => "d_1",
        :d_2 => "d_2", :dr => "death_rate", :fix2 => "fix2", :fix => "fix",
        :fix_L_1 => "fix_L_1", :fix_L_2 => "fix_L_2", :in => "in",
        :w => "w", :w_bar_1 => "w_bar_1", :w_bar_2 => "w_bar_2",
        :α_1 => "alpha_1", :α_2 => "alpha_2", :β_b => "beta_b",
        :γ => "GAMMA", :γ_T => "gamma_T", :γ_π => "gamma_pi",
        :δ => "DELTA", :δ_0 => "delta_0", :δ_1 => "delta_1", :δ_00 => "delta_00",
        :ε_w => "eps_w", :ζ => "zeta", :η => "eta", :θ_b => "theta_b",
        :ι_1 => "iota_1", :ι_2 => "iota_2", :κ => "kappa",
        :λ_1 => "lambda_1", :λ_2 => "lambda_2", :λ_aux => "Lambda_aux",
        :λ_b_aux => "Lambda_b_aux", :ξ => "xi", :π_bar => "pi_bar",
        :π_cb => "pi_cb", :ρ => "rho", :ρ_A_g => "rho_A_g",
        :ρ_BB => "rho_BB", :ρ_B => "rho_B", :ρ_B_F => "rho_B_F",
        :ρ_D => "rho_D", :ρ_G => "rho_G", :ρ_R => "rho_R",
        :ρ_X_QE => "rho_X_QE", :ρ_ZZ_1 => "rho_ZZ_1",
        :ρ_ZZ_2 => "rho_ZZ_2", :ρ_ZZ_3 => "rho_ZZ_3",
        :ρ_ZZ_4 => "rho_ZZ_4", :ρ_Z => "rho_Z", :ρ_mp => "rho_MP",
        :ρ_w_1 => "rho_w_1", :ρ_w_2 => "rho_w_2", :ρ_η => "rho_eta",
        :ρ_ι => "rho_iota", :ρ_ψ_rp => "rho_PSI_RP",
        :ρ_ψ_w => "rho_PSI_W", :σ2 => "sigma2", :τ_L => "tau_L",
        :τ_P => "tau_P", :τ_a => "tau_a", :τ_cp => "tau_cp",
        :ψ_1 => "psi_1", :ψ_2 => "psi_2", :ω => "omega", :ϕ => "phi",
        :ϕ_b => "phi_b", :ϕ_u => "phi_u", :ϕ_u_QE => "phi_u_QE",
        :ϕ_π => "phi_pi", :ϕ_π_QE => "phi_pi_QE",
    )

    for (model_key, dict_key) in mapping
        if haskey(param, dict_key) && param[dict_key] isa Real
            m[model_key] = Float64(param[dict_key])
        end
    end
    return m
end

function parameters_agg_CSC(m::kCSC9, param::Dict, SS_stats::Dict)
    parameters_agg_CSC(param, SS_stats)
    sync_kshape_parameters!(m, param)
    return param
end
