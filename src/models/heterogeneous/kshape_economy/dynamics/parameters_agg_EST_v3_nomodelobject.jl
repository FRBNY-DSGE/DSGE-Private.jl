"""
    parameters_agg_EST_v3!(param, param_update; model=:zlb, SS_stats=nothing)

Update aggregate parameters in `param` with values from `param_update`.
Modifies `param` in place.

`model` selects variant-specific overrides:
- `:csc`  — two-sector wage names, phi_u=0, ZZ shock persistence, capital
             utilisation auxiliary params. Requires `SS_stats`.
- `:mBBQ` — single-sector mBBQ-specific params (alpha_lk, iota_1/iota_2).
"""
function parameters_agg_EST_v3!(param::Dict, param_update::Dict;
                                 model::Symbol = :zlb,
                                 SS_stats::Union{Dict,Nothing} = nothing)
    # TF

    param["kappa"]          = param_update["kappa"]
    # param["Calvo"]        = param_update["Calvo"]
    param["GAMMA"]          = param_update["GAMMA"]
    param["phi"]            = param_update["phi"]
    param["phi_x"]          = param_update["phi_x"]
    param["rho_w"]          = param_update["rho_w"]
    param["d"]              = param_update["d"]
    param["rho_B"]          = param_update["rho_B"]
    # param["gamma_pi"]     = param_update["gamma_pi"]
    # param["gamma_T"]      = param_update["gamma_T"]
    param["rho_R"]          = param_update["rho_R"]
    param["phi_pi"]         = param_update["phi_pi"]
    param["phi_u"]          = param_update["phi_u"]
    param["rho_passive_QE"] = 0.95
    param["rho_X_QE"]       = param_update["rho_X_QE"]
    param["phi_pi_QE"]      = param_update["phi_pi_QE"]
    param["phi_u_QE"]       = param_update["phi_u_QE"]
    param["rho_Z"]          = param_update["rho_Z"]
    param["rho_PSI_RP"]     = param_update["rho_PSI_RP"]
    param["rho_eta"]        = param_update["rho_eta"]
    param["rho_BB"]         = param_update["rho_BB"]
    param["rho_D"]          = param_update["rho_D"]
    param["rho_G"]          = param_update["rho_G"]
    param["rho_PSI_RP"]     = param_update["rho_PSI_RP"]
    # param["rho_LT"]       = param_update["rho_LT"]
    # param["rho_TAU"]      = param_update["rho_TAU"]
    # param["rho_varphi"]   = param_update["rho_varphi"]
    param["rho_iota"]       = param_update["rho_iota"]
    param["sig_D"]          = param_update["sig_D"]
    param["sig_G"]          = param_update["sig_G"]
    param["sig_R"]          = param_update["sig_R"]
    # param["sig_QE"]       = param_update["sig_QE"]
    param["sig_Z"]          = param_update["sig_Z"]
    # param["sig_Q"]        = param_update["sig_Q"]
    param["sig_RP"]         = param_update["sig_RP"]
    param["sig_eta"]        = param_update["sig_eta"]
    param["sig_iota"]       = param_update["sig_iota"]
    param["sig_w"]          = param_update["sig_w"]
    param["sig_BB"]         = param_update["sig_BB"]
    param["sig_B_F"]        = param_update["sig_B_F"]
    param["rho_B_F"]        = param_update["rho_B_F"]
    param["rho_PSI_W"]      = param_update["rho_PSI_W"]
    param["rho_MP"]         = 0

    param["rho_A_g"]        = 0.95

    param["gamma_B"]        = param_update["gamma_B"]
    param["gamma_pi"]       = param_update["gamma_pi"]
    param["gamma_Y"]        = param_update["gamma_Y"]

    # param["gamma_B"]      = 0.144
    # param["gamma_pi"]     = -1.134
    # param["gamma_Y"]      = -0.176

    param["gamma_pi"]       = param_update["gamma_pi"]
    param["gamma_T"]        = param_update["gamma_T"]
    param["eps_w"]          = param_update["eps_w"]
    param["R_ZLB"]          = 1
    param["phi_B"]          = 0
    param["rho_MRS_ent"]    = 0
    param["sigma2"]         = 1.5
    # param["kappa"]        = (1-param["Calvo"]*param["beta"])*(1-param["Calvo"])/param["Calvo"]

    # ------------------------------------------------------------------
    # mBBQ-specific params
    # ------------------------------------------------------------------
    if model == :mBBQ
        param["alpha_lk"] = param_update["alpha_lk"]
        param["iota_1"]   = param_update["iota"]
        param["iota_2"]   = param_update["iota"]
    end

    # ------------------------------------------------------------------
    # CSC-specific overrides
    # ------------------------------------------------------------------
    if model == :csc
        # Two-sector wage names (sector 2 mirrors sector 1 at calibration)
        param["rho_w_1"] = param_update["rho_w"]
        param["rho_w_2"] = param_update["rho_w"]
        param["d_1"]     = param_update["d"]
        param["d_2"]     = param_update["d"]

        # eps_w aliased per sector
        param["eps_w_1"] = param["eps_w"]
        param["eps_w_2"] = param["eps_w"]

        # CSC sets phi_u to zero regardless of the estimated value
        param["phi_u"]   = 0.0

        # ZZ shock persistence (labour-market wedge processes)
        param["rho_ZZ_1"] = param["rho_PSI_W"]
        param["rho_ZZ_2"] = 0.9
        param["rho_ZZ_3"] = 0.9
        param["rho_ZZ_4"] = 0.9

        # Capital-utilisation auxiliary parameters derived from steady state
        if !isnothing(SS_stats)
            param["delta_1"] = SS_stats["r_k"] / param["delta_ss"] * param["v"]
            param["delta_0"] = param["delta_ss"] / param["v"]^param["delta_1"]
        end
    end

    return param
end
