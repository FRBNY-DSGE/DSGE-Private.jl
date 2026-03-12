"""
    parameters_agg!(param, SS_stats)

Set aggregate parameters in `param` using steady state statistics.
Modifies `param` in place.

Note: In MATLAB, QE_param.mat is loaded but its values are not actually used (commented out).
"""
function parameters_agg!(param::Dict, SS_stats::Dict)
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
