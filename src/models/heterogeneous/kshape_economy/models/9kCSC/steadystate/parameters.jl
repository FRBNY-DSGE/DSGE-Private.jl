function make_parameters()
    param = Dict{String, Any}()

    # 1) Households
    param["sigma"] = 1.5
    param["tau_L"] = 0.180
    param["tau_P"] = 0.102
    param["xi"] = 1.0 / 3.0
    param["sigma_chi"] = 3.4204542765
    param["logitratio"] = 2.6455668209
    param["mu_chi"] = param["logitratio"] * param["sigma_chi"]
    param["nu"] = 1.0
    param["guessadj"] = param["nu"] / (1 + exp(param["mu_chi"] / param["sigma_chi"]))
    param["n_1"] = 1.0
    param["n_2"] = 1.0
    param["beta"] = 0.9875
    param["death_rate"] = 1.0 / 180.0
    param["reinvest"] = 0.0
    param["GHH"] = (1 + 1 / param["xi"]) / (1 / param["xi"] + param["tau_P"])

    # Idiosyncratic productivity
    param["rhoS"] = 0.99
    param["sigmaS"] = 0.0282
    param["in"] = 0.001
    param["out"] = 0.20
    param["Eshare"] = param["in"] / (param["in"] + param["out"])
    param["in_s"] = 0.05
    param["out_s"] = 0.10
    param["in_l"] = 0.05
    param["out_l"] = 0.10
    param["super_low"] = 0.1811652190
    param["super_high"] = 5.4424929457

    # 2) Intermediate goods firms
    param["eta"] = 3.0
    param["pi_bar"] = 1.005
    param["Eratio"] = 0.3
    param["fix_ratio"] = 0.201538837260750
    param["fix"] = 0.0
    param["b_share"] = 0.0
    param["b_aux"] = 1.6335
    param["zeta"] = 0.401
    param["rho"] = -0.495
    param["a"] = 0.4
    param["w"] = 0.35
    param["Z"] = 3.0

    # 3) Capital service
    param["v"] = 1.0
    param["delta_ss"] = 0.01
    param["delta_0"] = param["delta_ss"]
    param["delta_1"] = 1.0
    param["delta_00"] = param["delta_ss"] - param["delta_0"]

    # 4) Labor agencies
    param["u_1"] = 0.054
    param["v_f_1"] = 0.312
    param["f_1"] = 0.85
    param["iota_1"] = 0.05
    param["lambda_1"] = 0.055
    param["fix_L_1"] = 4.643476652503947e-04

    param["u_2"] = 0.0215
    param["v_f_2"] = 0.610
    param["f_2"] = 0.664
    param["iota_2"] = 0.3
    param["lambda_2"] = 0.022
    param["fix_L_2"] = 4.643476652503947e-04

    param["earning_ratio"] = 2.016

    # 5) Fiscal policy    param["tau_a"] = 0.30
    param["tau_b"] = 0.0
    param["b_ratio"] = 0.4
    param["LT"] = 0.260227733085066
    param["B_gov_ratio"] = 0.6

    # 6) Monetary policy
    param["R_cb"] = 1.0075
    param["pi_cb"] = param["pi_bar"]
    param["Rprem"] = 0.038285657
    param["b_min"] = -1.3006141532
    param["A_g_ratio"] = 0.05

    # 7) Financial intermediary / MMMF
    param["theta_b"] = 0.90
    param["THETA"] = 3.0
    param["Ab_ratio"] = 0.271426003778368
    param["b_a_aux"] = 0.994444444444444
    param["b_a_aux2"] = 1.0
    param["beta_b"] = param["pi_cb"] / param["R_cb"] * (1 / (1 - param["death_rate"])) * param["b_a_aux"]
    param["B_MMMF_ratio"] = 0.783074980718057
    param["q"] = 1.0
    param["tau_cp"] = 0.0
    param["Lambda_bank"] = param["beta"]
    param["frac_b"] = 0.0

    # Shared-function defaults
    param["ss_obj_old"] = nothing

    return param
end
