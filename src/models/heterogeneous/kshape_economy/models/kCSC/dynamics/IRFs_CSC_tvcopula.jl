using LinearAlgebra

"""
    IRFs_CSC_tvcopula(hx, gx, Xss, Yss, Gamma_state, Gamma_control,
                      param, grid, SS_stats)

Compute impulse response functions for the CSC time-varying copula model.
Returns a dictionary of level series and log-deviation series using the
MATLAB CSC naming convention.
"""
function IRFs_CSC_tvcopula(hx, gx, Xss, Yss, Gamma_state, Gamma_control,
                           param, grid, SS_stats)

    if !haskey(grid, "maxlag")
        grid["maxlag"] = 1001
    end
    maxlag = Int(grid["maxlag"])

    numstates = Int(grid["numstates"])
    numcontrols = size(gx, 1)
    os = Int(grid["os"])
    oc = Int(grid["oc"])
    ns = Int(grid["ns"])

    x0 = zeros(numstates)
    x0[end - 3] = -0.0025 / 4

    MX = [Matrix{Float64}(I, numstates, numstates); gx]
    IRF_state_sparse = zeros(numstates + numcontrols, maxlag)
    x = copy(x0)
    for t in 1:maxlag
        IRF_state_sparse[:, t] = MX * x
        x = hx * x
    end

    Statenext_IRFs = repeat(Xss[(end - os + 1):end], 1, maxlag) .+
                     IRF_state_sparse[(numstates - os + 1):numstates, :]
    Control_IRFs = repeat(Yss[(end - oc + 1):end], 1, maxlag) .+
                   Gamma_control[(end - oc + 1):end, :] * IRF_state_sparse[(numstates + 1):end, :]

    Statenext_IRFs = exp.(Statenext_IRFs)
    Control_IRFs = exp.(Control_IRFs)

    # Local offsets within the observable state block.
    R_cb_ind = 1
    w_1_ind = 2
    w_2_ind = 3
    w_ind = 4
    A_b_ind = 5
    B_b_ind = 6
    A_g_ind = 7
    Q_ind = 8
    lev_ind = 9
    NW_b_ind = 10

    # Local offsets within the observable control block.
    MRS_ind = 1
    A_hh_ind = MRS_ind + 1
    B_hh_ind = A_hh_ind + 1
    C_ind = B_hh_ind + 1
    N_tilde_1_ind = C_ind + 1
    N_tilde_2_ind = N_tilde_1_ind + 1
    L_1_ind = N_tilde_2_ind + 1
    L_2_ind = L_1_ind + 1
    UB_ind = L_2_ind + 1
    unemp_1_ind = UB_ind + 1
    unemp_2_ind = unemp_1_ind + 1
    unemp_ind = unemp_2_ind + 1
    U_1_ind = unemp_ind + 1
    U_2_ind = U_1_ind + 1
    K_ind = U_2_ind + 1
    B_ind = K_ind + 1
    B_gov_ncp_ind = B_ind + 1
    T_ind = B_gov_ncp_ind + 1
    LT_ind = T_ind + 1
    G_ind = LT_ind + 1
    Lambda_ind = G_ind + 1
    pi_ind = Lambda_ind + 1
    V_1_ind = pi_ind + 1
    V_2_ind = V_1_ind + 1
    J_ind = V_2_ind .+ (1:ns)
    r_l_1_ind = J_ind[end] + 1
    r_l_2_ind = r_l_1_ind + 1
    v_ind = r_l_2_ind + 1
    Y_ind = v_ind + 1
    Profit_ind = Y_ind + 1
    r_k_ind = Profit_ind + 1
    r_a_ind = r_k_ind + 1
    MC_ind = r_a_ind + 1
    n_1_ind = MC_ind + 1
    n_2_ind = n_1_ind + 1
    M_1_ind = n_2_ind + 1
    M_2_ind = M_1_ind + 1
    f_1_ind = M_2_ind + 1
    f_2_ind = f_1_ind + 1
    zz_ind = f_2_ind + 1
    xx_ind = zz_ind + 1
    vv_ind = xx_ind + 1
    ee_ind = vv_ind + 1
    C_b_ind = ee_ind + 1
    Profit_FI_ind = C_b_ind + 1
    RRa_ind = Profit_FI_ind + 1
    RR_ind = RRa_ind + 1
    I_ind = RR_ind + 1
    x_k_ind = I_ind + 1

    state_rows = numstates - os
    total_rows = numstates + numcontrols
    control_rows = total_rows - oc

    state_series = [
        ("R_cb", R_cb_ind),
        ("w_1", w_1_ind),
        ("w_2", w_2_ind),
        ("w", w_ind),
        ("A_b", A_b_ind),
        ("B_b", B_b_ind),
        ("A_g", A_g_ind),
        ("Q", Q_ind),
        ("lev", lev_ind),
        ("NW_b", NW_b_ind),
    ]

    control_series = [
        ("MRS", MRS_ind),
        ("A_hh", A_hh_ind),
        ("B_hh", B_hh_ind),
        ("C", C_ind),
        ("N_tilde_1", N_tilde_1_ind),
        ("N_tilde_2", N_tilde_2_ind),
        ("L_1", L_1_ind),
        ("L_2", L_2_ind),
        ("UB", UB_ind),
        ("unemp_1", unemp_1_ind),
        ("unemp_2", unemp_2_ind),
        ("unemp", unemp_ind),
        ("U_1", U_1_ind),
        ("U_2", U_2_ind),
        ("K", K_ind),
        ("B", B_ind),
        ("B_gov_ncp", B_gov_ncp_ind),
        ("T", T_ind),
        ("LT", LT_ind),
        ("G", G_ind),
        ("Lambda", Lambda_ind),
        ("pi", pi_ind),
        ("V_1", V_1_ind),
        ("V_2", V_2_ind),
        ("J", J_ind),
        ("r_l_1", r_l_1_ind),
        ("r_l_2", r_l_2_ind),
        ("v", v_ind),
        ("Y", Y_ind),
        ("Profit", Profit_ind),
        ("r_k", r_k_ind),
        ("r_a", r_a_ind),
        ("MC", MC_ind),
        ("n_1", n_1_ind),
        ("n_2", n_2_ind),
        ("M_1", M_1_ind),
        ("M_2", M_2_ind),
        ("f_1", f_1_ind),
        ("f_2", f_2_ind),
        ("zz", zz_ind),
        ("xx", xx_ind),
        ("vv", vv_ind),
        ("ee", ee_ind),
        ("C_b", C_b_ind),
        ("Profit_FI", Profit_FI_ind),
        ("RRa", RRa_ind),
        ("RR", RR_ind),
        ("I", I_ind),
        ("x_k", x_k_ind),
    ]

    asrow(x) = ndims(x) == 1 ? reshape(x, 1, :) : x
    IRFs = Dict{String, Any}()

    for (name, idx) in state_series
        IRFs["$(name)_IRFs_p"] = asrow(Statenext_IRFs[idx, :])
        IRFs["IRFs_$(name)_p"] = asrow(100 .* IRF_state_sparse[state_rows .+ idx, :])
    end

    for (name, idx) in control_series
        IRFs["$(name)_IRFs_p"] = asrow(Control_IRFs[idx, :])
        IRFs["IRFs_$(name)_p"] = asrow(100 .* IRF_state_sparse[control_rows .+ idx, :])
    end

    return IRFs
end
