

#TODO: m only as input
function compute_system(m::mBBQ, StateSS, ControlSS, SS_stats, grid, param, Jacob_base)
    
    #update_ss_v5 #TODO need to make this model object only as input
    SS_stats_base, param_base = DSGE.update_ss_v5!(SS_stats, param, param_update, grid)
    
    #check certain values >0

    #state_reduc_tv_copula #TODO missing args
    reduc = state_reduc_tvcopula!(param_base, grid, SS_stats_base, mu_dist, Value, mutil_c, Va)
    Xss = reduc.Xss
    Yss = reduc.Yss
    DC = reduc.DC
    IDC = reduc.IDC
    DCD = reduc.DCD
    IDCD = reduc.IDCD
    Gamma_state = reduc.Gamma_state
    Gamma_control = reduc.Gamma_control
    InvGamma = reduc.InvGamma
    State = reduc.State
    State_m = reduc.State_m
    Contr = reduc.Contr
    Contr_m = reduc.Contr_m
    distrSS = reduc.distrSS
    CDF_SS = reduc.CDF_SS
    COP_SS = reduc.COP_SS
    distr_b_SS = reduc.distr_b_SS
    distr_a_SS = reduc.distr_a_SS
    distr_se_SS = reduc.distr_se_SS
    CDF_b_SS = reduc.CDF_b_SS
    CDF_a_SS = reduc.CDF_a_SS
    CDF_se_SS = reduc.CDF_se_SS
    compressionIndexesCOP = reduc.compressionIndexesCOP
    Poly = reduc.Poly
    InvCheb = reduc.InvCheb
    Gamma2 = reduc.Gamma2
    nPoly = reduc.nPoly
    nFullCtrl = reduc.nFullCtrl
    nRedCtrl = reduc.nRedCtrl
    nFullMarg = reduc.nFullMarg
    nRedMarg = reduc.nRedMarg
    nRedStates = reduc.nRedStates
    
    #maybe square some values?

    #compute jacob TESTING ONLY
    #F21_ad, F22_ad, F23_ad, F24_ad, F41_ad, F42_ad, F43_ad, F44_ad, F21_ad_zlb, F22_ad_zlb, F23_ad_zlb, F24_ad_zlb, F41_ad_zlb, F42_ad_zlb, F43_ad_zlb, F44_ad_zlb = DSGE.jacobian(m, Xss, Yss, SS_stats_base, grid, param_base)
    F21_ad = Matrix(CSV.read("test/csv/F21_ad.csv", DataFrame))
    F22_ad = Matrix(CSV.read("tests/csv/F22_ad.csv", DataFrame))
    F23_ad = Matrix(CSV.read("tests/csv/F23_ad.csv", DataFrame))
    F24_ad = Matrix(CSV.read("tests/csv/F24_ad.csv", DataFrame))
    F41_ad = Matrix(CSV.read("tests/csv/F41_ad.csv", DataFrame))
    F42_ad = Matrix(CSV.read("tests/csv/F42_ad.csv", DataFrame))
    F43_ad = Matrix(CSV.read("tests/csv/F43_ad.csv", DataFrame))
    F44_ad = Matrix(CSV.read("tests/csv/F44_ad.csv", DataFrame))

    F21_ad_zlb = Matrix(CSV.read("tests/csv/F21_ad_zlb.csv", DataFrame))
    F22_ad_zlb = Matrix(CSV.read("tests/csv/F22_ad_zlb.csv", DataFrame))
    F23_ad_zlb = Matrix(CSV.read("tests/csv/F23_ad_zlb.csv", DataFrame))
    F24_ad_zlb = Matrix(CSV.read("tests/csv/F24_ad_zlb.csv", DataFrame))
    F41_ad_zlb = Matrix(CSV.read("tests/csv/F41_ad_zlb.csv", DataFrame))
    F42_ad_zlb = Matrix(CSV.read("tests/csv/F42_ad_zlb.csv", DataFrame))
    F43_ad_zlb = Matrix(CSV.read("tests/csv/F43_ad_zlb.csv", DataFrame))
    F44_ad_zlb = Matrix(CSV.read("tests/csv/F44_ad_zlb.csv", DataFrame))



    # -----------------------------
    # SGU solver
    # -----------------------------
    param["overrideEigen"] = true
    hx, gx, F1_aux, F2_aux, F3_aux, F4_aux, param = DSGE.SGU_solver(F1, F2, F3, F4, grid)
    hx_zlb, gx_zlb, F1_aux_zlb, F2_aux_zlb, F3_aux_zlb, F4_aux_zlb, param_zlb = DSGE.SGU_solver(F1, F2, F3, F4, grid)

    # -----------------------------
    # Linearized system
    # -----------------------------
    nse = Int(grid["numstates_endo"])
    ns  = Int(grid["numstates"])
    nc  = Int(grid["numcontrols"])
    nsh = Int(grid["numstates_shocks"])

    F1 = vcat(F1_aux[1:nse, :], F1_aux[(ns+1):end, :])
    F2 = vcat(F2_aux[1:nse, :], F2_aux[(ns+1):end, :])
    F3 = vcat(F3_aux[1:nse, :], F3_aux[(ns+1):end, :])
    F4 = vcat(F4_aux[1:nse, :], F4_aux[(ns+1):end, :])

    F11 = F1[:, 1:nse]
    F31 = F3[:, 1:nse]
    F32 = F3[:, (nse+1):end]

    A_ref = hcat(zeros(nse + nc, nse), F2)
    B_ref = hcat(F11, F4)
    C_ref = hcat(F31, zeros(nse + nc, nc))
    E_ref = F32

    hx_aux = hx[1:nse, :]
    H_aux  = vcat(hx_aux, gx)
    H1_aux = H_aux[:, 1:nse]
    H2_aux = H_aux[:, (end - nsh + 1):end]

    P_ref  = hcat(H1_aux, zeros(nse + nc, nc))
    Q_ref  = H2_aux

    cof_ref = hcat(C_ref, B_ref, A_ref)
    J_ref   = E_ref


    #need to add some zlb-specific transforms


    #occbin



end
