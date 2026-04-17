

function compute_system(m::mBBQ, Jacob_base; H_obs=nothing)
    
    #update_ss_v5
    DSGE.update_ss_v5!(m) #TODO see if param can be moved to model
    
    @assert false
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

    #compute jacob TESTING ONLY, read csv jacob instead of actually compute jacob for testing
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
    param_base["overrideEigen"] = true
    hx, gx, F1_aux, F2_aux, F3_aux, F4_aux, param_base, indicator = DSGE.SGU_solver(param_base, grid, Jacob_base, F21_ad, F22_ad, F23_ad, F24_ad, F41_ad, F42_ad, F43_ad, F44_ad)
    F1_aux_zlb, F2_aux_zlb, F3_aux_zlb, F4_aux_zlb, param_zlb = DSGE.SGU_solver_zlb(param_base, grid, Jacob_base, F21_ad_zlb, F22_ad_zlb, F23_ad_zlb, F24_ad_zlb, F41_ad_zlb, F42_ad_zlb, F43_ad_zlb, F44_ad_zlb)

    #TODO TEST OUTPUTS SAME

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
    HH_aux = vcat(hx_aux, gx)
    H1_aux = HH_aux[:, 1:nse]
    H2_aux = HH_aux[:, (end - nsh + 1):end]

    P_ref  = hcat(H1_aux, zeros(nse + nc, nc))
    Q_ref  = H2_aux

    cof_ref = hcat(C_ref, B_ref, A_ref)
    J_ref   = E_ref


    F1_zlb = vcat(F1_aux_zlb[1:nse, :], F1_aux_zlb[(ns+1):end, :])
    F2_zlb = vcat(F2_aux_zlb[1:nse, :], F2_aux_zlb[(ns+1):end, :])
    F3_zlb = vcat(F3_aux_zlb[1:nse, :], F3_aux_zlb[(ns+1):end, :])
    F4_zlb = vcat(F4_aux_zlb[1:nse, :], F4_aux_zlb[(ns+1):end, :])

    F11_zlb = F1_zlb[:, 1:nse]
    F31_zlb = F3_zlb[:, 1:nse]
    F32_zlb = F3_zlb[:, (nse+1):end]

    A_zlb = hcat(zeros(nse + nc, nse), F2_zlb)
    B_zlb = hcat(F11_zlb, F4_zlb)
    C_zlb = hcat(F31_zlb, zeros(nse + nc, nc))
    E_zlb = F32_zlb
    cof_zlb = hcat(C_zlb, B_zlb, A_zlb)
    J_zlb = E_zlb

    #shock covariance matrices
    sig_B_F = get(param_base, "sig_B_F", 0.0)
    sig_BB = get(param_base, "sig_BB", 0.0)
    sig_Z = get(param_base, "sig_Z", 0.0)
    sig_G = get(param_base, "sig_G", 0.0)
    sig_D = get(param_base, "sig_D", 0.0)
    sig_R = get(param_base, "sig_R", 0.0)
    sig_iota = get(param_base, "sig_iota", 0.0)
    sig_eta = get(param_base, "sig_eta", 0.0)
    sig_w = get(param_base, "sig_w", 0.0)

    sigma_full = Matrix{Float64}(I, nsh, nsh)
    if nsh >= 1
        sigma_full[1, 1] = 1.0
    end
    if nsh >= 2
        sigma_full[2, 2] = sig_B_F^2
    end
    if nsh >= 3
        sigma_full[3, 3] = sig_BB^2
    end
    if nsh >= 4
        sigma_full[4, 4] = sig_Z^2
    end
    if nsh >= 5
        sigma_full[5, 5] = sig_G^2
    end
    if nsh >= 6
        sigma_full[6, 6] = sig_D^2
    end
    if nsh >= 7
        sigma_full[7, 7] = sig_R^2
    end
    if nsh >= 8
        sigma_full[8, 8] = sig_iota^2
    end
    if nsh >= 9
        sigma_full[9, 9] = sig_eta^2
    end
    if nsh >= 10
        sigma_full[10, 10] = sig_w^2
    end

    sigma = nsh > 1 ? sigma_full[2:end, 2:end] : zeros(0, 0)
    sigma_zlb = nsh > 2 ? sigma_full[2:end-1, 2:end-1] : zeros(0, 0)

    #OccBin call, commented out for now
    #Ps, Ds, E, P_1, D_1, E_1 = occbin(m)
    
    #Measurement-equation objects
    HQ = nothing
    H_lik = nothing
    Q_lik = nothing
    H_ZLB = nothing
    if H_obs !== nothing
        num_data = Int(get(grid, "num_data", 10))
        HQ = H_obs * Q_ref
        H_lik = H_obs[1:num_data-1, :]
        Q_lik = Q_ref[:, 2:end]

        H_ZLB = zeros(num_data-2, size(H_obs, 2))
        H_ZLB[1:4, :] = H_obs[1:4, :]
        H_ZLB[5:end, :] = H_obs[6:end-1, :]
    end

    #TODO TEST ALL VARIABLES SAME

    return (
        indicator = indicator,
        hx = hx,
        gx = gx,
        P_ref = P_ref,
        Q_ref = Q_ref,
        A_ref = A_ref,
        B_ref = B_ref,
        C_ref = C_ref,
        E_ref = E_ref,
        cof_ref = cof_ref,
        J_ref = J_ref,
        F1 = F1,
        F2 = F2,
        F3 = F3,
        F4 = F4,
        F1_zlb = F1_zlb,
        F2_zlb = F2_zlb,
        F3_zlb = F3_zlb,
        F4_zlb = F4_zlb,
        A_zlb = A_zlb,
        B_zlb = B_zlb,
        C_zlb = C_zlb,
        E_zlb = E_zlb,
        cof_zlb = cof_zlb,
        J_zlb = J_zlb,
        HH_aux = HH_aux,
        H_obs = H_obs,
        HQ = HQ,
        H_lik = H_lik,
        Q_lik = Q_lik,
        H_ZLB = H_ZLB,
        sigma_full = sigma_full,
        sigma = sigma,
        sigma_zlb = sigma_zlb
    )
end
