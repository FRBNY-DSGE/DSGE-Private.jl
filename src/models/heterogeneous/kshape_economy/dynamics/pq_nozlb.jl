using LinearAlgebra

"""
    pq(m, hx, gx, F1_aux, F2_aux, F3_aux, F4_aux)

Build the single-regime (no ZLB) objects used for filtering:
`H_aux`, `P_ref`, `Q_ref`, `HQ`, `SIGMA_full`, `SIGMA`.
"""


function pq(
    m,
    hx,
    gx,
    F1_aux,
    F2_aux,
    F3_aux,
    F4_aux)

    grid = m.dicts[:grid]

    nse = Int(grid["numstates_endo"])
    ns  = Int(grid["numstates"])
    nc  = Int(grid["numcontrols"])
    nsh = Int(grid["numstates_shocks"])
    nvars = Int(get(grid, "num_endo", nse + nc))

    #temp for testing with old jacobian
    # nse = 119
    # ns = 129
    # nc = 405
    # nsh = 10
    # nvars = 524

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

    P_ref = hcat(H1_aux, zeros(nse + nc, nc))
    Q_ref = H2_aux

    cof_ref = hcat(C_ref, B_ref, A_ref)
    J_ref = E_ref

    # Shock covariance (MATLAB: SIGMA_full / SIGMA)
    SIGMA_full = Matrix{Float64}(I, nsh, nsh)
    if nsh >= 1
        SIGMA_full[1, 1] = 1.0
    end
    if nsh >= 2
        SIGMA_full[2, 2] = m[:σ_B_F]^2
    end
    if nsh >= 3
        SIGMA_full[3, 3] = m[:σ_BB]^2
    end
    if nsh >= 4
        SIGMA_full[4, 4] = m[:σ_Z]^2
    end
    if nsh >= 5
        SIGMA_full[5, 5] = m[:σ_G]^2
    end
    if nsh >= 6
        SIGMA_full[6, 6] = m[:σ_D]^2
    end
    if nsh >= 7
        SIGMA_full[7, 7] = m[:σ_R]^2
    end
    if nsh >= 8
        SIGMA_full[8, 8] = m[:σ_ι]^2
    end
    if nsh >= 9
        SIGMA_full[9, 9] = m[:σ_η]^2
    end
    if nsh >= 10
        SIGMA_full[10, 10] = m[:σ_w]^2
    end

    SIGMA = nsh > 1 ? SIGMA_full[2:end, 2:end] : zeros(0, 0)

    #create H_aux
    oc = m.dicts[:grid]["oc"]

    H_aux = zeros(10, nvars)
    lenSS = length(m.grids[:StateSS])
    lenC = length(m.grids[:ControlSS])
    state_id, control_id = DSGE.build_indices(grid, lenSS, lenC)

    H_aux[1 ,end-lenC+control_id[:Y_t]]                = 1
    H_aux[1 ,end-lenC+control_id[:YY_lag_t]]           = -1

    H_aux[2 ,end-lenC+control_id[:C_t]]                = 1
    H_aux[2 ,end-lenC+control_id[:CC_lag_t]]           = -1

    H_aux[3 ,end-lenC+control_id[:I_t]]                = 1
    H_aux[3 ,end-lenC+control_id[:II_lag_t]]           = -1
    
    H_aux[4 ,end-lenC+control_id[:pi_t]]               = 1

    H_aux[5 ,ns - lenSS + state_id[:R_cb_t]]           = 1

    H_aux[6 ,ns - lenSS + state_id[:w_t]]             = 1
    H_aux[6 ,end-lenC+control_id[:w_lag_t]]           = -1

    H_aux[7 ,end-lenC+control_id[:unemp_t]]           = 1

    H_aux[8 ,end-lenC+control_id[:LT_t]]              = 1
    H_aux[8 ,end-lenC+control_id[:LT2_lag_t]]         = -1

    H_aux[9 ,end-lenC+control_id[:Profit_obs_t]]      = 1
    H_aux[9 ,end-lenC+control_id[:PPROFIT_lag_t]]    = -1

    H_aux[10 ,ns - lenSS + state_id[:A_gaux_t]]      = 1

    # Measurement helpers
    HQ = H_aux * Q_ref

    return H_aux,
        P_ref,
        Q_ref,
        HQ,
        SIGMA_full,
        SIGMA
end
