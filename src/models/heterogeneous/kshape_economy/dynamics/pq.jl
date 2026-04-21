using LinearAlgebra

"""
    pq(m, hx, gx, F1_aux, F2_aux, F3_aux, F4_aux, param, indicator_1,
       F1_ZLB_aux, F2_ZLB_aux, F3_ZLB_aux, F4_ZLB_aux;
       data_est = nothing,
       H_aux = nothing,
       ZLB_duration_1 = nothing,
       Fss_ZLB = nothing,
       R_cb_ind = nothing)

Build the objects used in `reference/post_density_covid_Kshape_MCMC_v2.m` for filtering:
`P_ref`, `Q_ref`, shock covariances, optional measurement helpers, ZLB indicator,
OccBin tables `Ps_aux/Ds_aux/Es_aux`, and `D_ZLB`.

Keyword inputs mirror the extra objects that exist only in the full posterior script.
If a keyword is omitted, the corresponding outputs are returned as `nothing` (except
`indicator`, which falls back to `indicator_1`).
"""


function pq(
    m::mBBQ,
    hx,
    gx,
    F1_aux,
    F2_aux,
    F3_aux,
    F4_aux,
    indicator_1,
    F1_ZLB_aux,
    F2_ZLB_aux,
    F3_ZLB_aux,
    F4_ZLB_aux,
    data_est,
    H_aux,
    ZLB_duration_1,
    Fss_ZLB)

    grid = m.dicts[:grid]

    nse = Int(grid["numstates_endo"])
    ns  = Int(grid["numstates"])
    nc  = Int(grid["numcontrols"])
    nsh = Int(grid["numstates_shocks"])
    nvars = Int(get(grid, "num_endo", nse + nc))

    #temp for testing with old jacobian
    nse = 119
    ns = 129
    nc = 405
    nsh = 10
    nvars = 524

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

    # ZLB linearization (same row reordering as baseline)
    F1_zlb = vcat(F1_ZLB_aux[1:nse, :], F1_ZLB_aux[(ns+1):end, :])
    F2_zlb = vcat(F2_ZLB_aux[1:nse, :], F2_ZLB_aux[(ns+1):end, :])
    F3_zlb = vcat(F3_ZLB_aux[1:nse, :], F3_ZLB_aux[(ns+1):end, :])
    F4_zlb = vcat(F4_ZLB_aux[1:nse, :], F4_ZLB_aux[(ns+1):end, :])

    F11_zlb = F1_zlb[:, 1:nse]
    F31_zlb = F3_zlb[:, 1:nse]
    F32_zlb = F3_zlb[:, (nse+1):end]

    A_zlb = hcat(zeros(nse + nc, nse), F2_zlb)
    B_zlb = hcat(F11_zlb, F4_zlb)
    C_zlb = hcat(F31_zlb, zeros(nse + nc, nc))
    E_zlb = F32_zlb
    cof_zlb = hcat(C_zlb, B_zlb, A_zlb)
    J_zlb = E_zlb

    function getp(p::AbstractDict, key::AbstractString)
        if haskey(p, key)
            return p[key]
        elseif haskey(p, Symbol(key))
            return p[Symbol(key)]
        end
        error("Missing parameter key: $key")
    end

    function as_float(x)
        x isa AbstractArray ? Float64(x[1]) : Float64(x)
    end

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

    # Measurement helpers
    HQ = H_aux * Q_ref

    # R_cb = as_float(getp(param, "R_cb"))
    R_cb = m.dicts[:SS_stats]["R_cb"]
    rates = vec(data_est[5, :])
    ZLB_indicator = exp.(rates) .* R_cb .< 1 + 1e-8

    D_ZLB = vcat(Fss_ZLB[1:nse, :], Fss_ZLB[(ns+1):end, :])

    # OccBin precomputation (unique ZLB durations)
    unique_EZLB_duration_1 = sort(unique(vec(ZLB_duration_1)))
    ndur = length(unique_EZLB_duration_1)

    Cb = cof_zlb[:, 1:nvars]
    Bb = cof_zlb[:, (nvars + 1):(2 * nvars)]
    Ab = cof_zlb[:, (2 * nvars + 1):(3 * nvars)]

    decrulea = P_ref#[:, 1:nse]

    Ps_aux = zeros(nvars, nvars, ndur)
    Ds_aux = zeros(nvars, ndur)
    Es_aux = zeros(size(J_zlb, 1), size(J_zlb, 2), ndur)

    Dv = vec(D_ZLB)

    for (kk, Tmax) in enumerate(unique_EZLB_duration_1)
        T = Int(Tmax)
        T < 1 && error("Invalid ZLB duration Tmax=$Tmax")

        # Terminal step at horizon T (MATLAB OccBin scripts)
        invmat = Ab * decrulea + Bb
        Ps1 = -(invmat \ Cb)
        Ds1 = -(invmat \ Dv)

        # Backward recursion to obtain P(:,:,1) at the start of the spell
        Ps_cur = Ps1
        Ds_cur = Ds1
        for _ in 1:(T - 1)
            invmat = Ab * Ps_cur + Bb
            Ps_cur = -(invmat \ Cb)
            Ds_cur = -(invmat \ (Ab * Ds_cur + Dv))
        end

        invmat = Ab * Ps_cur + Bb
        Es_cur = -(invmat \ J_zlb)

        Ps_aux[:, :, kk] = Ps_cur
        Ds_aux[:, kk] = Ds_cur
        Es_aux[:, :, kk] = Es_cur
    end

    indicator = indicator_1

    return H_aux,
        P_ref,
        Q_ref,
        HQ,
        SIGMA_full,
        SIGMA,
        ZLB_indicator,
        ZLB_duration_1,
        unique_EZLB_duration_1,
        Ps_aux,
        Ds_aux,
        Es_aux,
        D_ZLB,
        indicator
end
