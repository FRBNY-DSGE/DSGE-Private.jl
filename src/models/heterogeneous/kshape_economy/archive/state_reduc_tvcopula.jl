"""
    state_reduc_tvcopula!(m, m.dicts[:grid], m.dicts[:SS_stats], mu_dist, Value, mutil_c, Va)

Set up state reduction for the time-varying copula model.
Returns a named tuple with all computed variables.

Modifies `m.dicts[:grid]` and `m.dicts[:SS_stats]` in place.
"""
function state_reduc_tvcopula!(m::mBBQ, mu_dist::Array, Value::Array, mutil_c::Array, Va::Array)

    # Define inverse utility functions
    sigma = m.dicts[:param]["sigma"]
    invutil(u) = ((1 - sigma) .* u) .^ (1 / (1 - sigma))
    invmutil(mu) = (1 ./ mu) .^ (1 / sigma)

    # Build Xss based on adjust regime (TODO: SS/transition matrix object — handle separately)
    # adjust = string(get_setting(m, :adjust))
    adjust = "LT"
    if adjust == "G"
        Xss = vcat(
            vec(sum(sum(mu_dist, dims=2), dims=3)),
            vec(sum(sum(mu_dist, dims=1), dims=3)),
            vec(sum(sum(mu_dist, dims=2), dims=1)),
            vec(m.dicts[:SS_stats]["mu_dist"]),
            # TODO: SS/transition matrix object — handle separately (R_cb, q, u)
            log(m.dicts[:SS_stats]["R_cb"]), log(m.dicts[:param]["w_bar"]),
            log(m.dicts[:SS_stats]["A_b"]), log(m.dicts[:SS_stats]["B_b"]), log(m.dicts[:SS_stats]["A_g"] + 1),
            log(m.dicts[:SS_stats]["q"]), log(m.dicts[:SS_stats]["leverage"]), log(m.dicts[:SS_stats]["NW_b"]), log(m.dicts[:SS_stats]["R_cb"]), 0,
            log(m.dicts[:param]["pi_bar"]),
            log(m.dicts[:SS_stats]["Y"]), log(m.dicts[:SS_stats]["Chh"]), log(m.dicts[:SS_stats]["I2"]), log(m.dicts[:SS_stats]["Profit"] + m.dicts[:SS_stats]["Profit_FI"]), log(m.dicts[:SS_stats]["u"]), log(m.dicts[:SS_stats]["G"]), log(m.dicts[:SS_stats]["LT"]), log(m.dicts[:SS_stats]["R_cb"]), log(m.dicts[:SS_stats]["Y"] / (m.dicts[:SS_stats]["Y"] - m.dicts[:SS_stats]["B_F"])),
            log(1), log(1), log(m[:η]), log(1), log(m.dicts[:SS_stats]["Y"] / (m.dicts[:SS_stats]["Y"] - m.dicts[:SS_stats]["LT"] - m.dicts[:SS_stats]["C_b"])), log(1), log(1), log(1), log(1), log(m[:η] / (m[:η] - 1)),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )
    elseif adjust == "LT"
        Xss = vcat(
            vec(sum(sum(mu_dist, dims=2), dims=3)),
            vec(sum(sum(mu_dist, dims=1), dims=3)),
            vec(sum(sum(mu_dist, dims=2), dims=1)),
            vec(m.dicts[:SS_stats]["mu_dist"]),
            # TODO: SS/transition matrix object — handle separately (R_cb, q, u)
            log(m.dicts[:SS_stats]["R_cb"]), log(m.dicts[:param]["w_bar"]),
            log(m.dicts[:SS_stats]["A_b"]), log(m.dicts[:SS_stats]["B_b"]), log(m.dicts[:SS_stats]["A_g"] + 1),
            log(m.dicts[:SS_stats]["q"]), log(m.dicts[:SS_stats]["leverage"]), log(m.dicts[:SS_stats]["NW_b"]), log(m.dicts[:SS_stats]["R_cb"]), 0,
            log(m.dicts[:param]["pi_bar"]),
            log(m.dicts[:SS_stats]["Y"]), log(m.dicts[:SS_stats]["Chh"]), log(m.dicts[:SS_stats]["I2"]), log(m.dicts[:SS_stats]["Profit"] + m.dicts[:SS_stats]["Profit_FI"]), log(m.dicts[:SS_stats]["u"]), log(m.dicts[:SS_stats]["G"]), log(m.dicts[:SS_stats]["LT"]), log(m.dicts[:SS_stats]["R_cb"]), log(m.dicts[:SS_stats]["Y"] / (m.dicts[:SS_stats]["Y"] - m.dicts[:SS_stats]["B_F"])),
            log(1), log(1), log(m[:η]), log(1), log(m.dicts[:SS_stats]["Y"] / (m.dicts[:SS_stats]["Y"] - m.dicts[:SS_stats]["G"])), log(1), log(1), log(1), log(1), log(m[:η] / (m[:η] - 1)),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )
    else
        error("Unknown adjust value: $adjust")
    end

    # Build Yss based on adjust regime
    if adjust == "G"
        Yss = vcat(
            invutil(Value[:]),
            invmutil(mutil_c[:]),
            invmutil(Va[:]),
            log(m.dicts[:SS_stats]["MRS"]), log(m.dicts[:SS_stats]["A_hh"]), log(m.dicts[:SS_stats]["B_nb"] / (1 - m[:dr]) * m[:b_a_aux]), log(m.dicts[:SS_stats]["Chh"]),
            log(m.dicts[:SS_stats]["N"]), log(m.dicts[:SS_stats]["L"]), log(m.dicts[:SS_stats]["UB"]),
            log(m.dicts[:grid]["K"]), log(m.dicts[:SS_stats]["B2"]), log(m.dicts[:SS_stats]["B_gov_ncp"]),
            log(m.dicts[:SS_stats]["T"]), log(m.dicts[:SS_stats]["LT"]), log(m.dicts[:SS_stats]["G"]),
            log(m.dicts[:SS_stats]["Lambda"]), log(m.dicts[:param]["pi_bar"]),
            log(m.dicts[:SS_stats]["V"]), log.(m.dicts[:SS_stats]["J"]), log(m.dicts[:SS_stats]["r_l"]), log(m.dicts[:SS_stats]["v"]),
            log(m.dicts[:SS_stats]["Y"]), log(m.dicts[:SS_stats]["Profit"]), log(m.dicts[:SS_stats]["r_k"]), log(m.dicts[:SS_stats]["r_a"]), log(m.dicts[:SS_stats]["mc"]),
            log(m.dicts[:SS_stats]["u"]), log(m.dicts[:SS_stats]["n"]), log(m.dicts[:SS_stats]["M"]), log(m.dicts[:SS_stats]["f"]),  # TODO: SS/transition matrix object — handle separately (n)
            log(m.dicts[:SS_stats]["zz"]), log(m.dicts[:SS_stats]["xx"]), log(m.dicts[:SS_stats]["vv"]), log(m.dicts[:SS_stats]["ee"]), log(m.dicts[:SS_stats]["C_b"]), log(m.dicts[:SS_stats]["Profit_FI"]),
            log(m.dicts[:SS_stats]["R_a"]), log(m.dicts[:SS_stats]["R"]), log(m.dicts[:SS_stats]["I2"]),
            0,
            log(m.dicts[:SS_stats]["A_g"]), log(m.dicts[:SS_stats]["Y"]), log(m.dicts[:SS_stats]["C"]), log(m.dicts[:SS_stats]["I2"]), log(m.dicts[:param]["w_bar"]), log(m.dicts[:SS_stats]["Profit"] + m.dicts[:SS_stats]["Profit_FI"]),
            log(100 * m.dicts[:SS_stats]["u"]), log(m.dicts[:param]["pi_bar"]), log(m.dicts[:SS_stats]["R_cb"]), log(m.dicts[:param]["w_bar"]), log(m.dicts[:SS_stats]["G"]),  # TODO: SS/transition matrix object — handle separately (u, R_cb)
            log(m.dicts[:SS_stats]["Y"]), log(m.dicts[:SS_stats]["Chh"]), log(m.dicts[:SS_stats]["I2"]), log(m.dicts[:SS_stats]["Profit"] + m.dicts[:SS_stats]["Profit_FI"]), log(m.dicts[:SS_stats]["u"]), log(m.dicts[:SS_stats]["LT"]), log(m.dicts[:SS_stats]["A_g"]),
            log(m.dicts[:SS_stats]["lambda"]), 0, 0, log(m[:η]), 0, log(m.dicts[:SS_stats]["LT"]), log(m.dicts[:SS_stats]["G"]), log(m.dicts[:SS_stats]["LT"]), log(m.dicts[:SS_stats]["B_gov_ncp2"])  # TODO: SS/transition matrix object — handle separately (lambda)
        )
    elseif adjust == "LT"
        Yss = vcat(
            invutil(Value[:]),
            invmutil(mutil_c[:]),
            invmutil(Va[:]),
            log(m.dicts[:SS_stats]["MRS"]), log(m.dicts[:SS_stats]["A_hh"]), log(m.dicts[:SS_stats]["B_nb"] / (1 - m[:dr]) * m[:b_a_aux]), log(m.dicts[:SS_stats]["Chh"]),
            log(m.dicts[:SS_stats]["N"]), log(m.dicts[:SS_stats]["L"]), log(m.dicts[:SS_stats]["UB"]),
            log(m.dicts[:grid]["K"]), log(m.dicts[:SS_stats]["B2"]), log(m.dicts[:SS_stats]["B_gov_ncp"]),
            log(m.dicts[:SS_stats]["T"]), log(m.dicts[:SS_stats]["LT"]), log(m.dicts[:SS_stats]["G"]),
            log(m.dicts[:SS_stats]["Lambda"]), log(m.dicts[:param]["pi_bar"]),
            log(m.dicts[:SS_stats]["V"]), log.(m.dicts[:SS_stats]["J"]), log(m.dicts[:SS_stats]["r_l"]), log(m.dicts[:SS_stats]["v"]),
            log(m.dicts[:SS_stats]["Y"]), log(m.dicts[:SS_stats]["Profit"]), log(m.dicts[:SS_stats]["r_k"]), log(m.dicts[:SS_stats]["r_a"]), log(m.dicts[:SS_stats]["mc"]),
            log(m.dicts[:SS_stats]["u"]), log(m.dicts[:SS_stats]["n"]), log(m.dicts[:SS_stats]["M"]), log(m.dicts[:SS_stats]["f"]),  # TODO: SS/transition matrix object — handle separately (n)
            log(m.dicts[:SS_stats]["zz"]), log(m.dicts[:SS_stats]["xx"]), log(m.dicts[:SS_stats]["vv"]), log(m.dicts[:SS_stats]["ee"]), log(m.dicts[:SS_stats]["C_b"]), log(m.dicts[:SS_stats]["Profit_FI"]),
            log(m.dicts[:SS_stats]["R_a"]), log(m.dicts[:SS_stats]["R"]), log(m.dicts[:SS_stats]["I2"]),
            0,
            log(m.dicts[:SS_stats]["A_g"]), log(m.dicts[:SS_stats]["Y"]), log(m.dicts[:SS_stats]["C"]), log(m.dicts[:SS_stats]["I2"]), log(m.dicts[:param]["w_bar"]), log(m.dicts[:SS_stats]["Profit"] + m.dicts[:SS_stats]["Profit_FI"]),
            log(100 * m.dicts[:SS_stats]["u"]), log(m.dicts[:param]["pi_bar"]), log(m.dicts[:SS_stats]["R_cb"]), log(m.dicts[:param]["w_bar"]), log(m.dicts[:SS_stats]["G"]),  # TODO: SS/transition matrix object — handle separately (u, R_cb)
            log(m.dicts[:SS_stats]["Y"]), log(m.dicts[:SS_stats]["Chh"]), log(m.dicts[:SS_stats]["I2"]), log(m.dicts[:SS_stats]["Profit"] + m.dicts[:SS_stats]["Profit_FI"]), log(m.dicts[:SS_stats]["u"]), log(m.dicts[:SS_stats]["G"]), log(m.dicts[:SS_stats]["A_g"]),
            log(m.dicts[:SS_stats]["lambda"]), 0, 0, log(m[:η]), 0, log(m.dicts[:SS_stats]["LT"]), log(m.dicts[:SS_stats]["G"]), log(m.dicts[:SS_stats]["LT"]), log(m.dicts[:SS_stats]["B_gov_ncp2"])  # TODO: SS/transition matrix object — handle separately (lambda)
        )
    end

    # m.dicts[:grid] setup
    m.dicts[:grid]["maxdim"] = 10
    nb = Int(m.dicts[:grid]["nb"])
    na = Int(m.dicts[:grid]["na"])
    nse = Int(m.dicts[:grid]["nse"])
    m.dicts[:grid]["Ng"] = nb * na * nse
    Ng = m.dicts[:grid]["Ng"]

    m.dicts[:grid]["nb_copula"] = 10
    m.dicts[:grid]["na_copula"] = 10
    m.dicts[:grid]["nse_copula"] = 10

    # DCT matrices
    DC = Vector{Matrix{Float64}}(undef, 3)
    DC[1] = mydctmx(nb)
    DC[2] = mydctmx(na)
    DC[3] = mydctmx(nse)
    IDC = [transpose(DC[i]) for i in 1:3]

    DCD = Vector{Matrix{Float64}}(undef, 3)
    DCD[1] = mydctmx(m.dicts[:grid]["nb_copula"])
    DCD[2] = mydctmx(m.dicts[:grid]["na_copula"])
    DCD[3] = mydctmx(m.dicts[:grid]["nse_copula"])
    IDCD = [transpose(DCD[i]) for i in 1:3]

    # Distribution and copula
    distrSS = mu_dist
    CDF_SS = cumsum(cumsum(cumsum(distrSS, dims=1), dims=2), dims=3)
    COP_SS = CDF_SS

    m.dicts[:SS_stats]["COP_SS"] = COP_SS

    # Marginal distributions
    distr_b_SS = vec(sum(sum(distrSS, dims=2), dims=3))
    distr_b_SS = distr_b_SS / sum(distr_b_SS)
    distr_a_SS = vec(sum(sum(distrSS, dims=1), dims=3))
    distr_a_SS = distr_a_SS / sum(distr_a_SS)
    distr_se_SS = vec(sum(sum(distrSS, dims=1), dims=2))
    distr_se_SS = distr_se_SS / sum(distr_se_SS)

    CDF_b_SS = cumsum(distr_b_SS)
    CDF_a_SS = cumsum(distr_a_SS)
    CDF_se_SS = cumsum(distr_se_SS)

    # Copula marginal m.dicts[:grid]s
    m.dicts[:grid]["copula_marginal_b"] = copula_nodes_share_safe(distr_b_SS, vec(m.dicts[:grid]["b"]), m.dicts[:grid]["nb_copula"]; transform="logshift", alpha=0.3)
    m.dicts[:grid]["copula_marginal_a"] = copula_nodes_share_safe(distr_a_SS, vec(m.dicts[:grid]["a"]), m.dicts[:grid]["na_copula"]; transform="none", alpha=0.5)
    m.dicts[:grid]["copula_marginal_se"] = copula_nodes_share_safe(distr_se_SS, vec(m.dicts[:grid]["se"]), m.dicts[:grid]["nse_copula"]; transform="none", alpha=0.5, pinPenultimate=true)

    m.dicts[:grid]["reduc_copula"] = 10

    # Compression indices
    nb_cop = m.dicts[:grid]["nb_copula"]
    na_cop = m.dicts[:grid]["na_copula"]
    nse_cop = m.dicts[:grid]["nse_copula"]

    compressionIndexesCOP = Int[]
    for k in 1:nse_cop
        for j in 1:na_cop
            for i in 1:nb_cop
                if (i + j + k) <= m.dicts[:grid]["reduc_copula"] && !((i == 1 && j == 1) || (k == 1 && j == 1) || (k == 1 && i == 1))
                    linear_idx = i + (j - 1) * nb_cop + (k - 1) * nb_cop * na_cop
                    push!(compressionIndexesCOP, linear_idx)
                end
            end
        end
    end
    m.dicts[:grid]["compressionIndexesCOP"] = compressionIndexesCOP
    m.dicts[:grid]["nCOP"] = length(compressionIndexesCOP)

    s_m_b = m.dicts[:grid]["copula_marginal_b"]
    s_m_a = m.dicts[:grid]["copula_marginal_a"]
    s_m_se = m.dicts[:grid]["copula_marginal_se"]

    # Create sparse basis
    Poly, InvCheb, Gamma2 = createSparsebasis(m.dicts[:grid], m.dicts[:grid]["maxdim"], Xss)
    n1 = size(Poly)
    n2 = size(Gamma2)

    # m.dicts[:grid] parameters
    m.dicts[:grid]["os"] = length(Xss) - (m.dicts[:grid]["nb"] + m.dicts[:grid]["na"] + m.dicts[:grid]["nse"]) - (m.dicts[:grid]["nb"] * m.dicts[:grid]["na"] * m.dicts[:grid]["nse"]) #os means other states
    m.dicts[:grid]["oc"] = length(Yss) - 3 * n1[1] #oc means other controls
    m.dicts[:grid]["oc_summary"] = 7
    m.dicts[:grid]["oc_agg"] = m.dicts[:grid]["oc"] - m.dicts[:grid]["oc_summary"]
    m.dicts[:grid]["os_process"] = 17
    m.dicts[:grid]["os_agg"] = m.dicts[:grid]["os"] - m.dicts[:grid]["os_process"]

    nCOP = Int(m.dicts[:grid]["nCOP"])
    nPoly = Int(n1[2])
    nFullCtrl = Int(3 * Ng + m.dicts[:grid]["oc"])
    nRedCtrl = Int(3 * nPoly + m.dicts[:grid]["oc"])
    nFullMarg = Int(n2[1])
    nRedMarg = Int(n2[2])
    nRedStates = Int(nRedMarg + nCOP + m.dicts[:grid]["os"])

    Gamma_state = sparse(Gamma2)

    # InvGamma matrix
    InvGamma = spzeros(Float64, Int(nRedStates + nFullCtrl), Int(nRedStates + nRedCtrl))
    InvGamma[1:nRedStates, 1:nRedStates] = sparse(I, nRedStates, nRedStates)

    r0 = nRedStates
    c0 = nRedStates
    InvGamma[r0 .+ (1:Ng), c0 .+ (1:nPoly)] = transpose(InvCheb)
    InvGamma[r0 + Ng .+ (1:Ng), c0 + nPoly .+ (1:nPoly)] = transpose(InvCheb)
    InvGamma[r0 + 2*Ng .+ (1:Ng), c0 + 2*nPoly .+ (1:nPoly)] = transpose(InvCheb)
    InvGamma[r0 + 3*Ng .+ (1:m.dicts[:grid]["oc"]), c0 + 3*nPoly .+ (1:m.dicts[:grid]["oc"])] = sparse(I, m.dicts[:grid]["oc"], m.dicts[:grid]["oc"])
    InvGamma = transpose(InvGamma)

    # Gamma_control matrix
    Gamma_control = spzeros(Int(3 * n1[1] + m.dicts[:grid]["oc"]), Int(3 * n1[2] + m.dicts[:grid]["oc"]))
    Gamma_control[1:n1[1], 1:n1[2]] = Poly
    Gamma_control[n1[1] .+ (1:n1[1]), n1[2] .+ (1:n1[2])] = Poly
    Gamma_control[2*n1[1] .+ (1:n1[1]), 2*n1[2] .+ (1:n1[2])] = Poly
    Gamma_control[3*n1[1] .+ (1:m.dicts[:grid]["oc"]), 3*n1[2] .+ (1:m.dicts[:grid]["oc"])] = sparse(I, m.dicts[:grid]["oc"], m.dicts[:grid]["oc"])

    # m.dicts[:grid] state/control dimensions
    m.dicts[:grid]["numstates"] = (nb + na + nse - 3) + nCOP + m.dicts[:grid]["os"]
    m.dicts[:grid]["numstates_shocks"] = 10
    m.dicts[:grid]["numstates_endo"] = m.dicts[:grid]["numstates"] - m.dicts[:grid]["numstates_shocks"]
    m.dicts[:grid]["numcontrols"] = 3 * n1[2] + m.dicts[:grid]["oc"]
    m.dicts[:grid]["num_endo"] = m.dicts[:grid]["numstates_endo"] + m.dicts[:grid]["numcontrols"]

    # Initialize state and control vectors
    State = zeros(Int(m.dicts[:grid]["numstates"]))
    State_m = copy(State)
    Contr = zeros(Int(m.dicts[:grid]["numcontrols"]))
    Contr_m = copy(Contr)

    return (
        Xss = Xss,
        Yss = Yss,
        DC = DC,
        IDC = IDC,
        DCD = DCD,
        IDCD = IDCD,
        Gamma_state = Gamma_state,
        Gamma_control = Gamma_control,
        InvGamma = InvGamma,
        State = State,
        State_m = State_m,
        Contr = Contr,
        Contr_m = Contr_m,
        distrSS = distrSS,
        CDF_SS = CDF_SS,
        COP_SS = COP_SS,
        distr_b_SS = distr_b_SS,
        distr_a_SS = distr_a_SS,
        distr_se_SS = distr_se_SS,
        CDF_b_SS = CDF_b_SS,
        CDF_a_SS = CDF_a_SS,
        CDF_se_SS = CDF_se_SS,
        compressionIndexesCOP = compressionIndexesCOP,
        Poly = Poly,
        InvCheb = InvCheb,
        Gamma2 = Gamma2,
        nPoly = nPoly,
        nFullCtrl = nFullCtrl,
        nRedCtrl = nRedCtrl,
        nFullMarg = nFullMarg,
        nRedMarg = nRedMarg,
        nRedStates = nRedStates
    )
end
