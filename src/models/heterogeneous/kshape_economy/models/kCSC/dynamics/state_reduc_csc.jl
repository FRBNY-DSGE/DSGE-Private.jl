using SparseArrays
using LinearAlgebra

# include("../../helpers/mydctmx.jl")
# include("../../helpers/copula_nodes_share_safe.jl")
# include("../../helpers/createSparsebasis.jl")

"""
    state_reduc_tvcopula!(param, grid, SS_stats, mu_dist, Value, mutil_c, Va)

Set up state reduction for the CSC time-varying copula model.
Returns a named tuple with all computed variables.

Modifies `grid` and `SS_stats` in place.
"""
function state_reduc_csc(param::Dict, grid::Dict, SS_stats::Dict,
                               mu_dist::Array, Value::Array, mutil_c::Array, Va::Array)

    sigma = param["sigma"]
    invutil(u) = ((1 - sigma) .* u) .^ (1 / (1 - sigma))
    invmutil(mu) = (1 ./ mu) .^ (1 / sigma)

    Xss = vcat(
        vec(sum(sum(mu_dist, dims=2), dims=3)),
        vec(sum(sum(mu_dist, dims=1), dims=3)),
        vec(sum(sum(mu_dist, dims=2), dims=1)),
        vec(SS_stats["mu_dist"]),
        log(param["R_cb"]), log(param["w_bar_1"]), log(param["w_bar_2"]), log(SS_stats["w"]),
        log(SS_stats["A_b"]), log(SS_stats["B_b"]), log(SS_stats["A_g"]),
        log(param["q"]), log(SS_stats["leverage"]), log(SS_stats["NW_b"]), log(param["R_cb"]), 0,
        log(param["pi_bar"]),
        log(SS_stats["Y"]), log(SS_stats["Chh"]), log(SS_stats["I2"]),
        log(SS_stats["Profit"] + SS_stats["Profit_FI"]),
        log(param["u_1"]), log(param["u_2"]), log(SS_stats["u"]),
        log(SS_stats["G"]), log(SS_stats["LT"]), log(param["R_cb"]),
        0, 0, 0, 0,
        log(SS_stats["Y"] / (SS_stats["Y"] - SS_stats["B_F"])),
        log(param["Z"]), log(1), log(param["eta"]), log(1),
        log(SS_stats["Y"] / (SS_stats["Y"] - SS_stats["LT"] - SS_stats["C_b"])),
        log(1), log(1), log(1), log(1),
        log(param["eta"] / (param["eta"] - 1)),
        zeros(15)
    )

    Yss = vcat(
        invutil(Value[:]),
        invmutil(mutil_c[:]),
        invmutil(Va[:]),
        log(SS_stats["MRS"]), log(SS_stats["A_hh"]),
        log(SS_stats["B_nb"] / (1 - param["death_rate"]) * param["b_a_aux"]),
        log(SS_stats["Chh"]),
        log(SS_stats["N_tilde_1"]), log(SS_stats["N_tilde_2"]),
        log(SS_stats["L_1"]), log(SS_stats["L_2"]), log(SS_stats["UB"]),
        log(SS_stats["u_1"]), log(SS_stats["u_2"]), log(SS_stats["u"]),
        log(SS_stats["U_1"]), log(SS_stats["U_2"]),
        log(grid["K"]), log(SS_stats["B2"]), log(SS_stats["B_gov_ncp"]),
        log(SS_stats["T"]), log(SS_stats["LT"]), log(SS_stats["G"]),
        log(SS_stats["Lambda"]), log(param["pi_bar"]),
        log(SS_stats["V_1"]), log(SS_stats["V_2"]), vec(log.(SS_stats["J"])),
        log(SS_stats["r_l_1"]), log(SS_stats["r_l_2"]), log(SS_stats["v"]),
        log(SS_stats["Y"]), log(SS_stats["Profit"]), log(SS_stats["r_k"]),
        log(SS_stats["r_a"]), log(SS_stats["mc"]),
        log(param["n_1"]), log(param["n_2"]),
        log(SS_stats["M_1"]), log(SS_stats["M_2"]),
        log(SS_stats["f_1"]), log(SS_stats["f_2"]),
        log(SS_stats["zz"]), log(SS_stats["xx"]), log(SS_stats["vv"]),
        log(SS_stats["ee"]), log(SS_stats["C_b"]), log(SS_stats["Profit_FI"]),
        log(SS_stats["R_a"]), log(SS_stats["R"]), log(SS_stats["I2"]),
        0,
        log(SS_stats["A_g"]), log(SS_stats["Y"]), log(SS_stats["C"]), log(SS_stats["I2"]),
        log(param["w_bar_1"]), log(param["w_bar_2"]), log(SS_stats["w"]),
        log(SS_stats["Profit"] + SS_stats["Profit_FI"]),
        log(100 * param["u_1"]), log(100 * param["u_2"]), log(100 * SS_stats["u"]),
        log(param["pi_bar"]), log(param["R_cb"]),
        log(param["w_bar_1"]), log(param["w_bar_2"]), log(SS_stats["w"]),
        log(SS_stats["G"]),
        log(SS_stats["Y"]), log(SS_stats["Chh"]), log(SS_stats["I2"]),
        log(SS_stats["Profit"] + SS_stats["Profit_FI"]),
        log(param["u_1"]), log(param["u_2"]), log(SS_stats["u"]),
        log(SS_stats["LT"]), log(SS_stats["A_g"]),
        log(param["lambda_1"]), log(param["lambda_2"]),
        0, 0, log(param["eta"]), 0,
        log(SS_stats["LT"]), log(SS_stats["G"]), log(SS_stats["LT"]),
        log(SS_stats["B_gov_ncp2"])
    )

    grid["maxdim"] = 10
    nb = Int(grid["nb"])
    na = Int(grid["na"])
    nse = Int(grid["nse"])
    grid["Ng"] = nb * na * nse
    Ng = grid["Ng"]

    grid["nb_copula"] = 10
    grid["na_copula"] = 10
    grid["nse_copula"] = 10

    DC = Vector{Matrix{Float64}}(undef, 3)
    DC[1] = mydctmx(nb)
    DC[2] = mydctmx(na)
    DC[3] = mydctmx(nse)
    IDC = [transpose(DC[i]) for i in 1:3]

    DCD = Vector{Matrix{Float64}}(undef, 3)
    DCD[1] = mydctmx(grid["nb_copula"])
    DCD[2] = mydctmx(grid["na_copula"])
    DCD[3] = mydctmx(grid["nse_copula"])
    IDCD = [transpose(DCD[i]) for i in 1:3]

    distrSS = mu_dist
    CDF_SS = cumsum(cumsum(cumsum(distrSS, dims=1), dims=2), dims=3)
    COP_SS = CDF_SS

    SS_stats["COP_SS"] = COP_SS

    distr_b_SS = vec(sum(sum(distrSS, dims=2), dims=3))
    distr_b_SS = distr_b_SS / sum(distr_b_SS)
    distr_a_SS = vec(sum(sum(distrSS, dims=1), dims=3))
    distr_a_SS = distr_a_SS / sum(distr_a_SS)
    distr_se_SS = vec(sum(sum(distrSS, dims=1), dims=2))
    distr_se_SS = distr_se_SS / sum(distr_se_SS)

    CDF_b_SS = cumsum(distr_b_SS)
    CDF_a_SS = cumsum(distr_a_SS)
    CDF_se_SS = cumsum(distr_se_SS)

    grid["copula_marginal_b"] = copula_nodes_share_safe(
        distr_b_SS, vec(grid["b"]), grid["nb_copula"];
        transform="logshift", alpha=0.3
    )
    grid["copula_marginal_a"] = copula_nodes_share_safe(
        distr_a_SS, vec(grid["a"]), grid["na_copula"];
        transform="none", alpha=0.5
    )
    grid["copula_marginal_se"] = copula_nodes_share_safe(
        distr_se_SS, vec(grid["se"]), grid["nse_copula"];
        transform="none", alpha=0.5, pinPenultimate=true
    )

    grid["reduc_copula"] = 10

    nb_cop = grid["nb_copula"]
    na_cop = grid["na_copula"]
    nse_cop = grid["nse_copula"]

    compressionIndexesCOP = Int[]
    for k in 1:nse_cop
        for j in 1:na_cop
            for i in 1:nb_cop
                if (i + j + k) <= grid["reduc_copula"] &&
                   !((i == 1 && j == 1) || (k == 1 && j == 1) || (k == 1 && i == 1))
                    linear_idx = i + (j - 1) * nb_cop + (k - 1) * nb_cop * na_cop
                    push!(compressionIndexesCOP, linear_idx)
                end
            end
        end
    end
    grid["compressionIndexesCOP"] = compressionIndexesCOP
    grid["nCOP"] = length(compressionIndexesCOP)

    s_m_b = grid["copula_marginal_b"]
    s_m_a = grid["copula_marginal_a"]
    s_m_se = grid["copula_marginal_se"]

    Poly, InvCheb, Gamma2 = createSparsebasis(grid, grid["maxdim"], Xss)
    n1 = size(Poly)
    n2 = size(Gamma2)

    grid["os"] = length(Xss) - (grid["nb"] + grid["na"] + grid["nse"]) -
                 (grid["nb"] * grid["na"] * grid["nse"])
    grid["oc"] = length(Yss) - 3 * n1[1]
    grid["oc_summary"] = 14
    grid["oc_agg"] = grid["oc"] - grid["oc_summary"]
    grid["os_process"] = 30
    grid["os_agg"] = grid["os"] - grid["os_process"]

    nCOP = Int(grid["nCOP"])
    nPoly = Int(n1[2])
    nFullCtrl = Int(3 * Ng + grid["oc"])
    nRedCtrl = Int(3 * nPoly + grid["oc"])
    nFullMarg = Int(n2[1])
    nRedMarg = Int(n2[2])
    nRedStates = Int(nRedMarg + nCOP + grid["os"])

    Gamma_state = sparse(Gamma2)

    InvGamma = spzeros(Float64, Int(nRedStates + nFullCtrl), Int(nRedStates + nRedCtrl))
    InvGamma[1:nRedStates, 1:nRedStates] = sparse(I, nRedStates, nRedStates)

    r0 = nRedStates
    c0 = nRedStates
    InvGamma[r0 .+ (1:Ng), c0 .+ (1:nPoly)] = transpose(InvCheb)
    InvGamma[r0 + Ng .+ (1:Ng), c0 + nPoly .+ (1:nPoly)] = transpose(InvCheb)
    InvGamma[r0 + 2 * Ng .+ (1:Ng), c0 + 2 * nPoly .+ (1:nPoly)] = transpose(InvCheb)
    InvGamma[r0 + 3 * Ng .+ (1:grid["oc"]), c0 + 3 * nPoly .+ (1:grid["oc"])] =
        sparse(I, grid["oc"], grid["oc"])
    InvGamma = transpose(InvGamma)

    Gamma_control = spzeros(Int(3 * n1[1] + grid["oc"]), Int(3 * n1[2] + grid["oc"]))
    Gamma_control[1:n1[1], 1:n1[2]] = Poly
    Gamma_control[n1[1] .+ (1:n1[1]), n1[2] .+ (1:n1[2])] = Poly
    Gamma_control[2 * n1[1] .+ (1:n1[1]), 2 * n1[2] .+ (1:n1[2])] = Poly
    Gamma_control[3 * n1[1] .+ (1:grid["oc"]), 3 * n1[2] .+ (1:grid["oc"])] =
        sparse(I, grid["oc"], grid["oc"])

    grid["numstates"] = (nb + na + nse - 3) + nCOP + grid["os"]
    grid["numstates_shocks"] = 15
    grid["numstates_endo"] = grid["numstates"] - grid["numstates_shocks"]
    grid["numcontrols"] = 3 * n1[2] + grid["oc"]
    grid["num_endo"] = grid["numstates_endo"] + grid["numcontrols"]

    State = zeros(Int(grid["numstates"]))
    State_m = copy(State)
    Contr = zeros(Int(grid["numcontrols"]))
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



function state_reduc_csc(m::kCSC)
    param = m.dicts[:param]
    grid = m.dicts[:grid]
    SS_stats = m.dicts[:SS_stats]
    mu_dist = m.grids[:mu_dist]
    Value = m.grids[:Value]
    mutil_c = m.grids[:mutil_c]
    Va = m.grids[:Va]
    

    sigma = param["sigma"]
    invutil(u) = ((1 - sigma) .* u) .^ (1 / (1 - sigma))
    invmutil(mu) = (1 ./ mu) .^ (1 / sigma)

    Xss = vcat(
        vec(sum(sum(mu_dist, dims=2), dims=3)),
        vec(sum(sum(mu_dist, dims=1), dims=3)),
        vec(sum(sum(mu_dist, dims=2), dims=1)),
        vec(SS_stats["mu_dist"]),
        log(param["R_cb"]), log(param["w_bar_1"]), log(param["w_bar_2"]), log(SS_stats["w"]),
        log(SS_stats["A_b"]), log(SS_stats["B_b"]), log(SS_stats["A_g"]),
        log(param["q"]), log(SS_stats["leverage"]), log(SS_stats["NW_b"]), log(param["R_cb"]), 0,
        log(param["pi_bar"]),
        log(SS_stats["Y"]), log(SS_stats["Chh"]), log(SS_stats["I2"]),
        log(SS_stats["Profit"] + SS_stats["Profit_FI"]),
        log(param["u_1"]), log(param["u_2"]), log(SS_stats["u"]),
        log(SS_stats["G"]), log(SS_stats["LT"]), log(param["R_cb"]),
        0, 0, 0, 0,
        log(SS_stats["Y"] / (SS_stats["Y"] - SS_stats["B_F"])),
        log(param["Z"]), log(1), log(param["eta"]), log(1),
        log(SS_stats["Y"] / (SS_stats["Y"] - SS_stats["LT"] - SS_stats["C_b"])),
        log(1), log(1), log(1), log(1),
        log(param["eta"] / (param["eta"] - 1)),
        zeros(15)
    )

    Yss = vcat(
        invutil(Value[:]),
        invmutil(mutil_c[:]),
        invmutil(Va[:]),
        log(SS_stats["MRS"]), log(SS_stats["A_hh"]),
        log(SS_stats["B_nb"] / (1 - param["death_rate"]) * param["b_a_aux"]),
        log(SS_stats["Chh"]),
        log(SS_stats["N_tilde_1"]), log(SS_stats["N_tilde_2"]),
        log(SS_stats["L_1"]), log(SS_stats["L_2"]), log(SS_stats["UB"]),
        log(SS_stats["u_1"]), log(SS_stats["u_2"]), log(SS_stats["u"]),
        log(SS_stats["U_1"]), log(SS_stats["U_2"]),
        log(grid["K"]), log(SS_stats["B2"]), log(SS_stats["B_gov_ncp"]),
        log(SS_stats["T"]), log(SS_stats["LT"]), log(SS_stats["G"]),
        log(SS_stats["Lambda"]), log(param["pi_bar"]),
        log(SS_stats["V_1"]), log(SS_stats["V_2"]), vec(log.(SS_stats["J"])),
        log(SS_stats["r_l_1"]), log(SS_stats["r_l_2"]), log(SS_stats["v"]),
        log(SS_stats["Y"]), log(SS_stats["Profit"]), log(SS_stats["r_k"]),
        log(SS_stats["r_a"]), log(SS_stats["mc"]),
        log(param["n_1"]), log(param["n_2"]),
        log(SS_stats["M_1"]), log(SS_stats["M_2"]),
        log(SS_stats["f_1"]), log(SS_stats["f_2"]),
        log(SS_stats["zz"]), log(SS_stats["xx"]), log(SS_stats["vv"]),
        log(SS_stats["ee"]), log(SS_stats["C_b"]), log(SS_stats["Profit_FI"]),
        log(SS_stats["R_a"]), log(SS_stats["R"]), log(SS_stats["I2"]),
        0,
        log(SS_stats["A_g"]), log(SS_stats["Y"]), log(SS_stats["C"]), log(SS_stats["I2"]),
        log(param["w_bar_1"]), log(param["w_bar_2"]), log(SS_stats["w"]),
        log(SS_stats["Profit"] + SS_stats["Profit_FI"]),
        log(100 * param["u_1"]), log(100 * param["u_2"]), log(100 * SS_stats["u"]),
        log(param["pi_bar"]), log(param["R_cb"]),
        log(param["w_bar_1"]), log(param["w_bar_2"]), log(SS_stats["w"]),
        log(SS_stats["G"]),
        log(SS_stats["Y"]), log(SS_stats["Chh"]), log(SS_stats["I2"]),
        log(SS_stats["Profit"] + SS_stats["Profit_FI"]),
        log(param["u_1"]), log(param["u_2"]), log(SS_stats["u"]),
        log(SS_stats["LT"]), log(SS_stats["A_g"]),
        log(param["lambda_1"]), log(param["lambda_2"]),
        0, 0, log(param["eta"]), 0,
        log(SS_stats["LT"]), log(SS_stats["G"]), log(SS_stats["LT"]),
        log(SS_stats["B_gov_ncp2"])
    )

    grid["maxdim"] = 10
    nb = Int(grid["nb"])
    na = Int(grid["na"])
    nse = Int(grid["nse"])
    grid["Ng"] = nb * na * nse
    Ng = grid["Ng"]

    grid["nb_copula"] = 10
    grid["na_copula"] = 10
    grid["nse_copula"] = 10

    DC = Vector{Matrix{Float64}}(undef, 3)
    DC[1] = mydctmx(nb)
    DC[2] = mydctmx(na)
    DC[3] = mydctmx(nse)
    IDC = [transpose(DC[i]) for i in 1:3]

    DCD = Vector{Matrix{Float64}}(undef, 3)
    DCD[1] = mydctmx(grid["nb_copula"])
    DCD[2] = mydctmx(grid["na_copula"])
    DCD[3] = mydctmx(grid["nse_copula"])
    IDCD = [transpose(DCD[i]) for i in 1:3]

    distrSS = mu_dist
    CDF_SS = cumsum(cumsum(cumsum(distrSS, dims=1), dims=2), dims=3)
    COP_SS = CDF_SS

    SS_stats["COP_SS"] = COP_SS

    distr_b_SS = vec(sum(sum(distrSS, dims=2), dims=3))
    distr_b_SS = distr_b_SS / sum(distr_b_SS)
    distr_a_SS = vec(sum(sum(distrSS, dims=1), dims=3))
    distr_a_SS = distr_a_SS / sum(distr_a_SS)
    distr_se_SS = vec(sum(sum(distrSS, dims=1), dims=2))
    distr_se_SS = distr_se_SS / sum(distr_se_SS)

    CDF_b_SS = cumsum(distr_b_SS)
    CDF_a_SS = cumsum(distr_a_SS)
    CDF_se_SS = cumsum(distr_se_SS)

    grid["copula_marginal_b"] = copula_nodes_share_safe(
        distr_b_SS, vec(grid["b"]), grid["nb_copula"];
        transform="logshift", alpha=0.3
    )
    grid["copula_marginal_a"] = copula_nodes_share_safe(
        distr_a_SS, vec(grid["a"]), grid["na_copula"];
        transform="none", alpha=0.5
    )
    grid["copula_marginal_se"] = copula_nodes_share_safe(
        distr_se_SS, vec(grid["se"]), grid["nse_copula"];
        transform="none", alpha=0.5, pinPenultimate=true
    )

    grid["reduc_copula"] = 10

    nb_cop = grid["nb_copula"]
    na_cop = grid["na_copula"]
    nse_cop = grid["nse_copula"]

    compressionIndexesCOP = Int[]
    for k in 1:nse_cop
        for j in 1:na_cop
            for i in 1:nb_cop
                if (i + j + k) <= grid["reduc_copula"] &&
                   !((i == 1 && j == 1) || (k == 1 && j == 1) || (k == 1 && i == 1))
                    linear_idx = i + (j - 1) * nb_cop + (k - 1) * nb_cop * na_cop
                    push!(compressionIndexesCOP, linear_idx)
                end
            end
        end
    end
    grid["compressionIndexesCOP"] = compressionIndexesCOP
    grid["nCOP"] = length(compressionIndexesCOP)

    s_m_b = grid["copula_marginal_b"]
    s_m_a = grid["copula_marginal_a"]
    s_m_se = grid["copula_marginal_se"]

    Poly, InvCheb, Gamma2 = createSparsebasis(grid, grid["maxdim"], Xss)
    n1 = size(Poly)
    n2 = size(Gamma2)

    grid["os"] = length(Xss) - (grid["nb"] + grid["na"] + grid["nse"]) -
                 (grid["nb"] * grid["na"] * grid["nse"])
    grid["oc"] = length(Yss) - 3 * n1[1]
    grid["oc_summary"] = 14
    grid["oc_agg"] = grid["oc"] - grid["oc_summary"]
    grid["os_process"] = 30
    grid["os_agg"] = grid["os"] - grid["os_process"]

    nCOP = Int(grid["nCOP"])
    nPoly = Int(n1[2])
    nFullCtrl = Int(3 * Ng + grid["oc"])
    nRedCtrl = Int(3 * nPoly + grid["oc"])
    nFullMarg = Int(n2[1])
    nRedMarg = Int(n2[2])
    nRedStates = Int(nRedMarg + nCOP + grid["os"])

    Gamma_state = sparse(Gamma2)

    InvGamma = spzeros(Float64, Int(nRedStates + nFullCtrl), Int(nRedStates + nRedCtrl))
    InvGamma[1:nRedStates, 1:nRedStates] = sparse(I, nRedStates, nRedStates)

    r0 = nRedStates
    c0 = nRedStates
    InvGamma[r0 .+ (1:Ng), c0 .+ (1:nPoly)] = transpose(InvCheb)
    InvGamma[r0 + Ng .+ (1:Ng), c0 + nPoly .+ (1:nPoly)] = transpose(InvCheb)
    InvGamma[r0 + 2 * Ng .+ (1:Ng), c0 + 2 * nPoly .+ (1:nPoly)] = transpose(InvCheb)
    InvGamma[r0 + 3 * Ng .+ (1:grid["oc"]), c0 + 3 * nPoly .+ (1:grid["oc"])] =
        sparse(I, grid["oc"], grid["oc"])
    InvGamma = transpose(InvGamma)

    Gamma_control = spzeros(Int(3 * n1[1] + grid["oc"]), Int(3 * n1[2] + grid["oc"]))
    Gamma_control[1:n1[1], 1:n1[2]] = Poly
    Gamma_control[n1[1] .+ (1:n1[1]), n1[2] .+ (1:n1[2])] = Poly
    Gamma_control[2 * n1[1] .+ (1:n1[1]), 2 * n1[2] .+ (1:n1[2])] = Poly
    Gamma_control[3 * n1[1] .+ (1:grid["oc"]), 3 * n1[2] .+ (1:grid["oc"])] =
        sparse(I, grid["oc"], grid["oc"])

    grid["numstates"] = (nb + na + nse - 3) + nCOP + grid["os"]
    grid["numstates_shocks"] = 15
    grid["numstates_endo"] = grid["numstates"] - grid["numstates_shocks"]
    grid["numcontrols"] = 3 * n1[2] + grid["oc"]
    grid["num_endo"] = grid["numstates_endo"] + grid["numcontrols"]

    State = zeros(Int(grid["numstates"]))
    State_m = copy(State)
    Contr = zeros(Int(grid["numcontrols"]))
    Contr_m = copy(Contr)

    m.grids[:ControlSS] = Yss    
    m.grids[:StateSS] = Xss
    m.dicts[:grid] = grid
    m.dicts[:param] = param
    m.dicts[:SS_stats] = SS_stats
    m.grids[:mu_dist] = mu_dist
    m.grids[:Value] = Value
m.grids[:mutil_c] = mutil_c
m.grids[:Va] = Va

end
