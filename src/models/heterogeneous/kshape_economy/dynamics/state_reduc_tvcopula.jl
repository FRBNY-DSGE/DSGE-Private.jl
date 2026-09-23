using SparseArrays
using LinearAlgebra
# include("../helpers/DCT/mydctmx.jl")
# include("../helpers/grids/copula_nodes_share_safe.jl")
# include("../helpers/createSparsebasis.jl")
"""
    state_reduc_tvcopula!(m.dicts[:param], m.dicts[:grid], m.dicts[:SS_stats], m.grids[:mu_dist], Value, m.grids[:mutil_c], Va)

Set up state reduction for the time-varying copula model.
Returns a named tuple with all computed variables.

Modifies `m.dicts[:grid]` and `m.dicts[:SS_stats]` in place.
"""
function state_reduc_tvcopula!(m::mBBQ)
    param = m.dicts[:param]
    grid = m.dicts[:grid]
    ss = m.dicts[:SS_stats]
    mu_dist = m.grids[:mu_dist]
    cache = get!(m.dicts, :state_reduc_cache, Dict{Symbol, Any}())

    # Define inverse utility functions
    sigma = param["sigma"]
    invutil(u) = ((1 - sigma) .* u) .^ (1 / (1 - sigma))
    invmutil(mu) = (1 ./ mu) .^ (1 / sigma)

    # Precompute repeated marginal sums once.
    mu_sum_b = vec(sum(sum(mu_dist, dims=2), dims=3))
    mu_sum_a = vec(sum(sum(mu_dist, dims=1), dims=3))
    mu_sum_se = vec(sum(sum(mu_dist, dims=2), dims=1))

    function push_scalars!(v::Vector{Float64}, vals...)
        for x in vals
            push!(v, Float64(x))
        end
        return v
    end

    # Build Xss based on m.dicts[:param]["adjust"]
    adjust_raw = param["adjust"]
    adjust = adjust_raw isa Symbol ? adjust_raw : Symbol(adjust_raw)
    Xss = Float64[]
    sizehint!(Xss, length(mu_sum_b) + length(mu_sum_a) + length(mu_sum_se) + length(vec(mu_dist)) + 48)
    append!(Xss, mu_sum_b)
    append!(Xss, mu_sum_a)
    append!(Xss, mu_sum_se)

    if adjust == :G
        append!(Xss, vec(mu_dist))
        push_scalars!(Xss,
            log(param["R_cb"]), log(param["w_bar"]),
            log(ss["A_b"]), log(ss["B_b"]), log(ss["A_g"] + 1),
            log(param["q"]), log(ss["leverage"]), log(ss["NW_b"]), log(param["R_cb"]), 0,
            log(param["pi_bar"]),
            log(ss["Y"]), log(ss["Chh"]), log(ss["I2"]), log(ss["Profit"] + ss["Profit_FI"]), log(param["u"]), log(ss["G"]), log(ss["LT"]), log(param["R_cb"]), log(ss["Y"] / (ss["Y"] - ss["B_F"])),
            log(1), log(1), log(m[:η]), log(1), log(ss["Y"] / (ss["Y"] - ss["LT"] - ss["C_b"])), log(1), log(1), log(1), log(1), log(m[:η] / (m[:η] - 1)),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )
    elseif adjust == :LT
        append!(Xss, vec(ss["mu_dist"]))
        push_scalars!(Xss,
            log(param["R_cb"]), log(param["w_bar"]),
            log(ss["A_b"]), log(ss["B_b"]), log(ss["A_g"] + 1),
            log(param["q"]), log(ss["leverage"]), log(ss["NW_b"]), log(param["R_cb"]), 0,
            log(param["pi_bar"]),
            log(ss["Y"]), log(ss["Chh"]), log(ss["I2"]), log(ss["Profit"] + ss["Profit_FI"]), log(param["u"]), log(ss["G"]), log(ss["LT"]), log(param["R_cb"]), log(ss["Y"] / (ss["Y"] - ss["B_F"])),
            log(1), log(1), log(m[:η]), log(1), log(ss["Y"] / (ss["Y"] - ss["G"])), log(1), log(1), log(1), log(1), log(m[:η] / (m[:η] - 1)),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )
    else
        error("Unknown adjust value: $adjust_raw")
    end

    # Build Yss based on m.dicts[:param]["adjust"]
    Yss = Float64[]
    sizehint!(Yss, length(m.grids[:Value]) + length(m.grids[:mutil_c]) + length(m.grids[:Va]) + 80)
    append!(Yss, vec(invutil(m.grids[:Value])))
    append!(Yss, vec(invmutil(m.grids[:mutil_c])))
    append!(Yss, vec(invmutil(m.grids[:Va])))

    if adjust == :G
        push_scalars!(Yss,
            log(ss["MRS"]), log(ss["A_hh"]), log(ss["B_nb"] / (1 - m[:dr]) * m[:b_a_aux]), log(ss["Chh"]),
            log(ss["N"]), log(ss["L"]), log(ss["UB"]),
            log(grid["K"]), log(ss["B2"]), log(ss["B_gov_ncp"]),
            log(ss["T"]), log(ss["LT"]), log(ss["G"]),
            log(ss["Lambda"]), log(param["pi_bar"]),
            log(ss["V"])
        )
        append!(Yss, log.(ss["J"]))
        push_scalars!(Yss,
            log(ss["r_l"]), log(ss["v"]),
            log(ss["Y"]), log(ss["Profit"]), log(ss["r_k"]), log(ss["r_a"]), log(ss["mc"]),
            log(ss["u"]), log(param["n"]), log(ss["M"]), log(ss["f"]),
            log(ss["zz"]), log(ss["xx"]), log(ss["vv"]), log(ss["ee"]), log(ss["C_b"]), log(ss["Profit_FI"]),
            log(ss["R_a"]), log(ss["R"]), log(ss["I2"]),
            0,
            log(ss["A_g"]), log(ss["Y"]), log(ss["C"]), log(ss["I2"]), log(param["w_bar"]), log(ss["Profit"] + ss["Profit_FI"]),
            log(100 * param["u"]), log(param["pi_bar"]), log(param["R_cb"]), log(param["w_bar"]), log(ss["G"]),
            log(ss["Y"]), log(ss["Chh"]), log(ss["I2"]), log(ss["Profit"] + ss["Profit_FI"]), log(param["u"]), log(ss["LT"]), log(ss["A_g"]),
            log(param["lambda"]), 0, 0, log(m[:η]), 0, log(ss["LT"]), log(ss["G"]), log(ss["LT"]), log(ss["B_gov_ncp2"])
        )
    elseif adjust == :LT
        push_scalars!(Yss,
            log(ss["MRS"]), log(ss["A_hh"]), log(ss["B_nb"] / (1 - m[:dr]) * m[:b_a_aux]), log(ss["Chh"]),
            log(ss["N"]), log(ss["L"]), log(ss["UB"]),
            log(grid["K"]), log(ss["B2"]), log(ss["B_gov_ncp"]),
            log(ss["T"]), log(ss["LT"]), log(ss["G"]),
            log(ss["Lambda"]), log(param["pi_bar"]),
            log(ss["V"])
        )
        append!(Yss, log.(ss["J"]))
        push_scalars!(Yss,
            log(ss["r_l"]), log(ss["v"]),
            log(ss["Y"]), log(ss["Profit"]), log(ss["r_k"]), log(ss["r_a"]), log(ss["mc"]),
            log(ss["u"]), log(param["n"]), log(ss["M"]), log(ss["f"]),
            log(ss["zz"]), log(ss["xx"]), log(ss["vv"]), log(ss["ee"]), log(ss["C_b"]), log(ss["Profit_FI"]),
            log(ss["R_a"]), log(ss["R"]), log(ss["I2"]),
            0,
            log(ss["A_g"]), log(ss["Y"]), log(ss["C"]), log(ss["I2"]), log(param["w_bar"]), log(ss["Profit"] + ss["Profit_FI"]),
            log(100 * param["u"]), log(param["pi_bar"]), log(param["R_cb"]), log(param["w_bar"]), log(ss["G"]),
            log(ss["Y"]), log(ss["Chh"]), log(ss["I2"]), log(ss["Profit"] + ss["Profit_FI"]), log(param["u"]), log(ss["G"]), log(ss["A_g"]),
            log(param["lambda"]), 0, 0, log(m[:η]), 0, log(ss["LT"]), log(ss["G"]), log(ss["LT"]), log(ss["B_gov_ncp2"])
        )
    end

    # m.dicts[:grid] setup
    grid["maxdim"] = 10
    nb = Int(grid["nb"])
    na = Int(grid["na"])
    nse = Int(grid["nse"])
    grid["Ng"] = nb * na * nse
    Ng = grid["Ng"]

    grid["nb_copula"] = 10
    grid["na_copula"] = 10
    grid["nse_copula"] = 10

    # DCT matrices
    nb_cop = Int(grid["nb_copula"])
    na_cop = Int(grid["na_copula"])
    nse_cop = Int(grid["nse_copula"])
    dct_key = (nb, na, nse, nb_cop, na_cop, nse_cop)
    if get(cache, :dct_key, nothing) != dct_key
        DC = Matrix{Float64}[mydctmx(nb), mydctmx(na), mydctmx(nse)]
        IDC = Matrix{Float64}[transpose(DC[1]), transpose(DC[2]), transpose(DC[3])]
        DCD = Matrix{Float64}[mydctmx(nb_cop), mydctmx(na_cop), mydctmx(nse_cop)]
        IDCD = Matrix{Float64}[transpose(DCD[1]), transpose(DCD[2]), transpose(DCD[3])]
        cache[:dct_key] = dct_key
        cache[:DC] = DC
        cache[:IDC] = IDC
        cache[:DCD] = DCD
        cache[:IDCD] = IDCD
    end

    # Distribution and copula
    distrSS = mu_dist
    CDF_SS = cumsum(cumsum(cumsum(distrSS, dims=1), dims=2), dims=3)
    COP_SS = CDF_SS

    ss["COP_SS"] = COP_SS

    # Marginal distributions
    distr_b_SS = copy(mu_sum_b)
    distr_b_SS = distr_b_SS / sum(distr_b_SS)
    distr_a_SS = copy(mu_sum_a)
    distr_a_SS = distr_a_SS / sum(distr_a_SS)
    distr_se_SS = copy(mu_sum_se)
    distr_se_SS = distr_se_SS / sum(distr_se_SS)

    CDF_b_SS = cumsum(distr_b_SS)
    CDF_a_SS = cumsum(distr_a_SS)
    CDF_se_SS = cumsum(distr_se_SS)

    # Copula marginal m.dicts[:grid]s
    grid["copula_marginal_b"] = copula_nodes_share_safe(distr_b_SS, vec(grid["b"]), grid["nb_copula"]; transform="logshift", alpha=0.3)
    grid["copula_marginal_a"] = copula_nodes_share_safe(distr_a_SS, vec(grid["a"]), grid["na_copula"]; transform="none", alpha=0.5)
    grid["copula_marginal_se"] = copula_nodes_share_safe(distr_se_SS, vec(grid["se"]), grid["nse_copula"]; transform="none", alpha=0.5, pinPenultimate=true)

    grid["reduc_copula"] = 10

    # Compression indices
    reduc_copula = Int(grid["reduc_copula"])
    cop_key = (nb_cop, na_cop, nse_cop, reduc_copula)
    if get(cache, :compression_key, nothing) != cop_key
        compressionIndexesCOP = Int[]
        sizehint!(compressionIndexesCOP, nb_cop * na_cop * nse_cop)
        for k in 1:nse_cop
            for j in 1:na_cop
                for i in 1:nb_cop
                    if (i + j + k) <= reduc_copula && !((i == 1 && j == 1) || (k == 1 && j == 1) || (k == 1 && i == 1))
                        linear_idx = i + (j - 1) * nb_cop + (k - 1) * nb_cop * na_cop
                        push!(compressionIndexesCOP, linear_idx)
                    end
                end
            end
        end
        cache[:compression_key] = cop_key
        cache[:compression_indexes] = compressionIndexesCOP
    end
    compressionIndexesCOP = cache[:compression_indexes]
    grid["compressionIndexesCOP"] = compressionIndexesCOP
    grid["nCOP"] = length(compressionIndexesCOP)

    s_m_b = grid["copula_marginal_b"]
    s_m_a = grid["copula_marginal_a"]
    s_m_se = grid["copula_marginal_se"]

    # Create sparse basis
    basis_key = (grid["maxdim"], nb, na, nse, nb_cop, na_cop, nse_cop, reduc_copula, hash(Xss))
    if get(cache, :basis_key, nothing) == basis_key
        Poly = cache[:Poly]
        InvCheb = cache[:InvCheb]
        Gamma2 = cache[:Gamma2]
    else
        Poly, InvCheb, Gamma2 = createSparsebasis(grid, grid["maxdim"], Xss)
        cache[:basis_key] = basis_key
        cache[:Poly] = Poly
        cache[:InvCheb] = InvCheb
        cache[:Gamma2] = Gamma2
    end
    n1 = size(Poly)
    n2 = size(Gamma2)

    # m.dicts[:grid] m.dicts[:param]eters
    grid["os"] = length(Xss) - (grid["nb"] + grid["na"] + grid["nse"]) - (grid["nb"] * grid["na"] * grid["nse"]) #os means other states
    grid["oc"] = length(Yss) - 3 * n1[1] #oc means other controls
    grid["oc_summary"] = 7
    grid["oc_agg"] = grid["oc"] - grid["oc_summary"]
    grid["os_process"] = 17
    grid["os_agg"] = grid["os"] - grid["os_process"]

    nCOP = Int(grid["nCOP"])
    nPoly = Int(n1[2])
    nFullCtrl = Int(3 * Ng + grid["oc"])
    nRedCtrl = Int(3 * nPoly + grid["oc"])
    nFullMarg = Int(n2[1])
    nRedMarg = Int(n2[2])
    nRedStates = Int(nRedMarg + nCOP + grid["os"])

    Gamma_state = sparse(Gamma2)

    # InvGamma matrix
    InvGamma = spzeros(Float64, Int(nRedStates + nFullCtrl), Int(nRedStates + nRedCtrl))
    InvGamma[1:nRedStates, 1:nRedStates] = sparse(I, nRedStates, nRedStates)

    r0 = nRedStates
    c0 = nRedStates
    InvGamma[r0 .+ (1:Ng), c0 .+ (1:nPoly)] = transpose(InvCheb)
    InvGamma[r0 + Ng .+ (1:Ng), c0 + nPoly .+ (1:nPoly)] = transpose(InvCheb)
    InvGamma[r0 + 2*Ng .+ (1:Ng), c0 + 2*nPoly .+ (1:nPoly)] = transpose(InvCheb)
    InvGamma[r0 + 3*Ng .+ (1:grid["oc"]), c0 + 3*nPoly .+ (1:grid["oc"])] = sparse(I, grid["oc"], grid["oc"])
    InvGamma = transpose(InvGamma)

    # Gamma_control matrix
    Gamma_control = spzeros(Int(3 * n1[1] + grid["oc"]), Int(3 * n1[2] + grid["oc"]))
    Gamma_control[1:n1[1], 1:n1[2]] = Poly
    Gamma_control[n1[1] .+ (1:n1[1]), n1[2] .+ (1:n1[2])] = Poly
    Gamma_control[2*n1[1] .+ (1:n1[1]), 2*n1[2] .+ (1:n1[2])] = Poly
    Gamma_control[3*n1[1] .+ (1:grid["oc"]), 3*n1[2] .+ (1:grid["oc"])] = sparse(I, grid["oc"], grid["oc"])

    # m.dicts[:grid] state/control dimensions
    grid["numstates"] = (nb + na + nse - 3) + nCOP + grid["os"]
    grid["numstates_shocks"] = 10
    grid["numstates_endo"] = grid["numstates"] - grid["numstates_shocks"]
    grid["numcontrols"] = 3 * n1[2] + grid["oc"]
    grid["num_endo"] = grid["numstates_endo"] + grid["numcontrols"]

    # Initialize state and control vectors
    State = zeros(Int(grid["numstates"]))
    State_m = copy(State)
    Contr = zeros(Int(grid["numcontrols"]))
    Contr_m = copy(Contr)
    

    m.grids[:ControlSS] = Yss    
    m.grids[:StateSS] = Xss
    m.dicts[:grid] = grid
    m.dicts[:param] = param
    m.dicts[:SS_stats] = ss
    m.grids[:mu_dist] = mu_dist

    # return (
    #     Xss = Xss,
    #     Yss = Yss,
    #     DC = DC,
    #     IDC = IDC,
    #     DCD = DCD,
    #     IDCD = IDCD,
    #     Gamma_state = Gamma_state,
    #     Gamma_control = Gamma_control,
    #     InvGamma = InvGamma,
    #     State = State,
    #     State_m = State_m,
    #     Contr = Contr,
    #     Contr_m = Contr_m,
    #     distrSS = distrSS,
    #     CDF_SS = CDF_SS,
    #     COP_SS = COP_SS,
    #     distr_b_SS = distr_b_SS,
    #     distr_a_SS = distr_a_SS,
    #     distr_se_SS = distr_se_SS,
    #     CDF_b_SS = CDF_b_SS,
    #     CDF_a_SS = CDF_a_SS,
    #     CDF_se_SS = CDF_se_SS,
    #     compressionIndexesCOP = compressionIndexesCOP,
    #     Poly = Poly,
    #     InvCheb = InvCheb,
    #     Gamma2 = Gamma2,
    #     nPoly = nPoly,
    #     nFullCtrl = nFullCtrl,
    #     nRedCtrl = nRedCtrl,
    #     nFullMarg = nFullMarg,
    #     nRedMarg = nRedMarg,
    #     nRedStates = nRedStates
    # )
end
