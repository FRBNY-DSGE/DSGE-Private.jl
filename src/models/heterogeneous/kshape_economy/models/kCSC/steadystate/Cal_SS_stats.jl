using LinearAlgebra
using SparseArrays
using Statistics
using Interpolations

function _weighted_gini(values::Vector{Float64}, weights::Vector{Float64})
    ix = sortperm(values)
    sorted_values = values[ix]
    sorted_weights = weights[ix]
    running_sum = cumsum(sorted_weights .* sorted_values)
    running_sum = vcat(0.0, running_sum)
    gini = 1 - sum(sorted_weights .* (running_sum[1:end-1] .+ running_sum[2:end])) / running_sum[end]
    return gini, sorted_values, sorted_weights
end

function Cal_SS_stats(mu_dist::Array{Float64,3}, c_a_guess::Array{Float64,3},
                      c_n_guess::Array{Float64,3}, AProb::Array{Float64,3},
                      b_n_star::Array{Float64,3}, b_a_star::Array{Float64,3},
                      a_a_star::Array{Float64,3},
                      grid::Dict{String,Any}, param::Dict{String,Any}, meshes::Dict{String,Array{Float64,3}},
                      TT::SparseMatrixCSC{Float64,Int64},
                      Y::Float64, RR::Float64, w_1::Float64, w_2::Float64,
                      r_l_1::Float64, r_l_2::Float64, r_k::Float64, Profit::Float64,
                      n_1::Float64, n_2::Float64, u_1::Float64, u_2::Float64,
                      v::Float64, J::Vector{Float64},
                      V_1::Float64, V_2::Float64, M_1::Float64, M_2::Float64,
                      f_1::Float64, f_2::Float64, mc::Float64, Lambda::Float64,
                      NWb::Float64, THETA::Float64, Profit_FI::Float64, Ab::Float64,
                      B_FI::Float64, B_MMMF::Float64, T_MMMF::Float64, B_gov::Float64,
                      B_g::Float64, K_g_agg::Float64, TAX_revenue::Float64, UB::Float64,
                      mutil_c::Array{Float64,3}, mutil_c_a::Array{Float64,3},
                      mutil_c_n::Array{Float64,3}, Va::Array{Float64,3},
                      Value::Array{Float64,3})

    SS_stats = Dict{String, Any}()

    nb = Int(grid["nb"])
    na = Int(grid["na"])
    ns = Int(grid["ns"])
    ns_1 = Int(grid["ns_1"])
    ns_2 = Int(grid["ns_2"])
    nse = Int(grid["nse"])

    mesh_a = repeat(grid["a"]', nb, 1)
    mesh_b = repeat(grid["b"], 1, na)
    mesh = Dict("a" => mesh_a, "b" => mesh_b)

    grid["se_aux"] = vcat(grid["s"], min.(param["b_ratio"] .* grid["s"], grid["s_bar"]), 1.0)
    _, _, meshes_se_aux = ndgrid(grid["b"], grid["a"], grid["se_aux"])
    meshes["se_aux"] = meshes_se_aux

    NW = fill(param["xi"] / (1 + param["xi"]) * param["GHH"] * n_1 * param["w_bar_1"], nb, na, nse)
    NW[:, :, (ns_1 + 1):ns] .= param["xi"] / (1 + param["xi"]) * param["GHH"] * n_2 * param["w_bar_2"]
    NW[:, :, (ns + ns_1 + 1):(ns + ns)] .= param["xi"] / (1 + param["xi"]) * param["GHH"] * n_2 * param["w_bar_2"]
    WW = (1 - param["tau_w"]) .* NW
    WW[:, :, end] .= (1 - param["tau_w"]) * (param["Eratio"] * Profit) / param["Eshare"]

    RR_aux = RR + param["death_rate"] / (1 - param["death_rate"]) * param["q"]
    RBRB = param["R_cb"] / param["pi_cb"] .+ (meshes["b"] .< 0) .* (param["Rprem"] / param["pi_cb"])
    RBRB_aux = RBRB / (1 - param["death_rate"]) * param["b_a_aux"]
    inc_transfer = fill(param["LT"], nb, na, nse)

    mu_redux = dropdims(sum(mu_dist, dims=3), dims=3)
    mu_dist_b = dropdims(sum(sum(mu_dist, dims=2), dims=3), dims=(2, 3))
    mu_dist_a = dropdims(sum(sum(mu_dist, dims=1), dims=3), dims=(1, 3))
    mu_dist_se = dropdims(sum(sum(mu_dist, dims=1), dims=2), dims=(1, 2))

    H_tilde = kron(sparse(param["P_SS3"]'), sparse(1:(nb * na), 1:(nb * na), ones(nb * na), nb * na, nb * na))

    mu_dist_next = reshape((mu_dist[:]') * TT, (nb, na, nse))
    mu_dist_tilde = reshape(H_tilde * mu_dist[:], (nb, na, nse))
    mu_dist_tilde_next = reshape(H_tilde * mu_dist_next[:], (nb, na, nse))

    mu_dist_next_b = dropdims(sum(sum(mu_dist_next, dims=2), dims=3), dims=(2, 3))
    mu_dist_next_a = dropdims(sum(sum(mu_dist_next, dims=1), dims=3), dims=(1, 3))
    mu_dist_next_se = dropdims(sum(sum(mu_dist_next, dims=1), dims=2), dims=(1, 2))
    mu_dist_tilde_b = dropdims(sum(sum(mu_dist_tilde, dims=2), dims=3), dims=(2, 3))
    mu_dist_tilde_a = dropdims(sum(sum(mu_dist_tilde, dims=1), dims=3), dims=(1, 3))
    mu_dist_tilde_se = dropdims(sum(sum(mu_dist_tilde, dims=1), dims=2), dims=(1, 2))

    SS_stats["H_tilde"] = H_tilde
    SS_stats["Y"] = Y
    SS_stats["v"] = v
    SS_stats["u_1"] = u_1
    SS_stats["u_2"] = u_2
    SS_stats["u"] = 1 - sum(mu_dist_tilde_se[1:ns]) / (sum(mu_dist_tilde_se[1:ns]) + sum(mu_dist_tilde_se[(ns + 1):(ns + ns)]))
    SS_stats["Lambda"] = Lambda
    SS_stats["f_1"] = f_1
    SS_stats["f_2"] = f_2
    SS_stats["r_a"] = RR
    SS_stats["w_1"] = w_1
    SS_stats["w_2"] = w_2
    SS_stats["r_l_1"] = r_l_1
    SS_stats["r_l_2"] = r_l_2
    SS_stats["J"] = J
    SS_stats["M_1"] = M_1
    SS_stats["M_2"] = M_2
    SS_stats["V_1"] = V_1
    SS_stats["V_2"] = V_2
    SS_stats["L_1"] = grid["L_1"]
    SS_stats["L_2"] = grid["L_2"]
    SS_stats["N_tilde_1"] = sum(mu_dist_tilde_se[1:ns_1])
    SS_stats["N_tilde_2"] = sum(mu_dist_tilde_se[(ns_1 + 1):ns])
    SS_stats["N_1"] = sum(grid["se_dist_2"][1:ns_1])
    SS_stats["N_2"] = sum(grid["se_dist_2"][(ns_1 + 1):ns])
    SS_stats["U_1"] = sum(grid["se_dist_2"][(ns + 1):(ns + ns_1)])
    SS_stats["U_2"] = sum(grid["se_dist_2"][(ns + ns_1 + 1):(ns + ns)])
    SS_stats["n_1"] = ((1 - param["tau_w"]) * param["w_bar_1"] / param["psi_1"])^(1 / param["xi"])
    SS_stats["n_2"] = ((1 - param["tau_w"]) * param["w_bar_2"] / param["psi_2"])^(1 / param["xi"])
    SS_stats["r_k"] = r_k
    SS_stats["mc"] = mc
    SS_stats["w"] = (SS_stats["L_1"] * SS_stats["w_1"] + SS_stats["L_2"] * SS_stats["w_2"]) / (SS_stats["L_1"] + SS_stats["L_2"])
    SS_stats["Profit"] = Profit

    grid["n_1"] = SS_stats["n_1"]
    grid["n_2"] = SS_stats["n_2"]
    param["fix"] = param["fix_ratio"] * SS_stats["Y"]

    SS_stats["r_a_aux"] = SS_stats["r_a"] + param["death_rate"] / (1 - param["death_rate"]) * param["q"]
    SS_stats["RRa"] = 1 + SS_stats["r_a"]
    SS_stats["RR"] = param["R_cb"] / param["pi_bar"]

    SS_stats["NW_b"] = NWb
    SS_stats["leverage"] = THETA
    SS_stats["B_FI"] = B_FI
    SS_stats["B_MMMF"] = B_MMMF
    SS_stats["A_b"] = Ab
    SS_stats["A_I"] = SS_stats["NW_b"] * SS_stats["leverage"]
    SS_stats["A_g"] = K_g_agg
    SS_stats["A_F"] = 0.0
    SS_stats["A_NI"] = grid["K"] - SS_stats["A_I"] - SS_stats["A_F"] - SS_stats["A_g"]
    SS_stats["Profit_FI"] = Profit_FI
    param["fix2"] = SS_stats["Profit_FI"]

    SS_stats["ShareBorrower"] = sum((grid["b"] .< 0) .* mu_dist_b)
    SS_stats["K"] = grid["K"]

    SS_stats["B_gov_ncp"] = param["B_gov_ratio"] * SS_stats["Y"]
    SS_stats["B_g"] = B_g
    SS_stats["B_gov"] = SS_stats["B_gov_ncp"] + SS_stats["B_g"]
    SS_stats["B_F"] = 0.0

    SS_stats["B_neg"] = sum((grid["b"] .< 0) .* grid["b"] .* mu_dist_b)
    SS_stats["B_pos"] = sum((grid["b"] .>= 0) .* grid["b"] .* mu_dist_b)
    SS_stats["B_b"] = SS_stats["B_gov"] + (SS_stats["B_FI"] -
                     SS_stats["B_pos"] / (1 - param["death_rate"]) * param["b_a_aux"] -
                     SS_stats["B_neg"] / (1 - param["death_rate"]) * param["b_a_aux"] - SS_stats["B_F"])

    SS_stats["B_nb"] = sum(grid["b"] .* mu_dist_b)
    SS_stats["B"] = SS_stats["B_nb"] + SS_stats["B_b"] + SS_stats["B_F"]
    SS_stats["B2"] = SS_stats["B_nb"] / (1 - param["death_rate"]) * param["b_a_aux"] + SS_stats["B_b"] + SS_stats["B_F"]

    grid["B"] = SS_stats["B"]
    SS_stats["BoverK"] = SS_stats["B"] / SS_stats["K"]
    SS_stats["KY"] = SS_stats["K"] / Y / 4
    SS_stats["BY"] = SS_stats["B"] / Y / 4

    BCaux_b = mu_dist_b
    SS_stats["b_bc"] = BCaux_b[1]
    SS_stats["b_0"] = BCaux_b[findfirst(==(0.0), grid["b"])]
    BCaux_a = mu_dist_a
    SS_stats["a_bc"] = BCaux_a[1]
    aux_ba = dropdims(sum(mu_dist, dims=3), dims=3)
    SS_stats["WtH_b0"] = sum(aux_ba[(mesh["b"] .== 0) .& (mesh["a"] .> 0)])
    SS_stats["Zero"] = sum(aux_ba[(mesh["b"] .== 0) .& (mesh["a"] .== 0)])
    SS_stats["WtH_bnonpos"] = sum(aux_ba[(mesh["b"] .<= 0) .& (mesh["a"] .> 0)])

    SS_stats["Profit_int"] = SS_stats["Y"] - SS_stats["mc"] * SS_stats["Y"] - param["fix"]
    SS_stats["Profit_L"] = (SS_stats["r_l_1"] - param["fix_L_1"] - param["w_bar_1"]) * grid["L_1"] - param["iota_1"] * SS_stats["V_1"] +
                           (SS_stats["r_l_2"] - param["fix_L_2"] - param["w_bar_2"]) * grid["L_2"] - param["iota_2"] * SS_stats["V_2"]
    SS_stats["Profit_K"] = (SS_stats["r_k"] * SS_stats["v"] - param["delta_ss"]) * grid["K"]
    SS_stats["Profit2"] = SS_stats["Profit_int"] + SS_stats["Profit_L"] + SS_stats["Profit_K"]

    SS_stats["AdjProb"] = sum(AProb[:] .* mu_dist_tilde[:])
    SS_stats["W_1"] = param["w_bar_1"]
    SS_stats["W_2"] = param["w_bar_2"]
    SS_stats["tau_w"] = param["tau_w"]
    SS_stats["tau_a"] = param["tau_a"]
    SS_stats["PROFIT"] = SS_stats["Profit"]
    SS_stats["R_A"] = SS_stats["r_a"]
    SS_stats["Q"] = 1.0
    SS_stats["R_cbminus"] = param["R_cb"]
    SS_stats["R_cb"] = param["R_cb"]
    SS_stats["PI"] = param["pi_bar"]
    SS_stats["PInext"] = param["pi_bar"]
    SS_stats["Va_next_input"] = Va
    SS_stats["mutil_c_next_input"] = mutil_c
    SS_stats["VALUEnext"] = Value
    SS_stats["LT"] = param["LT"]

    X_a_RHS = WW .* meshes["se"] .+ (param["q"] .+ RR_aux) .* meshes["a"] .+ RBRB_aux .* meshes["b"] .-
              param["q"] .* a_a_star .- b_a_star .+ SS_stats["LT"]
    X_n_RHS = WW .* meshes["se"] .+ (param["q"] .+ RR_aux) .* meshes["a"] .+ RBRB_aux .* meshes["b"] .-
              param["q"] .* meshes["a"] .- b_n_star .+ SS_stats["LT"]

    X_a_agg = sum(X_a_RHS .* mu_dist_tilde .* AProb)
    X_n_agg = sum(X_n_RHS .* mu_dist_tilde .* (1 .- AProb))
    X_agg2 = X_a_agg + X_n_agg
    X_agg = sum(mu_dist_tilde[:] .* AProb[:] .* c_a_guess[:] .+ mu_dist_tilde[:] .* (1 .- AProb[:]) .* c_n_guess[:])

    B_tilde_agg = sum(mu_dist_tilde[:] .* AProb[:] .* b_a_star[:] .+ mu_dist_tilde[:] .* (1 .- AProb[:]) .* b_n_star[:])
    K_tilde_agg = sum((AProb[:] .* a_a_star[:] .+ (1 .- AProb[:]) .* meshes["a"][:]) .* mu_dist_tilde[:])

    SS_stats["K_tilde"] = K_tilde_agg
    SS_stats["B_tilde"] = B_tilde_agg

    B_agg = (1 - param["death_rate"]) * B_tilde_agg
    K_agg = (1 - param["death_rate"]) * K_tilde_agg

    NW_aux = param["w_bar_2"] * param["n_2"] .* meshes["se"]
    NW_aux[:, :, 1:ns_1] .= param["w_bar_1"] * param["n_1"] .* meshes["se"][:, :, 1:ns_1]

    if param["GHH"] == 1
        C_agg = X_agg + sum((1 - param["tau_w"]) / (1 + param["xi"]) .* NW_aux[:, :, 1:ns] .* meshes["se"][:, :, 1:ns] .* mu_dist_tilde[:, :, 1:ns])
        C_agg2 = X_agg2 + sum((1 - param["tau_w"]) / (1 + param["xi"]) .* NW_aux[:, :, 1:ns] .* meshes["se"][:, :, 1:ns] .* mu_dist_tilde[:, :, 1:ns])
    else
        C_agg = X_agg
        C_agg2 = X_agg2
    end

    I_agg = K_tilde_agg + SS_stats["A_g"] + SS_stats["A_b"] - (1 - param["delta_ss"]) * grid["K"]
    I_agg2 = (1 - param["death_rate"]) * K_tilde_agg + SS_stats["A_g"] + SS_stats["A_b"] - (1 - param["delta_ss"]) * grid["K"]
    SS_stats["I"] = I_agg2
    SS_stats["I2"] = I_agg2

    absorp_annuity = param["death_rate"] * K_tilde_agg - param["death_rate"] / (1 - param["death_rate"]) * SS_stats["A_NI"]
    SS_stats["C"] = C_agg
    SS_stats["Chh"] = C_agg
    SS_stats["C_hh"] = SS_stats["Chh"]

    SS_stats["T"] = TAX_revenue
    SS_stats["C_b"] = T_MMMF
    SS_stats["UB"] = UB
    SS_stats["LT"] = param["LT"] - SS_stats["C_b"]

    SS_stats["G"] = SS_stats["T"] + SS_stats["B_gov"] - param["R_cb"] / param["pi_cb"] * SS_stats["B_gov"] +
                    (param["q"] + SS_stats["r_a"]) / param["q"] * SS_stats["A_g"] - param["q"] * SS_stats["A_g"] -
                    SS_stats["UB"] - SS_stats["LT"] - param["MMMF_cont_ratio"] * SS_stats["T"] - param["tau_cp"] * SS_stats["A_g"]
    SS_stats["GtoY"] = SS_stats["G"] / Y

    Int_cost = sum((abs.(grid["b"]) .* (grid["b"] .< 0)) .* mu_dist_b) * param["Rprem"] / param["pi_cb"] / (1 - param["death_rate"]) * param["b_a_aux"]
    G_agg = SS_stats["G"]

    SS_stats["R_a"] = (param["q"] + SS_stats["r_a"]) / param["q"]
    SS_stats["R"] = param["R_cb"] / param["pi_cb"]
    SS_stats["A_hh"] = K_tilde_agg * (1 - param["death_rate"])
    SS_stats["B_hh"] = SS_stats["B_nb"]

    SS_stats["zz"] = (SS_stats["R_a"] - param["b_a_aux2"] * SS_stats["R"]) * SS_stats["leverage"] + param["b_a_aux2"] * SS_stats["R"]
    SS_stats["xx"] = SS_stats["zz"]
    SS_stats["Lambda_b"] = param["Lambda_bank"]
    SS_stats["vv"] = ((1 - param["theta_b"]) * SS_stats["Lambda_b"] * (SS_stats["R_a"] - param["b_a_aux2"] * SS_stats["R"])) /
                     (1 - param["theta_b"] * SS_stats["Lambda_b"] * SS_stats["xx"])
    SS_stats["ee"] = (1 - param["theta_b"]) * SS_stats["Lambda_b"] * param["b_a_aux2"] * SS_stats["R"] /
                     (1 - param["theta_b"] * SS_stats["Lambda_b"] * SS_stats["zz"])
    SS_stats["x_a"] = SS_stats["A_b"] / SS_stats["A_hh"]
    SS_stats["x_b"] = SS_stats["B_b"] / SS_stats["B_hh"]

    if param["b_a_aux2"] < 1
        diff_mk = SS_stats["Y"] - C_agg - param["tau_cp"] * SS_stats["A_g"] - param["frac_b"] * SS_stats["C_b"] - I_agg -
                  G_agg - param["iota_1"] * SS_stats["V_1"] - param["iota_2"] * SS_stats["V_2"] - Int_cost - param["fix"] -
                  absorp_annuity + param["death_rate"] * K_tilde_agg - param["fix2"] -
                  (param["R_cb"] / param["pi_bar"] - 1) * SS_stats["B_F"] - SS_stats["r_a"] * SS_stats["A_F"] -
                  param["fix_L_1"] * SS_stats["L_1"] - param["fix_L_2"] * SS_stats["L_2"] -
                  param["death_rate"] * SS_stats["B_hh"] * param["R_cb"] / param["pi_bar"] + param["death_rate"] * SS_stats["B_FI"]
        diff_mk2 = SS_stats["Y"] - C_agg2 - param["tau_cp"] * SS_stats["A_g"] - param["frac_b"] * SS_stats["C_b"] - I_agg -
                   G_agg - param["iota_1"] * SS_stats["V_1"] - param["iota_2"] * SS_stats["V_2"] - Int_cost - param["fix"] -
                   absorp_annuity + param["death_rate"] * K_tilde_agg - param["fix2"] -
                   (param["R_cb"] / param["pi_bar"] - 1) * SS_stats["B_F"] - SS_stats["r_a"] * SS_stats["A_F"] -
                   param["fix_L_1"] * SS_stats["L_1"] - param["fix_L_2"] * SS_stats["L_2"] -
                   param["death_rate"] * SS_stats["B_hh"] * param["R_cb"] / param["pi_bar"] + param["death_rate"] * SS_stats["B_FI"]
    else
        diff_mk = SS_stats["Y"] - C_agg - param["tau_cp"] * SS_stats["A_g"] - param["frac_b"] * SS_stats["C_b"] - I_agg -
                  G_agg - param["iota_1"] * SS_stats["V_1"] - param["iota_2"] * SS_stats["V_2"] - Int_cost - param["fix"] -
                  absorp_annuity + param["death_rate"] * K_tilde_agg - param["fix2"] -
                  (param["R_cb"] / param["pi_bar"] - 1) * SS_stats["B_F"] - SS_stats["r_a"] * SS_stats["A_F"] -
                  param["fix_L_1"] * SS_stats["L_1"] - param["fix_L_2"] * SS_stats["L_2"] -
                  param["death_rate"] * SS_stats["B_hh"] * param["R_cb"] / param["pi_bar"]
        diff_mk2 = SS_stats["Y"] - C_agg2 - param["tau_cp"] * SS_stats["A_g"] - param["frac_b"] * SS_stats["C_b"] - I_agg -
                   G_agg - param["iota_1"] * SS_stats["V_1"] - param["iota_2"] * SS_stats["V_2"] - Int_cost - param["fix"] -
                   absorp_annuity + param["death_rate"] * K_tilde_agg - param["fix2"] -
                   (param["R_cb"] / param["pi_bar"] - 1) * SS_stats["B_F"] - SS_stats["r_a"] * SS_stats["A_F"] -
                   param["fix_L_1"] * SS_stats["L_1"] - param["fix_L_2"] * SS_stats["L_2"] -
                   param["death_rate"] * SS_stats["B_hh"] * param["R_cb"] / param["pi_bar"]
    end
    SS_stats["diff_mk"] = diff_mk
    SS_stats["diff_mk2"] = diff_mk2

    wealth_values = vec(mesh["a"] * param["q"] .+ mesh["b"])
    SS_stats["GiniW"], total_wealth, total_wealth_pdf = _weighted_gini(wealth_values, vec(mu_redux))
    SS_stats["NegNetWorth"] = sum((total_wealth .< 0) .* total_wealth_pdf)

    SS_stats["GiniLI"], liquid_sort, liquid_pdf = _weighted_gini(vec(mesh["b"]), vec(mu_redux))
    SS_stats["Negliquid"] = sum((liquid_sort .< 0) .* liquid_pdf)

    SS_stats["GiniIL"], _, _ = _weighted_gini(vec(mesh["a"]), vec(mu_redux))

    NW_Gini = param["w_bar_2"] * param["n_2"] .* meshes["se"]
    NW_Gini[:, :, 1:ns_1] .= param["w_bar_1"] * param["n_1"] .* meshes["se"][:, :, 1:ns_1]
    WW_Gini = (1 - param["tau_w"]) .* NW_Gini
    WW_Gini[:, :, end] .= (1 - param["tau_a"]) * (param["Eratio"] * SS_stats["Profit"]) / param["Eshare"]

    income_unsorted = WW_Gini .* meshes["se_aux"] .+ SS_stats["r_a_aux"] .* meshes["a"] .+
                      (param["R_cb"] / param["pi_cb"] - 1) .* meshes["b"] .* (meshes["b"] .>= 0) .+
                      ((param["R_cb"] + param["Rprem"]) / param["pi_cb"] - 1) .* meshes["b"] .* (meshes["b"] .< 0) .+
                      inc_transfer
    SS_stats["GiniIncome"], _, _ = _weighted_gini(vec(income_unsorted), vec(mu_dist_tilde))

    labor_income_unsorted = WW_Gini .* meshes["se_aux"]
    SS_stats["GiniLIncome"], _, _ = _weighted_gini(vec(labor_income_unsorted), vec(mu_dist_tilde))

    nonlabor_income_unsorted = SS_stats["r_a_aux"] .* meshes["a"] .+
                               (param["R_cb"] / param["pi_cb"] / (1 - param["death_rate"]) * param["b_a_aux"] - 1) .* meshes["b"] .* (meshes["b"] .>= 0) .+
                               ((param["R_cb"] + param["Rprem"]) / param["pi_cb"] / (1 - param["death_rate"]) * param["b_a_aux"] - 1) .* meshes["b"] .* (meshes["b"] .< 0)
    SS_stats["GiniNLIncome"], _, _ = _weighted_gini(vec(nonlabor_income_unsorted), vec(mu_dist_tilde))

    cons_aux = vcat(vec(c_a_guess), vec(c_n_guess))
    mu_dist_tilde_aux = vcat(vec(AProb .* mu_dist_tilde), vec((1 .- AProb) .* mu_dist_tilde))
    SS_stats["GiniCons"], _, _ = _weighted_gini(cons_aux, mu_dist_tilde_aux)

    meshaux_b, meshaux_a, meshaux_se_aux = ndgrid(grid["b"], grid["a"], collect(1:nse))
    P_transition_exp = param["P_SS2"] * param["P_SS3"]
    mutil_c_aux = reshape(reshape(mutil_c, (nb * na, nse)) * P_transition_exp', (nb, na, nse))
    itp = interpolate((grid["b"], grid["a"], collect(1:nse)), mutil_c_aux, Gridded(Linear()))
    mutil_c_next = extrapolate(itp, Line())

    MRS_a = param["beta"] * (1 - param["death_rate"]) .* mutil_c_next.(b_a_star, a_a_star, meshaux_se_aux) ./ mutil_c_a .*
            AProb .* mu_dist_tilde .* (a_a_star ./ SS_stats["A_hh"])
    MRS_n = param["beta"] * (1 - param["death_rate"]) .* mutil_c_next.(b_n_star, meshaux_a, meshaux_se_aux) ./ mutil_c_n .*
            (1 .- AProb) .* mu_dist_tilde .* (meshaux_a ./ SS_stats["A_hh"])

    MRS_a_2 = param["beta"] * (1 - param["death_rate"]) .* mutil_c_next.(b_a_star, a_a_star, meshaux_se_aux) ./ mutil_c_a .*
              AProb .* mu_dist_tilde .* (meshaux_a ./ SS_stats["A_hh"])
    MRS_n_2 = param["beta"] * (1 - param["death_rate"]) .* mutil_c_next.(b_n_star, meshaux_a, meshaux_se_aux) ./ mutil_c_n .*
              (1 .- AProb) .* mu_dist_tilde .* (meshaux_a ./ SS_stats["A_hh"])

    MRS = sum(MRS_a) + sum(MRS_n)
    MRS_2 = sum(MRS_a_2) + sum(MRS_n_2)

    MRS_a_aux = param["beta"] * (1 - param["death_rate"]) .* mutil_c_next.(b_a_star, a_a_star, meshaux_se_aux) ./ mutil_c_a .* AProb .* mu_dist_tilde
    MRS_n_aux = param["beta"] * (1 - param["death_rate"]) .* mutil_c_next.(b_n_star, meshaux_a, meshaux_se_aux) ./ mutil_c_n .* (1 .- AProb) .* mu_dist_tilde

    MRS_a_aux2 = param["beta"] * (1 - param["death_rate"]) .* mutil_c_next.(b_a_star, a_a_star, meshaux_se_aux) ./ mutil_c_a
    MRS_n_aux2 = param["beta"] * (1 - param["death_rate"]) .* mutil_c_next.(b_n_star, meshaux_a, meshaux_se_aux) ./ mutil_c_n

    MRS_ent_aux = sum(MRS_a_aux[:, :, end] + MRS_n_aux[:, :, end]) / param["Eshare"]

    SS_stats["MRS_aux_hh"] = MRS
    SS_stats["MRS_aux_hh_2"] = MRS_2
    SS_stats["MRS"] = param["pi_bar"] / param["R_cb"]
    param["Lambda_aux"] = SS_stats["Lambda"] / MRS_ent_aux
    param["Lambda_b_aux"] = param["pi_bar"] / param["R_cb"] / MRS_ent_aux
    param["lambda_hh_aux"] = SS_stats["Lambda"] / MRS
    SS_stats["MRS_hh"] = MRS * param["lambda_hh_aux"]
    SS_stats["MRS_a_aux2"] = MRS_a_aux2
    SS_stats["MRS_n_aux2"] = MRS_n_aux2

    J_aux, J_bar_1, J_bar_2 = _csc_job_value_stats(r_l_1, r_l_2, param["w_bar_1"], param["w_bar_2"], param["fix_L_1"], param["fix_L_2"], param, grid)
    SS_stats["J_aux"] = J_aux
    SS_stats["J_bar_1"] = J_bar_1
    SS_stats["J_bar_2"] = J_bar_2
    SS_stats["B_gov_ncp2"] = SS_stats["B_gov_ncp"] - SS_stats["B_F"]

    AC = param["sigma_chi"] .* ((1 .- AProb[:]) .* log.(1 .- AProb[:]) .+ AProb[:] .* log.(AProb[:])) .+
         param["mu_chi"] .* AProb[:] .-
         (param["mu_chi"] - param["sigma_chi"] * log(1 + exp(param["mu_chi"] / param["sigma_chi"])))
    AProb_scaled = AProb .* param["nu"]
    AC .*= param["nu"]

    SS_stats["AProb"] = reshape(AProb_scaled, (nb, na, nse))
    SS_stats["AC"] = AC
    SS_stats["VALUE"] = Value
    SS_stats["AvgC"] = (param["delta_ss"] * grid["K"] + param["w_bar_1"] * grid["L_1"] + param["w_bar_2"] * grid["L_2"] +
                        param["iota_1"] * SS_stats["V_1"] + param["iota_2"] * SS_stats["V_2"] + param["fix"]) / SS_stats["Y"]
    SS_stats["mu_dist_NE"] = copy(mu_dist)
    SS_stats["mu_dist_NE"][:, :, end] .= 0.0
    SS_stats["mu_dist_NE"] ./= sum(SS_stats["mu_dist_NE"])
    SS_stats["LG"] = SS_stats["LT"] + SS_stats["G"]

    param["C_b"] = SS_stats["C_b"]

    return SS_stats, grid, param
end
