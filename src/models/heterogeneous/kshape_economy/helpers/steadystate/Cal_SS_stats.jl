"""
    Cal_SS_stats(...)

Compute steady state statistics for the HANK model.

This function calculates distributions, aggregates, Gini coefficients,
and various steady-state statistics needed for the model.

Translated from MATLAB to Julia for the HANK model replication.
"""
function Cal_SS_stats(mu_dist, c_a_guess, c_n_guess, AProb, b_n_star, b_a_star, a_a_star,
                     grid, param, meshes, TT, Output, R_fc, W_fc, H_fc, Profits_fc,
                     n, u, v, J, V, M, f, mc, Lambda, NWb, THETA, Profit_FI, Ab, Bb, Cb,
                     mutil_c, mutil_c_a, mutil_c_n, Value)

    # Initialize SS_stats dictionary
    SS_stats = Dict{String, Any}()

    ## Create meshes
    # Note: MATLAB's meshgrid and ndgrid have different conventions
    # meshgrid: [X, Y] = meshgrid(x, y) creates X with rows of x, Y with columns of y
    # ndgrid: [X, Y, Z] = ndgrid(x, y, z) creates proper 3D grids

    # For 2D mesh (used later for Gini calculations)
    mesh_a = repeat(grid["a"]', grid["nb"], 1)
    mesh_b = repeat(grid["b"], 1, grid["na"])
    mesh = Dict("a" => mesh_a, "b" => mesh_b)

    # Auxiliary variables for wages
    auxWW = ones(grid["nb"], grid["na"], grid["nse"])
    auxWW[:, :, (grid["ns"]+1):end] .= 0
    auxWW = auxWW .* (meshes["se"].^(param["b_aux"] - 1))

    NW = param["xi"] / (1 + param["xi"]) * n * param["w_bar"]
    WW_aux = (1 - param["tau_w"]) * (NW * ones(grid["nb"], grid["na"], grid["nse"]))
    WW = (1 - param["tau_w"]) * (NW * ones(grid["nb"], grid["na"], grid["nse"]) .+
         param["b_share"] * Profits_fc / ((1 - param["u"]) * (1 - param["Eshare"]) * grid["s2_bar"]) .* auxWW)
    WW[:, :, end] = (1 - param["tau_w"]) * (param["Eratio"] * Profits_fc) / param["Eshare"] * ones(grid["nb"], grid["na"])

    RR = R_fc
    RR_aux = R_fc + (param["death_rate"]) / (1 - param["death_rate"]) * param["q"]
    RBRB = param["R_cb"] / param["pi_cb"] .+ (meshes["b"] .< 0) .* (param["Rprem"] / param["pi_cb"])
    RBRB_aux = (param["R_cb"] / param["pi_cb"] .+ (meshes["b"] .< 0) .* (param["Rprem"] / param["pi_cb"])) /
               (1 - param["death_rate"]) * param["b_a_aux"]

    inc_transfer = param["LT"] * ones(grid["nb"], grid["na"], grid["nse"])

    ## Compute marginal distributions
    mu_redux = dropdims(sum(mu_dist, dims=3), dims=3)

    mu_dist_b = dropdims(sum(sum(mu_dist, dims=2), dims=3), dims=(2,3))
    mu_dist_a = dropdims(sum(sum(mu_dist, dims=1), dims=3), dims=(1,3))
    mu_dist_se = dropdims(sum(sum(mu_dist, dims=1), dims=2), dims=(1,2))

    ## H_tilde and next period distribution
    H_tilde = kron(param["P_SS3"]', sparse(I, grid["nb"] * grid["na"], grid["nb"] * grid["na"]))

    mu_dist_next = (mu_dist[:]') * TT
    mu_dist_next = reshape(mu_dist_next, (grid["nb"], grid["na"], grid["nse"]))

    mu_dist_next_b = dropdims(sum(sum(mu_dist_next, dims=2), dims=3), dims=(2,3))
    mu_dist_next_a = dropdims(sum(sum(mu_dist_next, dims=1), dims=3), dims=(1,3))
    mu_dist_next_se = dropdims(sum(sum(mu_dist_next, dims=1), dims=2), dims=(1,2))

    mu_dist_tilde = H_tilde * mu_dist[:]
    mu_dist_tilde = reshape(mu_dist_tilde, (grid["nb"], grid["na"], grid["nse"]))

    mu_dist_tilde_b = dropdims(sum(sum(mu_dist_tilde, dims=2), dims=3), dims=(2,3))
    mu_dist_tilde_a = dropdims(sum(sum(mu_dist_tilde, dims=1), dims=3), dims=(1,3))
    mu_dist_tilde_se = dropdims(sum(sum(mu_dist_tilde, dims=1), dims=2), dims=(1,2))

    mu_dist_tilde_next = H_tilde * mu_dist_next[:]
    mu_dist_tilde_next = reshape(mu_dist_tilde_next, (grid["nb"], grid["na"], grid["nse"]))

    ## Store basic statistics
    SS_stats["H_tilde"] = H_tilde
    SS_stats["Y"] = Output
    SS_stats["v"] = v
    SS_stats["u"] = u
    SS_stats["Lambda"] = Lambda
    SS_stats["f"] = f
    SS_stats["r_a"] = R_fc
    SS_stats["w"] = W_fc
    SS_stats["r_l"] = H_fc
    SS_stats["J"] = J
    SS_stats["M"] = M
    SS_stats["V"] = V
    SS_stats["L"] = grid["L"]
    SS_stats["N"] = grid["N"]
    SS_stats["n"] = ((1 - param["tau_w"]) * W_fc / param["psi"])^(1 / param["xi"])
    SS_stats["r_k"] = mc * param["theta"] * (v * grid["K"])^(param["theta"] - 1) *
                      (SS_stats["n"] * (1 - u) * (1 - param["Eshare"]) * grid["se_bar"])^param["theta_2"]
    SS_stats["mc"] = mc
    SS_stats["Profit"] = Profits_fc
    SS_stats["N2"] = (1 - param["lambda"] + param["lambda"] * SS_stats["f"]) * SS_stats["N"] +
                     SS_stats["f"] * (1 - SS_stats["N"] - param["Eshare"])

    grid["n"] = SS_stats["n"]
    param["fix"] = param["fix_ratio"] * SS_stats["Y"]
    SS_stats["h"] = SS_stats["r_l"]
    SS_stats["r_a_aux"] = SS_stats["r_a"] + param["death_rate"] / (1 - param["death_rate"]) * param["q"]
    SS_stats["RRa"] = (1 + SS_stats["r_a"])
    SS_stats["RR"] = param["R_cb"] / param["pi_bar"]

    ## Financial intermediary statistics
    SS_stats["NW_b"] = NWb
    SS_stats["leverage"] = THETA
    SS_stats["leverage_e"] = THETA / (1 - param["psi_cp"])
    SS_stats["B_FI"] = (THETA - 1) * NWb
    SS_stats["A_b"] = Ab
    SS_stats["A_I"] = SS_stats["NW_b"] * SS_stats["leverage"]
    SS_stats["A_g"] = param["A_g_ratio"] * SS_stats["Y"]
    SS_stats["A_F"] = param["A_F_ratio"] * grid["K"]
    SS_stats["A_NI"] = grid["K"] - SS_stats["A_I"] - SS_stats["A_F"] - SS_stats["A_g"]
    SS_stats["Profit_FI"] = Profit_FI
    SS_stats["K2"] = SS_stats["A_NI"] / (1 - param["death_rate"]) + SS_stats["A_g"] +
                     SS_stats["A_I"] + SS_stats["A_F"]

    param["fix2"] = SS_stats["Profit_FI"]

    SS_stats["ShareBorrower"] = sum((grid["b"] .< 0) .* dropdims(sum(sum(mu_dist, dims=2), dims=3), dims=(2,3)))
    SS_stats["K"] = grid["K"]

    ## Government bonds and aggregates
    SS_stats["B_gov_ncp"] = param["B_gov_ratio"] * SS_stats["Y"]
    SS_stats["B_cp"] = (1 + param["tau_cp"]) * SS_stats["A_g"]
    SS_stats["B_gov"] = SS_stats["B_gov_ncp"] + SS_stats["B_cp"]
    SS_stats["B_F"] = param["B_F_ratio"] * SS_stats["B_gov"]

    SS_stats["B_neg"] = sum((grid["b"] .< 0) .* grid["b"] .* mu_dist_b)
    SS_stats["B_pos"] = sum((grid["b"] .>= 0) .* grid["b"] .* mu_dist_b)

    SS_stats["B_b"] = SS_stats["B_gov"] + (SS_stats["B_FI"] - SS_stats["B_pos"] / (1 - param["death_rate"]) *
                      param["b_a_aux"] - SS_stats["B_neg"] / (1 - param["death_rate"]) * param["b_a_aux"] - SS_stats["B_F"])

    SS_stats["B_nb"] = sum(grid["b"] .* mu_dist_b)
    SS_stats["B"] = SS_stats["B_nb"] + SS_stats["B_b"] + SS_stats["B_F"]
    SS_stats["B_FI_nb"] = SS_stats["B_nb"] / (1 - param["death_rate"]) * param["b_a_aux"]
    SS_stats["B2"] = SS_stats["B_nb"] / (1 - param["death_rate"]) * param["b_a_aux"] +
                     SS_stats["B_b"] + SS_stats["B_F"]

    grid["B"] = SS_stats["B"]
    SS_stats["BoverK"] = SS_stats["B"] / SS_stats["K"]
    SS_stats["KY"] = SS_stats["K"] / Output / 4
    SS_stats["BY"] = SS_stats["B"] / Output / 4
    SS_stats["Y"] = Output

    ## Borrowing constraint statistics
    BCaux_b = dropdims(sum(sum(mu_dist, dims=2), dims=3), dims=(2,3))
    SS_stats["b_bc"] = BCaux_b[1]
    SS_stats["b_0"] = (BCaux_b[grid["b"] .== 0])[1]

    BCaux_a = dropdims(sum(sum(mu_dist, dims=1), dims=3), dims=(1,3))
    SS_stats["a_bc"] = BCaux_a[1]

    aux_ba = dropdims(sum(mu_dist, dims=3), dims=3)
    SS_stats["WtH_b0"] = sum(aux_ba[(mesh["b"] .== 0) .& (mesh["a"] .> 0)])
    SS_stats["Zero"] = sum(aux_ba[(mesh["b"] .== 0) .& (mesh["a"] .== 0)])
    SS_stats["WtH_bnonpos"] = sum(aux_ba[(mesh["b"] .<= 0) .& (mesh["a"] .> 0)])

    ## Profit calculations
    SS_stats["Profit_int"] = SS_stats["Y"] - SS_stats["mc"] * SS_stats["Y"] - param["fix"]
    SS_stats["Profit_L"] = (SS_stats["r_l"] - param["w_bar"] - param["fix_L"]) * grid["L"] - param["iota"] * SS_stats["V"]
    SS_stats["Profit_K"] = (SS_stats["r_k"] * SS_stats["v"] - param["delta_ss"]) * grid["K"]
    SS_stats["Profit2"] = SS_stats["Profit_int"] + SS_stats["Profit_L"] + SS_stats["Profit_K"]

    SS_stats["AdjProb"] = sum(AProb[:] .* mu_dist_tilde[:])

    ## More statistics
    SS_stats["nn"] = ((1 - param["tau_w"]) * SS_stats["w"] / param["psi"])^(1 / param["xi"])
    SS_stats["N"] = SS_stats["N"]
    SS_stats["W"] = param["w_bar"]
    SS_stats["tau_w"] = param["tau_w"]
    SS_stats["tau_a"] = param["tau_a"]
    SS_stats["PROFIT"] = SS_stats["Profit"]
    SS_stats["Ntilde"] = (1 - param["lambda"]) * SS_stats["N"] + SS_stats["M"]
    SS_stats["R_A"] = SS_stats["r_a"]
    SS_stats["Q"] = 1
    SS_stats["R_cbminus"] = param["R_cb"]
    SS_stats["R_cb"] = param["R_cb"]
    SS_stats["PI"] = param["pi_bar"]
    SS_stats["PInext"] = param["pi_bar"]
    SS_stats["LT"] = param["LT"]

    ## Aggregate consumption, investment, capital
    X_a_RHS = WW .* meshes["se"] .+ (param["q"] .+ RR_aux) .* meshes["a"] .+ RBRB_aux .* meshes["b"] .-
              param["q"] .* a_a_star .- b_a_star .+ SS_stats["LT"]
    X_n_RHS = WW .* meshes["se"] .+ (param["q"] .+ RR_aux) .* meshes["a"] .+ RBRB_aux .* meshes["b"] .-
              param["q"] .* meshes["a"] .- b_n_star .+ SS_stats["LT"]

    X_a_agg = sum(X_a_RHS .* mu_dist_tilde .* AProb)
    X_n_agg = sum(X_n_RHS .* mu_dist_tilde .* (1 .- AProb))
    X_agg2 = X_a_agg + X_n_agg

    X_agg = sum(mu_dist_tilde[:] .* AProb[:] .* c_a_guess[:] .+
                mu_dist_tilde[:] .* (1 .- AProb[:]) .* c_n_guess[:])

    B_tilde_agg = sum(mu_dist_tilde[:] .* AProb[:] .* b_a_star[:] .+
                      mu_dist_tilde[:] .* (1 .- AProb[:]) .* b_n_star[:])
    K_tilde_agg = sum((AProb[:] .* a_a_star[:] .+ (1 .- AProb[:]) .* meshes["a"][:]) .* mu_dist_tilde[:])

    SS_stats["K_tilde"] = K_tilde_agg
    SS_stats["B_tilde"] = B_tilde_agg

    B_agg = (1 - param["death_rate"]) * B_tilde_agg
    K_agg = (1 - param["death_rate"]) * K_tilde_agg

    # Create grid.se_aux for consumption calculations
    grid_s_aux = hcat(grid["s"], min.(param["b_ratio"]*grid["s"], grid["s_bar"]), [1])

    C_agg = X_agg + (1 - param["tau_w"]) / (1 + param["xi"]) * W_fc *
            sum(grid["se"][1:grid["ns"]] .* mu_dist_tilde_se[1:grid["ns"]]) * grid["n"]
    C_agg2 = X_agg2 + (1 - param["tau_w"]) / (1 + param["xi"]) * W_fc *
             sum(grid["se"][1:grid["ns"]] .* mu_dist_tilde_se[1:grid["ns"]]) * grid["n"]

    I_agg = K_tilde_agg + SS_stats["A_g"] + SS_stats["A_b"] - (1 - param["delta_ss"]) * grid["K"]
    I_agg2 = (1 - param["death_rate"]) * K_tilde_agg + SS_stats["A_g"] + SS_stats["A_b"] -
             (1 - param["delta_ss"]) * grid["K"]

    SS_stats["I"] = I_agg2
    SS_stats["I2"] = I_agg2

    absorp_annuity = param["death_rate"] * K_tilde_agg - param["death_rate"] / (1 - param["death_rate"]) * SS_stats["A_NI"]

    SS_stats["C2"] = C_agg + param["frac_b"] * Cb
    SS_stats["C"] = C_agg

    ## Taxes and government
    SS_stats["T"] = param["tau_w"] * param["w_bar"] * grid["n"] *
                    (sum(grid["se"][1:grid["ns"]] .* mu_dist_tilde_se[1:grid["ns"]]) +
                     param["xi"] / (1 + param["xi"]) * sum(grid["se"][(grid["ns"]+1):(2*grid["ns"])] .*
                     mu_dist_tilde_se[(grid["ns"]+1):(2*grid["ns"])])) +
                    param["tau_a"] * SS_stats["Profit"] + param["tau_a"] * (SS_stats["Profit_FI"] - param["fix2"])

    SS_stats["C_b"] = param["MMF_cont_ratio"] * SS_stats["T"] +
                      (param["R_cb"] / param["pi_bar"] - 1) * SS_stats["B_b"]
    SS_stats["LT"] = param["LT"] - SS_stats["C_b"]

    SS_stats["G"] = 0.2 * SS_stats["Y"]
    SS_stats["G"] = SS_stats["T"] + SS_stats["B_gov"] - param["R_cb"] / param["pi_cb"] * SS_stats["B_gov"] +
                    (param["q"] + SS_stats["r_a"]) / param["q"] * SS_stats["A_g"] - param["q"] * SS_stats["A_g"] -
                    param["xi"] / (1 + param["xi"]) * sum(grid["se"][(grid["ns"]+1):(2*grid["ns"])] .*
                    mu_dist_tilde_se[(grid["ns"]+1):(2*grid["ns"])]) * grid["n"] * param["w_bar"] -
                    SS_stats["LT"] - param["MMF_cont_ratio"] * SS_stats["T"] - param["tau_cp"] * SS_stats["A_g"]

    SS_stats["GtoY"] = SS_stats["G"] / Output
    SS_stats["UB"] = param["w_bar"] * param["n"] * param["xi"] / (1 + param["xi"]) *
                     sum(grid["se"][(grid["ns"]+1):(2*grid["ns"])] .* mu_dist_tilde_se[(grid["ns"]+1):(2*grid["ns"])])

    Int_cost = sum((abs.(grid["b"]) .* (grid["b"] .< 0)) .*
                   dropdims(sum(sum(mu_dist, dims=2), dims=3), dims=(2,3)) * param["Rprem"] / param["pi_cb"]) /
               (1 - param["death_rate"]) * param["b_a_aux"]

    G_agg = SS_stats["G"]

    SS_stats["Chh"] = C_agg
    SS_stats["C_hh"] = SS_stats["Chh"]
    SS_stats["R_a"] = (param["q"] + SS_stats["r_a"]) / param["q"]
    SS_stats["R"] = param["R_cb"] / param["pi_cb"]
    SS_stats["A_hh"] = K_tilde_agg * (1 - param["death_rate"])
    SS_stats["B_hh"] = SS_stats["B_nb"]

    ## Banker statistics
    SS_stats["zz"] = (SS_stats["R_a"] - param["b_a_aux2"] * SS_stats["R"]) * SS_stats["leverage"] +
                     param["b_a_aux2"] * SS_stats["R"]
    SS_stats["xx"] = SS_stats["zz"]
    SS_stats["Lambda_b"] = param["Lambda_bank"]
    SS_stats["vv"] = ((1 - param["theta_b"]) * SS_stats["Lambda_b"] *
                      (SS_stats["R_a"] - param["b_a_aux2"] * SS_stats["R"])) /
                     (1 - param["theta_b"] * SS_stats["Lambda_b"] * SS_stats["xx"])
    SS_stats["ee"] = (1 - param["theta_b"]) * SS_stats["Lambda_b"] * param["b_a_aux2"] * SS_stats["R"] /
                     (1 - param["theta_b"] * SS_stats["Lambda_b"] * SS_stats["zz"])

    param["G"] = SS_stats["G"]
    param["r_a"] = R_fc
    param["w_bar"] = W_fc
    param["profits"] = Profits_fc
    param["L"] = grid["L"]

    SS_stats["x_a"] = SS_stats["A_b"] / SS_stats["A_hh"]
    SS_stats["x_b"] = SS_stats["B_b"] / SS_stats["B_hh"]

    ## Market clearing check
    if param["b_a_aux2"] < 1
        diff_mk = SS_stats["Y"] - C_agg - param["tau_cp"] * SS_stats["A_g"] - param["frac_b"] * SS_stats["C_b"] - I_agg -
                  G_agg - param["iota"] * SS_stats["V"] - Int_cost - param["fix"] - absorp_annuity +
                  param["death_rate"] * K_tilde_agg - param["fix2"] -
                  (param["R_cb"] / param["pi_bar"] - 1) * SS_stats["B_F"] - SS_stats["r_a"] * SS_stats["A_F"] -
                  param["fix_L"] * SS_stats["L"] - param["death_rate"] * SS_stats["B_hh"] * param["R_cb"] / param["pi_bar"] +
                  param["death_rate"] * SS_stats["B_FI"]
        diff_mk2 = SS_stats["Y"] - C_agg2 - param["tau_cp"] * SS_stats["A_g"] - param["frac_b"] * SS_stats["C_b"] - I_agg -
                   G_agg - param["iota"] * SS_stats["V"] - Int_cost - param["fix"] - absorp_annuity +
                   param["death_rate"] * K_tilde_agg - param["fix2"] -
                   (param["R_cb"] / param["pi_bar"] - 1) * SS_stats["B_F"] - SS_stats["r_a"] * SS_stats["A_F"] -
                   param["fix_L"] * SS_stats["L"] - param["death_rate"] * SS_stats["B_hh"] * param["R_cb"] / param["pi_bar"] +
                   param["death_rate"] * SS_stats["B_FI"]
    else
        diff_mk = SS_stats["Y"] - C_agg - param["tau_cp"] * SS_stats["A_g"] - param["frac_b"] * SS_stats["C_b"] - I_agg -
                  G_agg - param["iota"] * SS_stats["V"] - Int_cost - param["fix"] - absorp_annuity +
                  param["death_rate"] * K_tilde_agg - param["fix2"] -
                  (param["R_cb"] / param["pi_bar"] - 1) * SS_stats["B_F"] - SS_stats["r_a"] * SS_stats["A_F"] -
                  param["fix_L"] * SS_stats["L"] - param["death_rate"] * SS_stats["B_hh"] * param["R_cb"] / param["pi_bar"]
        diff_mk2 = SS_stats["Y"] - C_agg2 - param["tau_cp"] * SS_stats["A_g"] - param["frac_b"] * SS_stats["C_b"] - I_agg -
                   G_agg - param["iota"] * SS_stats["V"] - Int_cost - param["fix"] - absorp_annuity +
                   param["death_rate"] * K_tilde_agg - param["fix2"] -
                   (param["R_cb"] / param["pi_bar"] - 1) * SS_stats["B_F"] - SS_stats["r_a"] * SS_stats["A_F"] -
                   param["fix_L"] * SS_stats["L"] - param["death_rate"] * SS_stats["B_hh"] * param["R_cb"] / param["pi_bar"]
    end

    ## Gini coefficients
    # Wealth Gini
    total_wealth = mesh["a"][:] * param["q"] + mesh["b"][:]
    IX = sortperm(total_wealth)
    total_wealth = total_wealth[IX]
    total_wealth_pdf = mu_redux[IX]
    total_wealth_cdf = cumsum(total_wealth_pdf)

    SS_stats["NegNetWorth"] = sum((total_wealth .< 0) .* total_wealth_pdf)

    mean_total_wealth = sum(total_wealth .* total_wealth_pdf)
    std_total_wealth = sqrt(sum(total_wealth.^2 .* total_wealth_pdf) - mean_total_wealth^2)

    S_W = cumsum(total_wealth_pdf .* total_wealth)
    S_W = vcat([0], S_W)
    SS_stats["GiniW"] = 1 - (sum(total_wealth_pdf .* (S_W[1:end-1] + S_W[2:end])) / S_W[end])

    # Liquid Gini
    liquid_sort = mesh["b"][:]
    IX = sortperm(liquid_sort)
    liquid_sort = liquid_sort[IX]
    liquid_pdf = mu_redux[IX]
    liquid_cdf = cumsum(liquid_pdf)
    SS_stats["Negliquid"] = sum((liquid_sort .< 0) .* liquid_pdf)

    S_LI = cumsum(liquid_pdf .* liquid_sort)
    S_LI = vcat([0], S_LI)
    SS_stats["GiniLI"] = 1 - (sum(liquid_pdf .* (S_LI[1:end-1] + S_LI[2:end])) / S_LI[end])

    # Illiquid Gini
    illiquid_sort = mesh["a"][:]
    IX = sortperm(illiquid_sort)
    illiquid_sort = illiquid_sort[IX]
    illiquid_pdf = mu_redux[IX]
    illiquid_cdf = cumsum(illiquid_pdf)

    S_IL = cumsum(illiquid_pdf .* illiquid_sort)
    S_IL = vcat([0], S_IL)
    SS_stats["GiniIL"] = 1 - (sum(illiquid_pdf .* (S_IL[1:end-1] + S_IL[2:end])) / S_IL[end])

    # Income Gini calculations
    # Create wage grids for Gini computation
    NW_Gini = n * param["w_bar"]
    auxWW = ones(grid["nb"], grid["na"], grid["nse"])
    WW_Gini = (1 - param["tau_w"]) * (NW_Gini * ones(grid["nb"], grid["na"], grid["nse"]) .+
               param["b_share"] * Profits_fc / ((1 - param["u"]) * (1 - param["Eshare"]) * grid["s2_bar"]) .* auxWW)
    WW_Gini[:, :, end] .= (1 - param["tau_a"]) * (param["Eratio"] * SS_stats["Profit"]) / param["Eshare"]

    # Create transfer grid
    inc_transfer = param["LT"] * ones(grid["nb"], grid["na"], grid["nse"])

    # Total income Gini
    income_unsorted = (WW_Gini .* meshes["se_aux"] .+
                       SS_stats["r_a_aux"] .* meshes["a"] .+
                       (param["R_cb"]/param["pi_cb"] - 1) .* meshes["b"] .* (meshes["b"] .>= 0) .+
                       ((param["R_cb"] + param["Rprem"])/param["pi_cb"] - 1) .* meshes["b"] .* (meshes["b"] .< 0) .+
                       inc_transfer)

    income_sort_vec = vec(income_unsorted)
    IX = sortperm(income_sort_vec)
    income_sort = income_sort_vec[IX]
    income_pdf = mu_dist_tilde[IX]
    income_cdf = cumsum(income_pdf)

    S_income = cumsum(income_pdf .* income_sort)
    S_income = vcat([0.0], S_income)
    SS_stats["GiniIncome"] = 1 - (sum(income_pdf .* (S_income[1:end-1] .+ S_income[2:end])) / S_income[end])

    # Labor income Gini
    Lincome_unsorted = WW_Gini .* meshes["se_aux"]
    Lincome_sort_vec = vec(Lincome_unsorted)
    IX = sortperm(Lincome_sort_vec)
    Lincome_sort = Lincome_sort_vec[IX]
    Lincome_pdf = mu_dist_tilde[IX]
    Lincome_cdf = cumsum(Lincome_pdf)

    S_Lincome = cumsum(Lincome_pdf .* Lincome_sort)
    S_Lincome = vcat([0.0], S_Lincome)
    SS_stats["GiniLIncome"] = 1 - (sum(Lincome_pdf .* (S_Lincome[1:end-1] .+ S_Lincome[2:end])) / S_Lincome[end])

    # Non-labor income Gini
    NLincome_unsorted = (SS_stats["r_a_aux"] .* meshes["a"] .+
                         (param["R_cb"]/param["pi_cb"]/(1-param["death_rate"])*param["b_a_aux"] - 1) .* meshes["b"] .* (meshes["b"] .>= 0) .+
                         ((param["R_cb"] + param["Rprem"])/param["pi_cb"]/(1-param["death_rate"])*param["b_a_aux"] - 1) .* meshes["b"] .* (meshes["b"] .< 0))
    NLincome_sort_vec = vec(NLincome_unsorted)
    IX = sortperm(NLincome_sort_vec)
    NLincome_sort = NLincome_sort_vec[IX]
    NLincome_pdf = mu_dist_tilde[IX]
    NLincome_cdf = cumsum(NLincome_pdf)

    S_NLincome = cumsum(NLincome_pdf .* NLincome_sort)
    S_NLincome = vcat([0.0], S_NLincome)
    SS_stats["GiniNLIncome"] = 1 - (sum(NLincome_pdf .* (S_NLincome[1:end-1] .+ S_NLincome[2:end])) / S_NLincome[end])

    # Consumption Gini
    cons_aux = vcat(vec(c_a_guess), vec(c_n_guess))
    mu_dist_tilde_aux = vcat(vec(AProb .* mu_dist_tilde), vec((1 .- AProb) .* mu_dist_tilde))

    IX = sortperm(cons_aux)
    cons_sort = cons_aux[IX]
    cons_pdf = mu_dist_tilde_aux[IX]
    cons_cdf = cumsum(cons_pdf)

    S_cons = cumsum(cons_pdf .* cons_sort)
    S_cons = vcat([0.0], S_cons)
    SS_stats["GiniCons"] = 1 - (sum(cons_pdf .* (S_cons[1:end-1] .+ S_cons[2:end])) / S_cons[end])

    SS_stats["LG"] = SS_stats["LT"] + SS_stats["G"]
    param["C_b"] = SS_stats["C_b"]

    ## MRS (Marginal Rate of Substitution) calculations
    meshaux_b, meshaux_a, meshaux_se_aux = ndgrid(grid["b"], grid["a"], 1:grid["nse"])
    meshaux = Dict("b" => meshaux_b, "a" => meshaux_a, "se_aux" => meshaux_se_aux)

    # Transition probability matrices
    P_transition_exp = param["P_SS2"] * param["P_SS3"]

    # Interpolate marginal utility of consumption to next period
    # First permute mutil_c dimensions and interpolate along a dimension
    mutil_c_reshaped = reshape(mutil_c, (grid["nb"], grid["na"], grid["nse"]))
    mutil_c_permuted = permutedims(mutil_c_reshaped, [2, 1, 3])  # Now (na, nb, nse)

    # Interpolate with identity scaling (min(1,1)*grid.a = grid.a)
    mutil_c_interp = zeros(grid["na"], grid["nb"], grid["nse"])
    for k in 1:grid["nse"]
        for j in 1:grid["nb"]
            mutil_c_interp[:, j, k] = mutil_c_permuted[:, j, k]
        end
    end

    # Permute back
    mutil_c_aux = permutedims(mutil_c_interp, [2, 1, 3])  # Back to (nb, na, nse)
    mutil_c_aux = vec(mutil_c_aux)

    # Apply transition probabilities
    mutil_c_aux = reshape(reshape(mutil_c_aux, (grid["nb"] * grid["na"], grid["nse"])) * P_transition_exp',
                         (grid["nb"], grid["na"], grid["nse"]))

    # Create interpolator for next period marginal utility
    #mutil_c_next = griddedInterpolant((grid["b"], grid["a"], 1:grid["nse"]), mutil_c_aux)
    #mutil_c_next = extrapolate(interpolate((grid["b"], grid["a"], 1:grid["nse"]), mutil_c_aux[1:end], Gridded(Linear())), Line())
    itp = interpolate((vec(meshaux_b[:, 1, 1]), vec(meshaux_a[1, :, 1]), vec(meshaux_se_aux[1, 1, :])),
                       mutil_c_aux, Gridded(Linear()))
    mutil_c_next = extrapolate(itp, Line()) #Flat()
    # MRS calculations with adjustment
    MRS_a = param["beta"] * (1 - param["death_rate"]) *
            mutil_c_next.(b_a_star, a_a_star, meshaux["se_aux"]) ./ mutil_c_a .*
            AProb .* mu_dist_tilde .* (a_a_star ./ SS_stats["A_hh"])

    MRS_n = param["beta"] * (1 - param["death_rate"]) *
            mutil_c_next.(b_n_star, meshaux["a"], meshaux["se_aux"]) ./ mutil_c_n .*
            (1 .- AProb) .* mu_dist_tilde .* (meshaux["a"] ./ SS_stats["A_hh"])

    # Alternative MRS using meshaux.a instead of a_a_star for adjustment case
    MRS_a_2 = param["beta"] * (1 - param["death_rate"]) *
              mutil_c_next.(b_a_star, a_a_star, meshaux["se_aux"]) ./ mutil_c_a .*
              AProb .* mu_dist_tilde .* (meshaux["a"] ./ SS_stats["A_hh"])

    MRS_n_2 = param["beta"] * (1 - param["death_rate"]) *
              mutil_c_next.(b_n_star, meshaux["a"], meshaux["se_aux"]) ./ mutil_c_n .*
              (1 .- AProb) .* mu_dist_tilde .* (meshaux["a"] ./ SS_stats["A_hh"])

    # Sum MRS components
    MRS = sum(MRS_a) + sum(MRS_n)
    MRS_2 = sum(MRS_a_2) + sum(MRS_n_2)

    # MRS auxiliary calculations (with distribution weighting but no asset weighting)
    MRS_a_aux = param["beta"] * (1 - param["death_rate"]) *
                mutil_c_next.(b_a_star, a_a_star, meshaux["se_aux"]) ./ mutil_c_a .*
                AProb .* mu_dist_tilde

    MRS_n_aux = param["beta"] * (1 - param["death_rate"]) *
                mutil_c_next.(b_n_star, meshaux["a"], meshaux["se_aux"]) ./ mutil_c_n .*
                (1 .- AProb) .* mu_dist_tilde

    # MRS auxiliary calculations (without distribution weighting) - needed for steady_anal
    MRS_a_aux2 = param["beta"] * (1 - param["death_rate"]) *
                 mutil_c_next.(b_a_star, a_a_star, meshaux["se_aux"]) ./ mutil_c_a

    MRS_n_aux2 = param["beta"] * (1 - param["death_rate"]) *
                 mutil_c_next.(b_n_star, meshaux["a"], meshaux["se_aux"]) ./ mutil_c_n

    # MRS for entrepreneurs
    MRS_ent_aux = sum(MRS_a_aux[:, :, end] + MRS_n_aux[:, :, end]) / param["Eshare"]

    # Store MRS statistics
    SS_stats["MRS_aux_hh"] = MRS
    SS_stats["MRS_aux_hh_2"] = MRS_2
    SS_stats["MRS"] = param["pi_bar"] / param["R_cb"]
    SS_stats["MRS_a_aux2"] = MRS_a_aux2
    SS_stats["MRS_n_aux2"] = MRS_n_aux2

    # Update param with Lambda ratios
    param["Lambda_aux"] = SS_stats["Lambda"] / MRS_ent_aux
    param["Lambda_b_aux"] = param["pi_bar"] / param["R_cb"] / MRS_ent_aux
    param["lambda_hh_aux"] = SS_stats["Lambda"] / MRS

    # J statistics
    I_mat = Matrix(1.0I, grid["ns"], grid["ns"])
    
    SS_stats["J_aux1"] = inv(I_mat - param["beta_L"] * (1 - param["death_rate"]) *
                             (1 - param["lambda"]) * (1 - param["in"]) * param["P_SS"]) *
                         (SS_stats["r_l"] - param["w_bar"]) * n * grid["s"]'
    SS_stats["J_aux2"] = inv(I_mat - param["beta_L"] * (1 - param["death_rate"]) *
                             (1 - param["lambda"]) * (1 - param["in"]) * param["P_SS"]) *
                         n * grid["s"]'

    SS_stats["J_bar1"] = (SS_stats["J_aux1"]' * grid["s_dist"])[1]
    SS_stats["J_bar2"] = (SS_stats["J_aux2"]' * grid["s_dist"])[1]

    SS_stats["B_gov_ncp2"] = SS_stats["B_gov_ncp"] - SS_stats["B_F"]

    # Adjustment cost and probability
    #=AC = param["sigma_chi"] .* ((1 .- AProb[:]) .* log.(1 .- AProb[:]) .+ AProb[:] .* log.(AProb[:])) .+
         param["mu_chi"] .* AProb[:] .-
         (param["mu_chi"] - param["sigma_chi"] * log(1 + exp(param["mu_chi"] / param["sigma_chi"])))
    AC = AC .* param["nu"] =#
    AC = param["sigma_chi"] .* ((1 .- AProb[:]) + AProb[:] .* log.(AProb[:]))
+ param["mu_chi"] .* AProb[:] .- (param["mu_chi"] - param["sigma_chi"] * log(1 + exp(param["mu_chi"] / param["sigma_chi"])))

AProb = AProb .* param["nu"]
AC = AC .* param["nu"]

    AProb_scaled = AProb .* param["nu"]
    SS_stats["AProb"] = reshape(AProb_scaled, (grid["nb"], grid["na"], grid["nse"]))

    # Load AC and VALUE from saved files (placeholder for now)
    # SS_stats["AC"] = AC_from_file
    # SS_stats["VALUE"] = VALUE_from_file
    #SS_stats["AC"] = reshape(AC, (grid["nb"], grid["na"], grid["nse"]))
    #SS_stats["VALUE"] = Value

    SS_stats["AvgC"] = (param["delta_ss"] * grid["K"] + param["w_bar"] * grid["L"] +
                        param["iota"] * SS_stats["V"] + param["fix"]) / SS_stats["Y"]

    SS_stats["mu_dist_NE"] = copy(mu_dist)
    SS_stats["mu_dist_NE"][:, :, end] .= 0
    SS_stats["mu_dist_NE"] = SS_stats["mu_dist_NE"] / sum(SS_stats["mu_dist_NE"][:])

    return SS_stats, grid, param
end
