"""
    steady_anal(grid, param, meshes, SS_stats, AProb, mu_dist, n, H_tilde,
                b_n_star, b_a_star, a_a_star, c_n_guess, c_a_guess)

Compute distributional analysis and quintile statistics for the steady state.

This function performs detailed quintile analysis of:
- Wealth distribution
- Income distribution
- Liquid assets distribution
- Consumption distribution
- MPCs (marginal propensities to consume)

# Arguments
- `grid`: Grid structure containing state space grids
- `param`: Parameter structure
- `meshes`: Mesh grids for state variables
- `SS_stats`: Steady state statistics from Cal_SS_stats
- `AProb`: Adjustment probability
- `mu_dist`: Household distribution
- `n`: Employment rate
- `H_tilde`: Transition matrix after productivity transition
- `b_n_star`: Bond policy (no adjustment)
- `b_a_star`: Bond policy (adjustment)
- `a_a_star`: Equity policy (adjustment)
- `c_n_guess`: Consumption policy (no adjustment)
- `c_a_guess`: Consumption policy (adjustment)

# Returns
A dictionary containing distributional statistics and quintile breakdowns

Translated from MATLAB to Julia for the HANK model replication.
"""
function steady_anal(grid, param, meshes, SS_stats, AProb, mu_dist, n, H_tilde,
                    b_n_star, b_a_star, a_a_star, c_n_guess, c_a_guess)

    # Initialize output dictionary
    anal_stats = Dict{String, Any}()

    ## 1. Create auxiliary productivity grids for income decomposition

    # grid.se_aux = [grid.s, min(param.b_ratio*grid.s, grid.s_bar), 1]
    grid_se_aux = hcat(grid["s"],
                      min.(param["b_ratio"] .* grid["s"], grid["s_bar"]),
                      [1])

    # grid.se1_aux = [grid.s, zeros(1,grid.ns), 0] - labor income only
    grid_se1_aux = hcat(grid["s"],
                       zeros(1, Int(grid["ns"])),
                       [0])

    # grid.se2_aux = [zeros(1,grid.ns), min(param.b_ratio*grid.s, grid.s_bar), 0] - UB income only
    grid_se2_aux = hcat(zeros(1, Int(grid["ns"])),
                       min.(param["b_ratio"] .* grid["s"], grid["s_bar"]),
                       [0])

    # grid.se3_aux = [zeros(1,2*grid.ns), 1] - entrepreneur income only
    grid_se3_aux = hcat(zeros(1, 2 * Int(grid["ns"])),
                       [1])

    ## 2. Create meshes for auxiliary grids
    nb_val = Int(grid["nb"])
    na_val = Int(grid["na"])
    nse_val = Int(grid["nse"])
    ns_val = Int(grid["ns"])

    # Create 3D meshes (ignoring first two outputs like MATLAB [~, ~, meshes.se_aux])
    meshes_se_aux = zeros(nb_val, na_val, size(grid_se_aux, 2))
    meshes_se1_aux = zeros(nb_val, na_val, size(grid_se1_aux, 2))
    meshes_se2_aux = zeros(nb_val, na_val, size(grid_se2_aux, 2))
    meshes_se3_aux = zeros(nb_val, na_val, size(grid_se3_aux, 2))

    for k in 1:size(grid_se_aux, 2)
        for j in 1:na_val
            for i in 1:nb_val
                meshes_se_aux[i, j, k] = grid_se_aux[k]
            end
        end
    end

    for k in 1:size(grid_se1_aux, 2)
        for j in 1:na_val
            for i in 1:nb_val
                meshes_se1_aux[i, j, k] = grid_se1_aux[k]
            end
        end
    end

    for k in 1:size(grid_se2_aux, 2)
        for j in 1:na_val
            for i in 1:nb_val
                meshes_se2_aux[i, j, k] = grid_se2_aux[k]
            end
        end
    end

    for k in 1:size(grid_se3_aux, 2)
        for j in 1:na_val
            for i in 1:nb_val
                meshes_se3_aux[i, j, k] = grid_se3_aux[k]
            end
        end
    end
    meshes["se_aux"] = meshes_se_aux
    meshes["se1_aux"] = meshes_se1_aux
    meshes["se2_aux"] = meshes_se2_aux
    meshes["se3_aux"] = meshes_se3_aux
    ## 3. Compute wage and income grids

    # auxWW = ones(...) .* (meshes.se.^(param.b_aux-1))
    auxWW = ones(nb_val, na_val, nse_val)
    auxWW[:, :, (ns_val+1):end] .= 0
    auxWW = auxWW .* (meshes["se"] .^ (param["b_aux"] - 1))

    # NW_aux = n*param.w_bar
    NW_aux = n * param["w_bar"]

    # WW_aux2 = (1-param.tau_w)*(NW_aux*ones(...))
    WW_aux2 = (1 - param["tau_w"]) .* (NW_aux .* ones(nb_val, na_val, nse_val))

    # WW_aux = (1-param.tau_w)*(NW_aux*ones(...) + param.b_share*SS_stats.Profit/...)
    WW_aux = (1 - param["tau_w"]) .* (NW_aux .* ones(nb_val, na_val, nse_val) .+
             param["b_share"] .* SS_stats["Profit"] /
             ((1 - param["u"]) .* (1 - param["Eshare"]) .* grid["s2_bar"]) .* auxWW)

    # WW_aux(:,:,end) = (1-param.tau_w)*(param.Eratio*SS_stats.Profit)/param.Eshare*...
    WW_aux[:, :, end] .= (1 - param["tau_w"]) * (param["Eratio"] * SS_stats["Profit"]) /
                         param["Eshare"]

    # inc.transfer = param.LT*ones(...)
    inc_transfer = param["LT"] * ones(nb_val, na_val, nse_val)

    ## 4. Compute income meshes for different sources

    # total_income_mesh_aux = WW_aux.*meshes.se_aux + SS_stats.r_a_aux.*meshes.a + ...
    # Bond returns with different rates for positive/negative balances
    bond_return_pos = ((param["R_cb"] / (1 - param["death_rate"]) * param["b_a_aux"]) /
                      param["pi_cb"] - 1)
    bond_return_neg = (((param["R_cb"] + param["Rprem"]) / (1 - param["death_rate"]) *
                      param["b_a_aux"]) / param["pi_cb"] - 1)

    total_income_mesh_aux = WW_aux .* meshes_se_aux .+
                           SS_stats["r_a_aux"] .* meshes["a"] .+
                           bond_return_pos .* meshes["b"] .* (meshes["b"] .>= 0) .+
                           bond_return_neg .* meshes["b"] .* (meshes["b"] .< 0) .+
                           inc_transfer

    # Decompose income by source
    meshes["w"] = param["q"] * meshes["a"] + meshes["b"]
    labor_income_mesh_aux = WW_aux .* meshes_se1_aux
    ubincome_mesh_aux = WW_aux .* meshes_se2_aux
    profit_income_mesh_aux = WW_aux .* meshes_se3_aux
    a_income_mesh_aux = SS_stats["r_a_aux"] .* meshes["a"]
    b_income_mesh_aux = bond_return_pos .* meshes["b"] .* (meshes["b"] .>= 0) .+
    bond_return_neg .* meshes["b"] .* (meshes["b"] .< 0)
    transfer_mesh_aux = inc_transfer

    ## 5. Apply productivity transition to distribution
    #H_tilde = SS_stats["H_tilde"]["data"]
    # mu_dist_tilde = H_tilde*mu_dist(:)
    mu_dist_tilde = H_tilde * mu_dist[:]
    mu_dist_tilde = reshape(mu_dist_tilde, (nb_val, na_val, nse_val))

    ## 6. Compute marginal distributions (like Cal_SS_stats does)
    mu_dist_b = dropdims(sum(sum(mu_dist, dims=2), dims=3), dims=(2,3))
    mu_dist_a = dropdims(sum(sum(mu_dist, dims=1), dims=3), dims=(1,3))
    mu_dist_se = dropdims(sum(sum(mu_dist, dims=1), dims=2), dims=(1,2))

    ## 7. Compute CDFs for wealth, income, liquid, and illiquid assets

    # Create mesh grids for b and a (like MATLAB's meshgrid)
    mesh_a = zeros(nb_val, na_val)
    mesh_b = zeros(nb_val, na_val)
    for j in 1:na_val
        for i in 1:nb_val
            mesh_a[i, j] = grid["a"][j]
            mesh_b[i, j] = grid["b"][i]
        end
    end

    # Reduce distribution by summing over se dimension
    mu_redux = dropdims(sum(mu_dist, dims=3), dims=3)

    # Total wealth distribution
    total_wealth = mesh_a[:] * param["q"] .+ mesh_b[:]
    IX = sortperm(total_wealth)
    total_wealth = total_wealth[IX]
    total_wealth_pdf = mu_redux[IX]
    total_wealth_cdf = cumsum(total_wealth_pdf)

    # Income distribution
    income_unsorted = WW_aux .* meshes_se_aux .+
    SS_stats["r_a_aux"] .* meshes["a"] .+
    bond_return_pos .* meshes["b"] .* (meshes["b"] .>= 0) .+
    bond_return_neg .* meshes["b"] .* (meshes["b"] .< 0) .+
    inc_transfer
    income_sort_vec = income_unsorted[:]
    IX = sortperm(income_sort_vec)
    income_sort = reshape(income_sort_vec[IX], :, 1)  # Column vector to match MATLAB (n, 1)
    anal_stats["income_sort"] = income_sort
    income_pdf = mu_dist_tilde[IX]
    income_cdf = cumsum(income_pdf)
    income_pdf = reshape(income_pdf, :, 1)
    anal_stats["income_pdf"] = income_pdf
    anal_stats["income_cdf"] = income_cdf
    anal_stats["income_cdf"] = reshape(anal_stats["income_cdf"], :, 1)

    # Liquid asset distribution (bonds)
    liquid_sort = mesh_b[:]
    IX = sortperm(liquid_sort)
    liquid_sort = liquid_sort[IX]
    liquid_pdf = mu_redux[IX]
    liquid_cdf = cumsum(liquid_pdf)

    # Illiquid asset distribution (equity)
    illiquid_sort = mesh_a[:]
    IX = sortperm(illiquid_sort)
    illiquid_sort = illiquid_sort[IX]
    illiquid_pdf = mu_redux[IX]
    illiquid_cdf = cumsum(illiquid_pdf)

    ## 8. Extract percentiles for wealth

    w0 = total_wealth[1]
    w1 = total_wealth[sum(total_wealth_cdf .< 0.01)]
    w10 = total_wealth[sum(total_wealth_cdf .< 0.1)]
    w20 = total_wealth[sum(total_wealth_cdf .< 0.2)]
    w25 = total_wealth[sum(total_wealth_cdf .< 0.25)]
    w30 = total_wealth[sum(total_wealth_cdf .< 0.3)]
    w40 = total_wealth[sum(total_wealth_cdf .< 0.4)]
    w50 = total_wealth[sum(total_wealth_cdf .< 0.5)]
    w60 = total_wealth[sum(total_wealth_cdf .< 0.6)]
    w70 = total_wealth[sum(total_wealth_cdf .< 0.7)]
    w80 = total_wealth[sum(total_wealth_cdf .< 0.8)]
    w90 = total_wealth[sum(total_wealth_cdf .< 0.9)]
    w99 = total_wealth[sum(total_wealth_cdf .< 0.99)]
    w999 = total_wealth[sum(total_wealth_cdf .< 0.999)]
    w100 = total_wealth[end]

    ## 9. Extract percentiles for income

    I0_ss = income_sort[1]
    idx_001 = sum(income_cdf .< 0.001)
    println("DEBUG I01_ss: index = $idx_001")
    println("DEBUG I01_ss: income_cdf[max(1,idx_001-2):min(end,idx_001+2)] = ", income_cdf[max(1,idx_001-2):min(end,idx_001+2)])
    println("DEBUG I01_ss: income_sort[max(1,idx_001-2):min(end,idx_001+2)] = ", income_sort[max(1,idx_001-2):min(end,idx_001+2)])
    println("DEBUG I01_ss: income_sort[idx_001] = ", income_sort[idx_001])
    println("DEBUG I01_ss: Are there duplicate values near idx_001? ", length(unique(income_sort[max(1,idx_001-5):min(end,idx_001+5)])), " unique out of ", min(11, length(income_sort)-max(1,idx_001-5)+1))
    I01_ss = income_sort[idx_001]
    anal_stats["I01_ss"] = I01_ss
    I1_ss = income_sort[sum(income_cdf .< 0.01)]
    I10_ss = income_sort[sum(income_cdf .< 0.1)]
    I20_ss = income_sort[sum(income_cdf .< 0.2)]
    I25_ss = income_sort[sum(income_cdf .< 0.25)]
    I30_ss = income_sort[sum(income_cdf .< 0.3)]
    I40_ss = income_sort[sum(income_cdf .< 0.4)]
    I50_ss = income_sort[sum(income_cdf .< 0.5)]
    I60_ss = income_sort[sum(income_cdf .< 0.6)]
    I70_ss = income_sort[sum(income_cdf .< 0.7)]
    I80_ss = income_sort[sum(income_cdf .< 0.8)]
    I90_ss = income_sort[sum(income_cdf .< 0.9)]
    I99_ss = income_sort[sum(income_cdf .< 0.99)]
    I999_ss = income_sort[sum(income_cdf .< 0.999)]
    I100_ss = income_sort[end]

    ## 10. Extract percentiles for liquid assets

    LI0 = liquid_sort[1]
    LI1 = liquid_sort[sum(liquid_cdf .< 0.01)]
    LI10 = liquid_sort[sum(liquid_cdf .< 0.1)]
    LI20 = liquid_sort[sum(liquid_cdf .< 0.2)]
    LI25 = liquid_sort[sum(liquid_cdf .< 0.25)]
    LI30 = liquid_sort[sum(liquid_cdf .< 0.3)]
    LI40 = liquid_sort[sum(liquid_cdf .< 0.4)]
    LI50 = liquid_sort[sum(liquid_cdf .< 0.5)]
    LI60 = liquid_sort[sum(liquid_cdf .< 0.6)]
    LI70 = liquid_sort[sum(liquid_cdf .< 0.7)]
    LI80 = liquid_sort[sum(liquid_cdf .< 0.8)]
    LI90 = liquid_sort[sum(liquid_cdf .< 0.9)]
    LI99 = liquid_sort[sum(liquid_cdf .< 0.99)]
    LI999 = liquid_sort[sum(liquid_cdf .< 0.999)]
    LI100 = liquid_sort[end]

    ## 11. Extract percentiles for illiquid assets

    IL0 = illiquid_sort[1]
    IL1 = illiquid_sort[sum(illiquid_cdf .< 0.01)]
    IL10 = illiquid_sort[sum(illiquid_cdf .< 0.1)]
    IL20 = illiquid_sort[sum(illiquid_cdf .< 0.2)]
    IL25 = illiquid_sort[sum(illiquid_cdf .< 0.25)]
    IL30 = illiquid_sort[sum(illiquid_cdf .< 0.3)]
    IL40 = illiquid_sort[sum(illiquid_cdf .< 0.4)]
    IL50 = illiquid_sort[sum(illiquid_cdf .< 0.5)]
    IL60 = illiquid_sort[sum(illiquid_cdf .< 0.6)]
    IL70 = illiquid_sort[sum(illiquid_cdf .< 0.7)]
    IL80 = illiquid_sort[sum(illiquid_cdf .< 0.8)]
    IL90 = illiquid_sort[sum(illiquid_cdf .< 0.9)]
    IL99 = illiquid_sort[sum(illiquid_cdf .< 0.99)]
    IL999 = illiquid_sort[sum(illiquid_cdf .< 0.999)]
    IL100 = illiquid_sort[end]

    ## 12. Store wealth percentiles in SS_stats

    SS_stats["w0"] = w0
    SS_stats["w1"] = w1
    SS_stats["w10"] = w10
    SS_stats["w20"] = w20
    SS_stats["w25"] = w25
    SS_stats["w30"] = w30
    SS_stats["w40"] = w40
    SS_stats["w50"] = w50
    SS_stats["w60"] = w60
    SS_stats["w70"] = w70
    SS_stats["w80"] = w80
    SS_stats["w90"] = w90
    SS_stats["w99"] = w99
    SS_stats["w999"] = w999
    SS_stats["w100"] = w100

    ## 13. Create wealth mesh and distribution masks by wealth quintiles

    meshes_w = param["q"] .* meshes["a"] .+ meshes["b"]

    SS_stats["mu_dist"] = mu_dist
    SS_stats["mu_dist_B01"] = (meshes_w .>= w0) .* (meshes_w .<= w0) .* mu_dist
    SS_stats["mu_dist_B1"] = (meshes_w .>= w0) .* (meshes_w .<= w1) .* mu_dist
    SS_stats["mu_dist_B10"] = (meshes_w .>= w0) .* (meshes_w .<= w10) .* mu_dist
    SS_stats["mu_dist_Q1"] = (meshes_w .>= w0) .* (meshes_w .<= w20) .* mu_dist
    SS_stats["mu_dist_Q2"] = (meshes_w .> w20) .* (meshes_w .<= w40) .* mu_dist
    SS_stats["mu_dist_Q3"] = (meshes_w .> w40) .* (meshes_w .<= w60) .* mu_dist
    SS_stats["mu_dist_Q4"] = (meshes_w .> w60) .* (meshes_w .<= w80) .* mu_dist
    SS_stats["mu_dist_Q5"] = (meshes_w .> w80) .* (meshes_w .<= w100) .* mu_dist
    SS_stats["mu_dist_P90"] = (meshes_w .> w90) .* (meshes_w .<= w100) .* mu_dist
    SS_stats["mu_dist_P99"] = (meshes_w .> w99) .* (meshes_w .<= w100) .* mu_dist
    SS_stats["mu_dist_P999"] = (meshes_w .> w999) .* (meshes_w .<= w100) .* mu_dist

    SS_stats["mu_dist_Emp"] = zeros(nb_val, na_val, nse_val)
    SS_stats["mu_dist_Emp"][:, :, 1:ns_val] = mu_dist[:, :, 1:ns_val]

    SS_stats["mu_dist_Unemp"] = zeros(nb_val, na_val, nse_val)
    SS_stats["mu_dist_Unemp"][:, :, (ns_val+1):(2*ns_val)] = mu_dist[:, :, (ns_val+1):(2*ns_val)]

    SS_stats["mu_dist_Ent"] = (meshes_se3_aux .== 1) .* mu_dist
    SS_stats["mu_dist_B60"] = (meshes_w .<= w60) .* mu_dist
    SS_stats["mu_dist_B90"] = (meshes_w .<= w90) .* mu_dist

    ## 14. Create distribution masks by wealth quintiles (using mu_dist_tilde)

    SS_stats["mu_dist_tilde"] = mu_dist_tilde
    SS_stats["mu_dist_tilde_B01"] = (meshes_w .>= w0) .* (meshes_w .<= w0) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_B1"] = (meshes_w .>= w0) .* (meshes_w .<= w1) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_B10"] = (meshes_w .>= w0) .* (meshes_w .<= w10) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_Q1"] = (meshes_w .>= w0) .* (meshes_w .<= w20) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_Q2"] = (meshes_w .> w20) .* (meshes_w .<= w40) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_Q3"] = (meshes_w .> w40) .* (meshes_w .<= w60) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_Q4"] = (meshes_w .> w60) .* (meshes_w .<= w80) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_Q5"] = (meshes_w .> w80) .* (meshes_w .<= w100) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_P90"] = (meshes_w .> w90) .* (meshes_w .<= w100) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_P99"] = (meshes_w .> w99) .* (meshes_w .<= w100) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_P999"] = (meshes_w .> w999) .* (meshes_w .<= w100) .* mu_dist_tilde

    ## 15. Apply H_tilde transition to employment status distributions

    SS_stats["mu_dist_tilde_Ent"] = H_tilde * SS_stats["mu_dist_Ent"][:]
    SS_stats["mu_dist_tilde_Emp"] = H_tilde * SS_stats["mu_dist_Emp"][:]
    SS_stats["mu_dist_tilde_Unemp"] = H_tilde * SS_stats["mu_dist_Unemp"][:]

    SS_stats["mu_dist_tilde_Ent"] = reshape(SS_stats["mu_dist_tilde_Ent"], (nb_val, na_val, nse_val))
    SS_stats["mu_dist_tilde_Emp"] = reshape(SS_stats["mu_dist_tilde_Emp"], (nb_val, na_val, nse_val))
    SS_stats["mu_dist_tilde_Unemp"] = reshape(SS_stats["mu_dist_tilde_Unemp"], (nb_val, na_val, nse_val))

    SS_stats["mu_dist_tilde_B60"] = (meshes_w .<= w60) .* mu_dist_tilde

    ## 16. Create Q_dists structure with income-based quintiles

    Q_dists = Dict{String, Any}()

    Q_dists["mu_dist_I_B01"] = (total_income_mesh_aux .>= I0_ss) .* (total_income_mesh_aux .<= I01_ss) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_B1"] = (total_income_mesh_aux .>= I0_ss) .* (total_income_mesh_aux .<= I1_ss) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_B10"] = (total_income_mesh_aux .>= I0_ss) .* (total_income_mesh_aux .<= I10_ss) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_Q1"] = (total_income_mesh_aux .>= I0_ss) .* (total_income_mesh_aux .<= I20_ss) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_Q2"] = (total_income_mesh_aux .> I20_ss) .* (total_income_mesh_aux .<= I40_ss) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_Q3"] = (total_income_mesh_aux .> I40_ss) .* (total_income_mesh_aux .<= I60_ss) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_Q4"] = (total_income_mesh_aux .> I60_ss) .* (total_income_mesh_aux .<= I80_ss) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_Q5"] = (total_income_mesh_aux .> I80_ss) .* (total_income_mesh_aux .<= I100_ss) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_P90"] = (total_income_mesh_aux .> I90_ss) .* (total_income_mesh_aux .<= I100_ss) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_P99"] = (total_income_mesh_aux .> I99_ss) .* (total_income_mesh_aux .<= I100_ss) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_P999"] = (total_income_mesh_aux .> I999_ss) .* (total_income_mesh_aux .<= I100_ss) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_B60"] = (total_income_mesh_aux .<= I60_ss) .* SS_stats["mu_dist"]

    ## 17. Copy wealth-based quintile distributions to Q_dists

    Q_dists["mu_dist"] = mu_dist
    Q_dists["mu_dist_Q1"] = SS_stats["mu_dist_Q1"]
    Q_dists["mu_dist_Q2"] = SS_stats["mu_dist_Q2"]
    Q_dists["mu_dist_Q3"] = SS_stats["mu_dist_Q3"]
    Q_dists["mu_dist_Q4"] = SS_stats["mu_dist_Q4"]
    Q_dists["mu_dist_Q5"] = SS_stats["mu_dist_Q5"]
    Q_dists["mu_dist_P90"] = SS_stats["mu_dist_P90"]
    Q_dists["mu_dist_P99"] = SS_stats["mu_dist_P99"]
    Q_dists["mu_dist_P999"] = SS_stats["mu_dist_P999"]
    Q_dists["mu_dist_Ent"] = SS_stats["mu_dist_Ent"]
    Q_dists["mu_dist_Emp"] = SS_stats["mu_dist_Emp"]
    Q_dists["mu_dist_Unemp"] = SS_stats["mu_dist_Unemp"]
    Q_dists["mu_dist_B60"] = SS_stats["mu_dist_B60"]

    # Store auxiliary versions
    Q_dists["mu_dist_aux"] = SS_stats["mu_dist"]
    Q_dists["mu_dist_aux_Q1"] = SS_stats["mu_dist_Q1"]
    Q_dists["mu_dist_aux_Q2"] = SS_stats["mu_dist_Q2"]
    Q_dists["mu_dist_aux_Q3"] = SS_stats["mu_dist_Q3"]
    Q_dists["mu_dist_aux_Q4"] = SS_stats["mu_dist_Q4"]
    Q_dists["mu_dist_aux_Q5"] = SS_stats["mu_dist_Q5"]
    Q_dists["mu_dist_aux_P90"] = SS_stats["mu_dist_P90"]
    Q_dists["mu_dist_aux_P99"] = SS_stats["mu_dist_P99"]
    Q_dists["mu_dist_aux_P999"] = SS_stats["mu_dist_P999"]
    Q_dists["mu_dist_aux_Ent"] = SS_stats["mu_dist_Ent"]
    Q_dists["mu_dist_aux_Emp"] = SS_stats["mu_dist_Emp"]
    Q_dists["mu_dist_aux_Unemp"] = SS_stats["mu_dist_Unemp"]
    Q_dists["mu_dist_aux_B60"] = SS_stats["mu_dist_B60"]

    ## 18. Store SS_stats auxiliary versions

    SS_stats["mu_dist_aux"] = SS_stats["mu_dist"]
    SS_stats["mu_dist_aux_Q1"] = SS_stats["mu_dist_Q1"]
    SS_stats["mu_dist_aux_Q2"] = SS_stats["mu_dist_Q2"]
    SS_stats["mu_dist_aux_Q3"] = SS_stats["mu_dist_Q3"]
    SS_stats["mu_dist_aux_Q4"] = SS_stats["mu_dist_Q4"]
    SS_stats["mu_dist_aux_Q5"] = SS_stats["mu_dist_Q5"]
    SS_stats["mu_dist_aux_P90"] = SS_stats["mu_dist_P90"]
    SS_stats["mu_dist_aux_P99"] = SS_stats["mu_dist_P99"]
    SS_stats["mu_dist_aux_P999"] = SS_stats["mu_dist_P999"]
    SS_stats["mu_dist_aux_Ent"] = SS_stats["mu_dist_Ent"]
    SS_stats["mu_dist_aux_Emp"] = SS_stats["mu_dist_Emp"]
    SS_stats["mu_dist_aux_Unemp"] = SS_stats["mu_dist_Unemp"]
    SS_stats["mu_dist_aux_B60"] = SS_stats["mu_dist_B60"]

    ## 19. Store bottom percentile distributions in Q_dists

    Q_dists["mu_dist_B01"] = SS_stats["mu_dist_B01"]
    Q_dists["mu_dist_B1"] = SS_stats["mu_dist_B1"]
    Q_dists["mu_dist_B10"] = SS_stats["mu_dist_B10"]

    Q_dists["mu_dist_tilde_B01"] = SS_stats["mu_dist_tilde_B01"]
    Q_dists["mu_dist_tilde_B1"] = SS_stats["mu_dist_tilde_B1"]
    Q_dists["mu_dist_tilde_B10"] = SS_stats["mu_dist_tilde_B10"]

    ## 20. Compute marginal distributions for bonds (b) by quintiles

    Q_dists["mu_dist_b"] = mu_dist_b
    Q_dists["mu_dist_b_Q1"] = dropdims(sum(sum(SS_stats["mu_dist_Q1"], dims=2), dims=3), dims=(2,3))
    Q_dists["mu_dist_b_Q2"] = dropdims(sum(sum(SS_stats["mu_dist_Q2"], dims=2), dims=3), dims=(2,3))
    Q_dists["mu_dist_b_Q3"] = dropdims(sum(sum(SS_stats["mu_dist_Q3"], dims=2), dims=3), dims=(2,3))
    Q_dists["mu_dist_b_Q4"] = dropdims(sum(sum(SS_stats["mu_dist_Q4"], dims=2), dims=3), dims=(2,3))
    Q_dists["mu_dist_b_Q5"] = dropdims(sum(sum(SS_stats["mu_dist_Q5"], dims=2), dims=3), dims=(2,3))
    Q_dists["mu_dist_b_P90"] = dropdims(sum(sum(SS_stats["mu_dist_P90"], dims=2), dims=3), dims=(2,3))
    Q_dists["mu_dist_b_P99"] = dropdims(sum(sum(SS_stats["mu_dist_P99"], dims=2), dims=3), dims=(2,3))
    Q_dists["mu_dist_b_P999"] = dropdims(sum(sum(SS_stats["mu_dist_P999"], dims=2), dims=3), dims=(2,3))
    Q_dists["mu_dist_b_Ent"] = dropdims(sum(sum(SS_stats["mu_dist_Ent"], dims=2), dims=3), dims=(2,3))
    Q_dists["mu_dist_b_Emp"] = dropdims(sum(sum(SS_stats["mu_dist_Emp"], dims=2), dims=3), dims=(2,3))
    Q_dists["mu_dist_b_Unemp"] = dropdims(sum(sum(SS_stats["mu_dist_Unemp"], dims=2), dims=3), dims=(2,3))
    Q_dists["mu_dist_b_B60"] = dropdims(sum(sum(SS_stats["mu_dist_B60"], dims=2), dims=3), dims=(2,3))

    ## 21. Compute marginal distributions for equity (a) by quintiles

    Q_dists["mu_dist_a"] = mu_dist_a
    Q_dists["mu_dist_a_Q1"] = vec(dropdims(sum(sum(SS_stats["mu_dist_Q1"], dims=1), dims=3), dims=(1,3)))
    Q_dists["mu_dist_a_Q2"] = vec(dropdims(sum(sum(SS_stats["mu_dist_Q2"], dims=1), dims=3), dims=(1,3)))
    Q_dists["mu_dist_a_Q3"] = vec(dropdims(sum(sum(SS_stats["mu_dist_Q3"], dims=1), dims=3), dims=(1,3)))
    Q_dists["mu_dist_a_Q4"] = vec(dropdims(sum(sum(SS_stats["mu_dist_Q4"], dims=1), dims=3), dims=(1,3)))
    Q_dists["mu_dist_a_Q5"] = vec(dropdims(sum(sum(SS_stats["mu_dist_Q5"], dims=1), dims=3), dims=(1,3)))
    Q_dists["mu_dist_a_P90"] = vec(dropdims(sum(sum(SS_stats["mu_dist_P90"], dims=1), dims=3), dims=(1,3)))
    Q_dists["mu_dist_a_P99"] = vec(dropdims(sum(sum(SS_stats["mu_dist_P99"], dims=1), dims=3), dims=(1,3)))
    Q_dists["mu_dist_a_P999"] = vec(dropdims(sum(sum(SS_stats["mu_dist_P999"], dims=1), dims=3), dims=(1,3)))
    Q_dists["mu_dist_a_Ent"] = vec(dropdims(sum(sum(SS_stats["mu_dist_Ent"], dims=1), dims=3), dims=(1,3)))
    Q_dists["mu_dist_a_Emp"] = vec(dropdims(sum(sum(SS_stats["mu_dist_Emp"], dims=1), dims=3), dims=(1,3)))
    Q_dists["mu_dist_a_Unemp"] = vec(dropdims(sum(sum(SS_stats["mu_dist_Unemp"], dims=1), dims=3), dims=(1,3)))
    Q_dists["mu_dist_a_B60"] = vec(dropdims(sum(sum(SS_stats["mu_dist_B60"], dims=1), dims=3), dims=(1,3)))

    ## 22. Compute marginal distributions for productivity (se) by quintiles

    Q_dists["mu_dist_se"] = mu_dist_se
    Q_dists["mu_dist_se_Q1"] = dropdims(sum(sum(SS_stats["mu_dist_Q1"], dims=1), dims=2), dims=(1,2))
    Q_dists["mu_dist_se_Q2"] = dropdims(sum(sum(SS_stats["mu_dist_Q2"], dims=1), dims=2), dims=(1,2))
    Q_dists["mu_dist_se_Q3"] = dropdims(sum(sum(SS_stats["mu_dist_Q3"], dims=1), dims=2), dims=(1,2))
    Q_dists["mu_dist_se_Q4"] = dropdims(sum(sum(SS_stats["mu_dist_Q4"], dims=1), dims=2), dims=(1,2))
    Q_dists["mu_dist_se_Q5"] = dropdims(sum(sum(SS_stats["mu_dist_Q5"], dims=1), dims=2), dims=(1,2))
    Q_dists["mu_dist_se_P90"] = dropdims(sum(sum(SS_stats["mu_dist_P90"], dims=1), dims=2), dims=(1,2))
    Q_dists["mu_dist_se_P99"] = dropdims(sum(sum(SS_stats["mu_dist_P99"], dims=1), dims=2), dims=(1,2))
    Q_dists["mu_dist_se_P999"] = dropdims(sum(sum(SS_stats["mu_dist_P999"], dims=1), dims=2), dims=(1,2))
    Q_dists["mu_dist_se_Ent"] = dropdims(sum(sum(SS_stats["mu_dist_Ent"], dims=1), dims=2), dims=(1,2))
    Q_dists["mu_dist_se_Emp"] = dropdims(sum(sum(SS_stats["mu_dist_Emp"], dims=1), dims=2), dims=(1,2))
    Q_dists["mu_dist_se_Unemp"] = dropdims(sum(sum(SS_stats["mu_dist_Unemp"], dims=1), dims=2), dims=(1,2))
    Q_dists["mu_dist_se_B60"] = dropdims(sum(sum(SS_stats["mu_dist_B60"], dims=1), dims=2), dims=(1,2))

    ## 23. Store mu_dist_tilde versions in Q_dists

    Q_dists["mu_dist_tilde"] = mu_dist_tilde
    Q_dists["mu_dist_tilde_aux"] = mu_dist_tilde
    Q_dists["mu_dist_tilde_full"] = mu_dist_tilde
    Q_dists["mu_dist_tilde_Q1"] = SS_stats["mu_dist_tilde_Q1"]
    Q_dists["mu_dist_tilde_Q2"] = SS_stats["mu_dist_tilde_Q2"]
    Q_dists["mu_dist_tilde_Q3"] = SS_stats["mu_dist_tilde_Q3"]
    Q_dists["mu_dist_tilde_Q4"] = SS_stats["mu_dist_tilde_Q4"]
    Q_dists["mu_dist_tilde_Q5"] = SS_stats["mu_dist_tilde_Q5"]
    Q_dists["mu_dist_tilde_P90"] = SS_stats["mu_dist_tilde_P90"]
    Q_dists["mu_dist_tilde_P99"] = SS_stats["mu_dist_tilde_P99"]
    Q_dists["mu_dist_tilde_P999"] = SS_stats["mu_dist_tilde_P999"]
    Q_dists["mu_dist_tilde_Ent"] = SS_stats["mu_dist_tilde_Ent"]
    Q_dists["mu_dist_tilde_Emp"] = SS_stats["mu_dist_tilde_Emp"]
    Q_dists["mu_dist_tilde_Unemp"] = SS_stats["mu_dist_tilde_Unemp"]
    Q_dists["mu_dist_tilde_B60"] = SS_stats["mu_dist_tilde_B60"]

    ## 24. Extract quintile distributions from SS_stats for local use

    mu_dist_B01 = SS_stats["mu_dist_B01"]
    mu_dist_B1 = SS_stats["mu_dist_B1"]
    mu_dist_B10 = SS_stats["mu_dist_B10"]
    mu_dist_Q1 = SS_stats["mu_dist_Q1"]
    mu_dist_Q2 = SS_stats["mu_dist_Q2"]
    mu_dist_Q3 = SS_stats["mu_dist_Q3"]
    mu_dist_Q4 = SS_stats["mu_dist_Q4"]
    mu_dist_Q5 = SS_stats["mu_dist_Q5"]
    mu_dist_P90 = SS_stats["mu_dist_P90"]
    mu_dist_P99 = SS_stats["mu_dist_P99"]
    mu_dist_P999 = SS_stats["mu_dist_P999"]
    mu_dist_Ent = SS_stats["mu_dist_Ent"]
    mu_dist_B60 = SS_stats["mu_dist_B60"]

    mu_dist_tilde_B01 = SS_stats["mu_dist_tilde_B01"]
    mu_dist_tilde_B1 = SS_stats["mu_dist_tilde_B1"]
    mu_dist_tilde_B10 = SS_stats["mu_dist_tilde_B10"]
    mu_dist_tilde_Q1 = SS_stats["mu_dist_tilde_Q1"]
    mu_dist_tilde_Q2 = SS_stats["mu_dist_tilde_Q2"]
    mu_dist_tilde_Q3 = SS_stats["mu_dist_tilde_Q3"]
    mu_dist_tilde_Q4 = SS_stats["mu_dist_tilde_Q4"]
    mu_dist_tilde_Q5 = SS_stats["mu_dist_tilde_Q5"]
    mu_dist_tilde_P90 = SS_stats["mu_dist_tilde_P90"]
    mu_dist_tilde_P99 = SS_stats["mu_dist_tilde_P99"]
    mu_dist_tilde_P999 = SS_stats["mu_dist_tilde_P999"]
    mu_dist_tilde_Ent = SS_stats["mu_dist_tilde_Ent"]
    mu_dist_tilde_B60 = SS_stats["mu_dist_tilde_B60"]

    ## 25. Compute average wealth by quintile

    w_all = sum(meshes_w .* mu_dist_tilde) / sum(mu_dist_tilde)
    w_B60 = sum(meshes_w .* mu_dist_tilde_B60) / sum(mu_dist_tilde_B60)
    w_Q1 = sum(meshes_w .* mu_dist_tilde_Q1) / sum(mu_dist_tilde_Q1)
    w_Q2 = sum(meshes_w .* mu_dist_tilde_Q2) / sum(mu_dist_tilde_Q2)
    w_Q3 = sum(meshes_w .* mu_dist_tilde_Q3) / sum(mu_dist_tilde_Q3)
    w_Q4 = sum(meshes_w .* mu_dist_tilde_Q4) / sum(mu_dist_tilde_Q4)
    w_Q5 = sum(meshes_w .* mu_dist_tilde_Q5) / sum(mu_dist_tilde_Q5)
    w_P90 = sum(meshes_w .* mu_dist_tilde_P90) / sum(mu_dist_tilde_P90)
    w_P99 = sum(meshes_w .* mu_dist_tilde_P99) / sum(mu_dist_tilde_P99)
    w_P999 = sum(meshes_w .* mu_dist_tilde_P999) / sum(mu_dist_tilde_P999)
    w_Ent = sum(meshes_w .* mu_dist_tilde_Ent) / sum(mu_dist_tilde_Ent)

    ## 26. Compute average income by quintile

    income_all = sum(total_income_mesh_aux .* mu_dist_tilde) / sum(mu_dist_tilde)
    income_B01 = sum(total_income_mesh_aux .* mu_dist_tilde_B01) / sum(mu_dist_tilde_B01)
    income_B1 = sum(total_income_mesh_aux .* mu_dist_tilde_B1) / sum(mu_dist_tilde_B1)
    income_B10 = sum(total_income_mesh_aux .* mu_dist_tilde_B10) / sum(mu_dist_tilde_B10)
    income_B60 = sum(total_income_mesh_aux .* mu_dist_tilde_B60) / sum(mu_dist_tilde_B60)
    income_Q1 = sum(total_income_mesh_aux .* mu_dist_tilde_Q1) / sum(mu_dist_tilde_Q1)
    income_Q2 = sum(total_income_mesh_aux .* mu_dist_tilde_Q2) / sum(mu_dist_tilde_Q2)
    income_Q3 = sum(total_income_mesh_aux .* mu_dist_tilde_Q3) / sum(mu_dist_tilde_Q3)
    income_Q4 = sum(total_income_mesh_aux .* mu_dist_tilde_Q4) / sum(mu_dist_tilde_Q4)
    income_Q5 = sum(total_income_mesh_aux .* mu_dist_tilde_Q5) / sum(mu_dist_tilde_Q5)
    income_P90 = sum(total_income_mesh_aux .* mu_dist_tilde_P90) / sum(mu_dist_tilde_P90)
    income_P99 = sum(total_income_mesh_aux .* mu_dist_tilde_P99) / sum(mu_dist_tilde_P99)
    income_P999 = sum(total_income_mesh_aux .* mu_dist_tilde_P999) / sum(mu_dist_tilde_P999)
    income_Ent = sum(total_income_mesh_aux .* mu_dist_tilde_Ent) / sum(mu_dist_tilde_Ent)

    ## 27. Compute average labor income by quintile (only for employed workers)

    labor_all = sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde[:, :, 1:ns_val]) /
    sum(mu_dist_tilde[:, :, 1:ns_val])
    labor_B60 = sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_B60[:, :, 1:ns_val]) /
    sum(mu_dist_tilde_B60[:, :, 1:ns_val])
    labor_Q1 = sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_Q1[:, :, 1:ns_val]) /
    sum(mu_dist_tilde_Q1[:, :, 1:ns_val])
    labor_Q2 = sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_Q2[:, :, 1:ns_val]) /
    sum(mu_dist_tilde_Q2[:, :, 1:ns_val])
    labor_Q3 = sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_Q3[:, :, 1:ns_val]) /
    sum(mu_dist_tilde_Q3[:, :, 1:ns_val])
    labor_Q4 = sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_Q4[:, :, 1:ns_val]) /
    sum(mu_dist_tilde_Q4[:, :, 1:ns_val])
    labor_Q5 = sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_Q5[:, :, 1:ns_val]) /
    sum(mu_dist_tilde_Q5[:, :, 1:ns_val])
    labor_P90 = sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_P90[:, :, 1:ns_val]) /
    sum(mu_dist_tilde_P90[:, :, 1:ns_val])
    labor_P99 = sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_P99[:, :, 1:ns_val]) /
    sum(mu_dist_tilde_P99[:, :, 1:ns_val])
    labor_P999 = sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_P999[:, :, 1:ns_val]) /
    sum(mu_dist_tilde_P999[:, :, 1:ns_val])
    #labor_Ent = sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_Ent[:, :, 1:ns_val]) /
    #sum(mu_dist_tilde_Ent[:, :, 1:ns_val])

    labor_Ent =
    sum(WW_aux[:, :, 1:grid["ns"]] .* meshes["se_aux"][:, :, 1:grid["ns"]] .* mu_dist_tilde_Ent[:, :, 1:grid["ns"]]) /
    sum(mu_dist_tilde_Ent[:, :, 1:grid["ns"]])

    ## 28. Compute average unemployment benefit income by quintile (unemployed workers only)

    ubincome_range = (ns_val+1):(2*ns_val)
    ubincome_all = sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde[:, :, ubincome_range]) /
    sum(mu_dist_tilde[:, :, ubincome_range])
    ubincome_B60 = sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_B60[:, :, ubincome_range]) /
    sum(mu_dist_tilde_B60[:, :, ubincome_range])
    ubincome_Q1 = sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_Q1[:, :, ubincome_range]) /
    sum(mu_dist_tilde_Q1[:, :, ubincome_range])
    ubincome_Q2 = sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_Q2[:, :, ubincome_range]) /
    sum(mu_dist_tilde_Q2[:, :, ubincome_range])
    ubincome_Q3 = sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_Q3[:, :, ubincome_range]) /
    sum(mu_dist_tilde_Q3[:, :, ubincome_range])
    ubincome_Q4 = sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_Q4[:, :, ubincome_range]) /
    sum(mu_dist_tilde_Q4[:, :, ubincome_range])
    ubincome_Q5 = sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_Q5[:, :, ubincome_range]) /
    sum(mu_dist_tilde_Q5[:, :, ubincome_range])
    ubincome_P90 = sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_P90[:, :, ubincome_range]) /
    sum(mu_dist_tilde_P90[:, :, ubincome_range])
    ubincome_P99 = sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_P99[:, :, ubincome_range]) /
    sum(mu_dist_tilde_P99[:, :, ubincome_range])
    ubincome_P999 = sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_P999[:, :, ubincome_range]) /
    sum(mu_dist_tilde_P999[:, :, ubincome_range])
    ubincome_Ent = sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_Ent[:, :, ubincome_range]) /
    sum(mu_dist_tilde_Ent[:, :, ubincome_range])

    ## 29. Compute average profit income by quintile (entrepreneurs only)

    profit_income_all = sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde[:, :, end]) /
    sum(mu_dist_tilde[:, :, end])
    profit_income_B60 = sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_B60[:, :, end]) /
    sum(mu_dist_tilde_B60[:, :, end])
    profit_income_Q1 = sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_Q1[:, :, end]) /
    sum(mu_dist_tilde_Q1[:, :, end])
    profit_income_Q2 = sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_Q2[:, :, end]) /
    sum(mu_dist_tilde_Q2[:, :, end])
    profit_income_Q3 = sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_Q3[:, :, end]) /
    sum(mu_dist_tilde_Q3[:, :, end])
    profit_income_Q4 = sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_Q4[:, :, end]) /
    sum(mu_dist_tilde_Q4[:, :, end])
    profit_income_Q5 = sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_Q5[:, :, end]) /
    sum(mu_dist_tilde_Q5[:, :, end])
    profit_income_P90 = sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_P90[:, :, end]) /
    sum(mu_dist_tilde_P90[:, :, end])
    profit_income_P99 = sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_P99[:, :, end]) /
    sum(mu_dist_tilde_P99[:, :, end])
    profit_income_P999 = sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_P999[:, :, end]) /
    sum(mu_dist_tilde_P999[:, :, end])
    profit_income_Ent = sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_Ent[:, :, end]) /
    sum(mu_dist_tilde_Ent[:, :, end])

    ## 30. Compute average equity income by quintile

    a_income_all = sum(sum(sum(SS_stats["r_a_aux"] .* mu_dist_tilde))) / sum(sum(sum(mu_dist_tilde)))
    a_income_B60 = sum(sum(sum(SS_stats["r_a_aux"] .* mu_dist_tilde_B60))) / sum(sum(sum(mu_dist_tilde_B60)))
    a_income_Q1 = sum(sum(sum(SS_stats["r_a_aux"] .* mu_dist_tilde_Q1))) / sum(sum(sum(mu_dist_tilde_Q1)))
    a_income_Q2 = sum(sum(sum(SS_stats["r_a_aux"] .* mu_dist_tilde_Q2))) / sum(sum(sum(mu_dist_tilde_Q2)))
    a_income_Q3 = sum(sum(sum(SS_stats["r_a_aux"] .* mu_dist_tilde_Q3))) / sum(sum(sum(mu_dist_tilde_Q3)))
    a_income_Q4 = sum(sum(sum(SS_stats["r_a_aux"] .* mu_dist_tilde_Q4))) / sum(sum(sum(mu_dist_tilde_Q4)))
    a_income_Q5 = sum(sum(sum(SS_stats["r_a_aux"] .* mu_dist_tilde_Q5))) / sum(sum(sum(mu_dist_tilde_Q5)))
    a_income_P90 = sum(sum(sum(SS_stats["r_a_aux"] .* mu_dist_tilde_P90))) / sum(sum(sum(mu_dist_tilde_P90)))
    a_income_P99 = sum(sum(sum(SS_stats["r_a_aux"] .* mu_dist_tilde_P99))) / sum(sum(sum(mu_dist_tilde_P99)))
    a_income_P999 = sum(sum(sum(SS_stats["r_a_aux"] .* mu_dist_tilde_P999))) / sum(sum(sum(mu_dist_tilde_P999)))
    a_income_Ent = sum(sum(sum(SS_stats["r_a_aux"] .* mu_dist_tilde_Ent))) / sum(sum(sum(mu_dist_tilde_Ent)))

    ## 31. Compute average bond income by quintile

    b_income_all = sum(sum(sum(b_income_mesh_aux .* mu_dist_tilde))) / sum(sum(sum(mu_dist_tilde)))
    b_income_B60 = sum(sum(sum(b_income_mesh_aux .* mu_dist_tilde_B60))) / sum(sum(sum(mu_dist_tilde_B60)))
    b_income_Q1 = sum(sum(sum(b_income_mesh_aux .* mu_dist_tilde_Q1))) / sum(sum(sum(mu_dist_tilde_Q1)))
    b_income_Q2 = sum(sum(sum(b_income_mesh_aux .* mu_dist_tilde_Q2))) / sum(sum(sum(mu_dist_tilde_Q2)))
    b_income_Q3 = sum(sum(sum(b_income_mesh_aux .* mu_dist_tilde_Q3))) / sum(sum(sum(mu_dist_tilde_Q3)))
    b_income_Q4 = sum(sum(sum(b_income_mesh_aux .* mu_dist_tilde_Q4))) / sum(sum(sum(mu_dist_tilde_Q4)))
    b_income_Q5 = sum(sum(sum(b_income_mesh_aux .* mu_dist_tilde_Q5))) / sum(sum(sum(mu_dist_tilde_Q5)))
    b_income_P90 = sum(sum(sum(b_income_mesh_aux .* mu_dist_tilde_P90))) / sum(sum(sum(mu_dist_tilde_P90)))
    b_income_P99 = sum(sum(sum(b_income_mesh_aux .* mu_dist_tilde_P99))) / sum(sum(sum(mu_dist_tilde_P99)))
    b_income_P999 = sum(sum(sum(b_income_mesh_aux .* mu_dist_tilde_P999))) / sum(sum(sum(mu_dist_tilde_P999)))
    b_income_Ent = sum(sum(sum(b_income_mesh_aux .* mu_dist_tilde_Ent))) / sum(sum(sum(mu_dist_tilde_Ent)))

    ## 32. Compute average transfer income by quintile

    transfer_all = sum(transfer_mesh_aux .* mu_dist_tilde) / sum(mu_dist_tilde)
    transfer_B60 = sum(transfer_mesh_aux .* mu_dist_tilde_B60) / sum(mu_dist_tilde_B60)
    transfer_Q1 = sum(transfer_mesh_aux .* mu_dist_tilde_Q1) / sum(mu_dist_tilde_Q1)
    transfer_Q2 = sum(transfer_mesh_aux .* mu_dist_tilde_Q2) / sum(mu_dist_tilde_Q2)
    transfer_Q3 = sum(transfer_mesh_aux .* mu_dist_tilde_Q3) / sum(mu_dist_tilde_Q3)
    transfer_Q4 = sum(transfer_mesh_aux .* mu_dist_tilde_Q4) / sum(mu_dist_tilde_Q4)
    transfer_Q5 = sum(transfer_mesh_aux .* mu_dist_tilde_Q5) / sum(mu_dist_tilde_Q5)
    transfer_P90 = sum(transfer_mesh_aux .* mu_dist_tilde_P90) / sum(mu_dist_tilde_P90)
    transfer_P99 = sum(transfer_mesh_aux .* mu_dist_tilde_P99) / sum(mu_dist_tilde_P99)
    transfer_P999 = sum(transfer_mesh_aux .* mu_dist_tilde_P999) / sum(mu_dist_tilde_P999)
    transfer_Ent = sum(transfer_mesh_aux .* mu_dist_tilde_Ent) / sum(mu_dist_tilde_Ent)

    ## 33. Compute average liquid assets (bonds) by quintile

    b_all = sum(meshes["b"] .* mu_dist_tilde) / sum(mu_dist_tilde)
    b_B60 = sum(meshes["b"] .* mu_dist_tilde_B60) / sum(mu_dist_tilde_B60)
    b_Q1 = sum(meshes["b"] .* mu_dist_tilde_Q1) / sum(mu_dist_tilde_Q1)
    b_Q2 = sum(meshes["b"] .* mu_dist_tilde_Q2) / sum(mu_dist_tilde_Q2)
    b_Q3 = sum(meshes["b"] .* mu_dist_tilde_Q3) / sum(mu_dist_tilde_Q3)
    b_Q4 = sum(meshes["b"] .* mu_dist_tilde_Q4) / sum(mu_dist_tilde_Q4)
    b_Q5 = sum(meshes["b"] .* mu_dist_tilde_Q5) / sum(mu_dist_tilde_Q5)
    b_P90 = sum(meshes["b"] .* mu_dist_tilde_P90) / sum(mu_dist_tilde_P90)
    b_P99 = sum(meshes["b"] .* mu_dist_tilde_P99) / sum(mu_dist_tilde_P99)
    b_P999 = sum(meshes["b"] .* mu_dist_tilde_P999) / sum(mu_dist_tilde_P999)
    b_Ent = sum(meshes["b"] .* mu_dist_tilde_Ent) / sum(mu_dist_tilde_Ent)

    ## 34. Compute average illiquid assets (equity) by quintile

    a_all = sum(param["q"] .* meshes["a"] .* mu_dist_tilde) / sum(mu_dist_tilde)
    a_B60 = sum(param["q"] .* meshes["a"] .* mu_dist_tilde_B60) / sum(mu_dist_tilde_B60)
    a_Q1 = sum(param["q"] .* meshes["a"] .* mu_dist_tilde_Q1) / sum(mu_dist_tilde_Q1)
    a_Q2 = sum(param["q"] .* meshes["a"] .* mu_dist_tilde_Q2) / sum(mu_dist_tilde_Q2)
    a_Q3 = sum(param["q"] .* meshes["a"] .* mu_dist_tilde_Q3) / sum(mu_dist_tilde_Q3)
    a_Q4 = sum(param["q"] .* meshes["a"] .* mu_dist_tilde_Q4) / sum(mu_dist_tilde_Q4)
    a_Q5 = sum(param["q"] .* meshes["a"] .* mu_dist_tilde_Q5) / sum(mu_dist_tilde_Q5)
    a_P90 = sum(param["q"] .* meshes["a"] .* mu_dist_tilde_P90) / sum(mu_dist_tilde_P90)
    a_P99 = sum(param["q"] .* meshes["a"] .* mu_dist_tilde_P99) / sum(mu_dist_tilde_P99)
    a_P999 = sum(param["q"] .* meshes["a"] .* mu_dist_tilde_P999) / sum(mu_dist_tilde_P999)
    a_Ent = sum(param["q"] .* meshes["a"] .* mu_dist_tilde_Ent) / sum(mu_dist_tilde_Ent)

    ## 35. Compute average consumption by quintile

    c_all = sum((AProb .* c_a_guess .+ (1 .- AProb) .* c_n_guess) .* mu_dist_tilde) / sum(mu_dist_tilde)
    c_B60 = sum((AProb .* c_a_guess .+ (1 .- AProb) .* c_n_guess) .* mu_dist_tilde_B60) / sum(mu_dist_tilde_B60)
    c_Q1 = sum((AProb .* c_a_guess .+ (1 .- AProb) .* c_n_guess) .* mu_dist_tilde_Q1) / sum(mu_dist_tilde_Q1)
    c_Q2 = sum((AProb .* c_a_guess .+ (1 .- AProb) .* c_n_guess) .* mu_dist_tilde_Q2) / sum(mu_dist_tilde_Q2)
    c_Q3 = sum((AProb .* c_a_guess .+ (1 .- AProb) .* c_n_guess) .* mu_dist_tilde_Q3) / sum(mu_dist_tilde_Q3)
    c_Q4 = sum((AProb .* c_a_guess .+ (1 .- AProb) .* c_n_guess) .* mu_dist_tilde_Q4) / sum(mu_dist_tilde_Q4)
    c_Q5 = sum((AProb .* c_a_guess .+ (1 .- AProb) .* c_n_guess) .* mu_dist_tilde_Q5) / sum(mu_dist_tilde_Q5)
    c_P90 = sum((AProb .* c_a_guess .+ (1 .- AProb) .* c_n_guess) .* mu_dist_tilde_P90) / sum(mu_dist_tilde_P90)
    c_P99 = sum((AProb .* c_a_guess .+ (1 .- AProb) .* c_n_guess) .* mu_dist_tilde_P99) / sum(mu_dist_tilde_P99)
    c_P999 = sum((AProb .* c_a_guess .+ (1 .- AProb) .* c_n_guess) .* mu_dist_tilde_P999) / sum(mu_dist_tilde_P999)
    c_Ent = sum((AProb .* c_a_guess .+ (1 .- AProb) .* c_n_guess) .* mu_dist_tilde_Ent) / sum(mu_dist_tilde_Ent)

    ## 36. Compute "other income" (UB + transfers) by quintile

    other_income_all = ubincome_all + transfer_all
    other_income_B60 = ubincome_B60 + transfer_B60
    other_income_Q1 = ubincome_Q1 + transfer_Q1
    other_income_Q2 = ubincome_Q2 + transfer_Q2
    other_income_Q3 = ubincome_Q3 + transfer_Q3
    other_income_Q4 = ubincome_Q4 + transfer_Q4
    other_income_Q5 = ubincome_Q5 + transfer_Q5
    other_income_P90 = ubincome_P90 + transfer_P90
    other_income_P99 = ubincome_P99 + transfer_P99
    other_income_P999 = ubincome_P999 + transfer_P999

    ## 37. Compute total asset income (equity + bonds) by quintile

    asset_income_all = a_income_all + b_income_all
    asset_income_B60 = a_income_B60 + b_income_B60
    asset_income_Q1 = a_income_Q1 + b_income_Q1
    asset_income_Q2 = a_income_Q2 + b_income_Q2
    asset_income_Q3 = a_income_Q3 + b_income_Q3
    asset_income_Q4 = a_income_Q4 + b_income_Q4
    asset_income_Q5 = a_income_Q5 + b_income_Q5
    asset_income_P90 = a_income_P90 + b_income_P90
    asset_income_P99 = a_income_P99 + b_income_P99
    asset_income_P999 = a_income_P999 + b_income_P999
    asset_income_Ent = a_income_Ent + b_income_Ent

    ## 38. Compute total income aggregates by quintile (sums, not averages)

    total_income_all = sum(sum(sum(total_income_mesh_aux .* mu_dist_tilde)))
    total_income_B60 = sum(sum(sum(total_income_mesh_aux .* mu_dist_tilde_B60)))
    total_income_Q1 = sum(sum(sum(total_income_mesh_aux .* mu_dist_tilde_Q1)))
    total_income_Q2 = sum(sum(sum(total_income_mesh_aux .* mu_dist_tilde_Q2)))
    total_income_Q3 = sum(sum(sum(total_income_mesh_aux .* mu_dist_tilde_Q3)))
    total_income_Q4 = sum(sum(sum(total_income_mesh_aux .* mu_dist_tilde_Q4)))
    total_income_Q5 = sum(sum(sum(total_income_mesh_aux .* mu_dist_tilde_Q5)))
    total_income_P90 = sum(sum(sum(total_income_mesh_aux .* mu_dist_tilde_P90)))
    total_income_P99 = sum(sum(sum(total_income_mesh_aux .* mu_dist_tilde_P99)))
    total_income_P999 = sum(sum(sum(total_income_mesh_aux .* mu_dist_tilde_P999)))
    total_income_Ent = sum(sum(sum(total_income_mesh_aux .* mu_dist_tilde_Ent)))

    ## 39. Compute total labor income by quintile (sums)

    total_labor_all = sum(sum(sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde[:, :, 1:ns_val])))
    total_labor_B60 = sum(sum(sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_B60[:, :, 1:ns_val])))
    total_labor_Q1 = sum(sum(sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_Q1[:, :, 1:ns_val])))
    total_labor_Q2 = sum(sum(sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_Q2[:, :, 1:ns_val])))
    total_labor_Q3 = sum(sum(sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_Q3[:, :, 1:ns_val])))
    total_labor_Q4 = sum(sum(sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_Q4[:, :, 1:ns_val])))
    total_labor_Q5 = sum(sum(sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_Q5[:, :, 1:ns_val])))
    total_labor_P90 = sum(sum(sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_P90[:, :, 1:ns_val])))
    total_labor_P99 = sum(sum(sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_P99[:, :, 1:ns_val])))
    total_labor_P999 = sum(sum(sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_P999[:, :, 1:ns_val])))
    total_labor_Ent = sum(sum(sum(labor_income_mesh_aux[:, :, 1:ns_val] .* mu_dist_tilde_Ent[:, :, 1:ns_val])))

    ## 40. Compute total UB income by quintile (sums)

    total_ubincome_all = sum(sum(sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde[:, :, ubincome_range])))
    total_ubincome_B60 = sum(sum(sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_B60[:, :, ubincome_range])))
    total_ubincome_Q1 = sum(sum(sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_Q1[:, :, ubincome_range])))
    total_ubincome_Q2 = sum(sum(sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_Q2[:, :, ubincome_range])))
    total_ubincome_Q3 = sum(sum(sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_Q3[:, :, ubincome_range])))
    total_ubincome_Q4 = sum(sum(sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_Q4[:, :, ubincome_range])))
    total_ubincome_Q5 = sum(sum(sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_Q5[:, :, ubincome_range])))
    total_ubincome_P90 = sum(sum(sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_P90[:, :, ubincome_range])))
    total_ubincome_P99 = sum(sum(sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_P99[:, :, ubincome_range])))
    total_ubincome_P999 = sum(sum(sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_P999[:, :, ubincome_range])))
    total_ubincome_Ent = sum(sum(sum(ubincome_mesh_aux[:, :, ubincome_range] .* mu_dist_tilde_Ent[:, :, ubincome_range])))

    ## 41. Compute total asset income by quintile (sums)

    total_asset_income_all = sum(sum(sum((SS_stats["r_a_aux"] .* meshes["a"] .+
        (param["R_cb"] / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"] - 1) .* meshes["b"] .* (meshes["b"] .>= 0) .+
        ((param["R_cb"] + param["Rprem"]) / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"]) .* meshes["b"] .* (meshes["b"] .< 0)) .* mu_dist_tilde)))
    total_asset_income_B60 = sum(sum(sum((SS_stats["r_a_aux"] .* meshes["a"] .+
        (param["R_cb"] / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"] - 1) .* meshes["b"] .* (meshes["b"] .>= 0) .+
        ((param["R_cb"] + param["Rprem"]) / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"]) .* meshes["b"] .* (meshes["b"] .< 0)) .* mu_dist_tilde_B60)))
    total_asset_income_Q1 = sum(sum(sum((SS_stats["r_a_aux"] .* meshes["a"] .+
        (param["R_cb"] / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"] - 1) .* meshes["b"] .* (meshes["b"] .>= 0) .+
        ((param["R_cb"] + param["Rprem"]) / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"]) .* meshes["b"] .* (meshes["b"] .< 0)) .* mu_dist_tilde_Q1)))
    total_asset_income_Q2 = sum(sum(sum((SS_stats["r_a_aux"] .* meshes["a"] .+
        (param["R_cb"] / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"] - 1) .* meshes["b"] .* (meshes["b"] .>= 0) .+
        ((param["R_cb"] + param["Rprem"]) / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"]) .* meshes["b"] .* (meshes["b"] .< 0)) .* mu_dist_tilde_Q2)))
    total_asset_income_Q3 = sum(sum(sum((SS_stats["r_a_aux"] .* meshes["a"] .+
        (param["R_cb"] / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"] - 1) .* meshes["b"] .* (meshes["b"] .>= 0) .+
        ((param["R_cb"] + param["Rprem"]) / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"]) .* meshes["b"] .* (meshes["b"] .< 0)) .* mu_dist_tilde_Q3)))
    total_asset_income_Q4 = sum(sum(sum((SS_stats["r_a_aux"] .* meshes["a"] .+
        (param["R_cb"] / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"] - 1) .* meshes["b"] .* (meshes["b"] .>= 0) .+
        ((param["R_cb"] + param["Rprem"]) / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"]) .* meshes["b"] .* (meshes["b"] .< 0)) .* mu_dist_tilde_Q4)))
    total_asset_income_Q5 = sum(sum(sum((SS_stats["r_a_aux"] .* meshes["a"] .+
        (param["R_cb"] / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"] - 1) .* meshes["b"] .* (meshes["b"] .>= 0) .+
        ((param["R_cb"] + param["Rprem"]) / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"]) .* meshes["b"] .* (meshes["b"] .< 0)) .* mu_dist_tilde_Q5)))
    total_asset_income_P90 = sum(sum(sum((SS_stats["r_a_aux"] .* meshes["a"] .+
        (param["R_cb"] / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"] - 1) .* meshes["b"] .* (meshes["b"] .>= 0) .+
        ((param["R_cb"] + param["Rprem"]) / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"]) .* meshes["b"] .* (meshes["b"] .< 0)) .* mu_dist_tilde_P90)))
    total_asset_income_P99 = sum(sum(sum((SS_stats["r_a_aux"] .* meshes["a"] .+
        (param["R_cb"] / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"] - 1) .* meshes["b"] .* (meshes["b"] .>= 0) .+
        ((param["R_cb"] + param["Rprem"]) / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"]) .* meshes["b"] .* (meshes["b"] .< 0)) .* mu_dist_tilde_P99)))
    total_asset_income_P999 = sum(sum(sum((SS_stats["r_a_aux"] .* meshes["a"] .+
        (param["R_cb"] / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"] - 1) .* meshes["b"] .* (meshes["b"] .>= 0) .+
        ((param["R_cb"] + param["Rprem"]) / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"]) .* meshes["b"] .* (meshes["b"] .< 0)) .* mu_dist_tilde_P999)))
    total_asset_income_Ent = sum(sum(sum((SS_stats["r_a_aux"] .* meshes["a"] .+
        (param["R_cb"] / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"] - 1) .* meshes["b"] .* (meshes["b"] .>= 0) .+
        ((param["R_cb"] + param["Rprem"]) / (1 - param["death_rate"]) * param["b_a_aux"] / param["pi_cb"]) .* meshes["b"] .* (meshes["b"] .< 0)) .* mu_dist_tilde_Ent)))

    ## 42. Compute total profit income by quintile (sums)

    total_profit_income_all = sum(sum(sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde[:, :, end])))
    total_profit_income_B60 = sum(sum(sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_B60[:, :, end])))
    total_profit_income_Q1 = sum(sum(sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_Q1[:, :, end])))
    total_profit_income_Q2 = sum(sum(sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_Q2[:, :, end])))
    total_profit_income_Q3 = sum(sum(sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_Q3[:, :, end])))
    total_profit_income_Q4 = sum(sum(sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_Q4[:, :, end])))
    total_profit_income_Q5 = sum(sum(sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_Q5[:, :, end])))
    total_profit_income_P90 = sum(sum(sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_P90[:, :, end])))
    total_profit_income_P99 = sum(sum(sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_P99[:, :, end])))
    total_profit_income_P999 = sum(sum(sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_P999[:, :, end])))
    total_profit_income_Ent = sum(sum(sum(profit_income_mesh_aux[:, :, end] .* mu_dist_tilde_Ent[:, :, end])))

    ## 43. Compute total transfer income by quintile (sums)

    total_transfer_income_all = sum(sum(sum(transfer_mesh_aux .* mu_dist_tilde)))
    total_transfer_income_B60 = sum(sum(sum(transfer_mesh_aux .* mu_dist_tilde_B60)))
    total_transfer_income_Q1 = sum(sum(sum(transfer_mesh_aux .* mu_dist_tilde_Q1)))
    total_transfer_income_Q2 = sum(sum(sum(transfer_mesh_aux .* mu_dist_tilde_Q2)))
    total_transfer_income_Q3 = sum(sum(sum(transfer_mesh_aux .* mu_dist_tilde_Q3)))
    total_transfer_income_Q4 = sum(sum(sum(transfer_mesh_aux .* mu_dist_tilde_Q4)))
    total_transfer_income_Q5 = sum(sum(sum(transfer_mesh_aux .* mu_dist_tilde_Q5)))
    total_transfer_income_P90 = sum(sum(sum(transfer_mesh_aux .* mu_dist_tilde_P90)))
    total_transfer_income_P99 = sum(sum(sum(transfer_mesh_aux .* mu_dist_tilde_P99)))
    total_transfer_income_P999 = sum(sum(sum(transfer_mesh_aux .* mu_dist_tilde_P999)))
    total_transfer_income_Ent = sum(sum(sum(transfer_mesh_aux .* mu_dist_tilde_Ent)))

    ## 44. Compute total other income (transfers + UB) by quintile (sums)

    total_other_income_all = total_transfer_income_all + total_ubincome_all
    total_other_income_B60 = total_transfer_income_B60 + total_ubincome_B60
    total_other_income_Q1 = total_transfer_income_Q1 + total_ubincome_Q1
    total_other_income_Q2 = total_transfer_income_Q2 + total_ubincome_Q2
    total_other_income_Q3 = total_transfer_income_Q3 + total_ubincome_Q3
    total_other_income_Q4 = total_transfer_income_Q4 + total_ubincome_Q4
    total_other_income_Q5 = total_transfer_income_Q5 + total_ubincome_Q5
    total_other_income_P90 = total_transfer_income_P90 + total_ubincome_P90
    total_other_income_P99 = total_transfer_income_P99 + total_ubincome_P99
    total_other_income_P999 = total_transfer_income_P999 + total_ubincome_P999

    ## 45. Compute total asset holdings by quintile (equity + bonds, sums)

    total_asset_all = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* mu_dist_tilde)
    total_asset_B60 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* mu_dist_tilde_B60)
    total_asset_Q1 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* mu_dist_tilde_Q1)
    total_asset_Q2 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* mu_dist_tilde_Q2)
    total_asset_Q3 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* mu_dist_tilde_Q3)
    total_asset_Q4 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* mu_dist_tilde_Q4)
    total_asset_Q5 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* mu_dist_tilde_Q5)
    total_asset_P90 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* mu_dist_tilde_P90)
    total_asset_P99 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* mu_dist_tilde_P99)
    total_asset_P999 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* mu_dist_tilde_P999)
    total_asset_Ent = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* mu_dist_tilde_Ent)

    ## 46. Compute total equity holdings by quintile (sums)

    total_a_all = sum(a_income_mesh_aux .* mu_dist_tilde)
    total_a_B60 = sum(a_income_mesh_aux .* mu_dist_tilde_B60)
    total_a_Q1 = sum(a_income_mesh_aux .* mu_dist_tilde_Q1)
    total_a_Q2 = sum(a_income_mesh_aux .* mu_dist_tilde_Q2)
    total_a_Q3 = sum(a_income_mesh_aux .* mu_dist_tilde_Q3)
    total_a_Q4 = sum(a_income_mesh_aux .* mu_dist_tilde_Q4)
    total_a_Q5 = sum(a_income_mesh_aux .* mu_dist_tilde_Q5)
    total_a_P90 = sum(a_income_mesh_aux .* mu_dist_tilde_P90)
    total_a_P99 = sum(a_income_mesh_aux .* mu_dist_tilde_P99)
    total_a_P999 = sum(a_income_mesh_aux .* mu_dist_tilde_P999)
    total_a_Ent = sum(a_income_mesh_aux .* mu_dist_tilde_Ent)

    ## 47. Compute total bond holdings by quintile (sums)

    total_b_all = sum(b_income_mesh_aux .* mu_dist_tilde)
    total_b_B60 = sum(b_income_mesh_aux .* mu_dist_tilde_B60)
    total_b_Q1 = sum(b_income_mesh_aux .* mu_dist_tilde_Q1)
    total_b_Q2 = sum(b_income_mesh_aux .* mu_dist_tilde_Q2)
    total_b_Q3 = sum(b_income_mesh_aux .* mu_dist_tilde_Q3)
    total_b_Q4 = sum(b_income_mesh_aux .* mu_dist_tilde_Q4)
    total_b_Q5 = sum(b_income_mesh_aux .* mu_dist_tilde_Q5)
    total_b_P90 = sum(b_income_mesh_aux .* mu_dist_tilde_P90)
    total_b_P99 = sum(b_income_mesh_aux .* mu_dist_tilde_P99)
    total_b_P999 = sum(b_income_mesh_aux .* mu_dist_tilde_P999)
    total_b_Ent = sum(b_income_mesh_aux .* mu_dist_tilde_Ent)

    ## 48. Compute income composition vectors (shares by source)

    # income_comp2: shares using average income
    income_comp2_all = [labor_all; profit_income_all; asset_income_all; other_income_all] ./
                       (labor_all + profit_income_all + asset_income_all + other_income_all)
    income_comp2_B60 = [labor_B60; profit_income_B60; asset_income_B60; other_income_B60] ./
                       (labor_B60 + profit_income_B60 + asset_income_B60 + other_income_B60)
    income_comp2_Q1 = [labor_Q1; profit_income_Q1; asset_income_Q1; other_income_Q1] ./
                      (labor_Q1 + profit_income_Q1 + asset_income_Q1 + other_income_Q1)
    income_comp2_Q2 = [labor_Q2; profit_income_Q2; asset_income_Q2; other_income_Q2] ./
                      (labor_Q2 + profit_income_Q2 + asset_income_Q2 + other_income_Q2)
    income_comp2_Q3 = [labor_Q3; profit_income_Q3; asset_income_Q3; other_income_Q3] ./
                      (labor_Q3 + profit_income_Q3 + asset_income_Q3 + other_income_Q3)
    income_comp2_Q4 = [labor_Q4; profit_income_Q4; asset_income_Q4; other_income_Q4] ./
                      (labor_Q4 + profit_income_Q4 + asset_income_Q4 + other_income_Q4)
    income_comp2_Q5 = [labor_Q5; profit_income_Q5; asset_income_Q5; other_income_Q5] ./
                      (labor_Q5 + profit_income_Q5 + asset_income_Q5 + other_income_Q5)
    income_comp2_P90 = [labor_P90; profit_income_P90; asset_income_P90; other_income_P90] ./
                       (labor_P90 + profit_income_P90 + asset_income_P90 + other_income_P90)
    income_comp2_P99 = [labor_P99; profit_income_P99; asset_income_P99; other_income_P99] ./
                       (labor_P99 + profit_income_P99 + asset_income_P99 + other_income_P99)
    income_comp2_P999 = [labor_P999; profit_income_P999; asset_income_P999; other_income_P999] ./
                        (labor_P999 + profit_income_P999 + asset_income_P999 + other_income_P999)

    # income_comp: shares using total income (pre-tax adjustment)
    tau_w = param["tau_w"]
    income_comp_all = [total_labor_all/(1-tau_w); total_profit_income_all/(1-tau_w); total_asset_income_all; total_other_income_all] ./
                      (total_labor_all/(1-tau_w) + total_profit_income_all/(1-tau_w) + total_asset_income_all + total_other_income_all)
    income_comp_B60 = [total_labor_B60/(1-tau_w); total_profit_income_B60/(1-tau_w); total_asset_income_B60; total_other_income_B60] ./
                      (total_labor_B60/(1-tau_w) + total_profit_income_B60/(1-tau_w) + total_asset_income_B60 + total_other_income_B60)
    income_comp_Q1 = [total_labor_Q1/(1-tau_w); total_profit_income_Q1/(1-tau_w); total_asset_income_Q1; total_other_income_Q1] ./
                     (total_labor_Q1/(1-tau_w) + total_profit_income_Q1/(1-tau_w) + total_asset_income_Q1 + total_other_income_Q1)
    income_comp_Q2 = [total_labor_Q2/(1-tau_w); total_profit_income_Q2/(1-tau_w); total_asset_income_Q2; total_other_income_Q2] ./
                     (total_labor_Q2/(1-tau_w) + total_profit_income_Q2/(1-tau_w) + total_asset_income_Q2 + total_other_income_Q2)
    income_comp_Q3 = [total_labor_Q3/(1-tau_w); total_profit_income_Q3/(1-tau_w); total_asset_income_Q3; total_other_income_Q3] ./
                     (total_labor_Q3/(1-tau_w) + total_profit_income_Q3/(1-tau_w) + total_asset_income_Q3 + total_other_income_Q3)
    income_comp_Q4 = [total_labor_Q4/(1-tau_w); total_profit_income_Q4/(1-tau_w); total_asset_income_Q4; total_other_income_Q4] ./
                     (total_labor_Q4/(1-tau_w) + total_profit_income_Q4/(1-tau_w) + total_asset_income_Q4 + total_other_income_Q4)
    income_comp_Q5 = [total_labor_Q5/(1-tau_w); total_profit_income_Q5/(1-tau_w); total_asset_income_Q5; total_other_income_Q5] ./
                     (total_labor_Q5/(1-tau_w) + total_profit_income_Q5/(1-tau_w) + total_asset_income_Q5 + total_other_income_Q5)
    income_comp_P90 = [total_labor_P90/(1-tau_w); total_profit_income_P90/(1-tau_w); total_asset_income_P90; total_other_income_P90] ./
                      (total_labor_P90/(1-tau_w) + total_profit_income_P90/(1-tau_w) + total_asset_income_P90 + total_other_income_P90)
    income_comp_P99 = [total_labor_P99/(1-tau_w); total_profit_income_P99/(1-tau_w); total_asset_income_P99; total_other_income_P99] ./
                      (total_labor_P99/(1-tau_w) + total_profit_income_P99/(1-tau_w) + total_asset_income_P99 + total_other_income_P99)
    income_comp_P999 = [total_labor_P999/(1-tau_w); total_profit_income_P999/(1-tau_w); total_asset_income_P999; total_other_income_P999] ./
                       (total_labor_P999/(1-tau_w) + total_profit_income_P999/(1-tau_w) + total_asset_income_P999 + total_other_income_P999)

    # income_comp_2: like income_comp but separates equity from bonds
    income_comp_all_2 = [total_labor_all/(1-tau_w); total_profit_income_all/(1-tau_w); total_a_all; total_b_all; total_other_income_all] ./
                        (total_labor_all/(1-tau_w) + total_profit_income_all/(1-tau_w) + total_asset_income_all + total_other_income_all)
    income_comp_B60_2 = [total_labor_B60/(1-tau_w); total_profit_income_B60/(1-tau_w); total_a_B60; total_b_B60; total_other_income_B60] ./
                        (total_labor_B60/(1-tau_w) + total_profit_income_B60/(1-tau_w) + total_asset_income_B60 + total_other_income_B60)
    income_comp_Q1_2 = [total_labor_Q1/(1-tau_w); total_profit_income_Q1/(1-tau_w); total_a_Q1; total_b_Q1; total_other_income_Q1] ./
                       (total_labor_Q1/(1-tau_w) + total_profit_income_Q1/(1-tau_w) + total_asset_income_Q1 + total_other_income_Q1)
    income_comp_Q2_2 = [total_labor_Q2/(1-tau_w); total_profit_income_Q2/(1-tau_w); total_a_Q2; total_b_Q2; total_other_income_Q2] ./
                       (total_labor_Q2/(1-tau_w) + total_profit_income_Q2/(1-tau_w) + total_asset_income_Q2 + total_other_income_Q2)
    income_comp_Q3_2 = [total_labor_Q3/(1-tau_w); total_profit_income_Q3/(1-tau_w); total_a_Q3; total_b_Q3; total_other_income_Q3] ./
                       (total_labor_Q3/(1-tau_w) + total_profit_income_Q3/(1-tau_w) + total_asset_income_Q3 + total_other_income_Q3)
    income_comp_Q4_2 = [total_labor_Q4/(1-tau_w); total_profit_income_Q4/(1-tau_w); total_a_Q4; total_b_Q4; total_other_income_Q4] ./
                       (total_labor_Q4/(1-tau_w) + total_profit_income_Q4/(1-tau_w) + total_asset_income_Q4 + total_other_income_Q4)
    income_comp_Q5_2 = [total_labor_Q5/(1-tau_w); total_profit_income_Q5/(1-tau_w); total_a_Q5; total_b_Q5; total_other_income_Q5] ./
                       (total_labor_Q5/(1-tau_w) + total_profit_income_Q5/(1-tau_w) + total_asset_income_Q5 + total_other_income_Q5)
    income_comp_P90_2 = [total_labor_P90/(1-tau_w); total_profit_income_P90/(1-tau_w); total_a_P90; total_b_P90; total_other_income_P90] ./
                        (total_labor_P90/(1-tau_w) + total_profit_income_P90/(1-tau_w) + total_asset_income_P90 + total_other_income_P90)
    income_comp_P99_2 = [total_labor_P99/(1-tau_w); total_profit_income_P99/(1-tau_w); total_a_P99; total_b_P99; total_other_income_P99] ./
                        (total_labor_P99/(1-tau_w) + total_profit_income_P99/(1-tau_w) + total_asset_income_P99 + total_other_income_P99)
    income_comp_P999_2 = [total_labor_P999/(1-tau_w); total_profit_income_P999/(1-tau_w); total_a_P999; total_b_P999; total_other_income_P999] ./
                         (total_labor_P999/(1-tau_w) + total_profit_income_P999/(1-tau_w) + total_asset_income_P999 + total_other_income_P999)

    ## 49. Create aggregate income composition matrices

    Q_income_comp_model = hcat(income_comp_B60, income_comp_Q4, income_comp_Q5, income_comp_P90, income_comp_P99, income_comp_P999)
    Q_income_comp_model_2 = hcat(income_comp_B60_2, income_comp_Q4_2, income_comp_Q5_2, income_comp_P90_2, income_comp_P99_2, income_comp_P999_2)

    # Simplified versions (labor, business+asset, other)
    Q_income_comp_model2 = vcat(Q_income_comp_model[1:1, :],
                                dropdims(sum(Q_income_comp_model[2:3, :], dims=1), dims=1)',
                                Q_income_comp_model[end:end, :])
    Q_income_comp_model2_2 = vcat(Q_income_comp_model_2[1:1, :],
                                  dropdims(sum(Q_income_comp_model_2[2:3, :], dims=1), dims=1)',
                                  Q_income_comp_model_2[4:end, :])

    ## 50. Store all income and asset statistics in anal_stats

    anal_stats["total_income_mesh"] = total_income_mesh_aux
    anal_stats["labor_income_mesh"] = labor_income_mesh_aux
    anal_stats["ub_income_mesh"] = ubincome_mesh_aux
    anal_stats["profit_income_mesh"] = profit_income_mesh_aux
    anal_stats["equity_income_mesh"] = a_income_mesh_aux
    anal_stats["bond_income_mesh"] = b_income_mesh_aux
    anal_stats["transfer_mesh"] = transfer_mesh_aux
    anal_stats["mu_dist_tilde"] = mu_dist_tilde
    anal_stats["SS_stats"] = SS_stats
    anal_stats["Q_dists"] = Q_dists
    anal_stats["meshes_w"] = meshes_w

    # Average wealth by quintile
    anal_stats["w_all"] = w_all
    anal_stats["w_B60"] = w_B60
    anal_stats["w_Q1"] = w_Q1
    anal_stats["w_Q2"] = w_Q2
    anal_stats["w_Q3"] = w_Q3
    anal_stats["w_Q4"] = w_Q4
    anal_stats["w_Q5"] = w_Q5
    anal_stats["w_P90"] = w_P90
    anal_stats["w_P99"] = w_P99
    anal_stats["w_P999"] = w_P999
    anal_stats["w_Ent"] = w_Ent

    # Average total income by quintile
    anal_stats["income_all"] = income_all
    anal_stats["income_B01"] = income_B01
    anal_stats["income_B1"] = income_B1
    anal_stats["income_B10"] = income_B10
    anal_stats["income_B60"] = income_B60
    anal_stats["income_Q1"] = income_Q1
    anal_stats["income_Q2"] = income_Q2
    anal_stats["income_Q3"] = income_Q3
    anal_stats["income_Q4"] = income_Q4
    anal_stats["income_Q5"] = income_Q5
    anal_stats["income_P90"] = income_P90
    anal_stats["income_P99"] = income_P99
    anal_stats["income_P999"] = income_P999
    anal_stats["income_Ent"] = income_Ent

    # Average labor income by quintile
    anal_stats["labor_all"] = labor_all
    anal_stats["labor_B60"] = labor_B60
    anal_stats["labor_Q1"] = labor_Q1
    anal_stats["labor_Q2"] = labor_Q2
    anal_stats["labor_Q3"] = labor_Q3
    anal_stats["labor_Q4"] = labor_Q4
    anal_stats["labor_Q5"] = labor_Q5
    anal_stats["labor_P90"] = labor_P90
    anal_stats["labor_P99"] = labor_P99
    anal_stats["labor_P999"] = labor_P999
    #anal_stats["labor_Ent"] = labor_Ent

    # Average UB income by quintile
    anal_stats["ubincome_all"] = ubincome_all
    anal_stats["ubincome_B60"] = ubincome_B60
    anal_stats["ubincome_Q1"] = ubincome_Q1
    anal_stats["ubincome_Q2"] = ubincome_Q2
    anal_stats["ubincome_Q3"] = ubincome_Q3
    anal_stats["ubincome_Q4"] = ubincome_Q4
    anal_stats["ubincome_Q5"] = ubincome_Q5
    anal_stats["ubincome_P90"] = ubincome_P90
    anal_stats["ubincome_P99"] = ubincome_P99
    anal_stats["ubincome_P999"] = ubincome_P999
    #anal_stats["ubincome_Ent"] = ubincome_Ent

    # Average profit income by quintile
    anal_stats["profit_income_all"] = profit_income_all
    anal_stats["profit_income_B60"] = profit_income_B60
    anal_stats["profit_income_Q1"] = profit_income_Q1
    anal_stats["profit_income_Q2"] = profit_income_Q2
    anal_stats["profit_income_Q3"] = profit_income_Q3
    anal_stats["profit_income_Q4"] = profit_income_Q4
    anal_stats["profit_income_Q5"] = profit_income_Q5
    anal_stats["profit_income_P90"] = profit_income_P90
    anal_stats["profit_income_P99"] = profit_income_P99
    anal_stats["profit_income_P999"] = profit_income_P999
    anal_stats["profit_income_Ent"] = profit_income_Ent

    # Average equity income by quintile
    anal_stats["a_income_all"] = a_income_all
    anal_stats["a_income_B60"] = a_income_B60
    anal_stats["a_income_Q1"] = a_income_Q1
    anal_stats["a_income_Q2"] = a_income_Q2
    anal_stats["a_income_Q3"] = a_income_Q3
    anal_stats["a_income_Q4"] = a_income_Q4
    anal_stats["a_income_Q5"] = a_income_Q5
    anal_stats["a_income_P90"] = a_income_P90
    anal_stats["a_income_P99"] = a_income_P99
    anal_stats["a_income_P999"] = a_income_P999
    anal_stats["a_income_Ent"] = a_income_Ent

    # Average bond income by quintile
    anal_stats["b_income_all"] = b_income_all
    anal_stats["b_income_B60"] = b_income_B60
    anal_stats["b_income_Q1"] = b_income_Q1
    anal_stats["b_income_Q2"] = b_income_Q2
    anal_stats["b_income_Q3"] = b_income_Q3
    anal_stats["b_income_Q4"] = b_income_Q4
    anal_stats["b_income_Q5"] = b_income_Q5
    anal_stats["b_income_P90"] = b_income_P90
    anal_stats["b_income_P99"] = b_income_P99
    anal_stats["b_income_P999"] = b_income_P999
    anal_stats["b_income_Ent"] = b_income_Ent

    # Average transfer income by quintile
    anal_stats["transfer_all"] = transfer_all
    anal_stats["transfer_B60"] = transfer_B60
    anal_stats["transfer_Q1"] = transfer_Q1
    anal_stats["transfer_Q2"] = transfer_Q2
    anal_stats["transfer_Q3"] = transfer_Q3
    anal_stats["transfer_Q4"] = transfer_Q4
    anal_stats["transfer_Q5"] = transfer_Q5
    anal_stats["transfer_P90"] = transfer_P90
    anal_stats["transfer_P99"] = transfer_P99
    anal_stats["transfer_P999"] = transfer_P999
    anal_stats["transfer_Ent"] = transfer_Ent

    # Average liquid assets (bonds) by quintile
    anal_stats["b_all"] = b_all
    anal_stats["b_B60"] = b_B60
    anal_stats["b_Q1"] = b_Q1
    anal_stats["b_Q2"] = b_Q2
    anal_stats["b_Q3"] = b_Q3
    anal_stats["b_Q4"] = b_Q4
    anal_stats["b_Q5"] = b_Q5
    anal_stats["b_P90"] = b_P90
    anal_stats["b_P99"] = b_P99
    anal_stats["b_P999"] = b_P999
    anal_stats["b_Ent"] = b_Ent

    # Average illiquid assets (equity) by quintile
    anal_stats["a_all"] = a_all
    anal_stats["a_B60"] = a_B60
    anal_stats["a_Q1"] = a_Q1
    anal_stats["a_Q2"] = a_Q2
    anal_stats["a_Q3"] = a_Q3
    anal_stats["a_Q4"] = a_Q4
    anal_stats["a_Q5"] = a_Q5
    anal_stats["a_P90"] = a_P90
    anal_stats["a_P99"] = a_P99
    anal_stats["a_P999"] = a_P999
    anal_stats["a_Ent"] = a_Ent

    # Average consumption by quintile
    anal_stats["c_all"] = c_all
    anal_stats["c_B60"] = c_B60
    anal_stats["c_Q1"] = c_Q1
    anal_stats["c_Q2"] = c_Q2
    anal_stats["c_Q3"] = c_Q3
    anal_stats["c_Q4"] = c_Q4
    anal_stats["c_Q5"] = c_Q5
    anal_stats["c_P90"] = c_P90
    anal_stats["c_P99"] = c_P99
    anal_stats["c_P999"] = c_P999
    anal_stats["c_Ent"] = c_Ent

    # Other income (UB + transfers)
    anal_stats["other_income_all"] = other_income_all
    anal_stats["other_income_B60"] = other_income_B60
    anal_stats["other_income_Q1"] = other_income_Q1
    anal_stats["other_income_Q2"] = other_income_Q2
    anal_stats["other_income_Q3"] = other_income_Q3
    anal_stats["other_income_Q4"] = other_income_Q4
    anal_stats["other_income_Q5"] = other_income_Q5
    anal_stats["other_income_P90"] = other_income_P90
    anal_stats["other_income_P99"] = other_income_P99
    anal_stats["other_income_P999"] = other_income_P999

    # Total asset income
    anal_stats["asset_income_all"] = asset_income_all
    anal_stats["asset_income_B60"] = asset_income_B60
    anal_stats["asset_income_Q1"] = asset_income_Q1
    anal_stats["asset_income_Q2"] = asset_income_Q2
    anal_stats["asset_income_Q3"] = asset_income_Q3
    anal_stats["asset_income_Q4"] = asset_income_Q4
    anal_stats["asset_income_Q5"] = asset_income_Q5
    anal_stats["asset_income_P90"] = asset_income_P90
    anal_stats["asset_income_P99"] = asset_income_P99
    anal_stats["asset_income_P999"] = asset_income_P999
    anal_stats["asset_income_Ent"] = asset_income_Ent

    # Total income aggregates (sums)
    anal_stats["total_income_all"] = total_income_all
    anal_stats["total_income_B60"] = total_income_B60
    anal_stats["total_income_Q1"] = total_income_Q1
    anal_stats["total_income_Q2"] = total_income_Q2
    anal_stats["total_income_Q3"] = total_income_Q3
    anal_stats["total_income_Q4"] = total_income_Q4
    anal_stats["total_income_Q5"] = total_income_Q5
    anal_stats["total_income_P90"] = total_income_P90
    anal_stats["total_income_P99"] = total_income_P99
    anal_stats["total_income_P999"] = total_income_P999
    anal_stats["total_income_Ent"] = total_income_Ent

    # Total labor income (sums)
    anal_stats["total_labor_all"] = total_labor_all
    anal_stats["total_labor_B60"] = total_labor_B60
    anal_stats["total_labor_Q1"] = total_labor_Q1
    anal_stats["total_labor_Q2"] = total_labor_Q2
    anal_stats["total_labor_Q3"] = total_labor_Q3
    anal_stats["total_labor_Q4"] = total_labor_Q4
    anal_stats["total_labor_Q5"] = total_labor_Q5
    anal_stats["total_labor_P90"] = total_labor_P90
    anal_stats["total_labor_P99"] = total_labor_P99
    anal_stats["total_labor_P999"] = total_labor_P999
    anal_stats["total_labor_Ent"] = total_labor_Ent

    # Total UB income (sums)
    anal_stats["total_ubincome_all"] = total_ubincome_all
    anal_stats["total_ubincome_B60"] = total_ubincome_B60
    anal_stats["total_ubincome_Q1"] = total_ubincome_Q1
    anal_stats["total_ubincome_Q2"] = total_ubincome_Q2
    anal_stats["total_ubincome_Q3"] = total_ubincome_Q3
    anal_stats["total_ubincome_Q4"] = total_ubincome_Q4
    anal_stats["total_ubincome_Q5"] = total_ubincome_Q5
    anal_stats["total_ubincome_P90"] = total_ubincome_P90
    anal_stats["total_ubincome_P99"] = total_ubincome_P99
    anal_stats["total_ubincome_P999"] = total_ubincome_P999
    anal_stats["total_ubincome_Ent"] = total_ubincome_Ent

    # Total asset income (sums)
    anal_stats["total_asset_income_all"] = total_asset_income_all
    anal_stats["total_asset_income_B60"] = total_asset_income_B60
    anal_stats["total_asset_income_Q1"] = total_asset_income_Q1
    anal_stats["total_asset_income_Q2"] = total_asset_income_Q2
    anal_stats["total_asset_income_Q3"] = total_asset_income_Q3
    anal_stats["total_asset_income_Q4"] = total_asset_income_Q4
    anal_stats["total_asset_income_Q5"] = total_asset_income_Q5
    anal_stats["total_asset_income_P90"] = total_asset_income_P90
    anal_stats["total_asset_income_P99"] = total_asset_income_P99
    anal_stats["total_asset_income_P999"] = total_asset_income_P999
    anal_stats["total_asset_income_Ent"] = total_asset_income_Ent

    # Total profit income (sums)
    anal_stats["total_profit_income_all"] = total_profit_income_all
    anal_stats["total_profit_income_B60"] = total_profit_income_B60
    anal_stats["total_profit_income_Q1"] = total_profit_income_Q1
    anal_stats["total_profit_income_Q2"] = total_profit_income_Q2
    anal_stats["total_profit_income_Q3"] = total_profit_income_Q3
    anal_stats["total_profit_income_Q4"] = total_profit_income_Q4
    anal_stats["total_profit_income_Q5"] = total_profit_income_Q5
    anal_stats["total_profit_income_P90"] = total_profit_income_P90
    anal_stats["total_profit_income_P99"] = total_profit_income_P99
    anal_stats["total_profit_income_P999"] = total_profit_income_P999
    anal_stats["total_profit_income_Ent"] = total_profit_income_Ent

    # Total transfer income (sums)
    anal_stats["total_transfer_income_all"] = total_transfer_income_all
    anal_stats["total_transfer_income_B60"] = total_transfer_income_B60
    anal_stats["total_transfer_income_Q1"] = total_transfer_income_Q1
    anal_stats["total_transfer_income_Q2"] = total_transfer_income_Q2
    anal_stats["total_transfer_income_Q3"] = total_transfer_income_Q3
    anal_stats["total_transfer_income_Q4"] = total_transfer_income_Q4
    anal_stats["total_transfer_income_Q5"] = total_transfer_income_Q5
    anal_stats["total_transfer_income_P90"] = total_transfer_income_P90
    anal_stats["total_transfer_income_P99"] = total_transfer_income_P99
    anal_stats["total_transfer_income_P999"] = total_transfer_income_P999
    anal_stats["total_transfer_income_Ent"] = total_transfer_income_Ent

    ## 51. Compute average liquid assets (b) by quintile
    b_all = sum(meshes["b"] .* mu_dist_tilde) / sum(mu_dist_tilde)
    b_B60 = sum(meshes["b"] .* mu_dist_tilde_B60) / sum(mu_dist_tilde_B60)
    b_Q1 = sum(meshes["b"] .* mu_dist_tilde_Q1) / sum(mu_dist_tilde_Q1)
    b_Q2 = sum(meshes["b"] .* mu_dist_tilde_Q2) / sum(mu_dist_tilde_Q2)
    b_Q3 = sum(meshes["b"] .* mu_dist_tilde_Q3) / sum(mu_dist_tilde_Q3)
    b_Q4 = sum(meshes["b"] .* mu_dist_tilde_Q4) / sum(mu_dist_tilde_Q4)
    b_Q5 = sum(meshes["b"] .* mu_dist_tilde_Q5) / sum(mu_dist_tilde_Q5)
    b_P90 = sum(meshes["b"] .* mu_dist_tilde_P90) / sum(mu_dist_tilde_P90)
    b_P99 = sum(meshes["b"] .* mu_dist_tilde_P99) / sum(mu_dist_tilde_P99)
    b_P999 = sum(meshes["b"] .* mu_dist_tilde_P999) / sum(mu_dist_tilde_P999)
    b_Ent = sum(meshes["b"] .* mu_dist_tilde_Ent) / sum(mu_dist_tilde_Ent)

    ## 52. Compute average illiquid assets valued at q (a_1) by quintile
    a_1_all = sum(param["q"] * meshes["a"] .* mu_dist_tilde) / sum(mu_dist_tilde)
    a_1_B60 = sum(param["q"] * meshes["a"] .* mu_dist_tilde_B60) / sum(mu_dist_tilde_B60)
    a_1_Q1 = sum(param["q"] * meshes["a"] .* mu_dist_tilde_Q1) / sum(mu_dist_tilde_Q1)
    a_1_Q2 = sum(param["q"] * meshes["a"] .* mu_dist_tilde_Q2) / sum(mu_dist_tilde_Q2)
    a_1_Q3 = sum(param["q"] * meshes["a"] .* mu_dist_tilde_Q3) / sum(mu_dist_tilde_Q3)
    a_1_Q4 = sum(param["q"] * meshes["a"] .* mu_dist_tilde_Q4) / sum(mu_dist_tilde_Q4)
    a_1_Q5 = sum(param["q"] * meshes["a"] .* mu_dist_tilde_Q5) / sum(mu_dist_tilde_Q5)
    a_1_P90 = sum(param["q"] * meshes["a"] .* mu_dist_tilde_P90) / sum(mu_dist_tilde_P90)
    a_1_P99 = sum(param["q"] * meshes["a"] .* mu_dist_tilde_P99) / sum(mu_dist_tilde_P99)
    a_1_P999 = sum(param["q"] * meshes["a"] .* mu_dist_tilde_P999) / sum(mu_dist_tilde_P999)
    a_1_Ent = sum(param["q"] * meshes["a"] .* mu_dist_tilde_Ent) / sum(mu_dist_tilde_Ent)

    ## 53. Compute average illiquid assets in units (a_2) by quintile
    a_2_all = sum(meshes["a"] .* mu_dist_tilde) / sum(mu_dist_tilde)
    a_2_B60 = sum(meshes["a"] .* mu_dist_tilde_B60) / sum(mu_dist_tilde_B60)
    a_2_Q1 = sum(meshes["a"] .* mu_dist_tilde_Q1) / sum(mu_dist_tilde_Q1)
    a_2_Q2 = sum(meshes["a"] .* mu_dist_tilde_Q2) / sum(mu_dist_tilde_Q2)
    a_2_Q3 = sum(meshes["a"] .* mu_dist_tilde_Q3) / sum(mu_dist_tilde_Q3)
    a_2_Q4 = sum(meshes["a"] .* mu_dist_tilde_Q4) / sum(mu_dist_tilde_Q4)
    a_2_Q5 = sum(meshes["a"] .* mu_dist_tilde_Q5) / sum(mu_dist_tilde_Q5)
    a_2_P90 = sum(meshes["a"] .* mu_dist_tilde_P90) / sum(mu_dist_tilde_P90)
    a_2_P99 = sum(meshes["a"] .* mu_dist_tilde_P99) / sum(mu_dist_tilde_P99)
    a_2_P999 = sum(meshes["a"] .* mu_dist_tilde_P999) / sum(mu_dist_tilde_P999)
    a_2_Ent = sum(meshes["a"] .* mu_dist_tilde_Ent) / sum(mu_dist_tilde_Ent)

    ## 54. Compute next period's illiquid assets (a_11) by quintile
    a_11_all = (1 - param["death_rate"]) * sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde)
    a_11_B60 = (1 - param["death_rate"]) * sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_B60)
    a_11_Q1 = (1 - param["death_rate"]) * sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_Q1)
    a_11_Q2 = (1 - param["death_rate"]) * sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_Q2)
    a_11_Q3 = (1 - param["death_rate"]) * sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_Q3)
    a_11_Q4 = (1 - param["death_rate"]) * sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_Q4)
    a_11_Q5 = (1 - param["death_rate"]) * sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_Q5)
    a_11_P90 = (1 - param["death_rate"]) * sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_P90)
    a_11_P99 = (1 - param["death_rate"]) * sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_P99)
    a_11_P999 = (1 - param["death_rate"]) * sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_P999)
    a_11_Ent = (1 - param["death_rate"]) * sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_Ent)

    ## 55. Compute illiquid asset to wealth ratios by quintile
    a_ratio_Q1 = a_1_Q1 / w_Q1
    a_ratio_Q2 = a_1_Q2 / w_Q2
    a_ratio_Q3 = a_1_Q3 / w_Q3
    a_ratio_B60 = a_1_B60 / w_B60
    a_ratio_Q4 = a_1_Q4 / w_Q4
    a_ratio_Q5 = a_1_Q5 / w_Q5
    a_ratio_P90 = a_1_P90 / w_P90
    a_ratio_P99 = a_1_P99 / w_P99
    a_ratio_P999 = a_1_P999 / w_P999
    a_ratio_Ent = a_1_Ent / w_Ent

    ## 56. Compute liquid asset to wealth ratios by quintile
    b_ratio_Q1 = b_Q1 / w_Q1
    b_ratio_Q2 = b_Q2 / w_Q2
    b_ratio_Q3 = b_Q3 / w_Q3
    b_ratio_B60 = b_B60 / w_B60
    b_ratio_Q4 = b_Q4 / w_Q4
    b_ratio_Q5 = b_Q5 / w_Q5
    b_ratio_P90 = b_P90 / w_P90
    b_ratio_P99 = b_P99 / w_P99
    b_ratio_P999 = b_P999 / w_P999
    b_ratio_Ent = b_Ent / w_Ent

    ## 57. Create portfolio composition matrix
    Q_portfol_comp_model = [a_ratio_B60, a_ratio_Q4, a_ratio_Q5, a_ratio_P90, a_ratio_P99, a_ratio_P999]
    mu_dist_tilde_Emp = Q_dists["mu_dist_tilde_Emp"]
    mu_dist_tilde_Unemp = Q_dists["mu_dist_tilde_Unemp"]
    mu_dist_tilde_aux = Q_dists["mu_dist_tilde_aux"]
    mu_dist_tilde_full = Q_dists["mu_dist_tilde_full"]
    ## 58. Compute consumption by quintile using q_cons2 function
    W_fc = SS_stats["w"]
    c_all = q_cons2(c_a_guess, c_n_guess, W_fc, n, AProb, mu_dist_tilde, meshes, param, grid, param["tau_w"])
    c_Q1 = q_cons2(c_a_guess, c_n_guess, W_fc, n, AProb, mu_dist_tilde_Q1, meshes, param, grid, param["tau_w"])
    c_Q2 = q_cons2(c_a_guess, c_n_guess, W_fc, n, AProb, mu_dist_tilde_Q2, meshes, param, grid, param["tau_w"])
    c_Q3 = q_cons2(c_a_guess, c_n_guess, W_fc, n, AProb, mu_dist_tilde_Q3, meshes, param, grid, param["tau_w"])
    c_Q4 = q_cons2(c_a_guess, c_n_guess, W_fc, n, AProb, mu_dist_tilde_Q4, meshes, param, grid, param["tau_w"])
    c_Q5 = q_cons2(c_a_guess, c_n_guess, W_fc, n, AProb, mu_dist_tilde_Q5, meshes, param, grid, param["tau_w"])
    c_P90 = q_cons2(c_a_guess, c_n_guess, W_fc, n, AProb, mu_dist_tilde_P90, meshes, param, grid, param["tau_w"])
    c_P99 = q_cons2(c_a_guess, c_n_guess, W_fc, n, AProb, mu_dist_tilde_P99, meshes, param, grid, param["tau_w"])
    c_P999 = q_cons2(c_a_guess, c_n_guess, W_fc, n, AProb, mu_dist_tilde_P999, meshes, param, grid, param["tau_w"])
    c_Ent = q_cons2(c_a_guess, c_n_guess, W_fc, n, AProb, mu_dist_tilde_Ent, meshes, param, grid, param["tau_w"])
    c_Emp = q_cons2(c_a_guess, c_n_guess, W_fc, n, AProb, mu_dist_tilde_Emp, meshes, param, grid, param["tau_w"])
    c_Unemp = q_cons2(c_a_guess, c_n_guess, W_fc, n, AProb, mu_dist_tilde_Unemp, meshes, param, grid, param["tau_w"])

    # Update anal_stats with correct consumption values (including home production)
    anal_stats["c_all"] = c_all
    anal_stats["c_Q1"] = c_Q1
    anal_stats["c_Q2"] = c_Q2
    anal_stats["c_Q3"] = c_Q3
    anal_stats["c_Q4"] = c_Q4
    anal_stats["c_Q5"] = c_Q5
    anal_stats["c_P90"] = c_P90
    anal_stats["c_P99"] = c_P99
    anal_stats["c_P999"] = c_P999
    anal_stats["c_Ent"] = c_Ent

    ## 59. Compute illiquid asset shares by percentile
    IL_Ent_share = sum(param["q"] * meshes["a"] .* mu_dist_tilde_Ent) / grid["K"]
    IL_TOP01_share = sum(param["q"] * meshes["a"] .* (meshes["a"] .> IL999) .* mu_dist_tilde) / SS_stats["A_NI"]
    IL_TOP1_share = sum(param["q"] * meshes["a"] .* (meshes["a"] .> IL99) .* mu_dist_tilde) / SS_stats["A_NI"]
    IL_TOP10_share = sum(param["q"] * meshes["a"] .* (meshes["a"] .> IL90) .* mu_dist_tilde) / SS_stats["A_NI"]
    IL_BOT50_share = sum(param["q"] * meshes["a"] .* (meshes["a"] .<= IL50) .* mu_dist_tilde) / SS_stats["A_NI"]
    IL_BOT25_share = sum(param["q"] * meshes["a"] .* (meshes["a"] .<= IL25) .* mu_dist_tilde) / SS_stats["A_NI"]

    ## 60. Compute liquid asset shares by percentile
    L_Ent_share = sum(meshes["b"] .* mu_dist_tilde_Ent) / grid["B"]
    L_TOP01_share = sum(meshes["b"] .* (meshes["b"] .> LI999) .* mu_dist_tilde) / SS_stats["B_nb"]
    L_TOP1_share = sum(meshes["b"] .* (meshes["b"] .> LI99) .* mu_dist_tilde) / SS_stats["B_nb"]
    L_TOP10_share = sum(meshes["b"] .* (meshes["b"] .> LI90) .* mu_dist_tilde) / SS_stats["B_nb"]
    L_BOT50_share = sum(meshes["b"] .* (meshes["b"] .<= LI50) .* mu_dist_tilde) / SS_stats["B_nb"]
    L_BOT25_share = sum(meshes["b"] .* (meshes["b"] .<= LI25) .* mu_dist_tilde) / SS_stats["B_nb"]

    ## 61. Compute Gini coefficients for wealth, liquid, and illiquid assets
    # Create 2D meshes (sum over productivity dimension)
    nb = Int(grid["nb"])
    na = Int(grid["na"])
    mesh_a = repeat(grid["a"]', nb, 1)
    mesh_b = repeat(grid["b"], 1, na)
    mu_redux = dropdims(sum(mu_dist, dims=3), dims=3)

    # Wealth Gini
    total_wealth = vec(mesh_a * param["q"] .+ mesh_b)
    IX = sortperm(total_wealth)
    total_wealth = total_wealth[IX]
    total_wealth_pdf = mu_redux[IX]
    total_wealth_cdf = cumsum(total_wealth_pdf)

    SS_stats["NegNetWorth"] = sum((total_wealth .< 0) .* total_wealth_pdf)

    S_NW = cumsum(total_wealth_pdf .* total_wealth)
    S_NW = vcat([0.0], S_NW)
    SS_stats["GiniW"] = 1 - (sum(total_wealth_pdf .* (S_NW[1:end-1] .+ S_NW[2:end])) / S_NW[end])

    # Liquid Gini
    liquid_sort = vec(mesh_b)
    IX = sortperm(liquid_sort)
    liquid_sort = liquid_sort[IX]
    liquid_pdf = mu_redux[IX]
    liquid_cdf = cumsum(liquid_pdf)
    SS_stats["Negliquid"] = sum((liquid_sort .< 0) .* liquid_pdf)

    S_LI = cumsum(liquid_pdf .* liquid_sort)
    S_LI = vcat([0.0], S_LI)
    SS_stats["GiniLI"] = 1 - (sum(liquid_pdf .* (S_LI[1:end-1] .+ S_LI[2:end])) / S_LI[end])

    # Illiquid Gini
    illiquid_sort = vec(mesh_a)
    IX = sortperm(illiquid_sort)
    illiquid_sort = illiquid_sort[IX]
    illiquid_pdf = mu_redux[IX]
    illiquid_cdf = cumsum(illiquid_pdf)

    S_IL = cumsum(illiquid_pdf .* illiquid_sort)
    S_IL = vcat([0.0], S_IL)
    SS_stats["GiniIL"] = 1 - (sum(illiquid_pdf .* (S_IL[1:end-1] .+ S_IL[2:end])) / S_IL[end])

    ## 62. Compute unemployed and entrepreneur shares by quintile
    ns = Int(grid["ns"])
    Q1_u_share = sum(mu_dist_tilde_Q1[:,:,ns+1:2*ns]) / sum(mu_dist_tilde_Q1[:,:,1:2*ns])
    Q2_u_share = sum(mu_dist_tilde_Q2[:,:,ns+1:2*ns]) / sum(mu_dist_tilde_Q2[:,:,1:2*ns])
    Q3_u_share = sum(mu_dist_tilde_Q3[:,:,ns+1:2*ns]) / sum(mu_dist_tilde_Q3[:,:,1:2*ns])
    Q4_u_share = sum(mu_dist_tilde_Q4[:,:,ns+1:2*ns]) / sum(mu_dist_tilde_Q4[:,:,1:2*ns])
    Q5_u_share = sum(mu_dist_tilde_Q5[:,:,ns+1:2*ns]) / sum(mu_dist_tilde_Q5[:,:,1:2*ns])
    P90_u_share = sum(mu_dist_tilde_P90[:,:,ns+1:2*ns]) / sum(mu_dist_tilde_P90[:,:,1:2*ns])
    P99_u_share = sum(mu_dist_tilde_P99[:,:,ns+1:2*ns]) / sum(mu_dist_tilde_P99[:,:,1:2*ns])
    P999_u_share = sum(mu_dist_tilde_P999[:,:,ns+1:2*ns]) / sum(mu_dist_tilde_P999[:,:,1:2*ns])

    Q1_e_share = sum(mu_dist_tilde_Q1[:,:,end]) / sum(mu_dist_tilde_Q1)
    Q2_e_share = sum(mu_dist_tilde_Q2[:,:,end]) / sum(mu_dist_tilde_Q2)
    Q3_e_share = sum(mu_dist_tilde_Q3[:,:,end]) / sum(mu_dist_tilde_Q3)
    Q4_e_share = sum(mu_dist_tilde_Q4[:,:,end]) / sum(mu_dist_tilde_Q4)
    Q5_e_share = sum(mu_dist_tilde_Q5[:,:,end]) / sum(mu_dist_tilde_Q5)
    P90_e_share = sum(mu_dist_tilde_P90[:,:,end]) / sum(mu_dist_tilde_P90)
    P99_e_share = sum(mu_dist_tilde_P99[:,:,end]) / sum(mu_dist_tilde_P99)
    P999_e_share = sum(mu_dist_tilde_P999[:,:,end]) / sum(mu_dist_tilde_P999)

    ## 63. Store income statistics in SS_stats
    SS_stats["income_all"] = income_all
    SS_stats["income_B01"] = income_B01
    SS_stats["income_B1"] = income_B1
    SS_stats["income_B10"] = income_B10
    SS_stats["income_B60"] = income_B60
    SS_stats["income_Q1"] = income_Q1
    SS_stats["income_Q2"] = income_Q2
    SS_stats["income_Q3"] = income_Q3
    SS_stats["income_Q4"] = income_Q4
    SS_stats["income_Q5"] = income_Q5
    SS_stats["income_P90"] = income_P90
    SS_stats["income_P99"] = income_P99
    SS_stats["income_P999"] = income_P999
    SS_stats["income_Ent"] = income_Ent

    ## 64. Store consumption statistics in SS_stats by quintile using q_cons2
    SS_stats["C_all"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, mu_dist_tilde, meshes, param, grid, param["tau_w"])
    SS_stats["C_B01"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, mu_dist_tilde_B01, meshes, param, grid, param["tau_w"])
    SS_stats["C_B1"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, mu_dist_tilde_B1, meshes, param, grid, param["tau_w"])
    SS_stats["C_B10"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, mu_dist_tilde_B10, meshes, param, grid, param["tau_w"])
    SS_stats["C_Q1"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, mu_dist_tilde_Q1, meshes, param, grid, param["tau_w"])
    SS_stats["C_Q2"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, mu_dist_tilde_Q2, meshes, param, grid, param["tau_w"])
    SS_stats["C_Q3"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, mu_dist_tilde_Q3, meshes, param, grid, param["tau_w"])
    SS_stats["C_Q4"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, mu_dist_tilde_Q4, meshes, param, grid, param["tau_w"])
    SS_stats["C_Q5"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, mu_dist_tilde_Q5, meshes, param, grid, param["tau_w"])
    SS_stats["C_P90"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, mu_dist_tilde_P90, meshes, param, grid, param["tau_w"])
    SS_stats["C_P99"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, mu_dist_tilde_P99, meshes, param, grid, param["tau_w"])
    SS_stats["C_P999"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, mu_dist_tilde_P999, meshes, param, grid, param["tau_w"])
    SS_stats["C_Ent"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, mu_dist_tilde_Ent, meshes, param, grid, param["tau_w"])
    SS_stats["C_Emp"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, mu_dist_tilde_Emp, meshes, param, grid, param["tau_w"])
    SS_stats["C_Unemp"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, mu_dist_tilde_Unemp, meshes, param, grid, param["tau_w"])

    ## 65. Store asset holdings in SS_stats by quintile
    SS_stats["A"] = sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde)
    SS_stats["A_B01"] = sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_B01)
    SS_stats["A_B10"] = sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_B10)
    SS_stats["A_B1"] = sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_B1)
    SS_stats["A_Q1"] = sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_Q1)
    SS_stats["A_Q2"] = sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_Q2)
    SS_stats["A_Q3"] = sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_Q3)
    SS_stats["A_Q4"] = sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_Q4)
    SS_stats["A_Q5"] = sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_Q5)
    SS_stats["A_P90"] = sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_P90)
    SS_stats["A_P99"] = sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_P99)
    SS_stats["A_P999"] = sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_P999)
    SS_stats["A_Ent"] = sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_Ent)
    SS_stats["A_Emp"] = sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_Emp)
    SS_stats["A_Unemp"] = sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_Unemp)
    SS_stats["A_all"] = sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_aux)
    SS_stats["A_all2"] = sum((AProb .* a_a_star .+ (1 .- AProb) .* meshes["a"]) .* mu_dist_tilde_full)

    SS_stats["B"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde)
    SS_stats["B_B01"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde_B01)
    SS_stats["B_B1"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde_B1)
    SS_stats["B_B10"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde_B10)
    SS_stats["B_Q1"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde_Q1)
    SS_stats["B_Q2"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde_Q2)
    SS_stats["B_Q3"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde_Q3)
    SS_stats["B_Q4"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde_Q4)
    SS_stats["B_Q5"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde_Q5)
    SS_stats["B_P90"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde_P90)
    SS_stats["B_P99"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde_P99)
    SS_stats["B_P999"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde_P999)
    SS_stats["B_Ent"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde_Ent)
    SS_stats["B_Emp"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde_Emp)
    SS_stats["B_Unemp"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde_Unemp)
    SS_stats["B_all"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde_aux)
    SS_stats["B_all2"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde_full)

    ## 66. Store wealth statistics in SS_stats
    SS_stats["w_all"] = sum(meshes["w"] .* mu_dist_tilde)
    SS_stats["w_B1"] = sum(meshes["w"] .* mu_dist_tilde_B1)
    SS_stats["w_B01"] = sum(meshes["w"] .* mu_dist_tilde_B01)
    SS_stats["w_B10"] = sum(meshes["w"] .* mu_dist_tilde_B10)
    SS_stats["w_Q1"] = sum(meshes["w"] .* mu_dist_tilde_Q1)
    SS_stats["w_Q2"] = sum(meshes["w"] .* mu_dist_tilde_Q2)
    SS_stats["w_Q3"] = sum(meshes["w"] .* mu_dist_tilde_Q3)
    SS_stats["w_Q4"] = sum(meshes["w"] .* mu_dist_tilde_Q4)
    SS_stats["w_Q5"] = sum(meshes["w"] .* mu_dist_tilde_Q5)
    SS_stats["w_P90"] = sum(meshes["w"] .* mu_dist_tilde_P90)
    SS_stats["w_P99"] = sum(meshes["w"] .* mu_dist_tilde_P99)
    SS_stats["w_P999"] = sum(meshes["w"] .* mu_dist_tilde_P999)

    SS_stats["wealth_B01"] = sum((meshes["a"] .+ meshes["b"]) .* mu_dist_tilde_B01) / sum(mu_dist_tilde_B01)
    SS_stats["wealth_B1"] = sum((meshes["a"] .+ meshes["b"]) .* mu_dist_tilde_B1) / sum(mu_dist_tilde_B1)
    SS_stats["wealth_B10"] = sum((meshes["a"] .+ meshes["b"]) .* mu_dist_tilde_B10) / sum(mu_dist_tilde_B10)
    SS_stats["wealth_Q1"] = sum((meshes["a"] .+ meshes["b"]) .* mu_dist_tilde_Q1) / sum(mu_dist_tilde_Q1)
    SS_stats["wealth_Q2"] = sum((meshes["a"] .+ meshes["b"]) .* mu_dist_tilde_Q2) / sum(mu_dist_tilde_Q2)
    SS_stats["wealth_Q3"] = sum((meshes["a"] .+ meshes["b"]) .* mu_dist_tilde_Q3) / sum(mu_dist_tilde_Q3)
    SS_stats["wealth_Q4"] = sum((meshes["a"] .+ meshes["b"]) .* mu_dist_tilde_Q4) / sum(mu_dist_tilde_Q4)
    SS_stats["wealth_Q5"] = sum((meshes["a"] .+ meshes["b"]) .* mu_dist_tilde_Q5) / sum(mu_dist_tilde_Q5)
    SS_stats["wealth_P90"] = sum((meshes["a"] .+ meshes["b"]) .* mu_dist_tilde_P90) / sum(mu_dist_tilde_P90)
    SS_stats["wealth_P99"] = sum((meshes["a"] .+ meshes["b"]) .* mu_dist_tilde_P99) / sum(mu_dist_tilde_P99)
    SS_stats["wealth_P999"] = sum((meshes["a"] .+ meshes["b"]) .* mu_dist_tilde_P999) / sum(mu_dist_tilde_P999)

    ## 67. Create income-based distribution masks
    # Compute steady-state income: labor + capital + interest + transfers
    income_ss = (WW_aux .* meshes_se_aux .+
                 SS_stats["r_a_aux"] .* meshes["a"] .+
                 (param["R_cb"]/param["pi_bar"] - 1) .* meshes["b"] .* (meshes["b"] .>= 0) .+
                 ((param["R_cb"] + param["Rprem"])/param["pi_bar"] - 1) .* meshes["b"] .* (meshes["b"] .< 0) .+
                 transfer_mesh_aux)

    SS_stats["mu_dist_I_B01"] = (income_ss .>= I0_ss) .* (income_ss .<= I01_ss) .* mu_dist
    SS_stats["mu_dist_I_B1"] = (income_ss .>= I0_ss) .* (income_ss .<= I1_ss) .* mu_dist
    SS_stats["mu_dist_I_B10"] = (income_ss .>= I0_ss) .* (income_ss .<= I10_ss) .* mu_dist
    SS_stats["mu_dist_I_Q1"] = (income_ss .>= I0_ss) .* (income_ss .<= I20_ss) .* mu_dist
    SS_stats["mu_dist_I_Q2"] = (income_ss .> I20_ss) .* (income_ss .<= I40_ss) .* mu_dist
    SS_stats["mu_dist_I_Q3"] = (income_ss .> I40_ss) .* (income_ss .<= I60_ss) .* mu_dist
    SS_stats["mu_dist_I_Q4"] = (income_ss .> I60_ss) .* (income_ss .<= I80_ss) .* mu_dist
    SS_stats["mu_dist_I_Q5"] = (income_ss .> I80_ss) .* (income_ss .<= I100_ss) .* mu_dist
    SS_stats["mu_dist_I_P90"] = (income_ss .> I90_ss) .* (income_ss .<= I100_ss) .* mu_dist
    SS_stats["mu_dist_I_P99"] = (income_ss .> I99_ss) .* (income_ss .<= I100_ss) .* mu_dist
    SS_stats["mu_dist_I_P999"] = (income_ss .> I999_ss) .* (income_ss .<= I100_ss) .* mu_dist
    SS_stats["mu_dist_I_B60"] = (income_ss .<= I60_ss) .* mu_dist

    ## 68. MPC calculations (Marginal Propensity to Consume)
    # Compute wage grids for MPC calculations
    WW_se = WW_aux[1, 1, :]  # Extract productivity-specific wages
    WW_se_mesh = WW_aux .* meshes_se_aux

    grid_se_aux = grid["se"]

    nb = Int(grid["nb"])
    na = Int(grid["na"])
    nse = Int(grid["nse"])

    # MPC with respect to liquid assets (b)
    MPC_a_b = zeros(nb, na, nse)
    MPC_n_b = zeros(nb, na, nse)

    # Compute gradients of consumption with respect to liquid assets
    b_return = param["R_cb"] / param["pi_cb"] * grid["b"] .+
               param["Rprem"] * grid["b"] .* (grid["b"] .< 0)
    dx = gradient(b_return)
    for kk in 1:na
        for hh in 1:nse
        #=    # Use central differences for gradient
            dc_a = diff(c_a_guess[:, kk, hh])
            db = diff(b_return)
            MPC_a_b[1:end-1, kk, hh] .= dc_a ./ db
            MPC_a_b[end, kk, hh] = MPC_a_b[end-1, kk, hh]  # Extrapolate last point

            dc_n = diff(c_n_guess[:, kk, hh])
            MPC_n_b[1:end-1, kk, hh] .= dc_n ./ db
            MPC_n_b[end, kk, hh] = MPC_n_b[end-1, kk, hh]  # Extrapolate last point
            =#
            dy_a = gradient(c_a_guess[:, kk, hh])
            dy_n = gradient(c_n_guess[:, kk, hh])
            MPC_a_b[:, kk, hh] .= dy_a ./ dx
            MPC_n_b[:, kk, hh] .= dy_n ./ dx
        end
    end

    # Adjust MPC by income-consumption ratio
    MPC_a_b = MPC_a_b .* (WW_se_mesh ./ c_a_guess)
    MPC_n_b = MPC_n_b .* (WW_se_mesh ./ c_n_guess)

    # MPC with respect to productivity shocks (se) - elasticity
    MPC_a_se = zeros(nb, na, nse)
    MPC_n_se = zeros(nb, na, nse)

    wage_grid = WW_se' .* grid_se_aux  # Wage at each productivity level
    log_wage_grid = log.(wage_grid)
#=
    for mm in 1:nb
        for kk in 1:na
            # Compute elasticity: d(log c) / d(log wage)
            dlog_c_a = diff(log.(c_a_guess[mm, kk, :]))
            dlog_wage = diff(log.(wage_grid[mm, kk, :]))
            MPC_a_se[mm, kk, 1:end-1] .= dlog_c_a ./ dlog_wage
            MPC_a_se[mm, kk, end] = MPC_a_se[mm, kk, end-1]  # Extrapolate

            dlog_c_n = diff(log.(c_n_guess[mm, kk, :]))
            MPC_n_se[mm, kk, 1:end-1] .= dlog_c_n ./ dlog_wage
            MPC_n_se[mm, kk, end] = MPC_n_se[mm, kk, end-1]  # Extrapolate
        end
    end
=#

for mm in 1:grid["nb"]
    for kk in 1:grid["na"]
        num_a = gradient(log.(vec(c_a_guess[mm, kk, :])))
        num_n = gradient(log.(vec(c_n_guess[mm, kk, :])))

        denom = gradient(log.(WW_se .* grid_se_aux))

        MPC_a_se[mm, kk, :] .= num_a ./ denom
        MPC_n_se[mm, kk, :] .= num_n ./ denom
    end
end

    # Aggregate MPCs
    MPC_se = AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se
    MPC_b = AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b

    # Expected MPC with respect to productivity by quintile
    EMPC_se = vec(mu_dist_tilde)' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(mu_dist_tilde)
    EMPC_se_B01 = vec(mu_dist_tilde_B01)' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(mu_dist_tilde_B01)
    EMPC_se_B1 = vec(mu_dist_tilde_B1)' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(mu_dist_tilde_B1)
    EMPC_se_B10 = vec(mu_dist_tilde_B10)' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(mu_dist_tilde_B10)
    EMPC_se_Q1 = vec(mu_dist_tilde_Q1)' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(mu_dist_tilde_Q1)
    EMPC_se_Q2 = vec(mu_dist_tilde_Q2)' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(mu_dist_tilde_Q2)
    EMPC_se_Q3 = vec(mu_dist_tilde_Q3)' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(mu_dist_tilde_Q3)
    EMPC_se_Q4 = vec(mu_dist_tilde_Q4)' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(mu_dist_tilde_Q4)
    EMPC_se_Q5 = vec(mu_dist_tilde_Q5)' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(mu_dist_tilde_Q5)
    EMPC_se_P90 = vec(mu_dist_tilde_P90)' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(mu_dist_tilde_P90)
    EMPC_se_P99 = vec(mu_dist_tilde_P99)' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(mu_dist_tilde_P99)
    EMPC_se_P999 = vec(mu_dist_tilde_P999)' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(mu_dist_tilde_P999)
    EMPC_se_Ent = vec(mu_dist_tilde_Ent)' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(mu_dist_tilde_Ent)
    EMPC_se_Emp = vec(mu_dist_tilde_Emp)' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(mu_dist_tilde_Emp)
    EMPC_se_Unemp = vec(mu_dist_tilde_Unemp)' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(mu_dist_tilde_Unemp)

    # Expected MPC with respect to liquid assets by quintile
    EMPC_b = vec(mu_dist_tilde)' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(mu_dist_tilde)
    EMPC_b_Q1 = vec(mu_dist_tilde_Q1)' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(mu_dist_tilde_Q1)
    EMPC_b_Q2 = vec(mu_dist_tilde_Q2)' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(mu_dist_tilde_Q2)
    EMPC_b_Q3 = vec(mu_dist_tilde_Q3)' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(mu_dist_tilde_Q3)
    EMPC_b_Q4 = vec(mu_dist_tilde_Q4)' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(mu_dist_tilde_Q4)
    EMPC_b_Q5 = vec(mu_dist_tilde_Q5)' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(mu_dist_tilde_Q5)
    EMPC_b_P90 = vec(mu_dist_tilde_P90)' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(mu_dist_tilde_P90)
    EMPC_b_P99 = vec(mu_dist_tilde_P99)' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(mu_dist_tilde_P99)
    EMPC_b_P999 = vec(mu_dist_tilde_P999)' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(mu_dist_tilde_P999)
    EMPC_b_Ent = vec(mu_dist_tilde_Ent)' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(mu_dist_tilde_Ent)
    EMPC_b_Emp = vec(mu_dist_tilde_Emp)' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(mu_dist_tilde_Emp)
    EMPC_b_Unemp = vec(mu_dist_tilde_Unemp)' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(mu_dist_tilde_Unemp)

    # Store MPC results in anal_stats
    anal_stats["MPC_se"] = MPC_se
    anal_stats["MPC_b"] = MPC_b
    anal_stats["MPC_a_b"] = MPC_a_b
    anal_stats["EMPC_se"] = EMPC_se
    anal_stats["EMPC_se_Q1"] = EMPC_se_Q1
    anal_stats["EMPC_se_Q2"] = EMPC_se_Q2
    anal_stats["EMPC_se_Q3"] = EMPC_se_Q3
    anal_stats["EMPC_se_Q4"] = EMPC_se_Q4
    anal_stats["EMPC_se_Q5"] = EMPC_se_Q5
    anal_stats["EMPC_se_P90"] = EMPC_se_P90
    anal_stats["EMPC_se_P99"] = EMPC_se_P99
    anal_stats["EMPC_se_P999"] = EMPC_se_P999
    anal_stats["EMPC_b"] = EMPC_b
    anal_stats["EMPC_b_Q1"] = EMPC_b_Q1
    anal_stats["EMPC_b_Q2"] = EMPC_b_Q2
    anal_stats["EMPC_b_Q3"] = EMPC_b_Q3
    anal_stats["EMPC_b_Q4"] = EMPC_b_Q4
    anal_stats["EMPC_b_Q5"] = EMPC_b_Q5
    anal_stats["EMPC_b_P90"] = EMPC_b_P90
    anal_stats["EMPC_b_P99"] = EMPC_b_P99
    anal_stats["EMPC_b_P999"] = EMPC_b_P999

    ## 69. MRS (Marginal Rate of Substitution) calculations by quintile
    # These calculations use MRS_a_aux2 and MRS_n_aux2 computed in Cal_SS_stats
    if haskey(SS_stats, "MRS_a_aux2") && haskey(SS_stats, "MRS_n_aux2")
        MRS_a_aux2 = SS_stats["MRS_a_aux2"]
        MRS_n_aux2 = SS_stats["MRS_n_aux2"]

        MRSs = vec(mu_dist_tilde)' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(mu_dist_tilde)
        MRSs_Q1 = vec(mu_dist_tilde_Q1)' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(mu_dist_tilde_Q1)
        MRSs_Q2 = vec(mu_dist_tilde_Q2)' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(mu_dist_tilde_Q2)
        MRSs_Q3 = vec(mu_dist_tilde_Q3)' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(mu_dist_tilde_Q3)
        MRSs_Q4 = vec(mu_dist_tilde_Q4)' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(mu_dist_tilde_Q4)
        MRSs_Q5 = vec(mu_dist_tilde_Q5)' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(mu_dist_tilde_Q5)
        MRSs_P90 = vec(mu_dist_tilde_P90)' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(mu_dist_tilde_P90)
        MRSs_P99 = vec(mu_dist_tilde_P99)' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(mu_dist_tilde_P99)
        MRSs_P999 = vec(mu_dist_tilde_P999)' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(mu_dist_tilde_P999)
        MRSs_Ent = vec(mu_dist_tilde_Ent)' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(mu_dist_tilde_Ent)
        MRSs_Emp = vec(mu_dist_tilde_Emp)' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(mu_dist_tilde_Emp)
        MRSs_Unemp = vec(mu_dist_tilde_Unemp)' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(mu_dist_tilde_Unemp)

        anal_stats["MRSs"] = MRSs
        anal_stats["MRSs_Q1"] = MRSs_Q1
        anal_stats["MRSs_Q2"] = MRSs_Q2
        anal_stats["MRSs_Q3"] = MRSs_Q3
        anal_stats["MRSs_Q4"] = MRSs_Q4
        anal_stats["MRSs_Q5"] = MRSs_Q5
        anal_stats["MRSs_P90"] = MRSs_P90
        anal_stats["MRSs_P99"] = MRSs_P99
        anal_stats["MRSs_P999"] = MRSs_P999
        anal_stats["MRSs_Ent"] = MRSs_Ent
        anal_stats["MRSs_Emp"] = MRSs_Emp
        anal_stats["MRSs_Unemp"] = MRSs_Unemp
    end

    println("Distributional statistics computed successfully.")
    println("Type of anal_stats: ", typeof(anal_stats))
    println("Type of SS_stats: ", typeof(SS_stats))
    println("Type of Q_dists: ", typeof(Q_dists))
    return anal_stats, SS_stats, Q_dists
end
