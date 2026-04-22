"""
    steady_anal(grid, param, meshes, SS_stats, AProb, mu_dist, n, H_tilde,
                b_n_star, b_a_star, a_a_star, c_n_guess, c_a_guess)

Compute distributional analysis and quintile statistics for the steady state.
Translated from MATLAB steady_anal.m

# Arguments
- `grid`: Grid structure containing state space grids
- `param`: Parameter structure
- `meshes`: Mesh grids for state variables
- `SS_stats`: Steady state statistics from Cal_SS_stats (will be modified)
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
Tuple of (anal_stats, SS_stats, Q_dists) dictionaries
"""
function steady_anal(grid, param, meshes, SS_stats, AProb, mu_dist, n, H_tilde,
                    b_n_star, b_a_star, a_a_star, c_n_guess, c_a_guess)

    # Get grid dimensions
    nb_val = Int(grid["nb"])
    na_val = Int(grid["na"])
    nse_val = Int(grid["nse"])
    ns_val = Int(grid["ns"])

    # Initialize output dictionaries
    anal_stats = Dict{String, Any}()
    Q_dists = Dict{String, Any}()

    ## 1. Create auxiliary productivity grids for income decomposition
    # grid.se_aux = [grid.s, min(param.b_ratio*grid.s, grid.s_bar), 1]
    grid_s = grid["s"]  # Should be 1×ns
    
    # Ensure grid_s is a row vector
    if ndims(grid_s) == 2 && size(grid_s, 1) == 1
        grid_s = vec(grid_s)
    end
    
    grid_se_aux = vcat(grid_s, min.(param["b_ratio"] .* grid_s, grid["s_bar"]), [1])
    grid_se1_aux = vcat(grid_s, zeros(ns_val), [0])
    grid_se2_aux = vcat(zeros(ns_val), min.(param["b_ratio"] .* grid_s, grid["s_bar"]), [0])
    grid_se3_aux = vcat(zeros(2*ns_val), [1])

    # [~, ~, meshes.se_aux] = ndgrid(grid.b, grid.a, grid.se_aux)
    _, _, meshes_se_aux = ndgrid(grid["b"], grid["a"], grid_se_aux)
    _, _, meshes_se1_aux = ndgrid(grid["b"], grid["a"], grid_se1_aux)
    _, _, meshes_se2_aux = ndgrid(grid["b"], grid["a"], grid_se2_aux)
    _, _, meshes_se3_aux = ndgrid(grid["b"], grid["a"], grid_se3_aux)

    # Store in meshes dict
    meshes["se_aux"] = meshes_se_aux
    meshes["se1_aux"] = meshes_se1_aux
    meshes["se2_aux"] = meshes_se2_aux
    meshes["se3_aux"] = meshes_se3_aux

    ## 2. Compute after-tax wages and income
    # auxWW = ones(grid.nb, grid.na, grid.nse)
    auxWW = ones(nb_val, na_val, nse_val)
    # auxWW(:,:,grid.ns+1:end) = zeros(grid.nb, grid.na, grid.ns+1)
    auxWW[:, :, ns_val+1:end] .= 0
    # auxWW = auxWW.*(meshes.se.^(param.b_aux-1))
    auxWW = auxWW .* (meshes["se"] .^ (param["b_aux"] - 1))

    NW_aux = n * param["w_bar"]
    WW_aux2 = (1 - param["tau_w"]) * (NW_aux * ones(nb_val, na_val, nse_val))
    WW_aux = (1 - param["tau_w"]) * (NW_aux * ones(nb_val, na_val, nse_val) .+
             param["b_share"] * SS_stats["Profit"] / ((1 - SS_stats["u"]) * (1 - param["Eshare"]) * grid["s2_bar"]) .* auxWW)
    # WW_aux(:,:,end) = (1-param.tau_w)*(param.Eratio*SS_stats.Profit)/param.Eshare*ones(grid.nb,grid.na)
    WW_aux[:, :, end] .= (1 - param["tau_w"]) * (param["Eratio"] * SS_stats["Profit"]) / param["Eshare"] * ones(nb_val, na_val)

    # inc.transfer = param.LT*ones(grid.nb, grid.na, grid.nse)
    inc_transfer = param["LT"] * ones(nb_val, na_val, nse_val)

    ## 3. Compute total income and its components
    # Bond return calculations
    bond_return_pos = ((param["R_cb"] / (1 - param["death_rate"]) * param["b_a_aux"]) / param["pi_cb"] - 1)
    bond_return_neg = (((param["R_cb"] + param["Rprem"]) / (1 - param["death_rate"]) * param["b_a_aux"]) / param["pi_cb"] - 1)

    total_income_mesh_aux = WW_aux .* meshes_se_aux .+
                           SS_stats["r_a_aux"] .* meshes["a"] .+
                           bond_return_pos .* meshes["b"] .* (meshes["b"] .>= 0) .+
                           bond_return_neg .* meshes["b"] .* (meshes["b"] .< 0) .+
                           inc_transfer

    labor_income_mesh_aux = WW_aux .* meshes_se1_aux
    ubincome_mesh_aux = WW_aux .* meshes_se2_aux
    profit_income_mesh_aux = WW_aux .* meshes_se3_aux
    a_income_mesh_aux = SS_stats["r_a_aux"] .* meshes["a"]
    b_income_mesh_aux = bond_return_pos .* meshes["b"] .* (meshes["b"] .>= 0) .+
                       bond_return_neg .* meshes["b"] .* (meshes["b"] .< 0)
    transfer_mesh_aux = inc_transfer

    ## 4. Apply productivity transition to distribution
    # mu_dist_tilde = H_tilde*mu_dist(:)
    mu_dist_tilde = H_tilde * vec(mu_dist)
    mu_dist_tilde = reshape(mu_dist_tilde, (nb_val, na_val, nse_val))

    ## 5. Compute marginal distributions
    mu_dist_b = dropdims(sum(sum(mu_dist, dims=2), dims=3), dims=(2,3))
    mu_dist_a = dropdims(sum(sum(mu_dist, dims=1), dims=3), dims=(1,3))
    mu_dist_se = dropdims(sum(sum(mu_dist, dims=1), dims=2), dims=(1,2))

    ## 6. Compute CDFs for wealth, income, liquid, and illiquid assets
    # Create mesh grids for b and a (like MATLAB's meshgrid)
    # [mesh.a, mesh.b] = meshgrid(grid.a, grid.b)
    # Note: MATLAB meshgrid swaps dimensions, Julia ndgrid doesn't
    # For meshgrid behavior: (rows vary b, cols vary a)
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
    total_wealth = vec(mesh_a) .* param["q"] .+ vec(mesh_b)
    IX = sortperm(total_wealth)
    total_wealth = total_wealth[IX]
    total_wealth_pdf = vec(mu_redux)[IX]
    total_wealth_cdf = cumsum(total_wealth_pdf)

    # Compute wealth thresholds
    w0 = total_wealth[1]
    w1 = total_wealth[sum(total_wealth_cdf .< 0.01) + 1]
    w10 = total_wealth[sum(total_wealth_cdf .< 0.1) + 1]
    w20 = total_wealth[sum(total_wealth_cdf .< 0.2) + 1]
    w25 = total_wealth[sum(total_wealth_cdf .< 0.25) + 1]
    w30 = total_wealth[sum(total_wealth_cdf .< 0.3) + 1]
    w40 = total_wealth[sum(total_wealth_cdf .< 0.4) + 1]
    w50 = total_wealth[sum(total_wealth_cdf .< 0.5) + 1]
    w60 = total_wealth[sum(total_wealth_cdf .< 0.6) + 1]
    w70 = total_wealth[sum(total_wealth_cdf .< 0.7) + 1]
    w80 = total_wealth[sum(total_wealth_cdf .< 0.8) + 1]
    w90 = total_wealth[sum(total_wealth_cdf .< 0.9) + 1]
    w99 = total_wealth[sum(total_wealth_cdf .< 0.99) + 1]
    w999 = total_wealth[sum(total_wealth_cdf .< 0.999) + 1]
    w100 = total_wealth[end]

    # Income distribution
    # meshes.income_ss will be computed later, need to compute income_sort first
    # For now, compute using total_income_mesh_aux
    income_mesh_flat = vec(total_income_mesh_aux)
    mu_dist_flat = vec(mu_dist)
    income_IX = sortperm(income_mesh_flat)
    income_sort = income_mesh_flat[income_IX]
    income_pdf = mu_dist_flat[income_IX]
    income_cdf = cumsum(income_pdf)

    I0_ss = income_sort[1]
    I01_ss = income_sort[sum(income_cdf .< 0.001) + 1]
    I1_ss = income_sort[sum(income_cdf .< 0.01) + 1]
    I10_ss = income_sort[sum(income_cdf .< 0.1) + 1]
    I20_ss = income_sort[sum(income_cdf .< 0.2) + 1]
    I25_ss = income_sort[sum(income_cdf .< 0.25) + 1]
    I30_ss = income_sort[sum(income_cdf .< 0.3) + 1]
    I40_ss = income_sort[sum(income_cdf .< 0.4) + 1]
    I50_ss = income_sort[sum(income_cdf .< 0.5) + 1]
    I60_ss = income_sort[sum(income_cdf .< 0.6) + 1]
    I70_ss = income_sort[sum(income_cdf .< 0.7) + 1]
    I80_ss = income_sort[sum(income_cdf .< 0.8) + 1]
    I90_ss = income_sort[sum(income_cdf .< 0.9) + 1]
    I99_ss = income_sort[sum(income_cdf .< 0.99) + 1]
    I999_ss = income_sort[sum(income_cdf .< 0.999) + 1]
    I100_ss = income_sort[end]

    # Liquid assets distribution
    liquid_sort_vec = vec(mesh_b)
    liquid_IX = sortperm(liquid_sort_vec)
    liquid_sort = liquid_sort_vec[liquid_IX]
    liquid_pdf = vec(mu_redux)[liquid_IX]
    liquid_cdf = cumsum(liquid_pdf)

    LI0 = liquid_sort[1]
    LI1 = liquid_sort[sum(liquid_cdf .< 0.01) + 1]
    LI10 = liquid_sort[sum(liquid_cdf .< 0.1) + 1]
    LI20 = liquid_sort[sum(liquid_cdf .< 0.2) + 1]
    LI25 = liquid_sort[sum(liquid_cdf .< 0.25) + 1]
    LI30 = liquid_sort[sum(liquid_cdf .< 0.3) + 1]
    LI40 = liquid_sort[sum(liquid_cdf .< 0.4) + 1]
    LI50 = liquid_sort[sum(liquid_cdf .< 0.5) + 1]
    LI60 = liquid_sort[sum(liquid_cdf .< 0.6) + 1]
    LI70 = liquid_sort[sum(liquid_cdf .< 0.7) + 1]
    LI80 = liquid_sort[sum(liquid_cdf .< 0.8) + 1]
    LI90 = liquid_sort[sum(liquid_cdf .< 0.9) + 1]
    LI99 = liquid_sort[sum(liquid_cdf .< 0.99) + 1]
    LI999 = liquid_sort[sum(liquid_cdf .< 0.999) + 1]
    LI100 = liquid_sort[end]

    # Illiquid assets distribution
    illiquid_sort_vec = vec(mesh_a)
    illiquid_IX = sortperm(illiquid_sort_vec)
    illiquid_sort = illiquid_sort_vec[illiquid_IX]
    illiquid_pdf = vec(mu_redux)[illiquid_IX]
    illiquid_cdf = cumsum(illiquid_pdf)

    IL0 = illiquid_sort[1]
    IL1 = illiquid_sort[sum(illiquid_cdf .< 0.01) + 1]
    IL10 = illiquid_sort[sum(illiquid_cdf .< 0.1) + 1]
    IL20 = illiquid_sort[sum(illiquid_cdf .< 0.2) + 1]
    IL25 = illiquid_sort[sum(illiquid_cdf .< 0.25) + 1]
    IL30 = illiquid_sort[sum(illiquid_cdf .< 0.3) + 1]
    IL40 = illiquid_sort[sum(illiquid_cdf .< 0.4) + 1]
    IL50 = illiquid_sort[sum(illiquid_cdf .< 0.5) + 1]
    IL60 = illiquid_sort[sum(illiquid_cdf .< 0.6) + 1]
    IL70 = illiquid_sort[sum(illiquid_cdf .< 0.7) + 1]
    IL80 = illiquid_sort[sum(illiquid_cdf .< 0.8) + 1]
    IL90 = illiquid_sort[sum(illiquid_cdf .< 0.9) + 1]
    IL99 = illiquid_sort[sum(illiquid_cdf .< 0.99) + 1]
    IL999 = illiquid_sort[sum(illiquid_cdf .< 0.999) + 1]
    IL100 = illiquid_sort[end]

    ## 7. Store wealth thresholds in SS_stats
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

    # meshes.w = param.q*meshes.a + meshes.b
    meshes["w"] = param["q"] .* meshes["a"] .+ meshes["b"]

    ## 8. Create distribution masks by wealth quintiles
    SS_stats["mu_dist"] = mu_dist
    SS_stats["mu_dist_B01"] = ((meshes["w"] .>= w0) .* (meshes["w"] .<= w0)) .* mu_dist
    SS_stats["mu_dist_B1"] = ((meshes["w"] .>= w0) .* (meshes["w"] .<= w1)) .* mu_dist
    SS_stats["mu_dist_B10"] = ((meshes["w"] .>= w0) .* (meshes["w"] .<= w10)) .* mu_dist
    SS_stats["mu_dist_Q1"] = ((meshes["w"] .>= w0) .* (meshes["w"] .<= w20)) .* mu_dist
    SS_stats["mu_dist_Q2"] = ((meshes["w"] .> w20) .* (meshes["w"] .<= w40)) .* mu_dist
    SS_stats["mu_dist_Q3"] = ((meshes["w"] .> w40) .* (meshes["w"] .<= w60)) .* mu_dist
    SS_stats["mu_dist_Q4"] = ((meshes["w"] .> w60) .* (meshes["w"] .<= w80)) .* mu_dist
    SS_stats["mu_dist_Q5"] = ((meshes["w"] .> w80) .* (meshes["w"] .<= w100)) .* mu_dist
    SS_stats["mu_dist_P90"] = ((meshes["w"] .> w90) .* (meshes["w"] .<= w100)) .* mu_dist
    SS_stats["mu_dist_P99"] = ((meshes["w"] .> w99) .* (meshes["w"] .<= w100)) .* mu_dist
    SS_stats["mu_dist_P999"] = ((meshes["w"] .> w999) .* (meshes["w"] .<= w100)) .* mu_dist
    SS_stats["mu_dist_Emp"] = zeros(nb_val, na_val, nse_val)
    SS_stats["mu_dist_Emp"][:, :, 1:ns_val] = mu_dist[:, :, 1:ns_val]
    SS_stats["mu_dist_Unemp"] = zeros(nb_val, na_val, nse_val)
    SS_stats["mu_dist_Unemp"][:, :, ns_val+1:2*ns_val] = mu_dist[:, :, ns_val+1:2*ns_val]
    SS_stats["mu_dist_Ent"] = (meshes_se3_aux .== 1) .* mu_dist
    SS_stats["mu_dist_B60"] = (meshes["w"] .<= w60) .* mu_dist

    ## 9. Create distribution masks using mu_dist_tilde
    SS_stats["mu_dist_tilde"] = mu_dist_tilde
    SS_stats["mu_dist_tilde_B01"] = ((meshes["w"] .>= w0) .* (meshes["w"] .<= w0)) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_B1"] = ((meshes["w"] .>= w0) .* (meshes["w"] .<= w1)) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_B10"] = ((meshes["w"] .>= w0) .* (meshes["w"] .<= w10)) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_Q1"] = ((meshes["w"] .>= w0) .* (meshes["w"] .<= w20)) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_Q2"] = ((meshes["w"] .> w20) .* (meshes["w"] .<= w40)) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_Q3"] = ((meshes["w"] .> w40) .* (meshes["w"] .<= w60)) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_Q4"] = ((meshes["w"] .> w60) .* (meshes["w"] .<= w80)) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_Q5"] = ((meshes["w"] .> w80) .* (meshes["w"] .<= w100)) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_P90"] = ((meshes["w"] .> w90) .* (meshes["w"] .<= w100)) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_P99"] = ((meshes["w"] .> w99) .* (meshes["w"] .<= w100)) .* mu_dist_tilde
    SS_stats["mu_dist_tilde_P999"] = ((meshes["w"] .> w999) .* (meshes["w"] .<= w100)) .* mu_dist_tilde

    # Apply H_tilde transition to employment status distributions
    SS_stats["mu_dist_tilde_Ent"] = H_tilde * vec(SS_stats["mu_dist_Ent"])
    SS_stats["mu_dist_tilde_Emp"] = H_tilde * vec(SS_stats["mu_dist_Emp"])
    SS_stats["mu_dist_tilde_Unemp"] = H_tilde * vec(SS_stats["mu_dist_Unemp"])

    SS_stats["mu_dist_tilde_Ent"] = reshape(SS_stats["mu_dist_tilde_Ent"], (nb_val, na_val, nse_val))
    SS_stats["mu_dist_tilde_Emp"] = reshape(SS_stats["mu_dist_tilde_Emp"], (nb_val, na_val, nse_val))
    SS_stats["mu_dist_tilde_Unemp"] = reshape(SS_stats["mu_dist_tilde_Unemp"], (nb_val, na_val, nse_val))

    SS_stats["mu_dist_tilde_B60"] = (meshes["w"] .<= w60) .* mu_dist_tilde

    ## 10. Create Q_dists structure with income-based quintiles
    # Note: meshes.income_ss computed later, using total_income_mesh_aux for now
    meshes["income_ss"] = total_income_mesh_aux  # Will be recomputed later with correct formula

    Q_dists["mu_dist_I_B01"] = ((total_income_mesh_aux .>= I0_ss) .* (total_income_mesh_aux .<= I01_ss)) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_B1"] = ((total_income_mesh_aux .>= I0_ss) .* (total_income_mesh_aux .<= I1_ss)) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_B10"] = ((total_income_mesh_aux .>= I0_ss) .* (total_income_mesh_aux .<= I10_ss)) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_Q1"] = ((total_income_mesh_aux .>= I0_ss) .* (total_income_mesh_aux .<= I20_ss)) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_Q2"] = ((total_income_mesh_aux .> I20_ss) .* (total_income_mesh_aux .<= I40_ss)) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_Q3"] = ((total_income_mesh_aux .> I40_ss) .* (total_income_mesh_aux .<= I60_ss)) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_Q4"] = ((total_income_mesh_aux .> I60_ss) .* (total_income_mesh_aux .<= I80_ss)) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_Q5"] = ((total_income_mesh_aux .> I80_ss) .* (total_income_mesh_aux .<= I100_ss)) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_P90"] = ((total_income_mesh_aux .> I90_ss) .* (total_income_mesh_aux .<= I100_ss)) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_P99"] = ((total_income_mesh_aux .> I99_ss) .* (total_income_mesh_aux .<= I100_ss)) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_P999"] = ((total_income_mesh_aux .> I999_ss) .* (total_income_mesh_aux .<= I100_ss)) .* SS_stats["mu_dist"]
    Q_dists["mu_dist_I_B60"] = (total_income_mesh_aux .<= I60_ss) .* SS_stats["mu_dist"]

    # Continue with Q_dists assignments (wealth-based)
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

    # Repeat assignments (appears twice in MATLAB code)
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

    Q_dists["mu_dist_B01"] = SS_stats["mu_dist_B01"]
    Q_dists["mu_dist_B1"] = SS_stats["mu_dist_B1"]
    Q_dists["mu_dist_B10"] = SS_stats["mu_dist_B10"]

    Q_dists["mu_dist_tilde_B01"] = SS_stats["mu_dist_tilde_B01"]
    Q_dists["mu_dist_tilde_B1"] = SS_stats["mu_dist_tilde_B1"]
    Q_dists["mu_dist_tilde_B10"] = SS_stats["mu_dist_tilde_B10"]

    # Marginal distributions for Q_dists
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

    Q_dists["mu_dist_a"] = mu_dist_a
    Q_dists["mu_dist_a_Q1"] = vec(dropdims(sum(sum(SS_stats["mu_dist_Q1"], dims=1), dims=3), dims=(1,3))')
    Q_dists["mu_dist_a_Q2"] = vec(dropdims(sum(sum(SS_stats["mu_dist_Q2"], dims=1), dims=3), dims=(1,3))')
    Q_dists["mu_dist_a_Q3"] = vec(dropdims(sum(sum(SS_stats["mu_dist_Q3"], dims=1), dims=3), dims=(1,3))')
    Q_dists["mu_dist_a_Q4"] = vec(dropdims(sum(sum(SS_stats["mu_dist_Q4"], dims=1), dims=3), dims=(1,3))')
    Q_dists["mu_dist_a_Q5"] = vec(dropdims(sum(sum(SS_stats["mu_dist_Q5"], dims=1), dims=3), dims=(1,3))')
    Q_dists["mu_dist_a_P90"] = vec(dropdims(sum(sum(SS_stats["mu_dist_P90"], dims=1), dims=3), dims=(1,3))')
    Q_dists["mu_dist_a_P99"] = vec(dropdims(sum(sum(SS_stats["mu_dist_P99"], dims=1), dims=3), dims=(1,3))')
    Q_dists["mu_dist_a_P999"] = vec(dropdims(sum(sum(SS_stats["mu_dist_P999"], dims=1), dims=3), dims=(1,3))')
    Q_dists["mu_dist_a_Ent"] = vec(dropdims(sum(sum(SS_stats["mu_dist_Ent"], dims=1), dims=3), dims=(1,3))')
    Q_dists["mu_dist_a_Emp"] = vec(dropdims(sum(sum(SS_stats["mu_dist_Emp"], dims=1), dims=3), dims=(1,3))')
    Q_dists["mu_dist_a_Unemp"] = vec(dropdims(sum(sum(SS_stats["mu_dist_Unemp"], dims=1), dims=3), dims=(1,3))')
    Q_dists["mu_dist_a_B60"] = vec(dropdims(sum(sum(SS_stats["mu_dist_B60"], dims=1), dims=3), dims=(1,3))')

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

    # mu_dist_tilde assignments
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

    ## 11. Compute Gini coefficients
    SS_stats["NegNetWorth"] = sum((total_wealth .< 0) .* total_wealth_pdf)
    mean_total_wealth = sum(total_wealth .* total_wealth_pdf)
    std_total_wealth = sqrt(sum(total_wealth .^ 2 .* total_wealth_pdf) - mean_total_wealth^2)

    # Wealth Gini
    S_W = cumsum(total_wealth_pdf .* total_wealth)'
    S_W = vcat([0.0], vec(S_W))
    SS_stats["GiniW"] = 1 - (sum(total_wealth_pdf .* (S_W[1:end-1] .+ S_W[2:end])) / S_W[end])

    # Liquid Gini (using already computed liquid_sort, liquid_pdf, liquid_cdf)
    SS_stats["Negliquid"] = sum((liquid_sort .< 0) .* liquid_pdf)
    S_LI = cumsum(liquid_pdf .* liquid_sort)'
    S_LI = vcat([0.0], vec(S_LI))
    SS_stats["GiniLI"] = 1 - (sum(liquid_pdf .* (S_LI[1:end-1] .+ S_LI[2:end])) / S_LI[end])

    # Illiquid Gini
    S_IL = cumsum(illiquid_pdf .* illiquid_sort)'
    S_IL = vcat([0.0], vec(S_IL))
    SS_stats["GiniIL"] = 1 - (sum(illiquid_pdf .* (S_IL[1:end-1] .+ S_IL[2:end])) / S_IL[end])

    # Note: MATLAB code has duplicate calculations - using S_NW = S_W, so same GiniW
    S_NW = S_W
    SS_stats["GiniW"] = 1 - (sum(total_wealth_pdf .* (S_NW[1:end-1] .+ S_NW[2:end])) / S_NW[end])

    ## 12. Compute wealth statistics by quintile
    # Use Q_dists instead of SS_stats for consistency with MATLAB
    w_all_sum = sum(meshes["w"] .* Q_dists["mu_dist_tilde"])
    w_all_denom = sum(Q_dists["mu_dist_tilde"])
    w_all = w_all_sum / (w_all_denom > 0 ? w_all_denom : 1.0)
    
    w_B60_sum = sum(meshes["w"] .* Q_dists["mu_dist_tilde_B60"])
    w_B60_denom = sum(Q_dists["mu_dist_tilde_B60"])
    w_B60 = w_B60_sum / (w_B60_denom > 0 ? w_B60_denom : 1.0)
    
    w_Q1_sum = sum(meshes["w"] .* Q_dists["mu_dist_tilde_Q1"])
    w_Q1_denom = sum(Q_dists["mu_dist_tilde_Q1"])
    w_Q1 = w_Q1_sum / (w_Q1_denom > 0 ? w_Q1_denom : 1.0)
    
    w_Q2_sum = sum(meshes["w"] .* Q_dists["mu_dist_tilde_Q2"])
    w_Q2_denom = sum(Q_dists["mu_dist_tilde_Q2"])
    w_Q2 = w_Q2_sum / (w_Q2_denom > 0 ? w_Q2_denom : 1.0)
    
    w_Q3_sum = sum(meshes["w"] .* Q_dists["mu_dist_tilde_Q3"])
    w_Q3_denom = sum(Q_dists["mu_dist_tilde_Q3"])
    w_Q3 = w_Q3_sum / (w_Q3_denom > 0 ? w_Q3_denom : 1.0)
    
    w_Q4_sum = sum(meshes["w"] .* Q_dists["mu_dist_tilde_Q4"])
    w_Q4_denom = sum(Q_dists["mu_dist_tilde_Q4"])
    w_Q4 = w_Q4_sum / (w_Q4_denom > 0 ? w_Q4_denom : 1.0)
    
    w_Q5_sum = sum(meshes["w"] .* Q_dists["mu_dist_tilde_Q5"])
    w_Q5_denom = sum(Q_dists["mu_dist_tilde_Q5"])
    w_Q5 = w_Q5_sum / (w_Q5_denom > 0 ? w_Q5_denom : 1.0)
    
    w_P90_sum = sum(meshes["w"] .* Q_dists["mu_dist_tilde_P90"])
    w_P90_denom = sum(Q_dists["mu_dist_tilde_P90"])
    w_P90 = w_P90_sum / (w_P90_denom > 0 ? w_P90_denom : 1.0)
    
    w_P99_sum = sum(meshes["w"] .* Q_dists["mu_dist_tilde_P99"])
    w_P99_denom = sum(Q_dists["mu_dist_tilde_P99"])
    w_P99 = w_P99_sum / (w_P99_denom > 0 ? w_P99_denom : 1.0)
    
    w_P999_sum = sum(meshes["w"] .* Q_dists["mu_dist_tilde_P999"])
    w_P999_denom = sum(Q_dists["mu_dist_tilde_P999"])
    w_P999 = w_P999_sum / (w_P999_denom > 0 ? w_P999_denom : 1.0)
    
    w_Ent_sum = sum(meshes["w"] .* Q_dists["mu_dist_tilde_Ent"])
    w_Ent_denom = sum(Q_dists["mu_dist_tilde_Ent"])
    w_Ent = w_Ent_sum / (w_Ent_denom > 0 ? w_Ent_denom : 1.0)

    ## 13. Compute income statistics by quintile
    # Total income (same formula as total_income_mesh_aux)
    income_all = sum(total_income_mesh_aux .* mu_dist_tilde) / sum(mu_dist_tilde)
    income_B01 = sum(total_income_mesh_aux .* SS_stats["mu_dist_tilde_B01"]) / sum(SS_stats["mu_dist_tilde_B01"])
    income_B1 = sum(total_income_mesh_aux .* SS_stats["mu_dist_tilde_B1"]) / sum(SS_stats["mu_dist_tilde_B1"])
    income_B10 = sum(total_income_mesh_aux .* SS_stats["mu_dist_tilde_B10"]) / sum(SS_stats["mu_dist_tilde_B10"])
    income_B60 = sum(total_income_mesh_aux .* SS_stats["mu_dist_tilde_B60"]) / sum(SS_stats["mu_dist_tilde_B60"])
    income_Q1 = sum(total_income_mesh_aux .* SS_stats["mu_dist_tilde_Q1"]) / sum(SS_stats["mu_dist_tilde_Q1"])
    income_Q2 = sum(total_income_mesh_aux .* SS_stats["mu_dist_tilde_Q2"]) / sum(SS_stats["mu_dist_tilde_Q2"])
    income_Q3 = sum(total_income_mesh_aux .* SS_stats["mu_dist_tilde_Q3"]) / sum(SS_stats["mu_dist_tilde_Q3"])
    income_Q4 = sum(total_income_mesh_aux .* SS_stats["mu_dist_tilde_Q4"]) / sum(SS_stats["mu_dist_tilde_Q4"])
    income_Q5 = sum(total_income_mesh_aux .* SS_stats["mu_dist_tilde_Q5"]) / sum(SS_stats["mu_dist_tilde_Q5"])
    income_P90 = sum(total_income_mesh_aux .* SS_stats["mu_dist_tilde_P90"]) / sum(SS_stats["mu_dist_tilde_P90"])
    income_P99 = sum(total_income_mesh_aux .* SS_stats["mu_dist_tilde_P99"]) / sum(SS_stats["mu_dist_tilde_P99"])
    income_P999 = sum(total_income_mesh_aux .* SS_stats["mu_dist_tilde_P999"]) / sum(SS_stats["mu_dist_tilde_P999"])
    income_Ent = sum(total_income_mesh_aux .* SS_stats["mu_dist_tilde_Ent"]) / sum(SS_stats["mu_dist_tilde_Ent"])

    # Labor income (for employed only, states 1:ns)
    labor_all = sum(WW_aux[:, :, 1:ns_val] .* meshes["se_aux"][:, :, 1:ns_val] .* mu_dist_tilde[:, :, 1:ns_val]) / 
                sum(mu_dist_tilde[:, :, 1:ns_val])
    labor_B60 = sum(WW_aux[:, :, 1:ns_val] .* meshes["se_aux"][:, :, 1:ns_val] .* SS_stats["mu_dist_tilde_B60"][:, :, 1:ns_val]) / 
                sum(SS_stats["mu_dist_tilde_B60"][:, :, 1:ns_val])
    labor_Q1 = sum(WW_aux[:, :, 1:ns_val] .* meshes["se_aux"][:, :, 1:ns_val] .* SS_stats["mu_dist_tilde_Q1"][:, :, 1:ns_val]) / 
               sum(SS_stats["mu_dist_tilde_Q1"][:, :, 1:ns_val])
    labor_Q2 = sum(WW_aux[:, :, 1:ns_val] .* meshes["se_aux"][:, :, 1:ns_val] .* SS_stats["mu_dist_tilde_Q2"][:, :, 1:ns_val]) / 
               sum(SS_stats["mu_dist_tilde_Q2"][:, :, 1:ns_val])
    labor_Q3 = sum(WW_aux[:, :, 1:ns_val] .* meshes["se_aux"][:, :, 1:ns_val] .* SS_stats["mu_dist_tilde_Q3"][:, :, 1:ns_val]) / 
               sum(SS_stats["mu_dist_tilde_Q3"][:, :, 1:ns_val])
    labor_Q4 = sum(WW_aux[:, :, 1:ns_val] .* meshes["se_aux"][:, :, 1:ns_val] .* SS_stats["mu_dist_tilde_Q4"][:, :, 1:ns_val]) / 
               sum(SS_stats["mu_dist_tilde_Q4"][:, :, 1:ns_val])
    labor_Q5 = sum(WW_aux[:, :, 1:ns_val] .* meshes["se_aux"][:, :, 1:ns_val] .* SS_stats["mu_dist_tilde_Q5"][:, :, 1:ns_val]) / 
               sum(SS_stats["mu_dist_tilde_Q5"][:, :, 1:ns_val])
    labor_P90 = sum(WW_aux[:, :, 1:ns_val] .* meshes["se_aux"][:, :, 1:ns_val] .* SS_stats["mu_dist_tilde_P90"][:, :, 1:ns_val]) / 
                sum(SS_stats["mu_dist_tilde_P90"][:, :, 1:ns_val])
    labor_P99 = sum(WW_aux[:, :, 1:ns_val] .* meshes["se_aux"][:, :, 1:ns_val] .* SS_stats["mu_dist_tilde_P99"][:, :, 1:ns_val]) / 
                sum(SS_stats["mu_dist_tilde_P99"][:, :, 1:ns_val])
    labor_P999 = sum(WW_aux[:, :, 1:ns_val] .* meshes["se_aux"][:, :, 1:ns_val] .* SS_stats["mu_dist_tilde_P999"][:, :, 1:ns_val]) / 
                 sum(SS_stats["mu_dist_tilde_P999"][:, :, 1:ns_val])
    labor_Ent = sum(WW_aux[:, :, 1:ns_val] .* meshes["se_aux"][:, :, 1:ns_val] .* SS_stats["mu_dist_tilde_Ent"][:, :, 1:ns_val]) / 
                sum(SS_stats["mu_dist_tilde_Ent"][:, :, 1:ns_val])

    # Unemployment benefit income (states ns+1:2*ns)
    ubincome_all = sum(WW_aux[:, :, ns_val+1:2*ns_val] .* meshes["se_aux"][:, :, ns_val+1:2*ns_val] .* mu_dist_tilde[:, :, ns_val+1:2*ns_val]) / 
                   sum(mu_dist_tilde[:, :, ns_val+1:2*ns_val])
    ubincome_B60 = sum(WW_aux[:, :, ns_val+1:2*ns_val] .* meshes["se_aux"][:, :, ns_val+1:2*ns_val] .* SS_stats["mu_dist_tilde_B60"][:, :, ns_val+1:2*ns_val]) / 
                   sum(SS_stats["mu_dist_tilde_B60"][:, :, ns_val+1:2*ns_val])
    ubincome_Q1 = sum(WW_aux[:, :, ns_val+1:2*ns_val] .* meshes["se_aux"][:, :, ns_val+1:2*ns_val] .* SS_stats["mu_dist_tilde_Q1"][:, :, ns_val+1:2*ns_val]) / 
                  sum(SS_stats["mu_dist_tilde_Q1"][:, :, ns_val+1:2*ns_val])
    ubincome_Q2 = sum(WW_aux[:, :, ns_val+1:2*ns_val] .* meshes["se_aux"][:, :, ns_val+1:2*ns_val] .* SS_stats["mu_dist_tilde_Q2"][:, :, ns_val+1:2*ns_val]) / 
                  sum(SS_stats["mu_dist_tilde_Q2"][:, :, ns_val+1:2*ns_val])
    ubincome_Q3 = sum(WW_aux[:, :, ns_val+1:2*ns_val] .* meshes["se_aux"][:, :, ns_val+1:2*ns_val] .* SS_stats["mu_dist_tilde_Q3"][:, :, ns_val+1:2*ns_val]) / 
                  sum(SS_stats["mu_dist_tilde_Q3"][:, :, ns_val+1:2*ns_val])
    ubincome_Q4 = sum(WW_aux[:, :, ns_val+1:2*ns_val] .* meshes["se_aux"][:, :, ns_val+1:2*ns_val] .* SS_stats["mu_dist_tilde_Q4"][:, :, ns_val+1:2*ns_val]) / 
                  sum(SS_stats["mu_dist_tilde_Q4"][:, :, ns_val+1:2*ns_val])
    ubincome_Q5 = sum(WW_aux[:, :, ns_val+1:2*ns_val] .* meshes["se_aux"][:, :, ns_val+1:2*ns_val] .* SS_stats["mu_dist_tilde_Q5"][:, :, ns_val+1:2*ns_val]) / 
                  sum(SS_stats["mu_dist_tilde_Q5"][:, :, ns_val+1:2*ns_val])
    ubincome_P90 = sum(WW_aux[:, :, ns_val+1:2*ns_val] .* meshes["se_aux"][:, :, ns_val+1:2*ns_val] .* SS_stats["mu_dist_tilde_P90"][:, :, ns_val+1:2*ns_val]) / 
                   sum(SS_stats["mu_dist_tilde_P90"][:, :, ns_val+1:2*ns_val])
    ubincome_P99 = sum(WW_aux[:, :, ns_val+1:2*ns_val] .* meshes["se_aux"][:, :, ns_val+1:2*ns_val] .* SS_stats["mu_dist_tilde_P99"][:, :, ns_val+1:2*ns_val]) / 
                   sum(SS_stats["mu_dist_tilde_P99"][:, :, ns_val+1:2*ns_val])
    ubincome_P999 = sum(WW_aux[:, :, ns_val+1:2*ns_val] .* meshes["se_aux"][:, :, ns_val+1:2*ns_val] .* SS_stats["mu_dist_tilde_P999"][:, :, ns_val+1:2*ns_val]) / 
                    sum(SS_stats["mu_dist_tilde_P999"][:, :, ns_val+1:2*ns_val])
    ubincome_Ent = sum(WW_aux[:, :, ns_val+1:2*ns_val] .* meshes["se_aux"][:, :, ns_val+1:2*ns_val] .* SS_stats["mu_dist_tilde_Ent"][:, :, ns_val+1:2*ns_val]) / 
                   sum(SS_stats["mu_dist_tilde_Ent"][:, :, ns_val+1:2*ns_val])

    # Profit income (entrepreneurs, last state)
    profit_income_all = sum(WW_aux[:, :, end] .* meshes_se_aux[:, :, end] .* mu_dist_tilde[:, :, end]) / 
                        sum(mu_dist_tilde[:, :, end])
    profit_income_B60 = sum(WW_aux[:, :, end] .* meshes_se_aux[:, :, end] .* SS_stats["mu_dist_tilde_B60"][:, :, end]) / 
                        sum(SS_stats["mu_dist_tilde_B60"][:, :, end])
    profit_income_Q1 = sum(WW_aux[:, :, end] .* meshes_se_aux[:, :, end] .* SS_stats["mu_dist_tilde_Q1"][:, :, end]) / 
                       sum(SS_stats["mu_dist_tilde_Q1"][:, :, end])
    profit_income_Q2 = sum(WW_aux[:, :, end] .* meshes_se_aux[:, :, end] .* SS_stats["mu_dist_tilde_Q2"][:, :, end]) / 
                       sum(SS_stats["mu_dist_tilde_Q2"][:, :, end])
    profit_income_Q3 = sum(WW_aux[:, :, end] .* meshes_se_aux[:, :, end] .* SS_stats["mu_dist_tilde_Q3"][:, :, end]) / 
                       sum(SS_stats["mu_dist_tilde_Q3"][:, :, end])
    profit_income_Q4 = sum(WW_aux[:, :, end] .* meshes_se_aux[:, :, end] .* SS_stats["mu_dist_tilde_Q4"][:, :, end]) / 
                       sum(SS_stats["mu_dist_tilde_Q4"][:, :, end])
    profit_income_Q5 = sum(WW_aux[:, :, end] .* meshes_se_aux[:, :, end] .* SS_stats["mu_dist_tilde_Q5"][:, :, end]) / 
                       sum(SS_stats["mu_dist_tilde_Q5"][:, :, end])
    profit_income_P90 = sum(WW_aux[:, :, end] .* meshes_se_aux[:, :, end] .* SS_stats["mu_dist_tilde_P90"][:, :, end]) / 
                        sum(SS_stats["mu_dist_tilde_P90"][:, :, end])
    profit_income_P99 = sum(WW_aux[:, :, end] .* meshes_se_aux[:, :, end] .* SS_stats["mu_dist_tilde_P99"][:, :, end]) / 
                        sum(SS_stats["mu_dist_tilde_P99"][:, :, end])
    profit_income_P999 = sum(WW_aux[:, :, end] .* meshes_se_aux[:, :, end] .* SS_stats["mu_dist_tilde_P999"][:, :, end]) / 
                         sum(SS_stats["mu_dist_tilde_P999"][:, :, end])
    profit_income_Ent = sum(WW_aux[:, :, end] .* meshes_se_aux[:, :, end] .* SS_stats["mu_dist_tilde_Ent"][:, :, end]) / 
                        sum(SS_stats["mu_dist_tilde_Ent"][:, :, end])

    # Transfer income
    transfer_income_all = sum(inc_transfer .* mu_dist_tilde) / sum(mu_dist_tilde)
    transfer_income_B60 = sum(inc_transfer .* SS_stats["mu_dist_tilde_B60"]) / sum(SS_stats["mu_dist_tilde_B60"])
    transfer_income_Q1 = sum(inc_transfer .* SS_stats["mu_dist_tilde_Q1"]) / sum(SS_stats["mu_dist_tilde_Q1"])
    transfer_income_Q2 = sum(inc_transfer .* SS_stats["mu_dist_tilde_Q2"]) / sum(SS_stats["mu_dist_tilde_Q2"])
    transfer_income_Q3 = sum(inc_transfer .* SS_stats["mu_dist_tilde_Q3"]) / sum(SS_stats["mu_dist_tilde_Q3"])
    transfer_income_Q4 = sum(inc_transfer .* SS_stats["mu_dist_tilde_Q4"]) / sum(SS_stats["mu_dist_tilde_Q4"])
    transfer_income_Q5 = sum(inc_transfer .* SS_stats["mu_dist_tilde_Q5"]) / sum(SS_stats["mu_dist_tilde_Q5"])
    transfer_income_P90 = sum(inc_transfer .* SS_stats["mu_dist_tilde_P90"]) / sum(SS_stats["mu_dist_tilde_P90"])
    transfer_income_P99 = sum(inc_transfer .* SS_stats["mu_dist_tilde_P99"]) / sum(SS_stats["mu_dist_tilde_P99"])
    transfer_income_P999 = sum(inc_transfer .* SS_stats["mu_dist_tilde_P999"]) / sum(SS_stats["mu_dist_tilde_P999"])
    transfer_income_Ent = sum(inc_transfer .* SS_stats["mu_dist_tilde_Ent"]) / sum(SS_stats["mu_dist_tilde_Ent"])

    # Other income (unemployment + transfer)
    other_income_all = ubincome_all + transfer_income_all
    other_income_B60 = ubincome_B60 + transfer_income_B60
    other_income_Q1 = ubincome_Q1 + transfer_income_Q1
    other_income_Q2 = ubincome_Q2 + transfer_income_Q2
    other_income_Q3 = ubincome_Q3 + transfer_income_Q3
    other_income_Q4 = ubincome_Q4 + transfer_income_Q4
    other_income_Q5 = ubincome_Q5 + transfer_income_Q5
    other_income_P90 = ubincome_P90 + transfer_income_P90
    other_income_P99 = ubincome_P99 + transfer_income_P99
    other_income_P999 = ubincome_P999 + transfer_income_P999

    # Asset income (equity + bond returns)
    asset_all = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* mu_dist_tilde) / sum(mu_dist_tilde)
    asset_B60 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* SS_stats["mu_dist_tilde_B60"]) / sum(SS_stats["mu_dist_tilde_B60"])
    asset_Q1 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* SS_stats["mu_dist_tilde_Q1"]) / sum(SS_stats["mu_dist_tilde_Q1"])
    asset_Q2 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* SS_stats["mu_dist_tilde_Q2"]) / sum(SS_stats["mu_dist_tilde_Q2"])
    asset_Q3 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* SS_stats["mu_dist_tilde_Q3"]) / sum(SS_stats["mu_dist_tilde_Q3"])
    asset_Q4 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* SS_stats["mu_dist_tilde_Q4"]) / sum(SS_stats["mu_dist_tilde_Q4"])
    asset_Q5 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* SS_stats["mu_dist_tilde_Q5"]) / sum(SS_stats["mu_dist_tilde_Q5"])
    asset_P90 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* SS_stats["mu_dist_tilde_P90"]) / sum(SS_stats["mu_dist_tilde_P90"])
    asset_P99 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* SS_stats["mu_dist_tilde_P99"]) / sum(SS_stats["mu_dist_tilde_P99"])
    asset_P999 = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* SS_stats["mu_dist_tilde_P999"]) / sum(SS_stats["mu_dist_tilde_P999"])
    asset_Ent = sum((a_income_mesh_aux .+ b_income_mesh_aux) .* SS_stats["mu_dist_tilde_Ent"]) / sum(SS_stats["mu_dist_tilde_Ent"])

    # Equity income only
    a_income_all = sum(a_income_mesh_aux .* mu_dist_tilde) / sum(mu_dist_tilde)
    a_income_Q1 = sum(a_income_mesh_aux .* SS_stats["mu_dist_tilde_Q1"]) / sum(SS_stats["mu_dist_tilde_Q1"])
    a_income_Q2 = sum(a_income_mesh_aux .* SS_stats["mu_dist_tilde_Q2"]) / sum(SS_stats["mu_dist_tilde_Q2"])
    a_income_Q3 = sum(a_income_mesh_aux .* SS_stats["mu_dist_tilde_Q3"]) / sum(SS_stats["mu_dist_tilde_Q3"])
    a_income_Q4 = sum(a_income_mesh_aux .* SS_stats["mu_dist_tilde_Q4"]) / sum(SS_stats["mu_dist_tilde_Q4"])
    a_income_Q5 = sum(a_income_mesh_aux .* SS_stats["mu_dist_tilde_Q5"]) / sum(SS_stats["mu_dist_tilde_Q5"])
    a_income_P90 = sum(a_income_mesh_aux .* SS_stats["mu_dist_tilde_P90"]) / sum(SS_stats["mu_dist_tilde_P90"])
    a_income_P99 = sum(a_income_mesh_aux .* SS_stats["mu_dist_tilde_P99"]) / sum(SS_stats["mu_dist_tilde_P99"])
    a_income_P999 = sum(a_income_mesh_aux .* SS_stats["mu_dist_tilde_P999"]) / sum(SS_stats["mu_dist_tilde_P999"])
    a_income_Ent = sum(a_income_mesh_aux .* SS_stats["mu_dist_tilde_Ent"]) / sum(SS_stats["mu_dist_tilde_Ent"])

    # Bond income only
    b_income_all = sum(b_income_mesh_aux .* mu_dist_tilde) / sum(mu_dist_tilde)
    b_income_Q1 = sum(b_income_mesh_aux .* SS_stats["mu_dist_tilde_Q1"]) / sum(SS_stats["mu_dist_tilde_Q1"])
    b_income_Q2 = sum(b_income_mesh_aux .* SS_stats["mu_dist_tilde_Q2"]) / sum(SS_stats["mu_dist_tilde_Q2"])
    b_income_Q3 = sum(b_income_mesh_aux .* SS_stats["mu_dist_tilde_Q3"]) / sum(SS_stats["mu_dist_tilde_Q3"])
    b_income_Q4 = sum(b_income_mesh_aux .* SS_stats["mu_dist_tilde_Q4"]) / sum(SS_stats["mu_dist_tilde_Q4"])
    b_income_Q5 = sum(b_income_mesh_aux .* SS_stats["mu_dist_tilde_Q5"]) / sum(SS_stats["mu_dist_tilde_Q5"])
    b_income_P90 = sum(b_income_mesh_aux .* SS_stats["mu_dist_tilde_P90"]) / sum(SS_stats["mu_dist_tilde_P90"])
    b_income_P99 = sum(b_income_mesh_aux .* SS_stats["mu_dist_tilde_P99"]) / sum(SS_stats["mu_dist_tilde_P99"])
    b_income_P999 = sum(b_income_mesh_aux .* SS_stats["mu_dist_tilde_P999"]) / sum(SS_stats["mu_dist_tilde_P999"])
    b_income_Ent = sum(b_income_mesh_aux .* SS_stats["mu_dist_tilde_Ent"]) / sum(SS_stats["mu_dist_tilde_Ent"])

    ## 14. Store income statistics in SS_stats
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

    ## 15. Compute consumption by quintile using q_cons2
    SS_stats["C_all"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, Q_dists["mu_dist_tilde"], meshes, param, grid, param["tau_w"])
    SS_stats["C_B01"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, Q_dists["mu_dist_tilde_B01"], meshes, param, grid, param["tau_w"])
    SS_stats["C_B1"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, Q_dists["mu_dist_tilde_B1"], meshes, param, grid, param["tau_w"])
    SS_stats["C_B10"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, Q_dists["mu_dist_tilde_B10"], meshes, param, grid, param["tau_w"])
    SS_stats["C_Q1"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, Q_dists["mu_dist_tilde_Q1"], meshes, param, grid, param["tau_w"])
    SS_stats["C_Q2"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, Q_dists["mu_dist_tilde_Q2"], meshes, param, grid, param["tau_w"])
    SS_stats["C_Q3"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, Q_dists["mu_dist_tilde_Q3"], meshes, param, grid, param["tau_w"])
    SS_stats["C_Q4"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, Q_dists["mu_dist_tilde_Q4"], meshes, param, grid, param["tau_w"])
    SS_stats["C_Q5"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, Q_dists["mu_dist_tilde_Q5"], meshes, param, grid, param["tau_w"])
    SS_stats["C_P90"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, Q_dists["mu_dist_tilde_P90"], meshes, param, grid, param["tau_w"])
    SS_stats["C_P99"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, Q_dists["mu_dist_tilde_P99"], meshes, param, grid, param["tau_w"])
    SS_stats["C_P999"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, Q_dists["mu_dist_tilde_P999"], meshes, param, grid, param["tau_w"])
    SS_stats["C_Ent"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, Q_dists["mu_dist_tilde_Ent"], meshes, param, grid, param["tau_w"])
    SS_stats["C_Emp"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, Q_dists["mu_dist_tilde_Emp"], meshes, param, grid, param["tau_w"])
    SS_stats["C_Unemp"] = q_cons2(c_a_guess, c_n_guess, SS_stats["w"], SS_stats["nn"], AProb, Q_dists["mu_dist_tilde_Unemp"], meshes, param, grid, param["tau_w"])

    ## 16. Compute asset holdings (A, B) by quintile
    # A = equity holdings (weighted by adjustment probability)
    SS_stats["A"] = sum(AProb .* Q_dists["mu_dist_tilde"] .* a_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde"] .* meshes["a"])
    SS_stats["A_B01"] = sum(AProb .* Q_dists["mu_dist_tilde_B01"] .* a_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_B01"] .* meshes["a"])
    SS_stats["A_B10"] = sum(AProb .* Q_dists["mu_dist_tilde_B10"] .* a_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_B10"] .* meshes["a"])
    SS_stats["A_B1"] = sum(AProb .* Q_dists["mu_dist_tilde_B1"] .* a_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_B1"] .* meshes["a"])
    SS_stats["A_Q1"] = sum(AProb .* Q_dists["mu_dist_tilde_Q1"] .* a_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_Q1"] .* meshes["a"])
    SS_stats["A_Q2"] = sum(AProb .* Q_dists["mu_dist_tilde_Q2"] .* a_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_Q2"] .* meshes["a"])
    SS_stats["A_Q3"] = sum(AProb .* Q_dists["mu_dist_tilde_Q3"] .* a_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_Q3"] .* meshes["a"])
    SS_stats["A_Q4"] = sum(AProb .* Q_dists["mu_dist_tilde_Q4"] .* a_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_Q4"] .* meshes["a"])
    SS_stats["A_Q5"] = sum(AProb .* Q_dists["mu_dist_tilde_Q5"] .* a_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_Q5"] .* meshes["a"])
    SS_stats["A_P90"] = sum(AProb .* Q_dists["mu_dist_tilde_P90"] .* a_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_P90"] .* meshes["a"])
    SS_stats["A_P99"] = sum(AProb .* Q_dists["mu_dist_tilde_P99"] .* a_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_P99"] .* meshes["a"])
    SS_stats["A_P999"] = sum(AProb .* Q_dists["mu_dist_tilde_P999"] .* a_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_P999"] .* meshes["a"])
    SS_stats["A_Ent"] = sum(AProb .* Q_dists["mu_dist_tilde_Ent"] .* a_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_Ent"] .* meshes["a"])
    SS_stats["A_Emp"] = sum(AProb .* Q_dists["mu_dist_tilde_Emp"] .* a_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_Emp"] .* meshes["a"])
    SS_stats["A_Unemp"] = sum(AProb .* Q_dists["mu_dist_tilde_Unemp"] .* a_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_Unemp"] .* meshes["a"])
    SS_stats["A_all"] = sum(AProb .* Q_dists["mu_dist_tilde_aux"] .* a_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_aux"] .* meshes["a"])
    SS_stats["A_all2"] = sum(AProb .* Q_dists["mu_dist_tilde_full"] .* a_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_full"] .* meshes["a"])

    # B = bond holdings (weighted by adjustment probability)
    SS_stats["B"] = sum(AProb .* Q_dists["mu_dist_tilde"] .* b_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde"] .* b_n_star)
    SS_stats["B_B01"] = sum(AProb .* Q_dists["mu_dist_tilde_B01"] .* b_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_B01"] .* b_n_star)
    SS_stats["B_B1"] = sum(AProb .* Q_dists["mu_dist_tilde_B1"] .* b_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_B1"] .* b_n_star)
    SS_stats["B_B10"] = sum(AProb .* Q_dists["mu_dist_tilde_B10"] .* b_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_B10"] .* b_n_star)
    SS_stats["B_Q1"] = sum(AProb .* Q_dists["mu_dist_tilde_Q1"] .* b_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_Q1"] .* b_n_star)
    SS_stats["B_Q2"] = sum(AProb .* Q_dists["mu_dist_tilde_Q2"] .* b_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_Q2"] .* b_n_star)
    SS_stats["B_Q3"] = sum(AProb .* Q_dists["mu_dist_tilde_Q3"] .* b_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_Q3"] .* b_n_star)
    SS_stats["B_Q4"] = sum(AProb .* Q_dists["mu_dist_tilde_Q4"] .* b_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_Q4"] .* b_n_star)
    SS_stats["B_Q5"] = sum(AProb .* Q_dists["mu_dist_tilde_Q5"] .* b_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_Q5"] .* b_n_star)
    SS_stats["B_P90"] = sum(AProb .* Q_dists["mu_dist_tilde_P90"] .* b_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_P90"] .* b_n_star)
    SS_stats["B_P99"] = sum(AProb .* Q_dists["mu_dist_tilde_P99"] .* b_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_P99"] .* b_n_star)
    SS_stats["B_P999"] = sum(AProb .* Q_dists["mu_dist_tilde_P999"] .* b_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_P999"] .* b_n_star)
    SS_stats["B_Ent"] = sum(AProb .* Q_dists["mu_dist_tilde_Ent"] .* b_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_Ent"] .* b_n_star)
    SS_stats["B_Emp"] = sum(AProb .* Q_dists["mu_dist_tilde_Emp"] .* b_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_Emp"] .* b_n_star)
    SS_stats["B_Unemp"] = sum(AProb .* Q_dists["mu_dist_tilde_Unemp"] .* b_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_Unemp"] .* b_n_star)
    SS_stats["B_all"] = sum(AProb .* Q_dists["mu_dist_tilde_aux"] .* b_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_aux"] .* b_n_star)
    SS_stats["B_all2"] = sum(AProb .* Q_dists["mu_dist_tilde_full"] .* b_a_star) + sum((1 .- AProb) .* Q_dists["mu_dist_tilde_full"] .* b_n_star)

    # Wealth (W) by quintile
    SS_stats["w_all"] = sum(meshes["w"] .* Q_dists["mu_dist_tilde"])
    SS_stats["w_B1"] = sum(meshes["w"] .* Q_dists["mu_dist_tilde_B1"])
    SS_stats["w_B01"] = sum(meshes["w"] .* Q_dists["mu_dist_tilde_B01"])
    SS_stats["w_B10"] = sum(meshes["w"] .* Q_dists["mu_dist_tilde_B10"])
    SS_stats["w_Q1"] = sum(meshes["w"] .* Q_dists["mu_dist_tilde_Q1"])
    SS_stats["w_Q2"] = sum(meshes["w"] .* Q_dists["mu_dist_tilde_Q2"])
    SS_stats["w_Q3"] = sum(meshes["w"] .* Q_dists["mu_dist_tilde_Q3"])
    SS_stats["w_Q4"] = sum(meshes["w"] .* Q_dists["mu_dist_tilde_Q4"])
    SS_stats["w_Q5"] = sum(meshes["w"] .* Q_dists["mu_dist_tilde_Q5"])
    SS_stats["w_P90"] = sum(meshes["w"] .* Q_dists["mu_dist_tilde_P90"])
    SS_stats["w_P99"] = sum(meshes["w"] .* Q_dists["mu_dist_tilde_P99"])
    SS_stats["w_P999"] = sum(meshes["w"] .* Q_dists["mu_dist_tilde_P999"])

    # Wealth statistics (using meshes.a + meshes.b)
    SS_stats["wealth_B01"] = sum((meshes["a"] .+ meshes["b"]) .* Q_dists["mu_dist_tilde_B01"]) / sum(Q_dists["mu_dist_tilde_B01"])
    SS_stats["wealth_B1"] = sum((meshes["a"] .+ meshes["b"]) .* Q_dists["mu_dist_tilde_B1"]) / sum(Q_dists["mu_dist_tilde_B1"])
    SS_stats["wealth_B10"] = sum((meshes["a"] .+ meshes["b"]) .* Q_dists["mu_dist_tilde_B10"]) / sum(Q_dists["mu_dist_tilde_B10"])
    SS_stats["wealth_Q1"] = sum((meshes["a"] .+ meshes["b"]) .* Q_dists["mu_dist_tilde_Q1"]) / sum(Q_dists["mu_dist_tilde_Q1"])
    SS_stats["wealth_Q2"] = sum((meshes["a"] .+ meshes["b"]) .* Q_dists["mu_dist_tilde_Q2"]) / sum(Q_dists["mu_dist_tilde_Q2"])
    SS_stats["wealth_Q3"] = sum((meshes["a"] .+ meshes["b"]) .* Q_dists["mu_dist_tilde_Q3"]) / sum(Q_dists["mu_dist_tilde_Q3"])
    SS_stats["wealth_Q4"] = sum((meshes["a"] .+ meshes["b"]) .* Q_dists["mu_dist_tilde_Q4"]) / sum(Q_dists["mu_dist_tilde_Q4"])
    SS_stats["wealth_Q5"] = sum((meshes["a"] .+ meshes["b"]) .* Q_dists["mu_dist_tilde_Q5"]) / sum(Q_dists["mu_dist_tilde_Q5"])
    SS_stats["wealth_P90"] = sum((meshes["a"] .+ meshes["b"]) .* Q_dists["mu_dist_tilde_P90"]) / sum(Q_dists["mu_dist_tilde_P90"])
    SS_stats["wealth_P99"] = sum((meshes["a"] .+ meshes["b"]) .* Q_dists["mu_dist_tilde_P99"]) / sum(Q_dists["mu_dist_tilde_P99"])
    SS_stats["wealth_P999"] = sum((meshes["a"] .+ meshes["b"]) .* Q_dists["mu_dist_tilde_P999"]) / sum(Q_dists["mu_dist_tilde_P999"])

    ## 17. Compute MPC (Marginal Propensity to Consume)
    WW_se = vec(WW_aux[1, 1, :])
    WW_se_mesh = WW_aux .* meshes["se_aux"]

    grid_se_aux_vec = grid_se_aux  # Use the same grid_se_aux from earlier

    MPC_a_b = zeros(nb_val, na_val, nse_val)
    MPC_n_b = zeros(nb_val, na_val, nse_val)
    for kk in 1:na_val
        for hh in 1:nse_val
            # Extract 1D slices (squeeze removes singleton dimensions)
            # c_a_guess[:, kk, hh] is (nb,) so it's already 1D
            c_a_slice = c_a_guess[:, kk, hh]
            c_n_slice = c_n_guess[:, kk, hh]
            b_grid = param["R_cb"] / param["pi_cb"] .* grid["b"] .+ param["Rprem"] .* grid["b"] .* (grid["b"] .< 0)
            MPC_a_b[:, kk, hh] = gradient(c_a_slice) ./ gradient(b_grid)
            MPC_n_b[:, kk, hh] = gradient(c_n_slice) ./ gradient(b_grid)
        end
    end

    MPC_a_b = MPC_a_b .* (WW_se_mesh ./ c_a_guess)
    MPC_n_b = MPC_n_b .* (WW_se_mesh ./ c_n_guess)

    MPC_a_se = zeros(nb_val, na_val, nse_val)
    MPC_n_se = zeros(nb_val, na_val, nse_val)
    # WW_se is a vector (nse,), grid_se_aux_vec should be (nse,)
    # MATLAB: WW_se' .* grid_se_aux - in MATLAB this is element-wise multiplication
    # In Julia, WW_se is already a vector, so just use element-wise multiplication
    WW_se_times_grid_se = WW_se .* grid_se_aux_vec  # Element-wise, results in (nse,)
    WW_se_log_vec = log.(WW_se_times_grid_se)  # (nse,)
    
    for mm in 1:nb_val
        for kk in 1:na_val
            # Extract 1D slices - c_a_guess[mm, kk, :] is (nse,) so already 1D
            c_a_slice = log.(c_a_guess[mm, kk, :])
            c_n_slice = log.(c_n_guess[mm, kk, :])
            # Both are 1D vectors now
            MPC_a_se[mm, kk, :] = gradient(c_a_slice) ./ gradient(WW_se_log_vec)
            MPC_n_se[mm, kk, :] = gradient(c_n_slice) ./ gradient(WW_se_log_vec)
        end
    end

    # Expected MPC
    EMPC_a_se = vec(mu_dist_tilde)' * vec(MPC_a_se)
    EMPC_a_b = vec(mu_dist_tilde)' * vec(MPC_a_b)
    EMPC_n_se = vec(mu_dist_tilde)' * vec(MPC_n_se)
    EMPC_n_b = vec(mu_dist_tilde)' * vec(MPC_n_b)

    # Expected MPC by quintile (with adjustment probability weighting)
    EMPC_se = vec(SS_stats["mu_dist_tilde"])' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(SS_stats["mu_dist_tilde"])
    EMPC_se_B01 = vec(SS_stats["mu_dist_tilde_B01"])' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(SS_stats["mu_dist_tilde_B01"])
    EMPC_se_B1 = vec(SS_stats["mu_dist_tilde_B1"])' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(SS_stats["mu_dist_tilde_B1"])
    EMPC_se_B10 = vec(SS_stats["mu_dist_tilde_B10"])' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(SS_stats["mu_dist_tilde_B10"])
    EMPC_se_Q1 = vec(SS_stats["mu_dist_tilde_Q1"])' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(SS_stats["mu_dist_tilde_Q1"])
    EMPC_se_Q2 = vec(SS_stats["mu_dist_tilde_Q2"])' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(SS_stats["mu_dist_tilde_Q2"])
    EMPC_se_Q3 = vec(SS_stats["mu_dist_tilde_Q3"])' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(SS_stats["mu_dist_tilde_Q3"])
    EMPC_se_Q4 = vec(SS_stats["mu_dist_tilde_Q4"])' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(SS_stats["mu_dist_tilde_Q4"])
    EMPC_se_Q5 = vec(SS_stats["mu_dist_tilde_Q5"])' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(SS_stats["mu_dist_tilde_Q5"])
    EMPC_se_P90 = vec(SS_stats["mu_dist_tilde_P90"])' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(SS_stats["mu_dist_tilde_P90"])
    EMPC_se_P99 = vec(SS_stats["mu_dist_tilde_P99"])' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(SS_stats["mu_dist_tilde_P99"])
    EMPC_se_P999 = vec(SS_stats["mu_dist_tilde_P999"])' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(SS_stats["mu_dist_tilde_P999"])
    EMPC_se_Ent = vec(SS_stats["mu_dist_tilde_Ent"])' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(SS_stats["mu_dist_tilde_Ent"])
    EMPC_se_Emp = vec(SS_stats["mu_dist_tilde_Emp"])' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(SS_stats["mu_dist_tilde_Emp"])
    EMPC_se_Unemp = vec(SS_stats["mu_dist_tilde_Unemp"])' * vec(AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se) / sum(SS_stats["mu_dist_tilde_Unemp"])

    EMPC_b = vec(SS_stats["mu_dist_tilde"])' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(SS_stats["mu_dist_tilde"])
    EMPC_b_Q1 = vec(SS_stats["mu_dist_tilde_Q1"])' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(SS_stats["mu_dist_tilde_Q1"])
    EMPC_b_Q2 = vec(SS_stats["mu_dist_tilde_Q2"])' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(SS_stats["mu_dist_tilde_Q2"])
    EMPC_b_Q3 = vec(SS_stats["mu_dist_tilde_Q3"])' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(SS_stats["mu_dist_tilde_Q3"])
    EMPC_b_Q4 = vec(SS_stats["mu_dist_tilde_Q4"])' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(SS_stats["mu_dist_tilde_Q4"])
    EMPC_b_Q5 = vec(SS_stats["mu_dist_tilde_Q5"])' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(SS_stats["mu_dist_tilde_Q5"])
    EMPC_b_P90 = vec(SS_stats["mu_dist_tilde_P90"])' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(SS_stats["mu_dist_tilde_P90"])
    EMPC_b_P99 = vec(SS_stats["mu_dist_tilde_P99"])' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(SS_stats["mu_dist_tilde_P99"])
    EMPC_b_P999 = vec(SS_stats["mu_dist_tilde_P999"])' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(SS_stats["mu_dist_tilde_P999"])
    EMPC_b_Ent = vec(SS_stats["mu_dist_tilde_Ent"])' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(SS_stats["mu_dist_tilde_Ent"])
    EMPC_b_Emp = vec(SS_stats["mu_dist_tilde_Emp"])' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(SS_stats["mu_dist_tilde_Emp"])
    EMPC_b_Unemp = vec(SS_stats["mu_dist_tilde_Unemp"])' * vec(AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b) / sum(SS_stats["mu_dist_tilde_Unemp"])

    MPC_se = AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se
    MPC_b = AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b

    ## 18. Compute MRS (Marginal Rate of Substitution) across wealth groups
    # Note: MRS_a_aux2 and MRS_n_aux2 should be computed from Cal_SS_stats
    # For now, we'll need these to be passed in or computed - checking if they exist in SS_stats
    if haskey(SS_stats, "MRS_a_aux2") && haskey(SS_stats, "MRS_n_aux2")
        MRS_a_aux2 = SS_stats["MRS_a_aux2"]
        MRS_n_aux2 = SS_stats["MRS_n_aux2"]
        
        MRSs = vec(SS_stats["mu_dist_tilde"])' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(SS_stats["mu_dist_tilde"])
        MRSs_Q1 = vec(SS_stats["mu_dist_tilde_Q1"])' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(SS_stats["mu_dist_tilde_Q1"])
        MRSs_Q2 = vec(SS_stats["mu_dist_tilde_Q2"])' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(SS_stats["mu_dist_tilde_Q2"])
        MRSs_Q3 = vec(SS_stats["mu_dist_tilde_Q3"])' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(SS_stats["mu_dist_tilde_Q3"])
        MRSs_Q4 = vec(SS_stats["mu_dist_tilde_Q4"])' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(SS_stats["mu_dist_tilde_Q4"])
        MRSs_Q5 = vec(SS_stats["mu_dist_tilde_Q5"])' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(SS_stats["mu_dist_tilde_Q5"])
        MRSs_P90 = vec(SS_stats["mu_dist_tilde_P90"])' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(SS_stats["mu_dist_tilde_P90"])
        MRSs_P99 = vec(SS_stats["mu_dist_tilde_P99"])' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(SS_stats["mu_dist_tilde_P99"])
        MRSs_P999 = vec(SS_stats["mu_dist_tilde_P999"])' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(SS_stats["mu_dist_tilde_P999"])
        MRSs_Ent = vec(SS_stats["mu_dist_tilde_Ent"])' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(SS_stats["mu_dist_tilde_Ent"])
        MRSs_Emp = vec(SS_stats["mu_dist_tilde_Emp"])' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(SS_stats["mu_dist_tilde_Emp"])
        MRSs_Unemp = vec(SS_stats["mu_dist_tilde_Unemp"])' * vec(AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2) / sum(SS_stats["mu_dist_tilde_Unemp"])
        
        # Store MRS in anal_stats
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

    ## 19. Compute income-based distribution masks (final calculation with correct pi_bar)
    # meshes.income_ss = WW_aux.*meshes.se_aux + SS_stats.r_a_aux.*meshes.a + (param.R_cb/param.pi_bar-1).*meshes.b.*(meshes.b>=0) + ((param.R_cb+param.Rprem)/param.pi_bar-1).*meshes.b.*(meshes.b<0) + inc.transfer
    meshes["income_ss"] = WW_aux .* meshes["se_aux"] .+ SS_stats["r_a_aux"] .* meshes["a"] .+ 
                          (param["R_cb"] / param["pi_bar"] - 1) .* meshes["b"] .* (meshes["b"] .>= 0) .+ 
                          ((param["R_cb"] + param["Rprem"]) / param["pi_bar"] - 1) .* meshes["b"] .* (meshes["b"] .< 0) .+ 
                          inc_transfer

    SS_stats["mu_dist_I_B01"] = ((meshes["income_ss"] .>= I0_ss) .* (meshes["income_ss"] .<= I01_ss)) .* mu_dist
    SS_stats["mu_dist_I_B1"] = ((meshes["income_ss"] .>= I0_ss) .* (meshes["income_ss"] .<= I1_ss)) .* mu_dist
    SS_stats["mu_dist_I_B10"] = ((meshes["income_ss"] .>= I0_ss) .* (meshes["income_ss"] .<= I10_ss)) .* mu_dist
    SS_stats["mu_dist_I_Q1"] = ((meshes["income_ss"] .>= I0_ss) .* (meshes["income_ss"] .<= I20_ss)) .* mu_dist
    SS_stats["mu_dist_I_Q2"] = ((meshes["income_ss"] .> I20_ss) .* (meshes["income_ss"] .<= I40_ss)) .* mu_dist
    SS_stats["mu_dist_I_Q3"] = ((meshes["income_ss"] .> I40_ss) .* (meshes["income_ss"] .<= I60_ss)) .* mu_dist
    SS_stats["mu_dist_I_Q4"] = ((meshes["income_ss"] .> I60_ss) .* (meshes["income_ss"] .<= I80_ss)) .* mu_dist
    SS_stats["mu_dist_I_Q5"] = ((meshes["income_ss"] .> I80_ss) .* (meshes["income_ss"] .<= I100_ss)) .* mu_dist
    SS_stats["mu_dist_I_P90"] = ((meshes["income_ss"] .> I90_ss) .* (meshes["income_ss"] .<= I100_ss)) .* mu_dist
    SS_stats["mu_dist_I_P99"] = ((meshes["income_ss"] .> I99_ss) .* (meshes["income_ss"] .<= I100_ss)) .* mu_dist
    SS_stats["mu_dist_I_P999"] = ((meshes["income_ss"] .> I999_ss) .* (meshes["income_ss"] .<= I100_ss)) .* mu_dist
    SS_stats["mu_dist_I_B60"] = (meshes["income_ss"] .<= I60_ss) .* mu_dist

    ## 20. Store wealth and asset statistics in anal_stats
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

    # Simple asset holdings (b, a_1) by quintile
    # Use Q_dists for consistency with MATLAB
    b_all_sum = sum(meshes["b"] .* Q_dists["mu_dist_tilde"])
    b_all_denom = sum(Q_dists["mu_dist_tilde"])
    b_all = b_all_sum / (b_all_denom > 0 ? b_all_denom : 1.0)
    
    b_Q1_sum = sum(meshes["b"] .* Q_dists["mu_dist_tilde_Q1"])
    b_Q1_denom = sum(Q_dists["mu_dist_tilde_Q1"])
    b_Q1 = b_Q1_sum / (b_Q1_denom > 0 ? b_Q1_denom : 1.0)
    
    b_Q2_sum = sum(meshes["b"] .* Q_dists["mu_dist_tilde_Q2"])
    b_Q2_denom = sum(Q_dists["mu_dist_tilde_Q2"])
    b_Q2 = b_Q2_sum / (b_Q2_denom > 0 ? b_Q2_denom : 1.0)
    
    b_Q3_sum = sum(meshes["b"] .* Q_dists["mu_dist_tilde_Q3"])
    b_Q3_denom = sum(Q_dists["mu_dist_tilde_Q3"])
    b_Q3 = b_Q3_sum / (b_Q3_denom > 0 ? b_Q3_denom : 1.0)
    
    b_Q4_sum = sum(meshes["b"] .* Q_dists["mu_dist_tilde_Q4"])
    b_Q4_denom = sum(Q_dists["mu_dist_tilde_Q4"])
    b_Q4 = b_Q4_sum / (b_Q4_denom > 0 ? b_Q4_denom : 1.0)
    
    b_Q5_sum = sum(meshes["b"] .* Q_dists["mu_dist_tilde_Q5"])
    b_Q5_denom = sum(Q_dists["mu_dist_tilde_Q5"])
    b_Q5 = b_Q5_sum / (b_Q5_denom > 0 ? b_Q5_denom : 1.0)
    
    b_P90_sum = sum(meshes["b"] .* Q_dists["mu_dist_tilde_P90"])
    b_P90_denom = sum(Q_dists["mu_dist_tilde_P90"])
    b_P90 = b_P90_sum / (b_P90_denom > 0 ? b_P90_denom : 1.0)
    
    b_P99_sum = sum(meshes["b"] .* Q_dists["mu_dist_tilde_P99"])
    b_P99_denom = sum(Q_dists["mu_dist_tilde_P99"])
    b_P99 = b_P99_sum / (b_P99_denom > 0 ? b_P99_denom : 1.0)

    a_1_all_sum = sum(param["q"] .* meshes["a"] .* Q_dists["mu_dist_tilde"])
    a_1_all_denom = sum(Q_dists["mu_dist_tilde"])
    a_1_all = a_1_all_sum / (a_1_all_denom > 0 ? a_1_all_denom : 1.0)
    
    a_1_B60_sum = sum(param["q"] .* meshes["a"] .* Q_dists["mu_dist_tilde_B60"])
    a_1_B60_denom = sum(Q_dists["mu_dist_tilde_B60"])
    a_1_B60 = a_1_B60_sum / (a_1_B60_denom > 0 ? a_1_B60_denom : 1.0)
    
    a_1_Q1_sum = sum(param["q"] .* meshes["a"] .* Q_dists["mu_dist_tilde_Q1"])
    a_1_Q1_denom = sum(Q_dists["mu_dist_tilde_Q1"])
    a_1_Q1 = a_1_Q1_sum / (a_1_Q1_denom > 0 ? a_1_Q1_denom : 1.0)
    
    a_1_Q2_sum = sum(param["q"] .* meshes["a"] .* Q_dists["mu_dist_tilde_Q2"])
    a_1_Q2_denom = sum(Q_dists["mu_dist_tilde_Q2"])
    a_1_Q2 = a_1_Q2_sum / (a_1_Q2_denom > 0 ? a_1_Q2_denom : 1.0)
    
    a_1_Q3_sum = sum(param["q"] .* meshes["a"] .* Q_dists["mu_dist_tilde_Q3"])
    a_1_Q3_denom = sum(Q_dists["mu_dist_tilde_Q3"])
    a_1_Q3 = a_1_Q3_sum / (a_1_Q3_denom > 0 ? a_1_Q3_denom : 1.0)
    
    a_1_Q4_sum = sum(param["q"] .* meshes["a"] .* Q_dists["mu_dist_tilde_Q4"])
    a_1_Q4_denom = sum(Q_dists["mu_dist_tilde_Q4"])
    a_1_Q4 = a_1_Q4_sum / (a_1_Q4_denom > 0 ? a_1_Q4_denom : 1.0)
    
    a_1_Q5_sum = sum(param["q"] .* meshes["a"] .* Q_dists["mu_dist_tilde_Q5"])
    a_1_Q5_denom = sum(Q_dists["mu_dist_tilde_Q5"])
    a_1_Q5 = a_1_Q5_sum / (a_1_Q5_denom > 0 ? a_1_Q5_denom : 1.0)
    
    a_1_P90_sum = sum(param["q"] .* meshes["a"] .* Q_dists["mu_dist_tilde_P90"])
    a_1_P90_denom = sum(Q_dists["mu_dist_tilde_P90"])
    a_1_P90 = a_1_P90_sum / (a_1_P90_denom > 0 ? a_1_P90_denom : 1.0)
    
    a_1_P99_sum = sum(param["q"] .* meshes["a"] .* Q_dists["mu_dist_tilde_P99"])
    a_1_P99_denom = sum(Q_dists["mu_dist_tilde_P99"])
    a_1_P99 = a_1_P99_sum / (a_1_P99_denom > 0 ? a_1_P99_denom : 1.0)
    
    a_1_P999_sum = sum(param["q"] .* meshes["a"] .* Q_dists["mu_dist_tilde_P999"])
    a_1_P999_denom = sum(Q_dists["mu_dist_tilde_P999"])
    a_1_P999 = a_1_P999_sum / (a_1_P999_denom > 0 ? a_1_P999_denom : 1.0)
    
    a_1_Ent_sum = sum(param["q"] .* meshes["a"] .* Q_dists["mu_dist_tilde_Ent"])
    a_1_Ent_denom = sum(Q_dists["mu_dist_tilde_Ent"])
    a_1_Ent = a_1_Ent_sum / (a_1_Ent_denom > 0 ? a_1_Ent_denom : 1.0)

    anal_stats["b_all"] = b_all
    anal_stats["b_Q1"] = b_Q1
    anal_stats["b_Q2"] = b_Q2
    anal_stats["b_Q3"] = b_Q3
    anal_stats["b_Q4"] = b_Q4
    anal_stats["b_Q5"] = b_Q5
    anal_stats["b_P90"] = b_P90
    anal_stats["b_P99"] = b_P99

    anal_stats["a_1_all"] = a_1_all
    anal_stats["a_1_B60"] = a_1_B60
    anal_stats["a_1_Q1"] = a_1_Q1
    anal_stats["a_1_Q5"] = a_1_Q5

    anal_stats["c_all"] = SS_stats["C_all"]
    anal_stats["c_Q1"] = SS_stats["C_Q1"]
    anal_stats["c_Q2"] = SS_stats["C_Q2"]
    anal_stats["c_Q3"] = SS_stats["C_Q3"]
    anal_stats["c_Q4"] = SS_stats["C_Q4"]
    anal_stats["c_Q5"] = SS_stats["C_Q5"]
    anal_stats["c_P90"] = SS_stats["C_P90"]
    anal_stats["c_P99"] = SS_stats["C_P99"]
    anal_stats["c_P999"] = SS_stats["C_P999"]
    anal_stats["c_Ent"] = SS_stats["C_Ent"]

    # Store MPC in anal_stats and SS_stats
    SS_stats["EMPC_se"] = EMPC_se
    SS_stats["EMPC_se_B01"] = EMPC_se_B01
    SS_stats["EMPC_se_B1"] = EMPC_se_B1
    SS_stats["EMPC_se_B10"] = EMPC_se_B10
    SS_stats["EMPC_se_Q1"] = EMPC_se_Q1
    SS_stats["EMPC_se_Q2"] = EMPC_se_Q2
    SS_stats["EMPC_se_Q3"] = EMPC_se_Q3
    SS_stats["EMPC_se_Q4"] = EMPC_se_Q4
    SS_stats["EMPC_se_Q5"] = EMPC_se_Q5
    SS_stats["EMPC_se_P90"] = EMPC_se_P90
    SS_stats["EMPC_se_P99"] = EMPC_se_P99
    SS_stats["EMPC_se_P999"] = EMPC_se_P999
    SS_stats["EMPC_se_Ent"] = EMPC_se_Ent
    SS_stats["EMPC_se_Emp"] = EMPC_se_Emp
    SS_stats["EMPC_se_Unemp"] = EMPC_se_Unemp

    SS_stats["EMPC_b"] = EMPC_b
    SS_stats["EMPC_b_Q1"] = EMPC_b_Q1
    SS_stats["EMPC_b_Q2"] = EMPC_b_Q2
    SS_stats["EMPC_b_Q3"] = EMPC_b_Q3
    SS_stats["EMPC_b_Q4"] = EMPC_b_Q4
    SS_stats["EMPC_b_Q5"] = EMPC_b_Q5
    SS_stats["EMPC_b_P90"] = EMPC_b_P90
    SS_stats["EMPC_b_P99"] = EMPC_b_P99
    SS_stats["EMPC_b_P999"] = EMPC_b_P999
    SS_stats["EMPC_b_Ent"] = EMPC_b_Ent
    SS_stats["EMPC_b_Emp"] = EMPC_b_Emp
    SS_stats["EMPC_b_Unemp"] = EMPC_b_Unemp

    SS_stats["MPC_se"] = MPC_se
    SS_stats["MPC_b"] = MPC_b

    # Store MRS in SS_stats if computed
    if haskey(SS_stats, "MRS_a_aux2")
        SS_stats["MRS_a_aux2"] = MRS_a_aux2
        SS_stats["MRS_n_aux2"] = MRS_n_aux2
    end

    # Return income_ss mesh for compatibility with helpers version
    income_ss = meshes["income_ss"]
    
    return anal_stats, SS_stats, Q_dists, income_ss
end
