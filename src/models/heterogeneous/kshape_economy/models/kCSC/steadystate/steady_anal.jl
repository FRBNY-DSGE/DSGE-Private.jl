include("q_cons.jl")
include("gradient.jl")

const _STEADY_ANAL_MAIN_GROUPS = ("Q1", "Q2", "Q3", "Q4", "Q5", "P90", "P99", "P999")
const _STEADY_ANAL_EXTENDED_GROUPS = ("Q1", "Q2", "Q3", "Q4", "Q5", "P90", "P99", "P999", "Ent", "Emp", "Unemp", "B60")

function _steady_anal_constant_state_mesh(values, nb::Int, na::Int)
    value_vec = Float64.(vec(values))
    out = Array{Float64}(undef, nb, na, length(value_vec))
    out .= reshape(value_vec, 1, 1, :)
    return out
end

function _steady_anal_auxiliary_productivity!(meshes, grid, param, nb::Int, na::Int)
    ns = Int(grid["ns"])
    unemployment_se = min.(param["b_ratio"] .* grid["s"], grid["s_bar"])

    grid_se_aux = vcat(grid["s"], unemployment_se, [1.0])
    grid_se1_aux = vcat(grid["s"], zeros(ns), [0.0])
    grid_se2_aux = vcat(zeros(ns), unemployment_se, [0.0])
    grid_se3_aux = vcat(zeros(2 * ns), [1.0])

    meshes_se_aux = _steady_anal_constant_state_mesh(grid_se_aux, nb, na)
    meshes_se1_aux = _steady_anal_constant_state_mesh(grid_se1_aux, nb, na)
    meshes_se2_aux = _steady_anal_constant_state_mesh(grid_se2_aux, nb, na)
    meshes_se3_aux = _steady_anal_constant_state_mesh(grid_se3_aux, nb, na)

    meshes["se_aux"] = meshes_se_aux
    meshes["se1_aux"] = meshes_se1_aux
    meshes["se2_aux"] = meshes_se2_aux
    meshes["se3_aux"] = meshes_se3_aux

    return (
        grid_se_aux = grid_se_aux,
        grid_se1_aux = grid_se1_aux,
        grid_se2_aux = grid_se2_aux,
        grid_se3_aux = grid_se3_aux,
        meshes_se_aux = meshes_se_aux,
        meshes_se1_aux = meshes_se1_aux,
        meshes_se2_aux = meshes_se2_aux,
        meshes_se3_aux = meshes_se3_aux,
    )
end

function _steady_anal_asset_meshes(grid, nb::Int, na::Int)
    mesh_a = Array{Float64}(undef, nb, na)
    mesh_b = Array{Float64}(undef, nb, na)
    mesh_a .= reshape(Float64.(vec(grid["a"])), 1, na)
    mesh_b .= reshape(Float64.(vec(grid["b"])), nb, 1)
    return mesh_a, mesh_b
end

function _steady_anal_sorted_distribution(values, weights)
    value_vec = vec(values)
    weight_vec = vec(weights)
    ix = sortperm(value_vec)
    sorted_values = value_vec[ix]
    pdf = weight_vec[ix]
    return sorted_values, pdf, cumsum(pdf)
end

function _steady_anal_percentile(sorted_values, cdf, p::Real)
    idx = searchsortedfirst(cdf, p) - 1
    return sorted_values[clamp(idx, firstindex(sorted_values), lastindex(sorted_values))]
end

function _steady_anal_range_dist(values, dist, lower, upper; lower_closed::Bool = true)
    out = similar(dist)
    z = zero(eltype(out))
    @inbounds for i in eachindex(values, dist)
        v = values[i]
        lower_ok = lower_closed ? v >= lower : v > lower
        out[i] = (lower_ok && v <= upper) ? dist[i] : z
    end
    return out
end

function _steady_anal_upper_dist(values, dist, upper)
    out = similar(dist)
    z = zero(eltype(out))
    @inbounds for i in eachindex(values, dist)
        out[i] = values[i] <= upper ? dist[i] : z
    end
    return out
end

function _steady_anal_transition_dist(H_tilde, dist, dims)
    return reshape(H_tilde * dist[:], dims)
end

function _steady_anal_store_wealth_groups!(SS_stats, meshes_w, mu_dist, mu_dist_tilde,
                                           meshes_se3_aux, H_tilde, cutoffs,
                                           nb::Int, na::Int, nse::Int, ns::Int)
    dims = (nb, na, nse)

    SS_stats["mu_dist"] = mu_dist
    SS_stats["mu_dist_B01"] = _steady_anal_range_dist(meshes_w, mu_dist, cutoffs["w0"], cutoffs["w0"])
    SS_stats["mu_dist_B1"] = _steady_anal_range_dist(meshes_w, mu_dist, cutoffs["w0"], cutoffs["w1"])
    SS_stats["mu_dist_B10"] = _steady_anal_range_dist(meshes_w, mu_dist, cutoffs["w0"], cutoffs["w10"])
    SS_stats["mu_dist_Q1"] = _steady_anal_range_dist(meshes_w, mu_dist, cutoffs["w0"], cutoffs["w20"])
    SS_stats["mu_dist_Q2"] = _steady_anal_range_dist(meshes_w, mu_dist, cutoffs["w20"], cutoffs["w40"]; lower_closed = false)
    SS_stats["mu_dist_Q3"] = _steady_anal_range_dist(meshes_w, mu_dist, cutoffs["w40"], cutoffs["w60"]; lower_closed = false)
    SS_stats["mu_dist_Q4"] = _steady_anal_range_dist(meshes_w, mu_dist, cutoffs["w60"], cutoffs["w80"]; lower_closed = false)
    SS_stats["mu_dist_Q5"] = _steady_anal_range_dist(meshes_w, mu_dist, cutoffs["w80"], cutoffs["w100"]; lower_closed = false)
    SS_stats["mu_dist_P90"] = _steady_anal_range_dist(meshes_w, mu_dist, cutoffs["w90"], cutoffs["w100"]; lower_closed = false)
    SS_stats["mu_dist_P99"] = _steady_anal_range_dist(meshes_w, mu_dist, cutoffs["w99"], cutoffs["w100"]; lower_closed = false)
    SS_stats["mu_dist_P999"] = _steady_anal_range_dist(meshes_w, mu_dist, cutoffs["w999"], cutoffs["w100"]; lower_closed = false)

    SS_stats["mu_dist_Emp"] = zeros(eltype(mu_dist), nb, na, nse)
    SS_stats["mu_dist_Emp"][:, :, 1:ns] = mu_dist[:, :, 1:ns]

    SS_stats["mu_dist_Unemp"] = zeros(eltype(mu_dist), nb, na, nse)
    SS_stats["mu_dist_Unemp"][:, :, (ns + 1):(2 * ns)] = mu_dist[:, :, (ns + 1):(2 * ns)]

    SS_stats["mu_dist_Ent"] = _steady_anal_range_dist(meshes_se3_aux, mu_dist, 1.0, 1.0)
    SS_stats["mu_dist_B60"] = _steady_anal_upper_dist(meshes_w, mu_dist, cutoffs["w60"])
    SS_stats["mu_dist_B90"] = _steady_anal_upper_dist(meshes_w, mu_dist, cutoffs["w90"])

    SS_stats["mu_dist_tilde"] = mu_dist_tilde
    SS_stats["mu_dist_tilde_B01"] = _steady_anal_range_dist(meshes_w, mu_dist_tilde, cutoffs["w0"], cutoffs["w0"])
    SS_stats["mu_dist_tilde_B1"] = _steady_anal_range_dist(meshes_w, mu_dist_tilde, cutoffs["w0"], cutoffs["w1"])
    SS_stats["mu_dist_tilde_B10"] = _steady_anal_range_dist(meshes_w, mu_dist_tilde, cutoffs["w0"], cutoffs["w10"])
    SS_stats["mu_dist_tilde_Q1"] = _steady_anal_range_dist(meshes_w, mu_dist_tilde, cutoffs["w0"], cutoffs["w20"])
    SS_stats["mu_dist_tilde_Q2"] = _steady_anal_range_dist(meshes_w, mu_dist_tilde, cutoffs["w20"], cutoffs["w40"]; lower_closed = false)
    SS_stats["mu_dist_tilde_Q3"] = _steady_anal_range_dist(meshes_w, mu_dist_tilde, cutoffs["w40"], cutoffs["w60"]; lower_closed = false)
    SS_stats["mu_dist_tilde_Q4"] = _steady_anal_range_dist(meshes_w, mu_dist_tilde, cutoffs["w60"], cutoffs["w80"]; lower_closed = false)
    SS_stats["mu_dist_tilde_Q5"] = _steady_anal_range_dist(meshes_w, mu_dist_tilde, cutoffs["w80"], cutoffs["w100"]; lower_closed = false)
    SS_stats["mu_dist_tilde_P90"] = _steady_anal_range_dist(meshes_w, mu_dist_tilde, cutoffs["w90"], cutoffs["w100"]; lower_closed = false)
    SS_stats["mu_dist_tilde_P99"] = _steady_anal_range_dist(meshes_w, mu_dist_tilde, cutoffs["w99"], cutoffs["w100"]; lower_closed = false)
    SS_stats["mu_dist_tilde_P999"] = _steady_anal_range_dist(meshes_w, mu_dist_tilde, cutoffs["w999"], cutoffs["w100"]; lower_closed = false)

    SS_stats["mu_dist_tilde_Ent"] = _steady_anal_transition_dist(H_tilde, SS_stats["mu_dist_Ent"], dims)
    SS_stats["mu_dist_tilde_Emp"] = _steady_anal_transition_dist(H_tilde, SS_stats["mu_dist_Emp"], dims)
    SS_stats["mu_dist_tilde_Unemp"] = _steady_anal_transition_dist(H_tilde, SS_stats["mu_dist_Unemp"], dims)
    SS_stats["mu_dist_tilde_B60"] = _steady_anal_upper_dist(meshes_w, mu_dist_tilde, cutoffs["w60"])

    return nothing
end

function _steady_anal_store_income_groups!(target, income_mesh, dist, cutoffs)
    target["mu_dist_I_B01"] = _steady_anal_range_dist(income_mesh, dist, cutoffs["I0"], cutoffs["I01"])
    target["mu_dist_I_B1"] = _steady_anal_range_dist(income_mesh, dist, cutoffs["I0"], cutoffs["I1"])
    target["mu_dist_I_B10"] = _steady_anal_range_dist(income_mesh, dist, cutoffs["I0"], cutoffs["I10"])
    target["mu_dist_I_Q1"] = _steady_anal_range_dist(income_mesh, dist, cutoffs["I0"], cutoffs["I20"])
    target["mu_dist_I_Q2"] = _steady_anal_range_dist(income_mesh, dist, cutoffs["I20"], cutoffs["I40"]; lower_closed = false)
    target["mu_dist_I_Q3"] = _steady_anal_range_dist(income_mesh, dist, cutoffs["I40"], cutoffs["I60"]; lower_closed = false)
    target["mu_dist_I_Q4"] = _steady_anal_range_dist(income_mesh, dist, cutoffs["I60"], cutoffs["I80"]; lower_closed = false)
    target["mu_dist_I_Q5"] = _steady_anal_range_dist(income_mesh, dist, cutoffs["I80"], cutoffs["I100"]; lower_closed = false)
    target["mu_dist_I_P90"] = _steady_anal_range_dist(income_mesh, dist, cutoffs["I90"], cutoffs["I100"]; lower_closed = false)
    target["mu_dist_I_P99"] = _steady_anal_range_dist(income_mesh, dist, cutoffs["I99"], cutoffs["I100"]; lower_closed = false)
    target["mu_dist_I_P999"] = _steady_anal_range_dist(income_mesh, dist, cutoffs["I999"], cutoffs["I100"]; lower_closed = false)
    target["mu_dist_I_B60"] = _steady_anal_upper_dist(income_mesh, dist, cutoffs["I60"])
    return nothing
end

function _steady_anal_copy_distribution_groups!(Q_dists, SS_stats, mu_dist, mu_dist_tilde)
    Q_dists["mu_dist"] = mu_dist
    for suffix in _STEADY_ANAL_EXTENDED_GROUPS
        Q_dists["mu_dist_$suffix"] = SS_stats["mu_dist_$suffix"]
        Q_dists["mu_dist_aux_$suffix"] = SS_stats["mu_dist_$suffix"]
        SS_stats["mu_dist_aux_$suffix"] = SS_stats["mu_dist_$suffix"]
    end

    Q_dists["mu_dist_aux"] = SS_stats["mu_dist"]
    SS_stats["mu_dist_aux"] = SS_stats["mu_dist"]

    for suffix in ("B01", "B1", "B10")
        Q_dists["mu_dist_$suffix"] = SS_stats["mu_dist_$suffix"]
        Q_dists["mu_dist_tilde_$suffix"] = SS_stats["mu_dist_tilde_$suffix"]
    end

    Q_dists["mu_dist_tilde"] = mu_dist_tilde
    Q_dists["mu_dist_tilde_aux"] = mu_dist_tilde
    Q_dists["mu_dist_tilde_full"] = mu_dist_tilde
    for suffix in _STEADY_ANAL_EXTENDED_GROUPS
        Q_dists["mu_dist_tilde_$suffix"] = SS_stats["mu_dist_tilde_$suffix"]
    end

    return nothing
end

_steady_anal_marginal_b(dist) = dropdims(sum(sum(dist, dims = 2), dims = 3), dims = (2, 3))
_steady_anal_marginal_a(dist) = vec(dropdims(sum(sum(dist, dims = 1), dims = 3), dims = (1, 3)))
_steady_anal_marginal_se(dist) = dropdims(sum(sum(dist, dims = 1), dims = 2), dims = (1, 2))

function _steady_anal_store_group_marginals!(Q_dists, SS_stats, mu_dist_b, mu_dist_a, mu_dist_se)
    Q_dists["mu_dist_b"] = mu_dist_b
    Q_dists["mu_dist_a"] = mu_dist_a
    Q_dists["mu_dist_se"] = mu_dist_se

    for suffix in _STEADY_ANAL_EXTENDED_GROUPS
        dist = SS_stats["mu_dist_$suffix"]
        Q_dists["mu_dist_b_$suffix"] = _steady_anal_marginal_b(dist)
        Q_dists["mu_dist_a_$suffix"] = _steady_anal_marginal_a(dist)
        Q_dists["mu_dist_se_$suffix"] = _steady_anal_marginal_se(dist)
    end

    return nothing
end

function _steady_anal_weighted_sum(field, dist)
    total = zero(promote_type(eltype(field), eltype(dist)))
    @inbounds for i in eachindex(field, dist)
        total += field[i] * dist[i]
    end
    return total
end

function _steady_anal_weighted_mean(field, dist)
    numerator = zero(promote_type(eltype(field), eltype(dist)))
    denominator = zero(eltype(dist))
    @inbounds for i in eachindex(field, dist)
        numerator += field[i] * dist[i]
        denominator += dist[i]
    end
    return numerator / denominator
end

function _steady_anal_weighted_gini(sorted_values, pdf)
    previous_share = zero(promote_type(eltype(sorted_values), eltype(pdf)))
    lorenz_area = zero(previous_share)
    @inbounds for i in eachindex(sorted_values, pdf)
        current_share = previous_share + pdf[i] * sorted_values[i]
        lorenz_area += pdf[i] * (previous_share + current_share)
        previous_share = current_share
    end
    return 1 - lorenz_area / previous_share
end

function _steady_anal_consumption_weight(c_a_guess, c_n_guess, AProb, meshes, param, grid, SS_stats)
    cons_weight = AProb .* c_a_guess .+ (1 .- AProb) .* c_n_guess

    if param["GHH"] == 1
        ns_1 = Int(grid["ns_1"])
        ns = Int(grid["ns"])
        nw_aux = SS_stats["w_2"] * SS_stats["n_2"] .* meshes["se"]
        nw_aux[:, :, 1:ns_1] .= SS_stats["w_1"] * SS_stats["n_1"] .* meshes["se"][:, :, 1:ns_1]
        cons_weight[:, :, 1:ns] .+= ((1 - param["tau_w"]) / (1 + param["xi"])) .*
                                     nw_aux[:, :, 1:ns] .* meshes["se"][:, :, 1:ns]
    end

    return cons_weight
end

function _steady_anal_group_sums(field, group_pairs)
    out = Dict{String, Float64}()
    for pair in group_pairs
        suffix = pair.first
        dist = pair.second
        out[suffix] = _steady_anal_weighted_sum(field, dist)
    end
    return out
end
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

    # Include helper function
    #include("q_cons.jl")

    # Initialize output dictionary
    anal_stats = Dict{String, Any}()

    ## 1-2. Create auxiliary productivity grids and meshes
    nb_val = Int(grid["nb"])
    na_val = Int(grid["na"])
    nse_val = Int(grid["nse"])
    ns_val = Int(grid["ns"])

    aux = _steady_anal_auxiliary_productivity!(meshes, grid, param, nb_val, na_val)
    grid_se_aux = aux.grid_se_aux
    meshes_se_aux = aux.meshes_se_aux
    meshes_se1_aux = aux.meshes_se1_aux
    meshes_se2_aux = aux.meshes_se2_aux
    meshes_se3_aux = aux.meshes_se3_aux
    ## 3. Compute wage and income grids

    # NW_aux = param.w_bar_2*param.n_2*meshes.se
    # NW_aux(:,:,1:ns_1) = param.w_bar_1*param.n_1*meshes.se(:,:,1:ns_1)
    NW_aux = param["w_bar_2"] * param["n_2"] .* meshes["se"]
    NW_aux[:, :, 1:Int(grid["ns_1"])] .= param["w_bar_1"] * param["n_1"] .* meshes["se"][:, :, 1:Int(grid["ns_1"])]

    # WW_aux = (1-param.tau_w)*NW_aux
    WW_aux = (1 - param["tau_w"]) .* NW_aux

    # WW_aux(:,:,end) = (1-param.tau_w)*(param.Eratio*SS_stats.Profit)/param.Eshare
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
    mesh_a, mesh_b = _steady_anal_asset_meshes(grid, nb_val, na_val)

    # Reduce distribution by summing over se dimension
    mu_redux = dropdims(sum(mu_dist, dims=3), dims=3)

    # Total wealth distribution
    total_wealth, total_wealth_pdf, total_wealth_cdf =
        _steady_anal_sorted_distribution(mesh_a[:] * param["q"] .+ mesh_b[:], mu_redux)

    # Income distribution
    income_unsorted = WW_aux .* meshes_se_aux .+
    SS_stats["r_a_aux"] .* meshes["a"] .+
    bond_return_pos .* meshes["b"] .* (meshes["b"] .>= 0) .+
    bond_return_neg .* meshes["b"] .* (meshes["b"] .< 0) .+
    inc_transfer
    income_sort_vec = income_unsorted[:]
    income_sort_vec, income_pdf_vec, income_cdf = _steady_anal_sorted_distribution(income_sort_vec, mu_dist_tilde)
    income_sort = reshape(income_sort_vec, :, 1)  # Column vector to match MATLAB (n, 1)
    anal_stats["income_sort"] = income_sort
    income_pdf = reshape(income_pdf_vec, :, 1)
    anal_stats["income_pdf"] = income_pdf
    anal_stats["income_cdf"] = income_cdf
    anal_stats["income_cdf"] = reshape(anal_stats["income_cdf"], :, 1)

    # Liquid asset distribution (bonds)
    liquid_sort, liquid_pdf, liquid_cdf = _steady_anal_sorted_distribution(mesh_b[:], mu_redux)

    # Illiquid asset distribution (equity)
    illiquid_sort, illiquid_pdf, illiquid_cdf = _steady_anal_sorted_distribution(mesh_a[:], mu_redux)

    ## 8. Extract percentiles for wealth

    w0 = total_wealth[1]
    w1 = _steady_anal_percentile(total_wealth, total_wealth_cdf, 0.01)
    w10 = _steady_anal_percentile(total_wealth, total_wealth_cdf, 0.1)
    w20 = _steady_anal_percentile(total_wealth, total_wealth_cdf, 0.2)
    w25 = _steady_anal_percentile(total_wealth, total_wealth_cdf, 0.25)
    w30 = _steady_anal_percentile(total_wealth, total_wealth_cdf, 0.3)
    w40 = _steady_anal_percentile(total_wealth, total_wealth_cdf, 0.4)
    w50 = _steady_anal_percentile(total_wealth, total_wealth_cdf, 0.5)
    w60 = _steady_anal_percentile(total_wealth, total_wealth_cdf, 0.6)
    w70 = _steady_anal_percentile(total_wealth, total_wealth_cdf, 0.7)
    w80 = _steady_anal_percentile(total_wealth, total_wealth_cdf, 0.8)
    w90 = _steady_anal_percentile(total_wealth, total_wealth_cdf, 0.9)
    w99 = _steady_anal_percentile(total_wealth, total_wealth_cdf, 0.99)
    w999 = _steady_anal_percentile(total_wealth, total_wealth_cdf, 0.999)
    w100 = total_wealth[end]

    ## 9. Extract percentiles for income

    I0_ss = income_sort[1]
    I01_ss = _steady_anal_percentile(income_sort_vec, income_cdf, 0.001)
    anal_stats["I01_ss"] = I01_ss
    I1_ss = _steady_anal_percentile(income_sort_vec, income_cdf, 0.01)
    I10_ss = _steady_anal_percentile(income_sort_vec, income_cdf, 0.1)
    I20_ss = _steady_anal_percentile(income_sort_vec, income_cdf, 0.2)
    I25_ss = _steady_anal_percentile(income_sort_vec, income_cdf, 0.25)
    I30_ss = _steady_anal_percentile(income_sort_vec, income_cdf, 0.3)
    I40_ss = _steady_anal_percentile(income_sort_vec, income_cdf, 0.4)
    I50_ss = _steady_anal_percentile(income_sort_vec, income_cdf, 0.5)
    I60_ss = _steady_anal_percentile(income_sort_vec, income_cdf, 0.6)
    I70_ss = _steady_anal_percentile(income_sort_vec, income_cdf, 0.7)
    I80_ss = _steady_anal_percentile(income_sort_vec, income_cdf, 0.8)
    I90_ss = _steady_anal_percentile(income_sort_vec, income_cdf, 0.9)
    I99_ss = _steady_anal_percentile(income_sort_vec, income_cdf, 0.99)
    I999_ss = _steady_anal_percentile(income_sort_vec, income_cdf, 0.999)
    I100_ss = income_sort[end]

    ## 10. Extract percentiles for liquid assets

    LI0 = liquid_sort[1]
    LI1 = _steady_anal_percentile(liquid_sort, liquid_cdf, 0.01)
    LI10 = _steady_anal_percentile(liquid_sort, liquid_cdf, 0.1)
    LI20 = _steady_anal_percentile(liquid_sort, liquid_cdf, 0.2)
    LI25 = _steady_anal_percentile(liquid_sort, liquid_cdf, 0.25)
    LI30 = _steady_anal_percentile(liquid_sort, liquid_cdf, 0.3)
    LI40 = _steady_anal_percentile(liquid_sort, liquid_cdf, 0.4)
    LI50 = _steady_anal_percentile(liquid_sort, liquid_cdf, 0.5)
    LI60 = _steady_anal_percentile(liquid_sort, liquid_cdf, 0.6)
    LI70 = _steady_anal_percentile(liquid_sort, liquid_cdf, 0.7)
    LI80 = _steady_anal_percentile(liquid_sort, liquid_cdf, 0.8)
    LI90 = _steady_anal_percentile(liquid_sort, liquid_cdf, 0.9)
    LI99 = _steady_anal_percentile(liquid_sort, liquid_cdf, 0.99)
    LI999 = _steady_anal_percentile(liquid_sort, liquid_cdf, 0.999)
    LI100 = liquid_sort[end]

    ## 11. Extract percentiles for illiquid assets

    IL0 = illiquid_sort[1]
    IL1 = _steady_anal_percentile(illiquid_sort, illiquid_cdf, 0.01)
    IL10 = _steady_anal_percentile(illiquid_sort, illiquid_cdf, 0.1)
    IL20 = _steady_anal_percentile(illiquid_sort, illiquid_cdf, 0.2)
    IL25 = _steady_anal_percentile(illiquid_sort, illiquid_cdf, 0.25)
    IL30 = _steady_anal_percentile(illiquid_sort, illiquid_cdf, 0.3)
    IL40 = _steady_anal_percentile(illiquid_sort, illiquid_cdf, 0.4)
    IL50 = _steady_anal_percentile(illiquid_sort, illiquid_cdf, 0.5)
    IL60 = _steady_anal_percentile(illiquid_sort, illiquid_cdf, 0.6)
    IL70 = _steady_anal_percentile(illiquid_sort, illiquid_cdf, 0.7)
    IL80 = _steady_anal_percentile(illiquid_sort, illiquid_cdf, 0.8)
    IL90 = _steady_anal_percentile(illiquid_sort, illiquid_cdf, 0.9)
    IL99 = _steady_anal_percentile(illiquid_sort, illiquid_cdf, 0.99)
    IL999 = _steady_anal_percentile(illiquid_sort, illiquid_cdf, 0.999)
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

    wealth_cutoffs = Dict(
        "w0" => w0, "w1" => w1, "w10" => w10, "w20" => w20, "w40" => w40,
        "w60" => w60, "w80" => w80, "w90" => w90, "w99" => w99,
        "w999" => w999, "w100" => w100,
    )

    ## 13-15. Create wealth mesh, distribution masks, and transitioned status masks
    meshes_w = param["q"] .* meshes["a"] .+ meshes["b"]

    _steady_anal_store_wealth_groups!(
        SS_stats, meshes_w, mu_dist, mu_dist_tilde, meshes_se3_aux, H_tilde,
        wealth_cutoffs, nb_val, na_val, nse_val, ns_val,
    )

    ## 16. Create Q_dists structure with income-based quintiles

    Q_dists = Dict{String, Any}()

    income_cutoffs = Dict(
        "I0" => I0_ss, "I01" => I01_ss, "I1" => I1_ss, "I10" => I10_ss,
        "I20" => I20_ss, "I40" => I40_ss, "I60" => I60_ss, "I80" => I80_ss,
        "I90" => I90_ss, "I99" => I99_ss, "I999" => I999_ss, "I100" => I100_ss,
    )
    _steady_anal_store_income_groups!(Q_dists, total_income_mesh_aux, SS_stats["mu_dist"], income_cutoffs)

    ## 17-23. Copy grouped distributions and compute marginals
    _steady_anal_copy_distribution_groups!(Q_dists, SS_stats, mu_dist, mu_dist_tilde)
    _steady_anal_store_group_marginals!(Q_dists, SS_stats, mu_dist_b, mu_dist_a, mu_dist_se)

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
    ## 58. Compute consumption by quintile using precomputed q_cons weights
    cons_weight = _steady_anal_consumption_weight(c_a_guess, c_n_guess, AProb, meshes, param, grid, SS_stats)
    cons_groups = _steady_anal_group_sums(cons_weight, (
        "all" => mu_dist_tilde,
        "B01" => mu_dist_tilde_B01,
        "B1" => mu_dist_tilde_B1,
        "B10" => mu_dist_tilde_B10,
        "Q1" => mu_dist_tilde_Q1,
        "Q2" => mu_dist_tilde_Q2,
        "Q3" => mu_dist_tilde_Q3,
        "Q4" => mu_dist_tilde_Q4,
        "Q5" => mu_dist_tilde_Q5,
        "P90" => mu_dist_tilde_P90,
        "P99" => mu_dist_tilde_P99,
        "P999" => mu_dist_tilde_P999,
        "Ent" => mu_dist_tilde_Ent,
        "Emp" => mu_dist_tilde_Emp,
        "Unemp" => mu_dist_tilde_Unemp,
    ))

    c_all = cons_groups["all"]
    c_Q1 = cons_groups["Q1"]
    c_Q2 = cons_groups["Q2"]
    c_Q3 = cons_groups["Q3"]
    c_Q4 = cons_groups["Q4"]
    c_Q5 = cons_groups["Q5"]
    c_P90 = cons_groups["P90"]
    c_P99 = cons_groups["P99"]
    c_P999 = cons_groups["P999"]
    c_Ent = cons_groups["Ent"]
    c_Emp = cons_groups["Emp"]
    c_Unemp = cons_groups["Unemp"]

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
    SS_stats["NegNetWorth"] = sum((total_wealth .< 0) .* total_wealth_pdf)
    SS_stats["GiniW"] = _steady_anal_weighted_gini(total_wealth, total_wealth_pdf)
    SS_stats["Negliquid"] = sum((liquid_sort .< 0) .* liquid_pdf)
    SS_stats["GiniLI"] = _steady_anal_weighted_gini(liquid_sort, liquid_pdf)
    SS_stats["GiniIL"] = _steady_anal_weighted_gini(illiquid_sort, illiquid_pdf)

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

    ## 64. Store consumption statistics in SS_stats by quintile
    SS_stats["C_all"] = cons_groups["all"]
    SS_stats["C_B01"] = cons_groups["B01"]
    SS_stats["C_B1"] = cons_groups["B1"]
    SS_stats["C_B10"] = cons_groups["B10"]
    SS_stats["C_Q1"] = cons_groups["Q1"]
    SS_stats["C_Q2"] = cons_groups["Q2"]
    SS_stats["C_Q3"] = cons_groups["Q3"]
    SS_stats["C_Q4"] = cons_groups["Q4"]
    SS_stats["C_Q5"] = cons_groups["Q5"]
    SS_stats["C_P90"] = cons_groups["P90"]
    SS_stats["C_P99"] = cons_groups["P99"]
    SS_stats["C_P999"] = cons_groups["P999"]
    SS_stats["C_Ent"] = cons_groups["Ent"]
    SS_stats["C_Emp"] = cons_groups["Emp"]
    SS_stats["C_Unemp"] = cons_groups["Unemp"]

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

    SS_stats["B_nb_hh"] = sum((AProb .* b_a_star .+ (1 .- AProb) .* b_n_star) .* mu_dist_tilde)
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

    _steady_anal_store_income_groups!(SS_stats, income_ss, mu_dist, income_cutoffs)

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

    denom_se = gradient(log.(WW_se .* grid_se_aux))
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

for mm in 1:nb
    for kk in 1:na
        num_a = gradient(log.(vec(c_a_guess[mm, kk, :])))
        num_n = gradient(log.(vec(c_n_guess[mm, kk, :])))

        MPC_a_se[mm, kk, :] .= num_a ./ denom_se
        MPC_n_se[mm, kk, :] .= num_n ./ denom_se
    end
end

    # Aggregate MPCs
    MPC_se = AProb .* MPC_a_se .+ (1 .- AProb) .* MPC_n_se
    MPC_b = AProb .* MPC_a_b .+ (1 .- AProb) .* MPC_n_b

    # Expected MPCs by group
    EMPC_se = _steady_anal_weighted_mean(MPC_se, mu_dist_tilde)
    EMPC_se_Q1 = _steady_anal_weighted_mean(MPC_se, mu_dist_tilde_Q1)
    EMPC_se_Q2 = _steady_anal_weighted_mean(MPC_se, mu_dist_tilde_Q2)
    EMPC_se_Q3 = _steady_anal_weighted_mean(MPC_se, mu_dist_tilde_Q3)
    EMPC_se_Q4 = _steady_anal_weighted_mean(MPC_se, mu_dist_tilde_Q4)
    EMPC_se_Q5 = _steady_anal_weighted_mean(MPC_se, mu_dist_tilde_Q5)
    EMPC_se_P90 = _steady_anal_weighted_mean(MPC_se, mu_dist_tilde_P90)
    EMPC_se_P99 = _steady_anal_weighted_mean(MPC_se, mu_dist_tilde_P99)
    EMPC_se_P999 = _steady_anal_weighted_mean(MPC_se, mu_dist_tilde_P999)

    EMPC_b = _steady_anal_weighted_mean(MPC_b, mu_dist_tilde)
    EMPC_b_Q1 = _steady_anal_weighted_mean(MPC_b, mu_dist_tilde_Q1)
    EMPC_b_Q2 = _steady_anal_weighted_mean(MPC_b, mu_dist_tilde_Q2)
    EMPC_b_Q3 = _steady_anal_weighted_mean(MPC_b, mu_dist_tilde_Q3)
    EMPC_b_Q4 = _steady_anal_weighted_mean(MPC_b, mu_dist_tilde_Q4)
    EMPC_b_Q5 = _steady_anal_weighted_mean(MPC_b, mu_dist_tilde_Q5)
    EMPC_b_P90 = _steady_anal_weighted_mean(MPC_b, mu_dist_tilde_P90)
    EMPC_b_P99 = _steady_anal_weighted_mean(MPC_b, mu_dist_tilde_P99)
    EMPC_b_P999 = _steady_anal_weighted_mean(MPC_b, mu_dist_tilde_P999)

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

        MRS_mix = AProb .* MRS_a_aux2 .+ (1 .- AProb) .* MRS_n_aux2
        MRSs = _steady_anal_weighted_mean(MRS_mix, mu_dist_tilde)
        MRSs_Q1 = _steady_anal_weighted_mean(MRS_mix, mu_dist_tilde_Q1)
        MRSs_Q2 = _steady_anal_weighted_mean(MRS_mix, mu_dist_tilde_Q2)
        MRSs_Q3 = _steady_anal_weighted_mean(MRS_mix, mu_dist_tilde_Q3)
        MRSs_Q4 = _steady_anal_weighted_mean(MRS_mix, mu_dist_tilde_Q4)
        MRSs_Q5 = _steady_anal_weighted_mean(MRS_mix, mu_dist_tilde_Q5)
        MRSs_P90 = _steady_anal_weighted_mean(MRS_mix, mu_dist_tilde_P90)
        MRSs_P99 = _steady_anal_weighted_mean(MRS_mix, mu_dist_tilde_P99)
        MRSs_P999 = _steady_anal_weighted_mean(MRS_mix, mu_dist_tilde_P999)
        MRSs_Ent = _steady_anal_weighted_mean(MRS_mix, mu_dist_tilde_Ent)
        MRSs_Emp = _steady_anal_weighted_mean(MRS_mix, mu_dist_tilde_Emp)
        MRSs_Unemp = _steady_anal_weighted_mean(MRS_mix, mu_dist_tilde_Unemp)

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

    return anal_stats, SS_stats, Q_dists
end
