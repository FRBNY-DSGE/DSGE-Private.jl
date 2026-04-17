@inline _mbbq_value(x) = hasproperty(x, :value) ? getproperty(x, :value) : x
@inline _mbbq_getvalue(m::mBBQ, key::Symbol, default = nothing) = haskey(m.keys, key) ? _mbbq_value(m[key]) : default

function _mbbq_ss_param_dict(m::mBBQ)
    param = Dict{String, Any}()

    param["A_F_ratio"] = _mbbq_getvalue(m, :A_F_ratio, 0.0)
    param["A_g_ratio"] = _mbbq_getvalue(m, :A_g_ratio, 0.0)
    param["Ab_ratio"] = _mbbq_getvalue(m, :Ab_ratio_init, 0.1)
    param["b_a_aux"] = _mbbq_getvalue(m, :b_a_aux)
    param["b_a_aux2"] = _mbbq_getvalue(m, :b_a_aux2)
    param["b_aux"] = _mbbq_getvalue(m, :b_aux)
    param["B_F_ratio"] = _mbbq_getvalue(m, :B_F_ratio, 0.5)
    param["B_gov_ratio"] = _mbbq_getvalue(m, :B_gov_ratio, 0.6)
    param["b_ratio"] = get_setting(m, :b_ratio)
    param["b_share"] = _mbbq_getvalue(m, :b_share)
    param["Bb_ratio"] = _mbbq_getvalue(m, :Bb_ratio, 0.5)
    param["beta"] = _mbbq_getvalue(m, :β)
    param["beta_L"] = _mbbq_getvalue(m, :β_L, _mbbq_getvalue(m, :β))
    param["death_rate"] = _mbbq_getvalue(m, :dr)
    param["DELTA"] = NaN
    param["delta_1"] = _mbbq_getvalue(m, :δ_1)
    param["delta_ss"] = NaN
    param["Eratio"] = _mbbq_getvalue(m, :Eratio)
    param["Eshare"] = _mbbq_getvalue(m, :Eshare)
    param["eta"] = _mbbq_getvalue(m, :η)
    param["fix"] = _mbbq_getvalue(m, :fix)
    param["fix_L"] = _mbbq_getvalue(m, :fix_L)
    param["guessadj"] = _mbbq_getvalue(m, :guessadj, 0.5)
    param["in"] = _mbbq_getvalue(m, :in)
    param["iota"] = _mbbq_getvalue(m, :ι)
    param["lambda"] = _mbbq_getvalue(m, :λ_sep, 0.03)
    param["Lambda_bank"] = _mbbq_getvalue(m, :β_b)
    param["LT"] = 0.0
    param["LT_ratio"] = _mbbq_getvalue(m, :LT_ratio, 0.0)
    param["MMF_cont_ratio"] = NaN
    param["mu_chi"] = _mbbq_getvalue(m, :μ_χ)
    param["n"] = get_setting(m, :n)
    param["nu"] = _mbbq_getvalue(m, :ν)
    param["omega"] = _mbbq_getvalue(m, :ω)
    param["out"] = get_setting(m, :out)
    param["P_SS"] = copy(m.grids[:P_SS])
    param["P_SS2"] = copy(m.grids[:P_SS2])
    param["pi_cb"] = _mbbq_getvalue(m, :π_cb)
    param["q"] = _mbbq_getvalue(m, :q_ss, 1.0)
    param["R_cb"] = _mbbq_getvalue(m, :R_cb_ss, 1.0069)
    param["Rprem"] = _mbbq_getvalue(m, :Rprem)
    param["sigma"] = _mbbq_getvalue(m, :σ_2)
    param["sigma_chi"] = _mbbq_getvalue(m, :σ_χ)
    param["ss_obj_old"] = haskey(m.settings, :ss_obj_old) ? m.settings[:ss_obj_old].value : nothing
    param["tau_a"] = _mbbq_getvalue(m, :τ_a)
    param["tau_w"] = _mbbq_getvalue(m, :τ_w)
    param["THETA"] = _mbbq_getvalue(m, :THETA_ss, 10.0)
    param["theta_2"] = _mbbq_getvalue(m, :θ_2)
    param["theta_b"] = _mbbq_getvalue(m, :θ_b)
    param["theta"] = _mbbq_getvalue(m, :θ)
    param["u"] = _mbbq_getvalue(m, :u_target, 0.058)
    param["v"] = _mbbq_getvalue(m, :v_util, 0.75)
    param["v_f"] = _mbbq_getvalue(m, :v_f_target, 0.7)
    param["w_bar"] = _mbbq_getvalue(m, :w_bar)
    param["wmarkdown"] = _mbbq_getvalue(m, :wmarkdown, 1.0)
    param["xi"] = _mbbq_getvalue(m, :ξ)

    return param
end

function _mbbq_ss_grid_dict(m::mBBQ)
    grid = Dict{String, Any}()

    grid["a"] = copy(get_gridpts(m, :a_grid))
    grid["b"] = copy(get_gridpts(m, :b_grid))
    grid["K"] = haskey(m.grids, :K) ? m.grids[:K] : _mbbq_getvalue(m, :K_ss_init, 30.0)
    grid["L"] = haskey(m.grids, :L) ? m.grids[:L] : NaN
    grid["N"] = haskey(m.grids, :N) ? m.grids[:N] : NaN
    grid["na"] = get_setting(m, :na)
    grid["nb"] = get_setting(m, :nb)
    grid["ns"] = get_setting(m, :ns)
    grid["nse"] = get_setting(m, :nse)
    grid["s"] = copy(m.grids[:s_grid])
    grid["s_bar"] = m.grids[:s_bar]
    grid["s_dist"] = copy(m.grids[:s_dist])
    grid["s2_bar"] = m.grids[:s2_bar]
    grid["se"] = copy(get_gridpts(m, :se_grid))
    grid["se_bar"] = haskey(m.grids, :se_bar) ? m.grids[:se_bar] : m.grids[:s_bar]
    grid["se_dist"] = haskey(m.grids, :se_dist) ? copy(m.grids[:se_dist]) : copy(m.grids[:s_dist])

    return grid
end

function _mbbq_meshes(m::mBBQ)
    return Dict{String, Any}(
        "b" => m.grids[:b_ndgrid],
        "a" => m.grids[:a_ndgrid],
        "se" => m.grids[:se_ndgrid],
    )
end

function ss_ump_new(m::mBBQ)
    c_n_guess, b_n_star, c_a_guess, b_a_star, a_a_star, psi_guess, mu_dist,
        AProb, Value, R_fc, H_fc, W_fc, r_k, mc, v, J, u, V, M, f, Profits_fc,
        Output, n, grid, param, P_SE, Lambda, TT, WW, THETA, NWb, Profit_FI, Ab, Bb, Cb =
        solve_SS(m)

    meshes = Dict{String, Any}()
    nb_val = Int(grid["nb"])
    na_val = Int(grid["na"])
    nse_val = Int(grid["nse"])

    meshes["b"] = zeros(nb_val, na_val, nse_val)
    meshes["a"] = zeros(nb_val, na_val, nse_val)
    meshes["se"] = zeros(nb_val, na_val, nse_val)

    for k in 1:nse_val
        for j in 1:na_val
            for i in 1:nb_val
                meshes["b"][i, j, k] = grid["b"][i]
                meshes["a"][i, j, k] = grid["a"][j]
                meshes["se"][i, j, k] = grid["se"][k]
            end
        end
    end

    grid_se_aux = hcat(grid["s"],
                      min.(param["b_ratio"] .* grid["s"], grid["s_bar"]),
                      [1])
    meshes["se_aux"] = zeros(nb_val, na_val, size(grid_se_aux, 2))
    for k in 1:size(grid_se_aux, 2)
        for j in 1:na_val
            for i in 1:nb_val
                meshes["se_aux"][i, j, k] = grid_se_aux[k]
            end
        end
    end

    RBRB = param["R_cb"] / param["pi_cb"] .+ (meshes["b"] .< 0) .* (param["Rprem"] / param["pi_cb"])

    mutil_c_n = 1 ./ (c_n_guess .^ param["sigma"])
    mutil_c_a = 1 ./ (c_a_guess .^ param["sigma"])
    mutil_c = AProb .* mutil_c_a .+ (1 .- AProb) .* mutil_c_n

    Vb = RBRB / (1 - param["death_rate"]) * param["b_a_aux"] .* mutil_c
    Vb = reshape(reshape(Vb, (nb_val * na_val, nse_val)), (nb_val, na_val, nse_val))

    r_a_aux = R_fc + param["death_rate"] / (1 - param["death_rate"]) * param["q"]

    Va = AProb .* (r_a_aux + param["q"]) .* mutil_c_a .+
         (1 .- AProb) .* r_a_aux .* mutil_c_n .+
         (1 .- AProb) .* psi_guess
    Va = reshape(reshape(Va, (nb_val * na_val, nse_val)), (nb_val, na_val, nse_val))

    cum_dist = cumsum(cumsum(cumsum(mu_dist, dims=1), dims=2), dims=3)
    marginal_b = cumsum(dropdims(sum(sum(mu_dist, dims=2), dims=3), dims=(2,3)))
    marginal_a = cumsum(dropdims(sum(sum(mu_dist, dims=1), dims=3), dims=(1,3)))
    marginal_se = cumsum(dropdims(sum(sum(mu_dist, dims=2), dims=1), dims=(1,2)))

    marginal_a_t = vec(marginal_a')
    Copula_itp = interpolate((marginal_b, marginal_a_t, marginal_se), cum_dist, Gridded(Linear()))
    Copula_itp_ext = extrapolate(Copula_itp, Line())
    meshes_b, meshes_a, meshes_se = ndgrid(marginal_b, marginal_a_t, marginal_se)
    Copula = Copula_itp_ext.(meshes_b, meshes_a, meshes_se)

    SS_stats, grid, param = Cal_SS_stats(
        mu_dist, c_a_guess, c_n_guess, AProb, b_n_star, b_a_star, a_a_star,
        grid, param, meshes, TT, Output, R_fc, W_fc, H_fc, Profits_fc,
        n, u, v, J, V, M, f, mc, Lambda, NWb, THETA, Profit_FI, Ab, Bb, Cb,
        mutil_c, mutil_c_a, mutil_c_n, Value
    )

    anal_stats, SS_stats_steady, Q_dists = steady_anal(grid, param, meshes, SS_stats, AProb, mu_dist, n, SS_stats["H_tilde"],
                    b_n_star, b_a_star, a_a_star, c_n_guess, c_a_guess)

    merge!(SS_stats, SS_stats_steady)

    return Dict{String, Any}(
        "c_n_guess" => c_n_guess,
        "b_n_star" => b_n_star,
        "c_a_guess" => c_a_guess,
        "b_a_star" => b_a_star,
        "a_a_star" => a_a_star,
        "psi_guess" => psi_guess,
        "mu_dist" => mu_dist,
        "AProb" => AProb,
        "Value" => Value,
        "R_fc" => R_fc,
        "H_fc" => H_fc,
        "W_fc" => W_fc,
        "r_k" => r_k,
        "mc" => mc,
        "v" => v,
        "J" => J,
        "u" => u,
        "V" => V,
        "M" => M,
        "f" => f,
        "Profits_fc" => Profits_fc,
        "Output" => Output,
        "n" => n,
        "grid" => grid,
        "param" => param,
        "P_SE" => P_SE,
        "Lambda" => Lambda,
        "TT" => TT,
        "WW" => WW,
        "THETA" => THETA,
        "NWb" => NWb,
        "Profit_FI" => Profit_FI,
        "Ab" => Ab,
        "Bb" => Bb,
        "Cb" => Cb,
        "mutil_c" => mutil_c,
        "mutil_c_a" => mutil_c_a,
        "mutil_c_n" => mutil_c_n,
        "Vb" => Vb,
        "Va" => Va,
        "meshes" => meshes,
        "SS_stats" => SS_stats,
        "Copula" => Copula,
        "anal_stats" => anal_stats,
        "Q_dists" => Q_dists,
    )
end

function _store_ss_results!(m::mBBQ, results::Dict{String, Any})
    m[:Output_star] = results["Output"]
    m[:w_bar_star] = results["W_fc"]
    m[:R_a_star] = results["R_fc"]
    m[:r_l_star] = results["H_fc"]
    m[:r_k_star] = results["r_k"]
    m[:mc_star] = results["mc"]
    m[:u_star] = results["u"]
    m[:n_star] = results["n"]
    m[:v_star] = results["v"]
    m[:Profit_star] = results["Profits_fc"]
    m[:THETA_star] = results["THETA"]
    m[:NWb_star] = results["NWb"]
    m[:Profit_FI_star] = results["Profit_FI"]
    m[:Lambda_star] = results["Lambda"]
    m[:f_star] = results["f"]
    m[:M_star] = results["M"]
    m[:Ab_star] = results["Ab"]
    m[:Bb_star] = results["Bb"]
    m[:Cb_star] = results["Cb"]
    m[:V_star] = results["V"]

    mu_dist = results["mu_dist"]
    pdf_b = vec(dropdims(sum(mu_dist, dims=(2, 3)), dims=(2, 3)))
    pdf_a = vec(dropdims(sum(mu_dist, dims=(1, 3)), dims=(1, 3)))
    pdf_se = vec(dropdims(sum(mu_dist, dims=(1, 2)), dims=(1, 2)))
    m[:mu_dist_star] = vec(mu_dist)
    m[:marginal_pdf_b_star] = pdf_b
    m[:marginal_pdf_a_star] = pdf_a
    m[:marginal_pdf_se_star] = pdf_se
    m[:marginal_cdf_b_star] = cumsum(pdf_b)
    m[:marginal_cdf_a_star] = cumsum(pdf_a)
    m[:marginal_cdf_se_star] = cumsum(pdf_se)

    m[:Value_star] = vec(results["Value"])
    m[:Vb_star] = vec(results["Vb"])
    m[:Va_star] = vec(results["Va"])
    m[:mutil_c_star] = vec(results["mutil_c"])

    m <= Setting(:copula, results["Copula"])
    m <= Setting(:SS_stats, results["SS_stats"])
    m <= Setting(:anal_stats, results["anal_stats"])
    m <= Setting(:Q_dists, results["Q_dists"])

    updated_grid = results["grid"]
    m.grids[:K] = updated_grid["K"]
    m.grids[:L] = updated_grid["L"]
    m.grids[:N] = updated_grid["N"]
    m.grids[:se_dist] = vec(updated_grid["se_dist"])
    m.grids[:se_bar] = updated_grid["se_bar"]

    return m
end

function steadystate!(m::mBBQ; verbose::Symbol = :none)
    if get_setting(m, :compute_full_steadystate)
        results = ss_ump_new(m)
        _store_ss_results!(m, results)
        m <= Setting(:compute_full_steadystate, false)
    end
    return m
end
