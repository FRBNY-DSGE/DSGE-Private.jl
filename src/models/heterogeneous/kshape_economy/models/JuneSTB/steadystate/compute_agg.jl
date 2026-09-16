using LinearAlgebra
using NLsolve

include(joinpath(@__DIR__, "..", "..", "helpers", "set_field!.jl"))
include(joinpath(@__DIR__, "find_fs.jl"))
include(joinpath(@__DIR__, "find_ws.jl"))

function find_alpha(in::Float64, V::Float64, vf::Float64, U::Float64, param::Dict{String,Any})
    alpha = exp(in)
    M = vf * V
    return M - U * V / ((U^alpha + V^alpha)^(1 / alpha))
end

function _solve_scalar_root(f, init::Float64)
    sol = nlsolve((F, x) -> (F[1] = f(x[1])), [init], show_trace=false, ftol=1e-15, xtol=1e-15)
    return sol.zero[1], sol
end

function compute_agg(meshes::Dict{String,Array{Float64,3}}, grid::Dict{String,Any}, param::Dict{String,Any})
    nb = Int(grid["nb"])
    na = Int(grid["na"])
    ns = Int(grid["ns"])
    ns_1 = Int(grid["ns_1"])
    ns_2 = Int(grid["ns_2"])
    nse = Int(grid["nse"])

    v = param["v"]
    Lambda = param["beta"]

    fs_init = log.([0.70, 0.70])
    fs_sol = nlsolve(
        (F, x) -> begin
            resid = find_fs(x, param, grid)
            F[1] = resid[1]
            F[2] = resid[2]
        end,
        fs_init;
        show_trace=false,
        ftol=1e-15,
        xtol=1e-15,
    )
    if !fs_sol.f_converged && !fs_sol.x_converged
        @warn "Could not find steady-state job-finding rates."
    end

    f_1 = exp(fs_sol.zero[1])
    f_2 = exp(fs_sol.zero[2])
    set_field!(param, :f_1, f_1)
    set_field!(param, :f_2, f_2)

    blocks = _csc_compute_labor_blocks(f_1, f_2, param, grid)

    set_field!(param, :P_SS3, blocks.P_SS3)
    set_field!(param, :P_SE, blocks.P_SE)
    set_field!(param, :P_SS4, blocks.P_SS4)
    set_field!(grid, :se_dist, blocks.se_dist)
    set_field!(grid, :se_bar_1, blocks.se_bar_1)
    set_field!(grid, :se_bar_2, blocks.se_bar_2)
    set_field!(grid, :se_dist_2, blocks.se_dist_2)
    set_field!(grid, :se_dist_3, blocks.se_dist_3)

    u_1 = blocks.U_tilde_1 / (blocks.N_tilde_1 + blocks.U_tilde_1)
    u_2 = blocks.U_tilde_2 / (blocks.N_tilde_2 + blocks.U_tilde_2)
    set_field!(param, :u_1, u_1)
    set_field!(param, :u_2, u_2)

    V_1 = blocks.M_1 / param["v_f_1"]
    V_2 = blocks.M_2 / param["v_f_2"]

    alpha_1, sol_alpha_1 = _solve_scalar_root(x -> find_alpha(x, V_1, param["v_f_1"], blocks.U_1, param), log(1.0))
    if !sol_alpha_1.f_converged && !sol_alpha_1.x_converged
        @warn "Could not find steady-state alpha_1."
    end
    set_field!(param, :alpha_1, exp(alpha_1))

    alpha_2, sol_alpha_2 = _solve_scalar_root(x -> find_alpha(x, V_2, param["v_f_2"], blocks.U_2, param), log(1.0))
    if !sol_alpha_2.f_converged && !sol_alpha_2.x_converged
        @warn "Could not find steady-state alpha_2."
    end
    set_field!(param, :alpha_2, exp(alpha_2))

    mc = (param["eta"] - 1) / param["eta"]

    n_1 = param["n_1"]
    n_2 = param["n_2"]

    L_1 = blocks.N_tilde_1 * blocks.se_bar_1 * n_1
    L_2 = blocks.N_tilde_2 * blocks.se_bar_2 * n_2

    V_aux = (param["w"] * grid["K"]^param["rho"] + (1 - param["w"]) * L_2^param["rho"])^(1 / param["rho"])
    F_aux = (param["a"] * L_1^param["zeta"] + (1 - param["a"]) * V_aux^param["zeta"])^(1 / param["zeta"])
    Y = param["Z"] * F_aux

    r_l_1 = mc * param["Z"] * F_aux^(1 - param["zeta"]) * param["a"] * L_1^(param["zeta"] - 1)
    r_k = mc * param["Z"] * F_aux^(1 - param["zeta"]) * (1 - param["a"]) * V_aux^(param["zeta"] - param["rho"]) * param["w"] * grid["K"]^(param["rho"] - 1)
    r_l_2 = mc * param["Z"] * F_aux^(1 - param["zeta"]) * (1 - param["a"]) * V_aux^(param["zeta"] - param["rho"]) * (1 - param["w"]) * L_2^(param["rho"] - 1)

    ws_init = [0.95 * r_l_1, 0.05]
    ws_sol = nlsolve(
        (F, x) -> begin
            resid = find_ws(x, r_l_1, r_l_2, V_1, V_2, blocks.M_1, blocks.M_2, param, grid)
            F[1] = resid[1]
            F[2] = resid[2]
        end,
        ws_init;
        show_trace=false,
        ftol=1e-15,
        xtol=1e-15,
    )
    if !ws_sol.f_converged && !ws_sol.x_converged
        @warn "Could not find steady-state wage and fixed labor cost."
    end

    w_bar_1 = ws_sol.zero[1]
    fix_L_2 = ws_sol.zero[2]
    w_bar_2 = w_bar_1 * param["earning_ratio"] / (blocks.se_bar_2 / blocks.se_bar_1)

    set_field!(param, :fix_L_1, 0.0)
    set_field!(param, :fix_L_2, fix_L_2)
    set_field!(param, :w_bar_1, w_bar_1)
    set_field!(param, :w_bar_2, w_bar_2)

    J_aux, _, _ = _csc_job_value_stats(r_l_1, r_l_2, w_bar_1, w_bar_2, param["fix_L_1"], param["fix_L_2"], param, grid)

    w_1 = param["w_bar_1"]
    w_2 = param["w_bar_2"]

    Profit_L = (r_l_1 - param["fix_L_1"] - param["w_bar_1"]) * L_1 - param["iota_1"] * V_1 +
               (r_l_2 - param["fix_L_2"] - param["w_bar_2"]) * L_2 - param["iota_2"] * V_2
    Profit_K = (r_k * v - param["delta_ss"]) * grid["K"]
    Profit_int = Y - mc * Y - param["fix"]
    Profit = Profit_int + Profit_L + Profit_K

    set_field!(param, :fix_ratio, param["fix"] / Y)

    r_a = (1 - param["tau_a"]) * (1 - param["Eratio"] - param["b_share"]) * Profit / grid["K"]

    tau_L = param["tau_L"]
    tau_P = param["tau_P"]
    xi = param["xi"]
    coeff = (1 / xi + tau_P) / (1 + 1 / xi) * param["GHH"]

    NW_pretax = fill(coeff * n_1 * param["w_bar_1"], nb, na, nse)
    NW_pretax[:, :, (ns_1 + 1):ns] .= coeff * n_2 * param["w_bar_2"]
    NW_pretax[:, :, (ns + ns_1 + 1):(ns + ns)] .= coeff * n_2 * param["w_bar_2"]
    WW_pretax = copy(NW_pretax)
    WW_pretax[:, :, end] .= (param["Eratio"] * Profit) / param["Eshare"]

    NW = fill(coeff * (n_1 * param["w_bar_1"])^(1 - tau_P), nb, na, nse)
    NW[:, :, (ns_1 + 1):ns] .= coeff * (n_2 * param["w_bar_2"])^(1 - tau_P)
    NW[:, :, (ns + ns_1 + 1):(ns + ns)] .= coeff * (n_2 * param["w_bar_2"])^(1 - tau_P)
    WW = (1 - tau_L) .* NW
    WW[:, :, end] .= (1 - tau_L) * ((param["Eratio"] * Profit) / param["Eshare"])^(1 - tau_P)

    se = grid["se"]
    dist = blocks.se_dist
    emp = 1:ns
    unemp = (ns + 1):(ns + ns)
    Lincome_pretax = dot(vec(WW_pretax[1, 1, emp]) .* se[emp], dist[emp])
    UB_pretax = dot(vec(WW_pretax[1, 1, unemp]) .* se[unemp], dist[unemp])
    Bincome_pretax = param["Eratio"] * Profit
    Lincome_aftertax = dot(vec(WW[1, 1, emp]) .* se[emp].^(1 - tau_P), dist[emp])
    UB_aftertax = dot(vec(WW[1, 1, unemp]) .* se[unemp].^(1 - tau_P), dist[unemp])
    Bincome_aftertax = param["Eshare"] * WW[1, 1, end]
    Income_Tax = Lincome_pretax - Lincome_aftertax + UB_pretax - UB_aftertax + Bincome_pretax - Bincome_aftertax
    TAX_revenue = Income_Tax + param["tau_a"] * (1 - param["Eratio"]) * Profit
    UB = UB_pretax
    B_gov = param["B_gov_ratio"] * Y

    set_field!(param, :Income_Tax, Income_Tax)
    set_field!(param, :MMMF_cont_ratio, 0.01 * Y / TAX_revenue)
    MMMF_cont = param["MMMF_cont_ratio"] * TAX_revenue

    set_field!(param, :psi_1, (1 - tau_L) * (1 - tau_P) * param["w_bar_1"]^(1 - tau_P) * (1 / param["n_1"])^((1 + xi * tau_P) / xi))
    set_field!(param, :psi_2, (1 - tau_L) * (1 - tau_P) * param["w_bar_2"]^(1 - tau_P) * (1 / param["n_2"])^((1 + xi * tau_P) / xi))

    RR = r_a
    RBRB = param["R_cb"] / param["pi_cb"] .+ (meshes["b"] .< 0) .* (param["Rprem"] / param["pi_cb"])

    K_I_agg = param["Ab_ratio"] * grid["K"]
    K_g_agg = param["A_g_ratio"] * Y
    B_g = K_g_agg

    R = param["R_cb"] / param["pi_cb"]
    R_a = (param["q"] + RR) / param["q"]
    Ab = K_I_agg
    THETA = param["THETA"]

    set_field!(param, :omega, (1 - param["theta_b"] * ((R_a - R) * THETA + R)) / THETA)

    zz = (R_a - R) * THETA + R
    xx = zz
    ee = (1 - param["theta_b"]) * Lambda * R / (1 - param["theta_b"] * zz * Lambda)
    vv = ((1 - param["theta_b"]) * (R_a - R) * Lambda) / (1 - param["theta_b"] * xx * Lambda)
    set_field!(param, :DELTA, vv + ee / THETA)

    NWb = Ab / THETA
    Profit_FI = (1 - param["theta_b"]) * ((R_a - R) * THETA + R) * NWb - param["omega"] * param["q"] * Ab
    B_FI = (THETA - 1) * NWb

    B_MMMF = B_FI
    T_MMMF = MMMF_cont + (R - 1) * B_MMMF

    return mc, L_1, L_2, n_1, n_2, r_l_1, r_l_2, r_k, Profit, v, J_aux, u_1, u_2, V_1, V_2, blocks.M_1, blocks.M_2,
           f_1, f_2, w_1, w_2, WW, RR, RBRB, Y, blocks.P_SE, Lambda, param, grid, THETA, NWb, Profit_FI, Ab, B_FI,
           B_MMMF, T_MMMF, B_gov, B_g, K_g_agg, TAX_revenue, UB
end
