using OrderedCollections: OrderedDict

function build_indices(grid, lenSSS, lenCSS)
    #must appear in exact order as indexation in F_sys_ref_tvcopula_QE
    agg_states = [:R_cb_t, :w_t, :A_b_t, :B_b_t, :A_gaux_t, :Q_t, :lev_t, :NW_b_t, :R_tilde_t, :x_cb_t,
    :π_past_t, :Y_past_t, :C_past_t, :I_past_t, :Profit_past_t, :unemp_past_t, :G_past_t, :LT_past_t,
    :R_star_t, :B_F_t, :Z_t, :ψ_rp_t, :η_t, :D_t, :GG_t, :ι_t, :BB_t, :ψ_w_t, :MP_t, :p_m_t]
    
    shock_states = [:eps_QE_t, :eps_RP_t,:eps_B_F_t, :eps_BB_t, :eps_Z_t, :eps_G_t, :eps_D_t,
    :eps_R_t, :eps_iota_t, :eps_eta_t, :eps_w_t]

    summary_controls = [:MRS_t, :A_hh_t, :B_hh_t, :C_t, :N_t, :L_t, :UB_t]

    other_controls = [:K_t, :B_t, :B_gov_ncp_t, :T_t, :LT_t, :G_t, :λ_t, :pi_t, :V_t,
    :J_t,
    :h_t, :v_t, :Y_t, :Profit_t, :r_k_t, :r_a_t, :MC_t, :unemp_t, :nn_t, :M_t, :f_t,
    :zz_t, :xx_t, :vv_t, :ee_t, :C_b_t, :Profit_FI_t, :RRa_t, :RR_t, :I_t, :x_k_t,]

    observable_controls = [:A_g_obs_t, :Y_obs_t, :C_obs_t, :I_obs_t, :w_obs_t, :Profit_obs_t, :unemp_obs_t,
    :pi_obs_t, :R_obs_t, :w_lag_t, :G_obs_t,
    # Past/auxiliary controls
    :YY_lag_t, :CC_lag_t, :II_lag_t, :PPROFIT_lag_t, :uu_lag_t, :GG_lag_t, :A_g_lag_t,
    :l_λ_t , :Q_lag_t, :x_I_t, :η2_t, :ι_2_t, :LT2_lag_t, :G2_lag_t, :LT_obs_t, :B_gov_ncp2_t]
    
    nb  = Int(grid[:nb])
    na  = Int(grid[:na])
    nse = Int(grid[:nse])
    ns  = Int(grid[:ns])
    nCOP = Int(grid[:nCOP])
    oc  = Int(grid[:oc])
    numstates = Int(grid[:numstates])

    NN = nb * na * nse
    # println("NN IN INDEX $NN")
    nRedMarg = nb + na + nse - 3
    NxNx = nRedMarg + nCOP

    ny = 3 * NN #+ oc

    # ─── state_id ─────────────────────────────────────────────────────────────
    state_id = OrderedDict{Symbol, Union{Int, UnitRange{Int}, Vector{Int}}}()

    # Distribution block: reduced marginals + copula coefficients
    #These indexes are all skipping an index but this is from the code so idk
    state_id[:marginal_pdf_b_t] = collect(1:nb-1)
    state_id[:marginal_pdf_a_t] = collect(nb:nb+na-2)
    state_id[:marginal_pdf_se_t] = collect(nb+na-1:nb+na+nse-3)

    state_id[:copula_t]         = nRedMarg+1:nRedMarg+nCOP

    next = lenSSS - (41 - 1) #TODO: how to get 41 without manually setting

    for sym in vcat(agg_states, shock_states)
        state_id[sym] = next
        next += 1
    end


    # @assert next - 1 == numstates "state_id covers $(next-1) indices but numstates = $numstates"

    # ─── control_id ───────────────────────────────────────────────────────────
    control_id = OrderedDict{Symbol, Union{Int, UnitRange{Int}, Vector{Int}}}()
    #next = 1

    # Distributional controls (full-grid, NN = nb*na*nse each) TODO: fix this
    control_id[:Value_t] = collect(1:NN)
    control_id[:mutil_c_t] = collect(NN+1:2*NN)
    control_id[:Va_t] = collect(2*NN+1:3*NN)

    next = ny + 1


    for sym in vcat(summary_controls, other_controls, observable_controls)
        if sym == :J_t
            control_id[sym] = next:(next + ns - 1)
            next += ns
        else
            control_id[sym] = next
            next += 1
        end
    end


    # @assert next - 1 == ny "control_id covers $(next-1) indices but ny (3*NN + oc) = $ny"

    return (state_id, control_id)
end

"""
    build_combined_indices(state_id, control_id, nState) -> OrderedDict

Merge `state_id` and `control_id` into a single index dict for use with the
combined vector `X = [states; controls]` (length `nState + nCtrl`).

State entries are unchanged. Control entries are offset by `nState` so that
`X[id[:Value_t]]` retrieves the period-t Value function deviations, and
`XPrime[id[:Value_t]]` retrieves the period-(t+1) deviations — same index,
different vector, just like BBL.
"""
function build_combined_indices(state_id::AbstractDict, control_id::AbstractDict, nState::Int)
    id = OrderedDict{Symbol, Union{Int, UnitRange{Int}}}()
    for (k, v) in state_id
        id[k] = v
    end
    for (k, v) in control_id
        id[k] = v isa Int ? v + nState : v .+ nState
    end
    return id
end



function build_ss(param::AbstractDict, SS_stats::AbstractDict)
    
    #TODO: need trim this because some are estimated params
    # param_ss_keys = [
    #     "R_cb", "pi_cb", "pi_bar", "w_bar", "eta", "fix", "fix_L", "fix2", "iota",
    #     "tau_cp", "tau_w", "tau_a", "b_a_aux", "b_a_aux2", "DELTA", "theta_b", "omega",
    #     "phi", "phi_pi_QE", "phi_u_QE", "rho_X_QE", "rho_B_F", "rho_Z", "rho_PSI_RP",
    #     "rho_D", "rho_G", "rho_iota", "rho_BB", "rho_PSI_W", "rho_MP", "rho_eta",
    #     "rho_R", "phi_pi", "phi_u", "rho_w", "d", "eps_w",
    #     "lambda", "beta_b", "phi_B", "sigma2", "death_rate", "xi", "Rprem", "frac_b",
    #     "b_aux", "P_SS", "P_SS2", "mu_chi", "sigma_chi", "nu", "beta",
    #     "gamma_pi", "gamma_T", "rho_B", "kappa", "GAMMA",
    #     "theta", "theta_2", "alpha_lk", "alpha_ll", "alpha_kk",
    #     "delta_0", "delta_1", "alpha", "psi", "Eratio", "b_share",
    #     "Lambda_aux", "Lambda_b_aux", "sigma", "in", "n", "u"
    # ]

    param_ss_keys = Dict(
        "R_cb" => "R_cb",
        "pi_cb" => "pi_cb",
        "pi_bar" => "π",
        "w_bar" => "w",
        "eta" => "eta",
        "fix" => "fix",
        "fix_L" => "fix_L",
        "fix2" => "fix2",
        "iota" => "iota",
        "tau_cp" => "tau_cp",
        "tau_w" => "tau_w",
        "tau_a" => "tau_a",
        "b_a_aux" => "b_a_aux",
        "b_a_aux2" => "b_a_aux2",
        "DELTA" => "δ",
        "theta_b" => "theta_b",
        "omega" => "omega",
        "phi" => "phi",
        "phi_pi_QE" => "phi_pi_QE",
        "phi_u_QE" => "phi_u_QE",
        "rho_X_QE" => "rho_X_QE",
        "rho_B_F" => "rho_B_F",
        "rho_Z" => "rho_Z",
        "rho_PSI_RP" => "rho_PSI_RP",
        "rho_D" => "rho_D",
        "rho_G" => "rho_G",
        "rho_iota" => "rho_iota",
        "rho_BB" => "rho_BB",
        "rho_PSI_W" => "rho_PSI_W",
        "rho_MP" => "rho_MP",
        "rho_eta" => "rho_eta",
        "rho_R" => "rho_R",
        "phi_pi" => "phi_pi",
        "phi_u" => "phi_u",
        "rho_w" => "rho_w",
        "d" => "d",
        "eps_w" => "eps_w",
        "lambda" => "lambda",
        "beta_b" => "beta_b",
        "phi_B" => "phi_B",
        "sigma2" => "sigma2",
        "death_rate" => "death_rate",
        "xi" => "xi",
        "Rprem" => "Rprem",
        "frac_b" => "frac_b",
        "b_aux" => "b_aux",
        "P_SS" => "P_SS",
        "P_SS2" => "P_SS2",
        "mu_chi" => "mu_chi",
        "sigma_chi" => "sigma_chi",
        "nu" => "nu",
        "beta" => "beta",
        "gamma_pi" => "gamma_pi",
        "gamma_T" => "gamma_T",
        "rho_B" => "rho_B",
        "kappa" => "kappa",
        "GAMMA" => "GAMMA",
        "theta" => "theta",
        "theta_2" => "theta_2",
        "alpha_lk" => "alpha_lk",
        "alpha_ll" => "alpha_ll",
        "alpha_kk" => "alpha_kk",
        "delta_0" => "delta_0",
        "delta_1" => "delta_1",
        "alpha" => "alpha",
        "psi" => "psi",
        "Eratio" => "Eratio",
        "b_share" => "b_share",
        "Lambda_aux" => "Lambda_aux",
        "Lambda_b_aux" => "Lambda_b_aux",
        "sigma" => "sigma",
        "in" => "in",
        "n" => "n",
        "u" => "u"
    )

    
    ss = Dict{Symbol, Any}()

    # keys for which we want to keep levels (no log), e.g. transition matrices
    no_log_keys = Set(["P_SS", "P_SS2", "tau_w", "lambda", "pi_bar", "tau_a"])

    for (k, va) in param_ss_keys
        if haskey(param, k)
            v = param[k]
            val = isa(v, Number) ? Float64(real(v)) : real.(v)

            key_sym = Symbol(va * "_t")
            if k in no_log_keys
                ss[key_sym] = val
            else
                ss[key_sym] = log(val)
            end
        end
    end

    #TODO: fix SS stats or make diff function for unpacking these
    # ss_stats_keys = [
    #     "u", "r_l", "Lambda", "Y", "B_F", "B_gov_ncp", "T", "G", "LT", "C_b",
    #     "A_F", "v", "L", "B_gov_ncp2", "COP_SS", "Chh", "I2", "Profit"
    # ]

    ss_stats_keys = Dict(
        "u" => "u",
        "r_l" => "r_l",
        "Lambda" => "λ",
        "Y" => "Y",
        "B_F" => "B_F",
        "B_gov_ncp" => "B_gov_ncp",
        "T" => "T",
        "G" => "G",
        "LT" => "LT",
        "C_b" => "C_b",
        "A_F" => "A_F",
        "A_g" => "A_g",
        "v" => "v",
        "L" => "L",
        "B_gov_ncp2" => "B_gov_ncp2",
        "COP_SS" => "COP_SS",
        "Chh" => "Chh",
        "I2" => "I2",
        "Profit" => "Profit"
    )

    for (k, va) in ss_stats_keys
        if haskey(SS_stats, k)
            v = SS_stats[k]
            val = isa(v, Number) ? Float64(v) : Float64(v[1])
            ss[Symbol(va * "_t")] = val
        end
    end
    
    # for k in ss_stats_keys
    #     if haskey(SS_stats, k)
    #         v = SS_stats[k]
    #         if v isa Number
    #             ss[Symbol(k)] = Float64(v)
    #         elseif v isa AbstractArray && length(v) > 0
    #             ss[Symbol(k)] = Float64(v[1])  # scalars only for ss dict
    #         end
    #     end
    # end

    return NamedTuple{Tuple(keys(ss))}(values(ss))
end

