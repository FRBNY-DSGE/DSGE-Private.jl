using OrderedCollections: OrderedDict

"""
    build_indices(grid) -> (state_id, control_id, eq)

Build named-index dictionaries for the state vector, control vector, and equation rows
used in `F_sys_ref_tvcopula_QE`.

Returns three `OrderedDict{Symbol, Union{Int, UnitRange{Int}}}`:
- `state_id`:    indices into the state vector (length `numstates`)
- `control_id`:  indices into the full-grid control vector (length `3*NN + oc`)
- `eq`:          row indices in the LHS/RHS equation system (length `numstates + ny`)

All sizes are derived from `grid`: `nb`, `na`, `nse`, `ns`, `nCOP`, `numstates`, `oc`.

Scalar keys carry a `_t` suffix (e.g. `:R_cb_t`).  Range keys for distributional
blocks (`:marginal_b`, `:VALUE`, etc.) are left unchanged.  Lag variables use
`*_lag_t` (e.g. `:C_lag_t`).

Equation-row keys strip the trailing `_t` before prepending `eq_`:
`state_id[:R_cb_t]` → `eq[:eq_R_cb]`.
"""
function build_indices(grid)
    nb  = Int(grid[:nb])
    na  = Int(grid[:na])
    nse = Int(grid[:nse])
    ns  = Int(grid[:ns])
    nCOP = Int(grid[:nCOP])
    oc  = Int(grid[:oc])
    numstates = Int(grid[:numstates])

    NN = nb * na * nse
    nRedMarg = nb + na + nse - 3
    NxNx = nRedMarg + nCOP

    ny = 3 * NN + oc   # full-grid control count

    # ─── state_id ─────────────────────────────────────────────────────────────
    state_id = OrderedDict{Symbol, Union{Int, UnitRange{Int}}}()
    next = 1

    # Distribution block: reduced marginals + copula coefficients
    state_id[:marginal_b]  = next:(next + nb - 2);   next += nb - 1
    state_id[:marginal_a]  = next:(next + na - 2);    next += na - 1
    state_id[:marginal_se] = next:(next + nse - 2);   next += nse - 1
    state_id[:COP]         = next:(next + nCOP - 1);  next += nCOP
    @assert next == NxNx + 1

    # Aggregate states, need to be same ordering as matlab
    for sym in (:R_cb_t, :w_t, :A_b_t, :B_b_t, :A_g_t, :Q_t, :lev_t, :NW_b_t, :R_tilde_t, :x_cb_t,
                :pi_lag_t, :Y_lag_t, :C_lag_t, :I_lag_t, :PROFIT_lag_t, :unemp_lag_t, :G_lag_t, :LT_lag_t,
                :R_star_t, :B_F_t, :Z_t, :PSI_RP_t, :eta_t, :D_t, :GG_t, :iota_t, :BB_t, :PSI_W_t, :MP_t, :p_mark_t)
        state_id[sym] = next
        next += 1
    end

    # Shock states
    for sym in (:eps_QE_t, :eps_RP_t, :eps_B_F_t, :eps_BB_t, :eps_Z_t, :eps_G_t, :eps_D_t,
                :eps_R_t, :eps_iota_t, :eps_eta_t, :eps_w_t)
        state_id[sym] = next
        next += 1
    end

    @assert next - 1 == numstates "state_id covers $(next-1) indices but numstates = $numstates"

    # ─── control_id ───────────────────────────────────────────────────────────
    control_id = OrderedDict{Symbol, Union{Int, UnitRange{Int}}}()
    next = 1

    # Distributional controls (full-grid, NN = nb*na*nse each)
    control_id[:VALUE]   = next:(next + NN - 1);  next += NN
    control_id[:mutil_c] = next:(next + NN - 1);  next += NN
    control_id[:Va]      = next:(next + NN - 1);  next += NN

    # Summary (household aggregates)
    for sym in (:MRS_t, :A_hh_t, :B_hh_t, :C_t, :N_t, :L_t, :UB_t)
        control_id[sym] = next
        next += 1
    end

    # Other aggregate controls
    for sym in (:K_t, :B_t, :B_gov_ncp_t, :T_t, :LT_t, :G_t, :Lambda_t, :pi_t, :V_t)
        control_id[sym] = next
        next += 1
    end

    control_id[:J] = next:(next + ns - 1);  next += ns

    for sym in (:h_t, :v_t, :Y_t, :Profit_t, :r_k_t, :r_a_t, :MC_t, :unemp_t, :nn_t, :M_t, :f_t,
                :zz_t, :xx_t, :vv_t, :ee_t, :C_b_t, :Profit_FI_t, :RRa_t, :RR_t, :I_t, :x_k_t)
        control_id[sym] = next
        next += 1
    end

    # Observables
    for sym in (:A_g_obs_t, :Y_obs_t, :C_obs_t, :I_obs_t, :w_obs_t, :PROFIT_obs_t, :unemp_obs_t,
                :inf_obs_t, :R_obs_t, :w_lag_t, :G_obs_t)
        control_id[sym] = next
        next += 1
    end

    # Past/auxiliary controls
    for sym in (:YY_lag_t, :CC_lag_t, :II_lag_t, :PPROFIT_lag_t, :uu_lag_t, :GG_lag_t, :A_g_lag_t,
                :l_lambda_t, :Q_lag_t, :x_I_t, :eta2_t, :iota2_t, :LT2_lag_t, :G2_lag_t, :LT_obs_t, :B_gov_ncp2_t)
        control_id[sym] = next
        next += 1
    end

    @assert next - 1 == ny "control_id covers $(next-1) indices but ny (3*NN + oc) = $ny"

    # ─── eq (equation rows) ───────────────────────────────────────────────────
    # State equations:   rows 1:nx          (same ranges as state_id)
    # Control equations: rows nx+1:nx+ny    (nx .+ control_id ranges)
    nx = numstates
    eq = OrderedDict{Symbol, Union{Int, UnitRange{Int}}}()

    # Strip trailing _t before prepending eq_:  :R_cb_t → :eq_R_cb
    function _eq_key(k::Symbol)
        s = string(k)
        base = endswith(s, "_t") ? s[1:end-2] : s
        Symbol(:eq_, base)
    end

    for (k, v) in state_id
        eq[_eq_key(k)] = v
    end

    for (k, v) in control_id
        if v isa Int
            eq[_eq_key(k)] = nx + v
        else
            eq[_eq_key(k)] = nx .+ v
        end
    end

    return (state_id, control_id, eq)
end

"""
    build_ss_levels(StateSS, ControlSS, state_id, control_id) -> Dict{Symbol, Float64}

Build a dictionary mapping variable names to their steady-state **level** values.

For each scalar (non-distributional) entry in `state_id` and `control_id`,
computes `exp(SS[idx])` to convert from log-level to actual level.

Keys match state_id/control_id: `ss[:R_cb_t]`, `ss[:K_t]`, etc.

Range-valued entries (distributional blocks like `:VALUE`, `:marginal_b`, etc.)
are skipped — only scalar aggregate variables are included.
"""
function build_ss_levels(StateSS::AbstractVector, ControlSS::AbstractVector,
                         state_id::AbstractDict, control_id::AbstractDict)
    ss = Dict{Symbol, Float64}()
    for (k, v) in state_id
        if v isa Int
            ss[k] = exp(StateSS[v])
        end
    end
    for (k, v) in control_id
        if v isa Int
            ss[k] = exp(ControlSS[v])
        end
    end
    return ss
end
