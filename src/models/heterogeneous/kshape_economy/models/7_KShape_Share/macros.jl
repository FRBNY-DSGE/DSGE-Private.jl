# The MATLAB original stores variables in log space; perturbation happens
# around the (nonzero) log steady state:
#
#     X_matlab = log(SS_level) + δ          (SS is nonzero)
#
# In the Julia replication, we perturb around ZERO:
#
#     X = 0  at steady state          (i.e.  X₀ = zeros(n_model_states))
#     X = δ  is the deviation itself
#
# To recover levels from (SS_vector, deviation_vector):
#
#     level = exp(SS_log[idx] + X[idx])     for log-linearised variables
#     value = SS[idx] + X[idx]              for additively-linearised variables
#
# In the MATLAB-translated F_sys, BOTH aggregate states AND aggregate controls
# are log-linearised.  The original code performs two steps:
#
#   1)  var_log  = StateSS[idx] + State[idx]      (log-level)
#   2)  var      = exp(var_log)                     (actual level)
#
# ── Which macro to use ───────────────────────────────────────────────────────
#
#   Aggregate states & shocks:
#     @unpack_states  R_cb_t, w_t, … = (State, Stateminus), state_id, StateSS
#       → R_cb_t    = exp(StateSS[idx] + Stateminus[idx])  (t   level)
#       → R_cb′_t   = exp(StateSS[idx] + State[idx])       (t+1 level)
#
#   Aggregate controls (oc block):
#     @unpack_controls  K_t, B_t, … = (Control, Controlnext), control_id, ControlSS
#       → K_t       = exp(ControlSS[idx] + Control[idx])       (t   level)
#       → K′_t      = exp(ControlSS[idx] + Controlnext[idx])   (t+1 level)
#
#   Distributional controls (VALUE, mutil_c, Va):
#     @extract  VALUE, mutil_c, Va = Control, control_id
#       → then apply util / mutil manually
#
#   Marginals / copula:
#     @extract  marginal_b, COP = Stateminus, state_id
#
#   Equation-row scalars:
#     @unpack_and_first  eq_R_cb, eq_K = eq
#
# ──────────────────────────────────────────────────────────────────────────────

"""
    _ssdev2level(ss, x, idx)

Additive deviation → value:  `ss[idx] + x[idx]`.
Returns the LOG-level for log-linearised variables (useful for LHS assignments
in the state block, where the SGU form needs the log value).
"""
@inline _ssdev2level(ss::AbstractVector, x::AbstractVector, idx::Int) =
    ss[idx] + x[idx]
@inline _ssdev2level(ss::AbstractVector, x::AbstractVector, idx::UnitRange{Int}) =
    @view(ss[idx]) .+ @view(x[idx])

"""
    _sslogdev2level(ss, x, idx)

Log deviation → level:  `exp(ss[idx] + x[idx])`.
`ss` holds `log(SS_level)` and `x` holds the log-deviation (zero at SS).
At SS, `x = 0` so `level = exp(log(SS_level)) = SS_level`.
"""
@inline _sslogdev2level(ss::AbstractVector, x::AbstractVector, idx::Int) =
    exp(ss[idx] + x[idx])
@inline _sslogdev2level(ss::AbstractVector, x::AbstractVector, idx::UnitRange{Int}) =
    exp.(@view(ss[idx]) .+ @view(x[idx]))

"""
    _extract(x, idx)

Raw extraction:  `x[idx]` (scalar) or `@view x[idx]` (range).
"""
@inline _extract(x::AbstractVector, idx::Int) = x[idx]
@inline _extract(x::AbstractVector, idx::UnitRange{Int}) = @view(x[idx])

"""
    _first_or_scalar(v)

Return `v` if scalar, `first(v)` if a range.  Used by `@unpack_and_first`.
"""
@inline _first_or_scalar(v::Int) = v
@inline _first_or_scalar(v::UnitRange{Int}) = first(v)

# ══════════════════════════════════════════════════════════════════════════════
#  MATLAB → Julia translation macros
# ══════════════════════════════════════════════════════════════════════════════

# ── Macro: @unpack_states ─────────────────────────────────────────────────────
"""
    @unpack_states var1_t, var2_t, … = (State, Stateminus), id, SS

Translate MATLAB log-linearised state variables to Julia levels.
For each listed name `k` (ending in `_t`), creates **two** local variables:

    k    = exp(SS[id[:k]] + Stateminus[id[:k]])   # t   level  (from Stateminus)
    k′   = exp(SS[id[:k]] + State[id[:k]])         # t+1 level  (from State)

where `k′` replaces the trailing `_t` with `′_t` (e.g. `R_cb_t` → `R_cb′_t`).

# Example — replaces ~100 lines of MATLAB-translated extraction + exp()
```julia
@unpack_states R_cb_t, w_t, A_b_t, B_b_t, A_g_t, Q_t, lev_t, NW_b_t, R_tilde_t, x_cb_t,
    pi_lag_t, Y_lag_t, C_lag_t, I_lag_t, PROFIT_lag_t, unemp_lag_t, G_lag_t, LT_lag_t,
    R_star_t, B_F_t, Z_t, PSI_RP_t, eta_t, D_t, GG_t, iota_t, BB_t, PSI_W_t, MP_t, p_mark_t =
    (State, Stateminus), state_id, StateSS

# Now available:
#   R_cb_t    — exp(StateSS[idx] + Stateminus[idx])    (t   level)
#   R_cb′_t   — exp(StateSS[idx] + State[idx])         (t+1 level)
#   w_t, w′_t, A_b_t, A_b′_t, …
```
"""
macro unpack_states(args)
    args.head != :(=) && error("@unpack_states: expected `vars = (Xnext, Xminus), id, SS`")
    varnames, rhs = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    vecs, id, ss = rhs.args
    xnext, xminus = vecs.args

    xi_n = gensym(); xi_m = gensym(); idi = gensym(); ssi = gensym()

    kd = Expr[]
    for key in varnames
        s = string(key)
        key_prime = endswith(s, "_t") ? Symbol(s[1:end-2], "′_t") : Symbol(s, "′")
        push!(kd, :($key       = $_sslogdev2level($ssi, $xi_m, $idi[$(QuoteNode(key))])))  # t from Stateminus
        push!(kd, :($key_prime = $_sslogdev2level($ssi, $xi_n, $idi[$(QuoteNode(key))])))  # t+1 from State
    end

    expr = quote
        local $xi_n = $xnext
        local $xi_m = $xminus
        local $idi  = $id
        local $ssi  = $ss
        $(kd...)
    end
    esc(expr)
end

# ── Macro: @unpack_controls ───────────────────────────────────────────────────
"""
    @unpack_controls var1_t, var2_t, … = (Control, Controlnext), id, SS

Translate MATLAB log-linearised control variables to Julia levels.
For each listed name `k` (ending in `_t`), creates **two** local variables:

    k    = exp(SS[id[:k]] + Control[id[:k]])        # t   level
    k′   = exp(SS[id[:k]] + Controlnext[id[:k]])    # t+1 level

where `k′` replaces the trailing `_t` with `′_t` (e.g. `K_t` → `K′_t`).

# Example — replaces ~100 lines of MATLAB-translated extraction
```julia
@unpack_controls K_t, B_t, B_gov_ncp_t, T_t, LT_t, G_t, Lambda_t, pi_t, V_t,
    h_t, v_t, Y_t, Profit_t, r_k_t, r_a_t, MC_t, unemp_t, nn_t, M_t, f_t,
    zz_t, xx_t, vv_t, ee_t, C_b_t, Profit_FI_t, RRa_t, RR_t, I_t, x_k_t,
    MRS_t, A_hh_t, B_hh_t, C_t, N_t, L_t, UB_t =
    (Control, Controlnext), control_id, ControlSS

# Now available:
#   K_t       — exp(ControlSS[idx] + Control[idx])         (t   level)
#   K′_t      — exp(ControlSS[idx] + Controlnext[idx])     (t+1 level)
#   B_t, B′_t, pi_t, pi′_t, …
```
"""
macro unpack_controls(args)
    args.head != :(=) && error("@unpack_controls: expected `vars = (X, Xnext), id, SS`")
    varnames, rhs = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    vecs, id, ss = rhs.args
    x, xnext = vecs.args

    xi = gensym(); xi_n = gensym(); idi = gensym(); ssi = gensym()

    kd = Expr[]
    for key in varnames
        s = string(key)
        key_prime = endswith(s, "_t") ? Symbol(s[1:end-2], "′_t") : Symbol(s, "′")
        push!(kd, :($key       = $_sslogdev2level($ssi, $xi,   $idi[$(QuoteNode(key))])))  # t from Control
        push!(kd, :($key_prime = $_sslogdev2level($ssi, $xi_n, $idi[$(QuoteNode(key))])))  # t+1 from Controlnext
    end

    expr = quote
        local $xi   = $x
        local $xi_n = $xnext
        local $idi  = $id
        local $ssi  = $ss
        $(kd...)
    end
    esc(expr)
end

# ══════════════════════════════════════════════════════════════════════════════
#  Lower-level building-block macros
# ══════════════════════════════════════════════════════════════════════════════

# ── Macro: @ssdeviations2levels ───────────────────────────────────────────────
"""
    @ssdeviations2levels var1, var2, … = X, id, XSS

Additive: `k = XSS[id[:k]] + X[id[:k]]` (log-value, before exp).
Useful for writing the state-equation LHS, which needs the log-level:

```julia
@ssdeviations2levels R_cb_log, w_log = State, state_id, StateSS
LHS[state_id[:R_cb]] = R_cb_log
LHS[state_id[:w]]    = w_log
```
"""
macro ssdeviations2levels(args)
    args.head != :(=) && error("@ssdeviations2levels: expected `vars = X, id, XSS`")
    varnames, rhs = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    x, id, ss = rhs.args
    xi = gensym(); idi = gensym(); ssi = gensym()
    kd = [:($key = $_ssdev2level($ssi, $xi, $idi[$(QuoteNode(key))])) for key in varnames]
    expr = quote
        local $xi  = $x
        local $idi = $id
        local $ssi = $ss
        $(kd...)
    end
    esc(expr)
end

# ── Macro: @sslogdeviations2levels ────────────────────────────────────────────
"""
    @sslogdeviations2levels var1, var2, … = X, id, XSS

Log-linearised: `k = exp(XSS[id[:k]] + X[id[:k]])`.
Single-vector version (one time period only).  Prefer `@unpack_states` /
`@unpack_controls` when you need both time periods.
"""
macro sslogdeviations2levels(args)
    args.head != :(=) && error("@sslogdeviations2levels: expected `vars = X, id, XSS`")
    varnames, rhs = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    x, id, ss = rhs.args
    xi = gensym(); idi = gensym(); ssi = gensym()
    kd = [:($key = $_sslogdev2level($ssi, $xi, $idi, $(QuoteNode(key)))) for key in varnames]
    expr = quote
        local $xi  = $x
        local $idi = $id
        local $ssi = $ss
        $(kd...)
    end
    esc(expr)
end

# ── Macro: @extract ───────────────────────────────────────────────────────────
"""
    @extract var1, var2, … = X, id

Raw extraction: `k = X[id[:k]]` (scalar) or `@view(X[id[:k]])` (range).
Use for distributional controls, marginals, copula blocks, or shocks
where a custom transform is applied afterwards.

```julia
@extract VALUE, mutil_c, Va = Control, control_id
VALUE_level = util(VALUE)       # apply custom transform manually
```
"""
macro extract(args)
    args.head != :(=) && error("@extract: expected `vars = X, id`")
    varnames, rhs = args.args
    varnames = isa(varnames, Symbol) ? [varnames] : varnames.args
    x, id = rhs.args
    xi = gensym(); idi = gensym()
    kd = [:($key = $_extract($xi, $idi[$(QuoteNode(key))])) for key in varnames]
    expr = quote
        local $xi  = $x
        local $idi = $id
        $(kd...)
    end
    esc(expr)
end

# ── Macro: @unpack_and_first ──────────────────────────────────────────────────
"""
    @unpack_and_first var1, var2, … = dict

Unpack dictionary entries, calling `first()` on range values to get a scalar.
Useful for extracting scalar equation-row indices from the `eq` dictionary.

```julia
@unpack_and_first eq_R_cb, eq_MRS = eq
```
"""
macro unpack_and_first(args)
    args.head != :(=) && error("@unpack_and_first: expected `vars = dict`")
    items, dict = args.args
    items = isa(items, Symbol) ? [items] : items.args
    di = gensym()
    kd = [:($key = $_first_or_scalar($di[$(QuoteNode(key))])) for key in items]
    expr = quote
        local $di = $dict
        $(kd...)
    end
    esc(expr)
end
