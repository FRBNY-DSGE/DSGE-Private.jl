"""
    indexes_tvcopula(param, grid) -> (idx, hh, shocks)

Constructs index maps (1-based) for a heterogeneous-agent model’s packed vectors.

This is a direct Julia translation of the MATLAB snippet that:
1) defines CRRA utility-related function handles,
2) builds contiguous index blocks for household-value-related objects of size `NN = nb * na * nse`,
3) defines indices for “aggregate/state/shock” variables, and
4) defines indices for “controls/observables/aux” variables.

# Inputs
- `param`: parameter container with field/property `sigma` (e.g. `param.sigma`).
- `grid`: grid container with fields/properties:
  - `nb`, `na`, `nse` (integers)
  - `ns` (integer) used to size the block `J_ind = V_ind .+ (1:ns)`

`param`/`grid` can be `NamedTuple`s, structs, or dictionaries as long as the above
fields are accessible (see notes).

# Outputs
Returns a 3-tuple:
- `idx::NamedTuple`: a large collection of integer indices and ranges matching the MATLAB code.
  - Scalar indices are `Int`
  - Block indices are `UnitRange{Int}` (e.g. `VALUE_ind`) or `Vector{Int}` (e.g. `J_ind`)
- `hh::NamedTuple`: utility functions:
  - `util(c)`, `mutil(c)`, `invutil(u)`, `invmutil(μ)`
- `shocks::NamedTuple`: convenience fields for `NN`, `VALUE_ind`, `mutil_c_ind`, `Va_ind`

# Notes on translation details
- MATLAB’s `1:NN` becomes Julia’s `1:NN` (`UnitRange`).
- `J_ind = V_ind + (1:grid.ns)'` becomes `J_ind = collect(V_ind .+ (1:ns))`.
  (We return it as a `Vector{Int}`; if you want a range-like object, keep it as a vector anyway
  since it’s not contiguous in general once you embed it in larger layouts.)
- The code below assumes your model uses 1-based indexing everywhere (Julia-native).
- If `param.sigma == 1`, CRRA utility is log utility; the original MATLAB code does not handle
  that special case, so this mirrors the original (it will divide by zero). If you need log case,
  I can patch it.

# Example
```julia
param = (sigma = 2.0,)
grid  = (nb = 50, na = 40, nse = 2, ns = 7)

idx, hh, shocks = indexes_tvcopula(param, grid)

idx.R_cb_ind        # 1
idx.VALUE_ind       # 1:NN
idx.J_ind           # Vector{Int} of length ns
hh.mutil(1.0)       # 1.0
"""
function indexes_tvcopula(param, grid)
    # --- small helper to support either structs/namedtuples OR dicts ---
    getf(x, s::Symbol) = getproperty(x, s)
    getf(x::AbstractDict, s::Symbol) = x[string(s)]
    getf(x::AbstractDict, s::AbstractString) = x[s]

    σ  = getf(param, :sigma)
    nb = Int(getf(grid, :nb))
    na = Int(getf(grid, :na))
    nse = Int(getf(grid, :nse))
    ns = Int(getf(grid, :ns))

    # --- utility functions (MATLAB function handles) ---
    util(c)     = (c .^ (1 - σ)) ./ (1 - σ)
    mutil(c)    = 1.0 ./ (c .^ σ)
    invutil(u)  = ((1 - σ) .* u) .^ (1 / (1 - σ))
    invmutil(μ) = (1.0 ./ μ) .^ (1 / σ)

    hh = (util = util, mutil = mutil, invutil = invutil, invmutil = invmutil)

    # --- household block sizes ---
    NN = nb * na * nse

    VALUE_ind   = 1:NN
    mutil_c_ind = NN .+ (1:NN)          # becomes a UnitRange after materialization? keep as vector/range:
    Va_ind      = 2NN .+ (1:NN)

    # Keep these in a small bundle too (often convenient downstream)
    shocks = (NN = NN, VALUE_ind = VALUE_ind, mutil_c_ind = mutil_c_ind, Va_ind = Va_ind)

    # --- "state/shock" indices (first block) ---
    R_cb_ind        = 1
    w_ind           = 2
    A_b_ind         = 3
    B_b_ind         = 4
    A_g_ind         = 5
    Q_ind           = 6
    lev_ind         = 7
    NW_b_ind        = 8
    R_tilde_ind     = 9
    x_cb_ind        = 10
    pastpi_ind      = 11
    pastY_ind       = 12
    pastC_ind       = 13
    pastI_ind       = 14
    pastPROFIT_ind  = 15
    pastunemp_ind   = 16
    pastG_ind       = 17
    pastLT_ind      = 18
    R_star_ind      = 19
    B_F_ind         = 20
    Z_ind           = 21
    PSI_RP_ind      = 22
    eta_ind         = 23
    D_ind           = 24
    GG_ind          = 25
    iota_ind        = 26
    BB_ind          = 27
    PSI_W_ind       = 28
    MP_ind          = 29
    p_mark_ind      = 30
    eps_QE_ind      = 31
    eps_RP_ind      = 32
    eps_B_F_ind     = 33
    eps_BB_ind      = 34
    eps_Z_ind       = 35
    eps_G_ind       = 36
    eps_D_ind       = 37
    eps_R_ind       = 38
    eps_iota_ind    = 39
    eps_eta_ind     = 40
    eps_w_ind       = 41

    # --- "controls/observables/aux" indices (second block) ---
    MRS_ind        = 1
    A_hh_ind       = MRS_ind       + 1
    B_hh_ind       = A_hh_ind      + 1
    C_ind          = B_hh_ind      + 1
    N_ind          = C_ind         + 1
    L_ind          = N_ind         + 1
    UB_ind         = L_ind         + 1
    K_ind          = UB_ind        + 1
    B_ind          = K_ind         + 1
    B_gov_ncp_ind  = B_ind         + 1
    T_ind          = B_gov_ncp_ind + 1
    LT_ind         = T_ind         + 1
    G_ind          = LT_ind        + 1
    Lambda_ind     = G_ind         + 1
    pi_ind         = Lambda_ind    + 1
    V_ind          = pi_ind        + 1

    J_ind          = collect(V_ind .+ (1:ns))  # length ns
    h_ind          = J_ind[end]    + 1
    v_ind          = h_ind         + 1
    Y_ind          = v_ind         + 1
    Profit_ind     = Y_ind         + 1
    r_k_ind        = Profit_ind    + 1
    r_a_ind        = r_k_ind       + 1
    MC_ind         = r_a_ind       + 1
    unemp_ind      = MC_ind        + 1
    nn_ind         = unemp_ind     + 1
    M_ind          = nn_ind        + 1
    f_ind          = M_ind         + 1
    zz_ind         = f_ind         + 1
    xx_ind         = zz_ind        + 1
    vv_ind         = xx_ind        + 1
    ee_ind         = vv_ind        + 1
    C_b_ind        = ee_ind        + 1
    Profit_FI_ind  = C_b_ind       + 1
    RRa_ind        = Profit_FI_ind + 1
    RR_ind         = RRa_ind       + 1
    I_ind          = RR_ind        + 1
    x_k_ind        = I_ind         + 1
    A_g_obs_ind    = x_k_ind       + 1
    Y_obs_ind      = A_g_obs_ind   + 1
    C_obs_ind      = Y_obs_ind     + 1
    I_obs_ind      = C_obs_ind     + 1
    w_obs_ind      = I_obs_ind     + 1
    PROFIT_obs_ind = w_obs_ind     + 1
    unemp_obs_ind  = PROFIT_obs_ind + 1
    inf_obs_ind    = unemp_obs_ind  + 1
    R_obs_ind      = inf_obs_ind    + 1
    pastw_ind      = R_obs_ind      + 1
    G_obs_ind      = pastw_ind      + 1

    pastYY_ind        = G_obs_ind        + 1
    pastCC_ind        = pastYY_ind       + 1
    pastII_ind        = pastCC_ind       + 1
    pastPPROFIT_ind   = pastII_ind       + 1
    pastuu_ind        = pastPPROFIT_ind  + 1
    pastGG_ind        = pastuu_ind       + 1
    pastA_g_ind       = pastGG_ind       + 1

    l_lambda_ind      = pastA_g_ind      + 1
    pastQ_ind         = l_lambda_ind     + 1
    x_I_ind           = pastQ_ind        + 1
    eta2_ind          = x_I_ind          + 1
    iota2_ind         = eta2_ind         + 1
    pastLT2_ind       = iota2_ind        + 1
    pastG2_ind        = pastLT2_ind      + 1
    LT_obs_ind        = pastG2_ind       + 1
    B_gov_ncp2_ind    = LT_obs_ind       + 1

    # --- pack everything into a single idx NamedTuple ---
    idx = (
        # household blocks
        NN = NN,
        VALUE_ind = VALUE_ind,
        mutil_c_ind = mutil_c_ind,
        Va_ind = Va_ind,

        # state/shock indices
        R_cb_ind = R_cb_ind,
        w_ind = w_ind,
        A_b_ind = A_b_ind,
        B_b_ind = B_b_ind,
        A_g_ind = A_g_ind,
        Q_ind = Q_ind,
        lev_ind = lev_ind,
        NW_b_ind = NW_b_ind,
        R_tilde_ind = R_tilde_ind,
        x_cb_ind = x_cb_ind,
        pastpi_ind = pastpi_ind,
        pastY_ind = pastY_ind,
        pastC_ind = pastC_ind,
        pastI_ind = pastI_ind,
        pastPROFIT_ind = pastPROFIT_ind,
        pastunemp_ind = pastunemp_ind,
        pastG_ind = pastG_ind,
        pastLT_ind = pastLT_ind,
        R_star_ind = R_star_ind,
        B_F_ind = B_F_ind,
        Z_ind = Z_ind,
        PSI_RP_ind = PSI_RP_ind,
        eta_ind = eta_ind,
        D_ind = D_ind,
        GG_ind = GG_ind,
        iota_ind = iota_ind,
        BB_ind = BB_ind,
        PSI_W_ind = PSI_W_ind,
        MP_ind = MP_ind,
        p_mark_ind = p_mark_ind,
        eps_QE_ind = eps_QE_ind,
        eps_RP_ind = eps_RP_ind,
        eps_B_F_ind = eps_B_F_ind,
        eps_BB_ind = eps_BB_ind,
        eps_Z_ind = eps_Z_ind,
        eps_G_ind = eps_G_ind,
        eps_D_ind = eps_D_ind,
        eps_R_ind = eps_R_ind,
        eps_iota_ind = eps_iota_ind,
        eps_eta_ind = eps_eta_ind,
        eps_w_ind = eps_w_ind,

        # control/obs/aux indices
        MRS_ind = MRS_ind,
        A_hh_ind = A_hh_ind,
        B_hh_ind = B_hh_ind,
        C_ind = C_ind,
        N_ind = N_ind,
        L_ind = L_ind,
        UB_ind = UB_ind,
        K_ind = K_ind,
        B_ind = B_ind,
        B_gov_ncp_ind = B_gov_ncp_ind,
        T_ind = T_ind,
        LT_ind = LT_ind,
        G_ind = G_ind,
        Lambda_ind = Lambda_ind,
        pi_ind = pi_ind,
        V_ind = V_ind,
        J_ind = J_ind,
        h_ind = h_ind,
        v_ind = v_ind,
        Y_ind = Y_ind,
        Profit_ind = Profit_ind,
        r_k_ind = r_k_ind,
        r_a_ind = r_a_ind,
        MC_ind = MC_ind,
        unemp_ind = unemp_ind,
        nn_ind = nn_ind,
        M_ind = M_ind,
        f_ind = f_ind,
        zz_ind = zz_ind,
        xx_ind = xx_ind,
        vv_ind = vv_ind,
        ee_ind = ee_ind,
        C_b_ind = C_b_ind,
        Profit_FI_ind = Profit_FI_ind,
        RRa_ind = RRa_ind,
        RR_ind = RR_ind,
        I_ind = I_ind,
        x_k_ind = x_k_ind,
        A_g_obs_ind = A_g_obs_ind,
        Y_obs_ind = Y_obs_ind,
        C_obs_ind = C_obs_ind,
        I_obs_ind = I_obs_ind,
        w_obs_ind = w_obs_ind,
        PROFIT_obs_ind = PROFIT_obs_ind,
        unemp_obs_ind = unemp_obs_ind,
        inf_obs_ind = inf_obs_ind,
        R_obs_ind = R_obs_ind,
        pastw_ind = pastw_ind,
        G_obs_ind = G_obs_ind,
        pastYY_ind = pastYY_ind,
        pastCC_ind = pastCC_ind,
        pastII_ind = pastII_ind,
        pastPPROFIT_ind = pastPPROFIT_ind,
        pastuu_ind = pastuu_ind,
        pastGG_ind = pastGG_ind,
        pastA_g_ind = pastA_g_ind,
        l_lambda_ind = l_lambda_ind,
        pastQ_ind = pastQ_ind,
        x_I_ind = x_I_ind,
        eta2_ind = eta2_ind,
        iota2_ind = iota2_ind,
        pastLT2_ind = pastLT2_ind,
        pastG2_ind = pastG2_ind,
        LT_obs_ind = LT_obs_ind,
        B_gov_ncp2_ind = B_gov_ncp2_ind
    )

    return idx, hh, shocks
end
