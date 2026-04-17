using LinearAlgebra
using MAT

"""
    IRFs_Taylor_QE_compare_tvcopula(hx, gx, Xss, Yss, Gamma_state, Gamma_control, 
                                     param, grid, SS_stats, MP_shock_aux=nothing)

Computes impulse response functions (IRFs) comparing Taylor rule and QE regimes.

# Arguments
- `hx`: State transition matrix (nk × nk)
- `gx`: Control policy matrix (numcontrols × nk)
- `Xss`: Steady state state vector
- `Yss`: Steady state control vector
- `Gamma_state`: Mapping for state perturbations
- `Gamma_control`: Mapping for control perturbations
- `param`: Parameter dictionary (must contain "regime", "adjust", "sigma_R", "Eshare", "lambda", "alpha", "tau_w", "psi", "xi", "delta_0", "delta_1", "fix", "fix2", "death_rate", "tau_cp")
- `grid`: Grid dictionary (must contain "numstates", "os", "oc", "nb", "na", "nse", "ns", "maxlag")
- `SS_stats`: Steady state statistics dictionary
- `MP_shock_aux`: Optional pre-computed monetary policy shock size (for QE regime)

# Returns
Dictionary containing:
- All IRF series (both levels and percentage deviations)
- Keys follow MATLAB naming convention (e.g., `IRFs_Rcb_p`, `IRFs_Y_p`, etc.)

# Notes
- For QE regime with 'G' adjustment, the function computes a shock size that matches
  a Taylor rule IRF (requires IRFs_Taylor_G.mat file)
- For other regimes, uses the provided or computed shock size
"""
function IRFs_Taylor_QE_compare_tvcopula(hx, gx, Xss, Yss, Gamma_state, Gamma_control,
                                         m::mBBQ, grid, SS_stats, MP_shock_aux=nothing)
    
    # Set maxlag if not provided
    if !haskey(grid, "maxlag")
        grid["maxlag"] = 1001
    end
    maxlag = Int(grid["maxlag"])
    
    # Initialize state vector
    numstates = Int(grid["numstates"])
    x0 = zeros(numstates)
    
    # Define indexes (matching indexes_tvcopula.m)
    # Statenext_IRFs has os rows; Control_IRFs has oc rows.
    # Indices must be local (1-based within those blocks), not global.
    nb = Int(grid["nb"])
    na = Int(grid["na"])
    nse = Int(grid["nse"])
    ns = Int(grid["ns"])
    NN = nb * na * nse
    
    os = Int(grid["os"])
    oc = Int(grid["oc"])
    
    # State indices: local 1-based offsets within observable state block (length os)
    R_cb_ind = 1
    w_ind = 2
    A_b_ind = 3
    B_b_ind = 4
    A_g_ind = 5
    Q_ind = 6
    lev_ind = 7
    NW_b_ind = 8
    R_tilde_ind = 9
    x_cb_ind = 10
    pastpi_ind = 11
    pastY_ind = 12
    pastC_ind = 13
    pastI_ind = 14
    pastPROFIT_ind = 15
    pastunemp_ind = 16
    pastG_ind = 17
    pastLT_ind = 18
    R_star_ind = 19
    B_F_ind = 20
    
    # Control indices: local 1-based offsets within observable control block (length oc)
    MRS_ind = 1
    A_hh_ind = MRS_ind + 1
    B_hh_ind = A_hh_ind + 1
    C_ind = B_hh_ind + 1
    N_ind = C_ind + 1
    L_ind = N_ind + 1
    UB_ind = L_ind + 1
    K_ind = UB_ind + 1
    B_ind = K_ind + 1
    B_gov_ncp_ind = B_ind + 1
    T_ind = B_gov_ncp_ind + 1
    LT_ind = T_ind + 1
    G_ind = LT_ind + 1
    Lambda_ind = G_ind + 1
    pi_ind = Lambda_ind + 1
    V_ind = pi_ind + 1
    J_ind = (V_ind + 1):(V_ind + ns)
    h_ind = V_ind + ns + 1
    v_ind = h_ind + 1
    Y_ind = v_ind + 1
    Profit_ind = Y_ind + 1
    r_k_ind = Profit_ind + 1
    r_a_ind = r_k_ind + 1
    MC_ind = r_a_ind + 1
    unemp_ind = MC_ind + 1
    nn_ind = unemp_ind + 1
    M_ind = nn_ind + 1
    f_ind = M_ind + 1
    zz_ind = f_ind + 1
    xx_ind = zz_ind + 1
    vv_ind = xx_ind + 1
    ee_ind = vv_ind + 1
    C_b_ind = ee_ind + 1
    Profit_FI_ind = C_b_ind + 1
    RRa_ind = Profit_FI_ind + 1
    RR_ind = RRa_ind + 1
    I_ind = RR_ind + 1
    
    # Handle different regimes
    # TODO: SS/transition matrix object — handle separately (regime, adjust)
    regime = string(get_setting(m, :regime))
    adjust = string(get_setting(m, :adjust))

    if regime == "QE"
        for jjj in 1:2
            if jjj == 1
                # First pass: compute shock size to match Taylor rule
                x0[end-3] = -m[:σ_R]
                
                # Compute IRF for shock size calculation
                numcontrols_aux = size(gx, 1)
                MX = [Matrix{Float64}(I, length(x0), length(x0)); gx]
                IRF_state_sparse = zeros(numstates + numcontrols_aux, maxlag)
                x = copy(x0)
                
                for t in 1:maxlag
                    IRF_state_sparse[:, t] = (MX * x)
                    x = hx * x
                end
                
                # Extract Y IRF
                Statenext_IRFs_aux = repeat(Xss[(end-os+1):end], 1, maxlag) .+ IRF_state_sparse[(numstates-os+1):numstates, :]
                Control_IRFs_aux = repeat(Yss[(end-oc+1):end], 1, maxlag) .+ Gamma_control[(end-oc+1):end, :] * IRF_state_sparse[(numstates+1):(numstates+numcontrols_aux), :]
                
                Statenext_IRFs_aux = exp.(Statenext_IRFs_aux)
                Control_IRFs_aux = exp.(Control_IRFs_aux)
                
                Y_IRFs_aux = Control_IRFs_aux[Y_ind, 1]
                IRFs_Y_aux = 100 * (Y_IRFs_aux / SS_stats["Y"] - 1)
                
                if adjust == "G"
                    # Load Taylor rule IRF to match
                    try
                        taylor_data = matread("IRFs_Taylor_G.mat")
                        IRFs_Taylor = taylor_data["IRFs_Taylor_G"]
                        MP_shock_aux = m[:σ_R] * IRFs_Taylor["IRFs_Y_p"][1] / IRFs_Y_aux

                        # Save for later use
                        matwrite("MP_shock_aux.mat", Dict("MP_shock_aux" => MP_shock_aux))
                    catch e
                        @warn "Could not load IRFs_Taylor_G.mat, using default shock size"
                        MP_shock_aux = m[:σ_R]
                    end
                elseif adjust == "LT"
                    # Load pre-computed shock size
                    try
                        shock_data = matread("MP_shock_aux.mat")
                        MP_shock_aux = shock_data["MP_shock_aux"]
                    catch e
                        @warn "Could not load MP_shock_aux.mat, using default shock size"
                        MP_shock_aux = m[:σ_R]
                    end
                end
            else
                # Second pass: use computed shock size
                x0[end-3] = -MP_shock_aux
            end
        end
    elseif regime == "Taylor"
        x0[end-3] = -m[:σ_R]
    elseif regime == "Partial QE"
        if MP_shock_aux === nothing
            try
                shock_data = matread("MP_shock_aux.mat")
                MP_shock_aux = shock_data["MP_shock_aux"]
            catch e
                @warn "Could not load MP_shock_aux.mat, using default shock size"
                MP_shock_aux = m[:σ_R]
            end
        end
        x0[end-3] = -MP_shock_aux
    else
        error("Unknown regime: $regime")
    end
    
    # Compute main IRF
    # MX stacks identity (states) and gx (controls), so IRF_state_sparse has numstates+numcontrols rows
    numcontrols = size(gx, 1)
    MX = [Matrix{Float64}(I, length(x0), length(x0)); gx]
    IRF_state_sparse = zeros(numstates + numcontrols, maxlag)
    x = copy(x0)
    
    for t in 1:maxlag
        IRF_state_sparse[:, t] = (MX * x)
        x = hx * x
    end
    
    # Compute state and control IRFs
    Statenext_IRFs = repeat(Xss[(end-os+1):end], 1, maxlag) .+ IRF_state_sparse[(numstates-os+1):numstates, :]
    Control_IRFs = repeat(Yss[(end-oc+1):end], 1, maxlag) .+ Gamma_control[(end-oc+1):end, :] * IRF_state_sparse[(numstates+1):end, :]
    
    Statenext_IRFs = exp.(Statenext_IRFs)
    Control_IRFs = exp.(Control_IRFs)
    
    # Extract individual IRF series
    Rcb_IRFs_p = Statenext_IRFs[R_cb_ind, :]
    W_IRFs_p = Statenext_IRFs[w_ind, :]
    Q_IRFs_p = Statenext_IRFs[Q_ind, :]
    A_g_IRFs_p = Statenext_IRFs[A_g_ind, :]
    B_F_IRFs_p = Statenext_IRFs[B_F_ind, :]
    Lambda_IRFs_p = Control_IRFs[Lambda_ind, :]
    pi_IRFs_p = Control_IRFs[pi_ind, :]
    V_IRFs_p = Control_IRFs[V_ind, :]
    L_IRFs_p = Control_IRFs[L_ind, :]
    J_IRFs_p = Control_IRFs[J_ind, :]
    h_IRFs_p = Control_IRFs[h_ind, :]
    v_IRFs_p = Control_IRFs[v_ind, :]
    Y_IRFs_p = Control_IRFs[Y_ind, :]
    Profit_IRFs_p = Control_IRFs[Profit_ind, :]
    r_k_IRFs_p = Control_IRFs[r_k_ind, :]
    r_a_IRFs_p = Control_IRFs[r_a_ind, :]
    T_IRFs_p = Control_IRFs[T_ind, :]
    B_IRFs_p = Control_IRFs[B_ind, :]
    K_IRFs_p = Control_IRFs[K_ind, :]
    N_IRFs_p = Control_IRFs[N_ind, :]
    LT_IRFs_p = Control_IRFs[LT_ind, :]
    G_IRFs_p = Control_IRFs[G_ind, :]
    C_IRFs_p = Control_IRFs[C_ind, :]
    MC_IRFs_p = Control_IRFs[MC_ind, :]
    RRa_IRFs_p = Control_IRFs[RRa_ind, :]
    RR_IRFs_p = Control_IRFs[RR_ind, :]
    I_IRFs_p = Control_IRFs[I_ind, :]
    A_hh_IRFs_p = Control_IRFs[A_hh_ind, :]
    unemp_IRFs_p = Control_IRFs[unemp_ind, :]
    B_hh_IRFs_p = Control_IRFs[B_hh_ind, :]
    MRS_IRFs_p = Control_IRFs[MRS_ind, :]
    
    # Compute derived series
    U_IRFs_p = 1 .- N_IRFs_p .- m[:Eshare]
    # TODO: SS/transition matrix object — handle separately (lambda)
    M_IRFs_p = (U_IRFs_p .+ SS_stats["lambda"] .* N_IRFs_p) .* V_IRFs_p ./
               (((U_IRFs_p .+ SS_stats["lambda"] .* N_IRFs_p) .^ m[:α] .+
                 V_IRFs_p .^ m[:α]) .^ (1 / m[:α]))
    f_IRFs_p = M_IRFs_p ./ (U_IRFs_p .+ SS_stats["lambda"] .* N_IRFs_p)
    nn_IRFs_p = ((1 - m[:τ_w]) .* W_IRFs_p ./ m[:ψ]) .^ (1 / m[:ξ])
    Ntilde_IRFs_p = (1 - SS_stats["lambda"]) .* N_IRFs_p .+ M_IRFs_p
    IRFs_M_p = 100 * (M_IRFs_p / SS_stats["M"] .- 1)
    IRFs_f_p = 100 * (f_IRFs_p / SS_stats["f"] .- 1)
    IRFs_Y_p = 100 * (Y_IRFs_p / SS_stats["Y"] .- 1)
    
    # Percentage deviations from steady state (log-linearized)
    # For states: MATLAB uses grid.numstates-grid.os+ind
    # For controls: MATLAB uses end-grid.oc+ind, where end = numstates+numcontrols
    total_rows = numstates + numcontrols
    IRFs_Rcb_p = 100 * IRF_state_sparse[(numstates-os+R_cb_ind), :]
    IRFs_W_p = 100 * IRF_state_sparse[(numstates-os+w_ind), :]
    IRFs_Q_p = 100 * IRF_state_sparse[(numstates-os+Q_ind), :]
    IRFs_A_g_p = 100 * IRF_state_sparse[(numstates-os+A_g_ind), :]
    IRFs_Lambda_p = 100 * IRF_state_sparse[(total_rows-oc+Lambda_ind), :]
    IRFs_pi_p = 100 * IRF_state_sparse[(total_rows-oc+pi_ind), :]
    IRFs_V_p = 100 * IRF_state_sparse[(total_rows-oc+V_ind), :]
    # J_ind is a range
    IRFs_J_p = 100 * IRF_state_sparse[(total_rows - oc) .+ J_ind, :]
    IRFs_h_p = 100 * IRF_state_sparse[(total_rows-oc+h_ind), :]
    IRFs_v_p = 100 * IRF_state_sparse[(total_rows-oc+v_ind), :]
    IRFs_Profit_p = 100 * IRF_state_sparse[(total_rows-oc+Profit_ind), :]
    IRFs_r_k_p = 100 * IRF_state_sparse[(total_rows-oc+r_k_ind), :]
    IRFs_r_a_p = 100 * IRF_state_sparse[(total_rows-oc+r_a_ind), :]
    IRFs_T_p = 100 * IRF_state_sparse[(total_rows-oc+T_ind), :]
    IRFs_B_p = 100 * IRF_state_sparse[(total_rows-oc+B_ind), :]
    IRFs_K_p = 100 * IRF_state_sparse[(total_rows-oc+K_ind), :]
    IRFs_N_p = 100 * IRF_state_sparse[(total_rows-oc+N_ind), :]
    IRFs_LT_p = 100 * IRF_state_sparse[(total_rows-oc+LT_ind), :]
    IRFs_G_p = 100 * IRF_state_sparse[(total_rows-oc+G_ind), :]
    IRFs_C_hh_p = 100 * IRF_state_sparse[(total_rows-oc+C_ind), :]
    IRFs_MC_p = 100 * IRF_state_sparse[(total_rows-oc+MC_ind), :]
    IRFs_RRa_p = 100 * IRF_state_sparse[(total_rows-oc+RRa_ind), :]
    IRFs_RR_p = 100 * IRF_state_sparse[(total_rows-oc+RR_ind), :]
    IRFs_I_p = 100 * IRF_state_sparse[(total_rows-oc+I_ind), :]
    IRFs_A_hh_p = 100 * IRF_state_sparse[(total_rows-oc+A_hh_ind), :]
    IRFs_unemp_p = 100 * (unemp_IRFs_p .- SS_stats["u"])
    IRFs_B_hh_p = 100 * IRF_state_sparse[(total_rows-oc+B_hh_ind), :]
    IRFs_MRS_p = 100 * IRF_state_sparse[(total_rows-oc+MRS_ind), :]
    
    # Financial sector variables
    A_b_IRFs_p = Statenext_IRFs[A_b_ind, :]
    B_b_IRFs_p = Statenext_IRFs[B_b_ind, :]
    lev_IRFs_p = Statenext_IRFs[lev_ind, :]
    NW_b_IRFs_p = Statenext_IRFs[NW_b_ind, :]
    zz_IRFs_p = Control_IRFs[zz_ind, :]
    xx_IRFs_p = Control_IRFs[xx_ind, :]
    Profit_FI_IRFs_p = Control_IRFs[Profit_FI_ind, :]
    vv_IRFs_p = Control_IRFs[vv_ind, :]
    ee_IRFs_p = Control_IRFs[ee_ind, :]
    C_b_IRFs_p = Control_IRFs[C_b_ind, :]
    
    IRFs_A_b_p = 100 * IRF_state_sparse[(numstates-os+A_b_ind), :]
    IRFs_B_b_p = 100 * IRF_state_sparse[(numstates-os+B_b_ind), :]
    IRFs_lev_p = 100 * IRF_state_sparse[(numstates-os+lev_ind), :]
    IRFs_NW_b_p = 100 * IRF_state_sparse[(numstates-os+NW_b_ind), :]
    IRFs_zz_p = 100 * IRF_state_sparse[(total_rows-oc+zz_ind), :]
    IRFs_xx_p = 100 * IRF_state_sparse[(total_rows-oc+xx_ind), :]
    IRFs_Profit_FI_p = 100 * IRF_state_sparse[(total_rows-oc+Profit_FI_ind), :]
    IRFs_vv_p = 100 * IRF_state_sparse[(total_rows-oc+vv_ind), :]
    IRFs_ee_p = 100 * IRF_state_sparse[(total_rows-oc+ee_ind), :]
    IRFs_C_b_p = 100 * IRF_state_sparse[(total_rows-oc+C_b_ind), :]
    
    # Markup calculations
    # TODO: SS/transition matrix object — handle separately (v in SS context)
    TC_ss = m[:w_bar] * SS_stats["L"] + m[:δ_0] * SS_stats["v"]^m[:δ_1] * grid["K"] +
            m[:ι] * SS_stats["V"] + m[:fix]
    Amarkup = 1 - TC_ss / SS_stats["Y"]

    Markup_IRFs_p = 1 .- (W_IRFs_p .* L_IRFs_p .+ m[:δ_0] .* v_IRFs_p .^ m[:δ_1] .* K_IRFs_p .+
                          m[:ι] .* V_IRFs_p .+ m[:fix]) ./ Y_IRFs_p
    IRFs_Markups_p = 100 * (Markup_IRFs_p .- Amarkup)
    IRFs_Profit_NF_p = 100 * ((Profit_IRFs_p .- Profit_FI_IRFs_p .+ m[:fix2]) / SS_stats["Profit"] .- 1)
    IRFs_Profit_F_p = 100 * ((SS_stats["Profit"] .- m[:fix2] .+ Profit_FI_IRFs_p) / SS_stats["Profit"] .- 1)

    # Average cost
    AC_IRFs_p = (m[:δ_0] .* v_IRFs_p[1:(end-1)] .^ m[:δ_1] .* K_IRFs_p[1:(end-1)] .+
                 W_IRFs_p[2:end] .* L_IRFs_p[1:(end-1)] .+ m[:fix] .+ m[:ι] .* V_IRFs_p[1:(end-1)]) ./
                Y_IRFs_p[1:(end-1)]
    IRFs_AC_p = 100 * (AC_IRFs_p / SS_stats["AvgC"] .- 1)

    IRFs_Profit_int_p = 100 * (((1 .- MC_IRFs_p) .* Y_IRFs_p .- m[:fix]) / SS_stats["Profit_int"] .- 1)

    # Government and financial sector bonds
    B_FI_IRFs_p = Q_IRFs_p .* A_b_IRFs_p .- NW_b_IRFs_p
    B_cp_IRFs_p = Q_IRFs_p .* (1 + m[:τ_cp]) .* (A_g_IRFs_p .- 1)
    IRFs_B_FI_p = 100 * (B_FI_IRFs_p / SS_stats["B_FI"] .- 1)
    B_gov_IRFs_p = B_b_IRFs_p[2:end] .+ B_hh_IRFs_p[1:(end-1)] / (1 - m[:dr]) .+
                   SS_stats["B_F"] .- B_FI_IRFs_p[2:end]
    B_gov_ncp_IRFs_p = B_gov_IRFs_p .- B_cp_IRFs_p[2:end]
    IRFs_B_gov_p = 100 * (B_gov_IRFs_p / SS_stats["B_gov"] .- 1)
    IRFs_B_gov_ncp_p = 100 * (B_gov_ncp_IRFs_p / SS_stats["B_gov_ncp"] .- 1)
    
    # Build output dictionary
    # MATLAB stores 1D series as 1×maxlag; ensure we match for comparison.
    asrow(x) = ndims(x) == 1 ? reshape(x, 1, :) : x
    IRFs = Dict{String, Any}()

    IRFs["AC_IRFs_p"] = asrow(AC_IRFs_p)
    IRFs["IRFs_AC_p"] = asrow(IRFs_AC_p)
    IRFs["IRFs_Profit_int_p"] = asrow(IRFs_Profit_int_p)

    IRFs["Rcb_IRFs_p"] = asrow(Rcb_IRFs_p)
    IRFs["W_IRFs_p"] = asrow(W_IRFs_p)
    IRFs["Q_IRFs_p"] = asrow(Q_IRFs_p)
    IRFs["A_g_IRFs_p"] = asrow(A_g_IRFs_p)
    IRFs["Lambda_IRFs_p"] = asrow(Lambda_IRFs_p)
    IRFs["pi_IRFs_p"] = asrow(pi_IRFs_p)
    IRFs["V_IRFs_p"] = asrow(V_IRFs_p)
    IRFs["J_IRFs_p"] = asrow(J_IRFs_p)
    IRFs["h_IRFs_p"] = asrow(h_IRFs_p)
    IRFs["v_IRFs_p"] = asrow(v_IRFs_p)
    IRFs["Y_IRFs_p"] = asrow(Y_IRFs_p)
    IRFs["Profit_IRFs_p"] = asrow(Profit_IRFs_p)
    IRFs["r_k_IRFs_p"] = asrow(r_k_IRFs_p)
    IRFs["r_a_IRFs_p"] = asrow(r_a_IRFs_p)
    IRFs["T_IRFs_p"] = asrow(T_IRFs_p)
    IRFs["B_IRFs_p"] = asrow(B_IRFs_p)
    IRFs["K_IRFs_p"] = asrow(K_IRFs_p)
    IRFs["N_IRFs_p"] = asrow(N_IRFs_p)
    IRFs["LT_IRFs_p"] = asrow(LT_IRFs_p)
    IRFs["G_IRFs_p"] = asrow(G_IRFs_p)
    IRFs["C_IRFs_p"] = asrow(C_IRFs_p)
    IRFs["MC_IRFs_p"] = asrow(MC_IRFs_p)
    IRFs["RRa_IRFs_p"] = asrow(RRa_IRFs_p)
    IRFs["RR_IRFs_p"] = asrow(RR_IRFs_p)
    IRFs["I_IRFs_p"] = asrow(I_IRFs_p)
    IRFs["A_hh_IRFs_p"] = asrow(A_hh_IRFs_p)
    IRFs["unemp_IRFs_p"] = asrow(unemp_IRFs_p)
    IRFs["B_hh_IRFs_p"] = asrow(B_hh_IRFs_p)
    IRFs["MRS_IRFs_p"] = asrow(MRS_IRFs_p)

    IRFs["Markup_IRFs_p"] = asrow(Markup_IRFs_p)
    IRFs["IRFs_Markups_p"] = asrow(IRFs_Markups_p)
    IRFs["IRFs_Profit_NF_p"] = asrow(IRFs_Profit_NF_p)
    IRFs["IRFs_Profit_F_p"] = asrow(IRFs_Profit_F_p)

    IRFs["U_IRFs_p"] = asrow(U_IRFs_p)
    IRFs["M_IRFs_p"] = asrow(M_IRFs_p)
    IRFs["f_IRFs_p"] = asrow(f_IRFs_p)
    IRFs["nn_IRFs_p"] = asrow(nn_IRFs_p)
    IRFs["Ntilde_IRFs_p"] = asrow(Ntilde_IRFs_p)
    IRFs["IRFs_M_p"] = asrow(IRFs_M_p)
    IRFs["IRFs_f_p"] = asrow(IRFs_f_p)
    IRFs["IRFs_Y_p"] = asrow(IRFs_Y_p)

    IRFs["IRFs_Rcb_p"] = asrow(IRFs_Rcb_p)
    IRFs["IRFs_W_p"] = asrow(IRFs_W_p)
    IRFs["IRFs_Q_p"] = asrow(IRFs_Q_p)
    IRFs["IRFs_A_g_p"] = asrow(IRFs_A_g_p)
    IRFs["IRFs_Lambda_p"] = asrow(IRFs_Lambda_p)
    IRFs["IRFs_pi_p"] = asrow(IRFs_pi_p)
    IRFs["IRFs_V_p"] = asrow(IRFs_V_p)
    IRFs["IRFs_J_p"] = asrow(IRFs_J_p)
    IRFs["IRFs_h_p"] = asrow(IRFs_h_p)
    IRFs["IRFs_v_p"] = asrow(IRFs_v_p)
    IRFs["IRFs_Profit_p"] = asrow(IRFs_Profit_p)
    IRFs["IRFs_r_k_p"] = asrow(IRFs_r_k_p)
    IRFs["IRFs_r_a_p"] = asrow(IRFs_r_a_p)
    IRFs["IRFs_T_p"] = asrow(IRFs_T_p)
    IRFs["IRFs_B_p"] = asrow(IRFs_B_p)
    IRFs["IRFs_K_p"] = asrow(IRFs_K_p)
    IRFs["IRFs_N_p"] = asrow(IRFs_N_p)
    IRFs["IRFs_LT_p"] = asrow(IRFs_LT_p)
    IRFs["IRFs_G_p"] = asrow(IRFs_G_p)
    IRFs["IRFs_C_hh_p"] = asrow(IRFs_C_hh_p)
    IRFs["IRFs_MC_p"] = asrow(IRFs_MC_p)
    IRFs["IRFs_RRa_p"] = asrow(IRFs_RRa_p)
    IRFs["IRFs_RR_p"] = asrow(IRFs_RR_p)
    IRFs["IRFs_I_p"] = asrow(IRFs_I_p)
    IRFs["IRFs_A_hh_p"] = asrow(IRFs_A_hh_p)
    IRFs["IRFs_unemp_p"] = asrow(IRFs_unemp_p)
    IRFs["IRFs_B_hh_p"] = asrow(IRFs_B_hh_p)
    IRFs["IRFs_MRS_p"] = asrow(IRFs_MRS_p)

    IRFs["A_b_IRFs_p"] = asrow(A_b_IRFs_p)
    IRFs["B_b_IRFs_p"] = asrow(B_b_IRFs_p)
    IRFs["lev_IRFs_p"] = asrow(lev_IRFs_p)
    IRFs["NW_b_IRFs_p"] = asrow(NW_b_IRFs_p)
    IRFs["zz_IRFs_p"] = asrow(zz_IRFs_p)
    IRFs["xx_IRFs_p"] = asrow(xx_IRFs_p)
    IRFs["Profit_FI_IRFs_p"] = asrow(Profit_FI_IRFs_p)
    IRFs["vv_IRFs_p"] = asrow(vv_IRFs_p)
    IRFs["ee_IRFs_p"] = asrow(ee_IRFs_p)
    IRFs["C_b_IRFs_p"] = asrow(C_b_IRFs_p)

    IRFs["IRFs_A_b_p"] = asrow(IRFs_A_b_p)
    IRFs["IRFs_B_b_p"] = asrow(IRFs_B_b_p)
    IRFs["IRFs_lev_p"] = asrow(IRFs_lev_p)
    IRFs["IRFs_NW_b_p"] = asrow(IRFs_NW_b_p)
    IRFs["IRFs_zz_p"] = asrow(IRFs_zz_p)
    IRFs["IRFs_xx_p"] = asrow(IRFs_xx_p)
    IRFs["IRFs_Profit_FI_p"] = asrow(IRFs_Profit_FI_p)
    IRFs["IRFs_vv_p"] = asrow(IRFs_vv_p)
    IRFs["IRFs_ee_p"] = asrow(IRFs_ee_p)
    IRFs["IRFs_C_b_p"] = asrow(IRFs_C_b_p)

    IRFs["B_FI_IRFs_p"] = asrow(B_FI_IRFs_p)
    IRFs["B_cp_IRFs_p"] = asrow(B_cp_IRFs_p)
    IRFs["IRFs_B_FI_p"] = asrow(IRFs_B_FI_p)
    IRFs["B_gov_IRFs_p"] = asrow(B_gov_IRFs_p)
    IRFs["B_gov_ncp_IRFs_p"] = asrow(B_gov_ncp_IRFs_p)
    IRFs["IRFs_B_gov_p"] = asrow(IRFs_B_gov_p)
    IRFs["IRFs_B_gov_ncp_p"] = asrow(IRFs_B_gov_ncp_p)
    
    return IRFs
end
