using LinearAlgebra
using SparseArrays
using Interpolations
include("../helpers/ndgrid.jl")
include("../helpers/genweight.jl")
include("../helpers/uncompress.jl")
include("../helpers/compress.jl")
include("../helpers/myinterpolate3.jl")
include("../helpers/cdf_to_pdf3D_forwarddiff.jl")
include("../helpers/sub2ind.jl")
include("../helpers/q_cons2.jl")
include("policies_update.jl")

"""
    _build_transition_matrix!(weight11, weight12, weight21, weight22,
                               Dist_b, idb, Dist_a, ida,
                               P_transition_dist, nb, na, nse)

Build a single transition matrix from bilinear weights and productivity transitions.
Mutates the pre-allocated weight arrays and returns the sparse transition matrix.
"""
function _build_transition_matrix!(weight11, weight12, weight21, weight22,
                                    Dist_b, idb, Dist_a, ida,
                                    P_transition_dist, nb, na, nse)
    idb_rep = repeat(vec(idb), 1, nse)
    ida_rep = repeat(vec(ida), 1, nse)
    idse = vec(kron(1:nse, ones(Int, 1, nb * na * nse))')'

    index11 = sub2ind((nb, na, nse), vec(idb_rep), vec(ida_rep), vec(idse))
    index12 = sub2ind((nb, na, nse), vec(idb_rep), vec(ida_rep) .+ 1, vec(idse))
    index21 = sub2ind((nb, na, nse), vec(idb_rep) .+ 1, vec(ida_rep), vec(idse))
    index22 = sub2ind((nb, na, nse), vec(idb_rep) .+ 1, vec(ida_rep) .+ 1, vec(idse))

    @inbounds for hh in 1:nse
        Db = @view Dist_b[:, :, hh]
        Da = @view Dist_a[:, :, hh]
        Pt = @view P_transition_dist[hh, :]

        for i in eachindex(Db)
            w_b = Db[i]
            w_a = Da[i]
            w11 = (1.0 - w_b) * (1.0 - w_a)
            w12 = (1.0 - w_b) * w_a
            w21 = w_b * (1.0 - w_a)
            w22 = w_b * w_a
            for s in 1:nse
                weight11[i, s, hh] = w11 * Pt[s]
                weight12[i, s, hh] = w12 * Pt[s]
                weight21[i, s, hh] = w21 * Pt[s]
                weight22[i, s, hh] = w22 * Pt[s]
            end
        end
    end

    # Permute (b×a, h', h) -> (b×a, h, h')
    weight11p = permutedims(weight11, (1, 3, 2))
    weight12p = permutedims(weight12, (1, 3, 2))
    weight21p = permutedims(weight21, (1, 3, 2))
    weight22p = permutedims(weight22, (1, 3, 2))
    rowindex = repeat(collect(1:nb*na*nse), 4*nse)

    return sparse(vec(rowindex),
                  [vec(index11); vec(index21); vec(index12); vec(index22)],
                  [vec(weight11p); vec(weight21p); vec(weight12p); vec(weight22p)],
                  nb * na * nse, nb * na * nse)
end

"""
    F_sys_ref_tvcopula_QE(State, Stateminus, Controlnext_sparse, Control_sparse, StateSS,
                           ControlSS, Gamma_state, Gamma_control, InvGamma,
                           param, grid, SS_stats,  DCD, IDCD, P_SE)

System of equations written in Schmitt-Grohe-Uribe generic form with states and controls.

# Returns
- `Difference`: Difference between LHS and RHS
- `LHS`: Left hand side of equations
- `RHS`: Right hand side of equations
- `MUnext`: Next period distribution
- `c_a_star`, `b_a_star`, `a_a_star`: Asset holder policies
- `c_n_star`, `b_n_star`: Non-asset holder policies
- `AC`: Adjustment costs
- `AProb`: Transition probabilities
- `P_transition_exp`: Expected transition matrix
- `P_transition_dist`: Distribution transition matrix
"""
function F_sys_ref_tvcopula_QE(State, Stateminus, Controlnext_sparse, Control_sparse, StateSS,
                                 ControlSS, Gamma_state, Gamma_control, InvGamma,
                                 param, grid, SS_stats,  DCD, IDCD, P_SE)

# System of equations written in Schmitt-Groh?-Uribe generic form with states and controls
# STATE: Vector of state variables t+1 (only marginal distributions for histogram)
# STATEMINUS: Vector of state variables t (only marginal distributions for histogram)
# CONTROL: Vector of state variables t+1 (only coefficients of sparse polynomial)
# CONTROLMINUS: Vector of state variables t (only coefficients of sparse polynomial)
# STATESS and CONTROLSS: Value of the state and control variables in steady
# state. For the Value functions these are at full grids.
# GAMMA_STATE: Mapping such that perturbationof marginals are still
# distributions (sum to 1).
# PARAM: Model and numerical parameters (structure)
# GRID: Liquid, illiquid and productivity grid
# SS_stats: Stores targets for government policy
# COPULA: Interpolant that allows to map marginals back to full-grid
# distribuitions
# P: steady state transition matrix
# aggshock: sets whether the Aggregate shock is Z or uncertainty
#
# =========================================================================
# Part of the Matlab code to aPSI_RPompany the paper
# 'Solving heterogeneous agent models in discrete time with
# many idiosyncratic states by perturbation methods', CEPR Discussion Paper
# DP13071
# by Christian Bayer and Ralph Luetticke
# http://www.erc-hump-uni-bonn.net/
# =========================================================================

## Initializations

util     = c -> (c .^ (1 - param["sigma"])) ./ (1 - param["sigma"])
mutil    = c -> 1.0 ./ (c .^ param["sigma"])
invutil  = u -> ((1 - param["sigma"]) .* u) .^ (1 / (1 - param["sigma"]))
invmutil = mu -> (1.0 ./ mu) .^ (1 / param["sigma"])

# Number of states, controls
nx   = Int(grid["numstates"])  # Number of states
ny   = length(ControlSS)       # Number of Controls

# Sizes for marginal & copula blocks
nFullMarg = size(Gamma_state, 1)      # = nb + na + nse
nRedMarg  = size(Gamma_state, 2)      # = nb + na + nse - 3
nCOP      = Int(grid["nCOP"])         # compressed-copula coefficients count

NxNx = nRedMarg + nCOP                # states-before-aggregates (reduced)

nb  = Int(grid["nb"])
na  = Int(grid["na"])
nse = Int(grid["nse"])
ns  = Int(grid["ns"])
NN  = nb * na * nse   # Number of points in the full grid
Ny  = 3 * NN

# Initialize LHS and RHS
LHS = zeros(nx + ny)
RHS = zeros(nx + ny) 

# Indices for LHS/RHS
# Indices for Controls

VALUE_ind   = 1:NN
mutil_c_ind = NN .+ (1:NN)
Va_ind      = 2*NN .+ (1:NN)

# Summary variables
MRS_ind       = Ny + 1
A_hh_ind      = MRS_ind + 1
B_hh_ind      = A_hh_ind + 1
C_ind         = B_hh_ind + 1
N_ind         = C_ind + 1
L_ind         = N_ind + 1
UB_ind        = L_ind + 1

# Other control variables
K_ind         = UB_ind + 1         # 1
B_ind         = K_ind + 1          # 2
B_gov_ncp_ind = B_ind + 1          # 3
T_ind         = B_gov_ncp_ind + 1  # 4
LT_ind        = T_ind + 1          # 5
G_ind         = LT_ind + 1         # 6
Lambda_ind    = G_ind + 1          # 7
pi_ind        = Lambda_ind + 1     # 8
V_ind         = pi_ind + 1         # 9
J_ind         = V_ind .+ (1:ns)    # 10,11,12,13,14
h_ind         = J_ind[end] + 1     # 15
v_ind         = h_ind + 1          # 16
Y_ind         = v_ind + 1          # 17
Profit_ind    = Y_ind + 1          # 18
r_k_ind       = Profit_ind + 1     # 19
r_a_ind       = r_k_ind + 1        # 20
MC_ind        = r_a_ind + 1        # 21
unemp_ind     = MC_ind + 1         # 22
nn_ind        = unemp_ind + 1      # 23
M_ind         = nn_ind + 1         # 24
f_ind         = M_ind + 1          # 25
zz_ind        = f_ind + 1          # 26
xx_ind        = zz_ind + 1         # 27
vv_ind        = xx_ind + 1         # 28
ee_ind        = vv_ind + 1         # 29
C_b_ind       = ee_ind + 1         # 30
Profit_FI_ind = C_b_ind + 1        # 31
RRa_ind       = Profit_FI_ind + 1  # 32
RR_ind        = RRa_ind + 1        # 33
I_ind         = RR_ind + 1         # 34
x_k_ind       = I_ind + 1          # 35

# Observables
A_g_obs_ind    = x_k_ind + 1        # 36
Y_obs_ind      = A_g_obs_ind + 1    # 37
C_obs_ind      = Y_obs_ind + 1      # 38
I_obs_ind      = C_obs_ind + 1      # 39
w_obs_ind      = I_obs_ind + 1      # 40
PROFIT_obs_ind = w_obs_ind + 1      # 41
unemp_obs_ind  = PROFIT_obs_ind + 1 # 42
inf_obs_ind    = unemp_obs_ind + 1
R_obs_ind      = inf_obs_ind + 1
pastw_ind      = R_obs_ind + 1
G_obs_ind      = pastw_ind + 1 #46

pastYY_ind        = G_obs_ind + 1 #47
pastCC_ind        = pastYY_ind + 1
pastII_ind        = pastCC_ind + 1
pastPPROFIT_ind   = pastII_ind + 1
pastuu_ind        = pastPPROFIT_ind + 1
pastGG_ind        = pastuu_ind + 1
pastA_g_ind       = pastGG_ind + 1
l_lambda_ind      = pastA_g_ind + 1
pastQ_ind         = l_lambda_ind + 1
x_I_ind           = pastQ_ind + 1
eta2_ind          = x_I_ind + 1
iota2_ind         = eta2_ind + 1
pastLT2_ind       = iota2_ind + 1
pastG2_ind        = pastLT2_ind + 1
LT_obs_ind        = pastG2_ind + 1
B_gov_ncp2_ind    = LT_obs_ind + 1 #62

# Indices for States
marginal_b_ind  = 1:(nb-1)
marginal_a_ind  = (nb-1) .+ (1:(na-1))
marginal_se_ind = (nb + na - 2) .+ (1:(nse-1))

COP_ind         = (nb + na + nse - 3) .+ (1:nCOP) 

R_cb_ind        = NxNx + 1
w_ind           = NxNx + 2
A_b_ind         = NxNx + 3
B_b_ind         = NxNx + 4
A_g_ind         = NxNx + 5
Q_ind           = NxNx + 6
lev_ind         = NxNx + 7
NW_b_ind        = NxNx + 8
R_tilde_ind     = NxNx + 9
x_cb_ind        = NxNx + 10
pastpi_ind      = NxNx + 11

pastY_ind       = NxNx + 12
pastC_ind       = NxNx + 13
pastI_ind       = NxNx + 14
pastPROFIT_ind  = NxNx + 15
pastunemp_ind   = NxNx + 16
pastG_ind       = NxNx + 17
pastLT_ind      = NxNx + 18
R_star_ind      = NxNx + 19

B_F_ind         = NxNx + 20
Z_ind           = NxNx + 21
PSI_RP_ind      = NxNx + 22
eta_ind         = NxNx + 23
D_ind           = NxNx + 24
GG_ind          = NxNx + 25
iota_ind        = NxNx + 26
BB_ind          = NxNx + 27
PSI_W_ind       = NxNx + 28
MP_ind          = NxNx + 29
p_mark_ind      = NxNx + 30

eps_QE_ind      = NxNx + 31
eps_RP_ind      = NxNx + 32  # Networth
eps_B_F_ind     = NxNx + 33
eps_BB_ind      = NxNx + 34  # Undetermined
eps_Z_ind       = NxNx + 35  # 1) TFP shock
eps_G_ind       = NxNx + 36  # 2) G shock
eps_D_ind       = NxNx + 37  # 3) Liquidity preference
eps_R_ind       = NxNx + 38  # 4) MP shock
eps_iota_ind    = NxNx + 39  # 5) MEI (or IST) shock
eps_eta_ind     = NxNx + 40  # 6) Price mark-up shock
eps_w_ind       = NxNx + 41  # 7) Wage (mark-up) shock

NxNx_aux = 41 


## Control Variables (For value function and derivatives, DCT is used)

oc = Int(grid["oc"])

Controlnext  = ControlSS .* (1 .+ Gamma_control * Controlnext_sparse)
Control      = ControlSS .* (1 .+ Gamma_control * Control_sparse)

Controlnext[end-oc+1:end] = ControlSS[end-oc+1:end] + Gamma_control[end-oc+1:end, :] * Controlnext_sparse
Control[end-oc+1:end]     = ControlSS[end-oc+1:end] + Gamma_control[end-oc+1:end, :] * Control_sparse 

## State Variables

# Marginals only (Gamma_state maps reduced marginals → full marginals)
Distribution      = StateSS[1:nFullMarg] + Gamma_state * State[1:nRedMarg]
Distributionminus = StateSS[1:nFullMarg] + Gamma_state * Stateminus[1:nRedMarg]

# Copula coefficients (kept separate; used when reconstructing the joint)
cop_coefs   = State[nRedMarg .+ (1:nCOP)]
cop_coefs_m = Stateminus[nRedMarg .+ (1:nCOP)] 


#  Fiscal and Monetary Policy Shock

R_cb           = StateSS[end-(NxNx_aux-R_cb_ind+NxNx)] + (State[end-(NxNx_aux-R_cb_ind+NxNx)]) 
R_cbminus      = StateSS[end-(NxNx_aux-R_cb_ind+NxNx)] + (Stateminus[end-(NxNx_aux-R_cb_ind+NxNx)]) 

Wminus         = StateSS[end-(NxNx_aux-w_ind+NxNx)] + (Stateminus[end-(NxNx_aux-w_ind+NxNx)]) 
W              = StateSS[end-(NxNx_aux-w_ind+NxNx)] + (State[end-(NxNx_aux-w_ind+NxNx)]) 

A_bnext        = StateSS[end-(NxNx_aux-A_b_ind+NxNx)] + (State[end-(NxNx_aux-A_b_ind+NxNx)]) 
A_b            = StateSS[end-(NxNx_aux-A_b_ind+NxNx)] + (Stateminus[end-(NxNx_aux-A_b_ind+NxNx)]) 

B_bnext           = StateSS[end-(NxNx_aux-B_b_ind+NxNx)] + (State[end-(NxNx_aux-B_b_ind+NxNx)]) 
B_b               = StateSS[end-(NxNx_aux-B_b_ind+NxNx)] + (Stateminus[end-(NxNx_aux-B_b_ind+NxNx)]) 

A_gauxnext        = StateSS[end-(NxNx_aux-A_g_ind+NxNx)] + (State[end-(NxNx_aux-A_g_ind+NxNx)]) 
A_gaux            = StateSS[end-(NxNx_aux-A_g_ind+NxNx)] + (Stateminus[end-(NxNx_aux-A_g_ind+NxNx)]) 

Q                 = StateSS[end-(NxNx_aux-Q_ind+NxNx)] + (State[end-(NxNx_aux-Q_ind+NxNx)]) 
Qminus            = StateSS[end-(NxNx_aux-Q_ind+NxNx)] + (Stateminus[end-(NxNx_aux-Q_ind+NxNx)]) 

lev               = StateSS[end-(NxNx_aux-lev_ind+NxNx)] + (State[end-(NxNx_aux-lev_ind+NxNx)]) 
levminus          = StateSS[end-(NxNx_aux-lev_ind+NxNx)] + (Stateminus[end-(NxNx_aux-lev_ind+NxNx)]) 

NW_b              = StateSS[end-(NxNx_aux-NW_b_ind+NxNx)] + (State[end-(NxNx_aux-NW_b_ind+NxNx)]) 
NW_bminus         = StateSS[end-(NxNx_aux-NW_b_ind+NxNx)] + (Stateminus[end-(NxNx_aux-NW_b_ind+NxNx)]) 

# inven             = StateSS[end-(NxNx_aux-inven_ind+NxNx]) + (State[end-(NxNx_aux-inven_ind+NxNx])) 
# invenminus        = StateSS[end-(NxNx_aux-inven_ind+NxNx]) + (Stateminus[end-(NxNx_aux-inven_ind+NxNx])) 

R_tilde           = StateSS[end-(NxNx_aux-R_tilde_ind+NxNx)] + (State[end-(NxNx_aux-R_tilde_ind+NxNx)]) 
R_tildeminus      = StateSS[end-(NxNx_aux-R_tilde_ind+NxNx)] + (Stateminus[end-(NxNx_aux-R_tilde_ind+NxNx)]) 

x_cb              = StateSS[end-(NxNx_aux-x_cb_ind+NxNx)] + (State[end-(NxNx_aux-x_cb_ind+NxNx)]) 
x_cbminus         = StateSS[end-(NxNx_aux-x_cb_ind+NxNx)] + (Stateminus[end-(NxNx_aux-x_cb_ind+NxNx)]) 

# pasts2s           = StateSS[end-(NxNx_aux-pasts2s_ind+NxNx]) + (State[end-(NxNx_aux-pasts2s_ind+NxNx])) 
# pasts2sminus      = StateSS[end-(NxNx_aux-pasts2s_ind+NxNx]) + (Stateminus[end-(NxNx_aux-pasts2s_ind+NxNx])) 

pastpi = StateSS[end - (NxNx_aux - pastpi_ind + NxNx)] +
         State[end - (NxNx_aux - pastpi_ind + NxNx)]

pastpiminus = StateSS[end - (NxNx_aux - pastpi_ind + NxNx)] +
              Stateminus[end - (NxNx_aux - pastpi_ind + NxNx)]


pastY = StateSS[end - (NxNx_aux - pastY_ind + NxNx)] +
        State[end - (NxNx_aux - pastY_ind + NxNx)]

pastYminus = StateSS[end - (NxNx_aux - pastY_ind + NxNx)] +
             Stateminus[end - (NxNx_aux - pastY_ind + NxNx)]


pastC = StateSS[end - (NxNx_aux - pastC_ind + NxNx)] +
        State[end - (NxNx_aux - pastC_ind + NxNx)]

pastCminus = StateSS[end - (NxNx_aux - pastC_ind + NxNx)] +
             Stateminus[end - (NxNx_aux - pastC_ind + NxNx)]


pastI = StateSS[end - (NxNx_aux - pastI_ind + NxNx)] +
        State[end - (NxNx_aux - pastI_ind + NxNx)]

pastIminus = StateSS[end - (NxNx_aux - pastI_ind + NxNx)] +
             Stateminus[end - (NxNx_aux - pastI_ind + NxNx)]


pastPROFIT = StateSS[end - (NxNx_aux - pastPROFIT_ind + NxNx)] +
             State[end - (NxNx_aux - pastPROFIT_ind + NxNx)]

pastPROFITminus = StateSS[end - (NxNx_aux - pastPROFIT_ind + NxNx)] +
                  Stateminus[end - (NxNx_aux - pastPROFIT_ind + NxNx)]


pastunemp = StateSS[end - (NxNx_aux - pastunemp_ind + NxNx)] +
            State[end - (NxNx_aux - pastunemp_ind + NxNx)]

pastunempminus = StateSS[end - (NxNx_aux - pastunemp_ind + NxNx)] +
                 Stateminus[end - (NxNx_aux - pastunemp_ind + NxNx)]


pastG = StateSS[end - (NxNx_aux - pastG_ind + NxNx)] +
        State[end - (NxNx_aux - pastG_ind + NxNx)]

pastGminus = StateSS[end - (NxNx_aux - pastG_ind + NxNx)] +
             Stateminus[end - (NxNx_aux - pastG_ind + NxNx)]


pastLT = StateSS[end - (NxNx_aux - pastLT_ind + NxNx)] +
         State[end - (NxNx_aux - pastLT_ind + NxNx)]

pastLTminus = StateSS[end - (NxNx_aux - pastLT_ind + NxNx)] +
              Stateminus[end - (NxNx_aux - pastLT_ind + NxNx)]


R_star = StateSS[end - (NxNx_aux - R_star_ind + NxNx)] +
         State[end - (NxNx_aux - R_star_ind + NxNx)]

R_starminus = StateSS[end - (NxNx_aux - R_star_ind + NxNx)] +
              Stateminus[end - (NxNx_aux - R_star_ind + NxNx)]


B_Fnext = StateSS[end - (NxNx_aux - B_F_ind + NxNx)] +
          State[end - (NxNx_aux - B_F_ind + NxNx)]

B_F = StateSS[end - (NxNx_aux - B_F_ind + NxNx)] +
      Stateminus[end - (NxNx_aux - B_F_ind + NxNx)]


Z = StateSS[end - (NxNx_aux - Z_ind + NxNx)] +
    State[end - (NxNx_aux - Z_ind + NxNx)]

Zminus = StateSS[end - (NxNx_aux - Z_ind + NxNx)] +
         Stateminus[end - (NxNx_aux - Z_ind + NxNx)]


PSI_RP = StateSS[end - (NxNx_aux - PSI_RP_ind + NxNx)] +
         State[end - (NxNx_aux - PSI_RP_ind + NxNx)]

PSI_RPminus = StateSS[end - (NxNx_aux - PSI_RP_ind + NxNx)] +
              Stateminus[end - (NxNx_aux - PSI_RP_ind + NxNx)]


eta = StateSS[end - (NxNx_aux - eta_ind + NxNx)] +
      State[end - (NxNx_aux - eta_ind + NxNx)]

etaminus = StateSS[end - (NxNx_aux - eta_ind + NxNx)] +
           Stateminus[end - (NxNx_aux - eta_ind + NxNx)]


D = StateSS[end - (NxNx_aux - D_ind + NxNx)] +
    State[end - (NxNx_aux - D_ind + NxNx)]

Dminus = StateSS[end - (NxNx_aux - D_ind + NxNx)] +
         Stateminus[end - (NxNx_aux - D_ind + NxNx)]


GG = StateSS[end - (NxNx_aux - GG_ind + NxNx)] +
     State[end - (NxNx_aux - GG_ind + NxNx)]

GGminus = StateSS[end - (NxNx_aux - GG_ind + NxNx)] +
          Stateminus[end - (NxNx_aux - GG_ind + NxNx)]


iota = StateSS[end - (NxNx_aux - iota_ind + NxNx)] +
       State[end - (NxNx_aux - iota_ind + NxNx)]

iotaminus = StateSS[end - (NxNx_aux - iota_ind + NxNx)] +
            Stateminus[end - (NxNx_aux - iota_ind + NxNx)]


BB = StateSS[end - (NxNx_aux - BB_ind + NxNx)] +
     State[end - (NxNx_aux - BB_ind + NxNx)]

BBminus = StateSS[end - (NxNx_aux - BB_ind + NxNx)] +
          Stateminus[end - (NxNx_aux - BB_ind + NxNx)]


PSI_W = StateSS[end - (NxNx_aux - PSI_W_ind + NxNx)] +
        State[end - (NxNx_aux - PSI_W_ind + NxNx)]

PSI_Wminus = StateSS[end - (NxNx_aux - PSI_W_ind + NxNx)] +
             Stateminus[end - (NxNx_aux - PSI_W_ind + NxNx)]


MP = StateSS[end - (NxNx_aux - MP_ind + NxNx)] +
     State[end - (NxNx_aux - MP_ind + NxNx)]

MPminus = StateSS[end - (NxNx_aux - MP_ind + NxNx)] +
          Stateminus[end - (NxNx_aux - MP_ind + NxNx)]


p_mark = StateSS[end - (NxNx_aux - p_mark_ind + NxNx)] +
         State[end - (NxNx_aux - p_mark_ind + NxNx)]

p_markminus = StateSS[end - (NxNx_aux - p_mark_ind + NxNx)] +
              Stateminus[end - (NxNx_aux - p_mark_ind + NxNx)]



#  Shock
eps_B_Fnext = StateSS[end - (NxNx_aux - eps_B_F_ind + NxNx)] +
              State[end - (NxNx_aux - eps_B_F_ind + NxNx)]

eps_B_F = StateSS[end - (NxNx_aux - eps_B_F_ind + NxNx)] +
          Stateminus[end - (NxNx_aux - eps_B_F_ind + NxNx)]


eps_RPnext = StateSS[end - (NxNx_aux - eps_RP_ind + NxNx)] +
             State[end - (NxNx_aux - eps_RP_ind + NxNx)]

eps_RP = StateSS[end - (NxNx_aux - eps_RP_ind + NxNx)] +
         Stateminus[end - (NxNx_aux - eps_RP_ind + NxNx)]


eps_etanext = StateSS[end - (NxNx_aux - eps_eta_ind + NxNx)] +
              State[end - (NxNx_aux - eps_eta_ind + NxNx)]

eps_eta = StateSS[end - (NxNx_aux - eps_eta_ind + NxNx)] +
          Stateminus[end - (NxNx_aux - eps_eta_ind + NxNx)]


eps_Dnext = StateSS[end - (NxNx_aux - eps_D_ind + NxNx)] +
            State[end - (NxNx_aux - eps_D_ind + NxNx)]

eps_D = StateSS[end - (NxNx_aux - eps_D_ind + NxNx)] +
        Stateminus[end - (NxNx_aux - eps_D_ind + NxNx)]


eps_Rnext = StateSS[end - (NxNx_aux - eps_R_ind + NxNx)] +
            State[end - (NxNx_aux - eps_R_ind + NxNx)]

eps_R = StateSS[end - (NxNx_aux - eps_R_ind + NxNx)] +
        Stateminus[end - (NxNx_aux - eps_R_ind + NxNx)]


eps_Znext = StateSS[end - (NxNx_aux - eps_Z_ind + NxNx)] +
            State[end - (NxNx_aux - eps_Z_ind + NxNx)]

eps_Z = StateSS[end - (NxNx_aux - eps_Z_ind + NxNx)] +
        Stateminus[end - (NxNx_aux - eps_Z_ind + NxNx)]


eps_Gnext = StateSS[end - (NxNx_aux - eps_G_ind + NxNx)] +
            State[end - (NxNx_aux - eps_G_ind + NxNx)]

eps_G = StateSS[end - (NxNx_aux - eps_G_ind + NxNx)] +
        Stateminus[end - (NxNx_aux - eps_G_ind + NxNx)]


eps_iotanext = StateSS[end - (NxNx_aux - eps_iota_ind + NxNx)] +
               State[end - (NxNx_aux - eps_iota_ind + NxNx)]

eps_iota = StateSS[end - (NxNx_aux - eps_iota_ind + NxNx)] +
           Stateminus[end - (NxNx_aux - eps_iota_ind + NxNx)]


eps_QEnext = StateSS[end - (NxNx_aux - eps_QE_ind + NxNx)] +
             State[end - (NxNx_aux - eps_QE_ind + NxNx)]

eps_QE = StateSS[end - (NxNx_aux - eps_QE_ind + NxNx)] +
         Stateminus[end - (NxNx_aux - eps_QE_ind + NxNx)]


eps_wnext = StateSS[end - (NxNx_aux - eps_w_ind + NxNx)] +
            State[end - (NxNx_aux - eps_w_ind + NxNx)]

eps_w = StateSS[end - (NxNx_aux - eps_w_ind + NxNx)] +
        Stateminus[end - (NxNx_aux - eps_w_ind + NxNx)]


eps_BBnext = StateSS[end - (NxNx_aux - eps_BB_ind + NxNx)] +
             State[end - (NxNx_aux - eps_BB_ind + NxNx)]

eps_BB = StateSS[end - (NxNx_aux - eps_BB_ind + NxNx)] +
         Stateminus[end - (NxNx_aux - eps_BB_ind + NxNx)]



## Split the Control vector into items with names

#  Controls

VALUEnext     = util(Controlnext[VALUE_ind]) 
VALUE         = util(Control[VALUE_ind]) 
Vanext        = mutil(Controlnext[Va_ind]) 
Va            = mutil(Control[Va_ind]) 
mutil_cnext   = mutil(Controlnext[mutil_c_ind]) 
mutil_c       = mutil(Control[mutil_c_ind]) 

#  Aggregate Controls (t+1) 

A_hhnext      = exp(Controlnext[A_hh_ind]) 
B_hhnext      = exp(Controlnext[B_hh_ind]) 
Cnext         = exp(Controlnext[C_ind]) 
Knext         = exp(Controlnext[K_ind]) 
Bnext         = exp(Controlnext[B_ind]) 
Nnext         = exp(Controlnext[N_ind]) 

LAMBDAnext    = exp(Controlnext[Lambda_ind]) 
PInext        = exp(Controlnext[pi_ind]) 
Vnext         = exp(Controlnext[V_ind]) 
Jnext         = exp.(Controlnext[J_ind]) 
Hnext         = exp(Controlnext[h_ind]) 
vnext         = exp(Controlnext[v_ind]) 
Ynext         = exp(Controlnext[Y_ind]) 
R_Knext       = exp(Controlnext[r_k_ind]) 
R_Anext       = exp(Controlnext[r_a_ind]) 
MCnext        = exp(Controlnext[MC_ind]) 
RRanext       = exp(Controlnext[RRa_ind]) 
RRnext        = exp(Controlnext[RR_ind]) 
Inext         = exp(Controlnext[I_ind]) 
unempnext     = exp(Controlnext[unemp_ind]) 
# salenext      = exp(Controlnext[sale_ind]) 
# s2snext       = exp(Controlnext[s2s_ind]) 
# stocknext     = exp(Controlnext[stock_ind]) 
x_knext       = exp(Controlnext[x_k_ind]) 
nnnext        = exp(Controlnext[nn_ind]) 
Mnext         = exp(Controlnext[M_ind]) 
fnext         = exp(Controlnext[f_ind]) 
B_gov_ncpnext = exp(Controlnext[B_gov_ncp_ind]) 
UBnext        = exp(Controlnext[UB_ind]) 
l_lambdanext  = exp(Controlnext[l_lambda_ind]) 
x_Inext       = exp(Controlnext[x_I_ind]) 
eta2next      = exp(Controlnext[eta2_ind]) 
iota2next     = exp(Controlnext[iota2_ind]) 
B_gov_ncp2next= exp(Controlnext[B_gov_ncp2_ind]) 


#  Aggregate Controls (t) 

A_hh       = exp(Control[A_hh_ind]) 
B_hh       = exp(Control[B_hh_ind]) 
C          = exp(Control[C_ind]) 
K          = exp(Control[K_ind]) 
B          = exp(Control[B_ind]) 
N          = exp(Control[N_ind]) 
L          = exp(Control[L_ind]) 
T          = exp(Control[T_ind]) 
LT         = exp(Control[LT_ind]) 
G          = exp(Control[G_ind]) 
MRS        = exp(Control[MRS_ind]) 
LAMBDA     = exp(Control[Lambda_ind]) 

PI         = exp(Control[pi_ind]) 
V          = exp(Control[V_ind]) 
J          = exp.(Control[J_ind]) 
H          = exp(Control[h_ind]) 
v          = exp(Control[v_ind]) 
Y          = exp(Control[Y_ind]) 
PROFIT     = exp(Control[Profit_ind]) 
R_K        = exp(Control[r_k_ind]) 
R_A        = exp(Control[r_a_ind]) 
MC         = exp(Control[MC_ind]) 
RRa        = exp(Control[RRa_ind]) 
RR         = exp(Control[RR_ind]) 
Inv        = exp(Control[I_ind]) 
unemp      = exp(Control[unemp_ind]) 
x_k        = exp(Control[x_k_ind]) 
nn         = exp(Control[nn_ind]) 
M          = exp(Control[M_ind]) 
f          = exp(Control[f_ind]) 
B_gov_ncp  = exp(Control[B_gov_ncp_ind]) 
UB         = exp(Control[UB_ind]) 

Y_obs      = exp(Control[Y_obs_ind]) 
C_obs      = exp(Control[C_obs_ind]) 
I_obs      = exp(Control[I_obs_ind]) 
w_obs      = exp(Control[w_obs_ind]) 
PROFIT_obs = exp(Control[PROFIT_obs_ind]) 
unemp_obs  = exp(Control[unemp_obs_ind]) 
inf_obs    = exp(Control[inf_obs_ind]) 
R_obs      = exp(Control[R_obs_ind]) 
pastw      = exp(Control[pastw_ind]) 
G_obs      = exp(Control[G_obs_ind]) 

pastYY      = exp(Control[pastYY_ind]) 
pastCC      = exp(Control[pastCC_ind]) 
pastII      = exp(Control[pastII_ind]) 
pastPPROFIT = exp(Control[pastPPROFIT_ind]) 
pastuu      = exp(Control[pastuu_ind]) 
pastGG      = exp(Control[pastGG_ind]) 
pastA_g     = exp(Control[pastA_g_ind]) 
pastLT2     = exp(Control[pastLT2_ind]) 
pastG2      = exp(Control[pastG2_ind]) 
LT_obs      = exp(Control[LT_obs_ind]) 
B_gov_ncp2  = exp(Control[B_gov_ncp2_ind]) 

l_lambda    = exp(Control[l_lambda_ind]) 
pastQ       = exp(Control[pastQ_ind]) 

x_I       = exp(Control[x_I_ind]) 
eta2      = exp(Control[eta2_ind]) 
iota2     = exp(Control[iota2_ind]) 

# Observables

A_g_obs    = exp(Control[A_g_obs_ind]) 
tau_w      = param["tau_w"] 
tau_a      = param["tau_a"] 

zznext         = exp(Controlnext[zz_ind]) 
xxnext         = exp(Controlnext[xx_ind]) 
Profit_FInext  = exp(Controlnext[Profit_FI_ind]) 
vvnext         = exp(Controlnext[vv_ind]) 
eenext         = exp(Controlnext[ee_ind]) 
C_bnext        = exp(Controlnext[C_b_ind]) 

zz             = exp(Control[zz_ind]) 
xx             = exp(Control[xx_ind]) 
Profit_FI      = exp(Control[Profit_FI_ind]) 
vv             = exp(Control[vv_ind]) 
ee             = exp(Control[ee_ind]) 
C_b            = exp(Control[C_b_ind]) 

## Write LHS values (t variables)

#  Controls

LHS[nx .+ VALUE_ind] = Control[VALUE_ind]
LHS[nx .+ mutil_c_ind] = Control[mutil_c_ind]
LHS[nx .+ Va_ind] = Control[Va_ind] 

LHS[nx+A_hh_ind] = A_hh 
LHS[nx+B_hh_ind] = B_hh 
LHS[nx+C_ind] = C 
LHS[nx+K_ind] = K 
LHS[nx+B_ind] = B 
LHS[nx+N_ind] = N 
LHS[nx+L_ind] = L 
LHS[nx+T_ind] = T 
LHS[nx+LT_ind] = LT 
LHS[nx+G_ind] = G 
LHS[nx+MRS_ind] = MRS 
LHS[nx+Lambda_ind] = LAMBDA 

LHS[nx+pi_ind] = PI 
LHS[nx+V_ind] = V 
LHS[nx .+ J_ind] = J 
LHS[nx+h_ind] = H 
LHS[nx+v_ind] = v 
LHS[nx+Y_ind] = Y 
LHS[nx+Profit_ind] = PROFIT 
LHS[nx+r_k_ind] = R_K 
LHS[nx+r_a_ind] = R_A 
LHS[nx+MC_ind] = MC 
LHS[nx+RRa_ind] = RRa 
LHS[nx+RR_ind] = RR 
LHS[nx+I_ind] = Inv 
LHS[nx+unemp_ind] = unemp 
# LHS[nx+sale_ind] = sale 
# LHS[nx+s2s_ind] = s2s 
# LHS[nx+stock_ind] = stock 
LHS[nx+x_k_ind] = x_k 

LHS[nx+nn_ind] = nn 
LHS[nx+M_ind] = M 
LHS[nx+f_ind] = f 
LHS[nx+B_gov_ncp_ind] = B_gov_ncp 
LHS[nx+UB_ind] = UB 

LHS[nx+zz_ind] = zz 
LHS[nx+xx_ind] = xx 
LHS[nx+Profit_FI_ind] = Profit_FI 
LHS[nx+vv_ind] = vv 
LHS[nx+ee_ind] = ee 
LHS[nx+C_b_ind] = C_b 
LHS[nx+A_g_obs_ind] = A_g_obs 

LHS[nx+Y_obs_ind] = Y_obs 
LHS[nx+C_obs_ind] = C_obs 
LHS[nx+I_obs_ind] = I_obs 
LHS[nx+w_obs_ind] = w_obs 
LHS[nx+PROFIT_obs_ind] = PROFIT_obs 
LHS[nx+unemp_obs_ind] = unemp_obs 
LHS[nx+inf_obs_ind] = inf_obs 
LHS[nx+R_obs_ind] = R_obs 
LHS[nx+pastw_ind] = pastw 
LHS[nx+G_obs_ind] = G_obs 

LHS[nx+pastYY_ind] = pastYY 
LHS[nx+pastCC_ind] = pastCC 
LHS[nx+pastII_ind] = pastII 
LHS[nx+pastPPROFIT_ind] = pastPPROFIT 
LHS[nx+pastuu_ind] = pastuu 
LHS[nx+pastGG_ind] = pastGG 
LHS[nx+pastA_g_ind] = pastA_g 
LHS[nx+pastLT2_ind] = pastLT2 
LHS[nx+pastG2_ind] = pastG2 
LHS[nx+LT_obs_ind] = LT_obs 
LHS[nx+B_gov_ncp2_ind] = B_gov_ncp2 

LHS[nx+l_lambda_ind] = l_lambda 
LHS[nx+pastQ_ind] = pastQ 
LHS[nx+x_I_ind] = x_I 
LHS[nx+eta2_ind] = eta2 
LHS[nx+iota2_ind] = iota2 

# LHS[nx+B_gov_obs_ind] = B_gov_obs 
# LHS[nx+Y_obs_ind] = Y_obs 
# LHS[nx+C_obs_ind] = C_obs 
# LHS[nx+I_obs_ind] = I_obs 
# LHS[nx+w_obs_ind] = w_obs 
# LHS[nx+PI_obs_ind] = PI_obs 
# LHS[nx+GG_obs_ind] = GG_obs 

# States
# Marginal Distributions

LHS[marginal_b_ind] = Distribution[1:nb-1]
LHS[marginal_a_ind] = Distribution[nb .+ (1:na-1)]
LHS[marginal_se_ind] = Distribution[(nb + na) .+ (1:nse-1)] 

LHS[COP_ind] = cop_coefs

LHS[R_cb_ind] = (R_cb) 
LHS[w_ind] = (W) 
LHS[A_b_ind] = (A_bnext) 
LHS[B_b_ind] = (B_bnext) 
LHS[A_g_ind] = (A_gauxnext) 
LHS[Q_ind] = (Q) 
LHS[lev_ind] = (lev) 
LHS[NW_b_ind] = (NW_b) 
# LHS[inven_ind] = (inven) 
LHS[R_tilde_ind] = (R_tilde) 
LHS[x_cb_ind] = (x_cb) 
# LHS[pasts2s_ind] = (pasts2s) 
LHS[pastpi_ind] = (pastpi) 

LHS[pastY_ind] = pastY 
LHS[pastC_ind] = pastC 
LHS[pastI_ind] = pastI 
LHS[pastPROFIT_ind] = pastPROFIT 
LHS[pastunemp_ind] = pastunemp 
LHS[pastG_ind] = pastG 
LHS[pastLT_ind] = pastLT 
LHS[R_star_ind] = R_star 
LHS[B_F_ind] = B_Fnext 

LHS[Z_ind] = (Z) 
LHS[PSI_RP_ind] = (PSI_RP) 
LHS[eta_ind] = (eta) 
LHS[D_ind] = (D) 
LHS[GG_ind] = (GG) 
LHS[iota_ind] = (iota) 
LHS[BB_ind] = (BB) 
LHS[PSI_W_ind] = (PSI_W) 
LHS[MP_ind] = (MP) 
LHS[p_mark_ind] = (p_mark) 

# LHS[gamma_Q_ind] = (gamma_Q) 

LHS[eps_QE_ind] = (eps_QEnext) 
LHS[eps_RP_ind] = (eps_RPnext) 
LHS[eps_B_F_ind] = (eps_B_Fnext) 
LHS[eps_BB_ind] = (eps_BBnext) 
LHS[eps_Z_ind] = (eps_Znext) 
LHS[eps_G_ind] = (eps_Gnext) 
LHS[eps_D_ind] = (eps_Dnext) 
LHS[eps_R_ind] = (eps_Rnext) 
LHS[eps_iota_ind] = (eps_iotanext) 
LHS[eps_eta_ind] = (eps_etanext) 
LHS[eps_w_ind] = (eps_wnext) 


# LHS[pastx_a_ind] = (pastx_a) 
#
# LHS[pastY_ind] = (pastY) 
# LHS[pastC_ind] = (pastC) 
# LHS[pastI_ind] = (pastI) 
# LHS[pastPROFIT_ind] = (pastPROFIT) 
# LHS[pastBgov_ind] = (pastBgov) 

# LHS[varphi_ind] = (varphi) 
# LHS[eps_varphi_ind] = (eps_varphinext) 

# LHS[pastTAX_ind] = (pastTAX) 
# LHS[TAU_ind] = (TAU) 
# LHS[eps_TAU_ind] = (eps_TAUnext) 

R_cb        = exp(R_cb)        ; R_cbminus        = exp(R_cbminus) 
W           = exp(W)           ; Wminus           = exp(Wminus) 
A_bnext     = exp(A_bnext)     ; A_b              = exp(A_b) 
B_bnext     = exp(B_bnext)     ; B_b              = exp(B_b) 
A_gaux      = exp(A_gaux)      ; A_gauxnext       = exp(A_gauxnext) 
Q           = exp(Q)           ; Qminus           = exp(Qminus) 
lev         = exp(lev)         ; levminus         = exp(levminus) 
NW_b        = exp(NW_b)        ; NW_bminus        = exp(NW_bminus) 
# inven       = exp(inven)       ; invenminus       = exp(invenminus) 
R_tilde     = exp(R_tilde)     ; R_tildeminus     = exp(R_tildeminus) 
x_cb        = exp(x_cb)        ; x_cbminus        = exp(x_cbminus) 
# pasts2s     = exp(pasts2s)     ; pasts2sminus     = exp(pasts2sminus) 
pastpi      = exp(pastpi)      ; pastpiminus      = exp(pastpiminus) 
pastY       = exp(pastY)       ; pastYminus       = exp(pastYminus) 
pastC       = exp(pastC)       ; pastCminus       = exp(pastCminus) 
pastI       = exp(pastI)       ; pastIminus       = exp(pastIminus) 
pastPROFIT  = exp(pastPROFIT)  ; pastPROFITminus  = exp(pastPROFITminus) 
pastunemp   = exp(pastunemp)   ; pastunempminus   = exp(pastunempminus) 
pastG       = exp(pastG)       ; pastGminus       = exp(pastGminus) 
pastLT      = exp(pastLT)      ; pastLTminus      = exp(pastLTminus) 
B_Fnext     = exp(B_Fnext)     ; B_F              = exp(B_F) 
R_star      = exp(R_star)      ; R_starminus      = exp(R_starminus) 
Z           = exp(Z)           ; Zminus           = exp(Zminus) 
PSI_RP      = exp(PSI_RP)      ; PSI_RPminus      = exp(PSI_RPminus) 
eta         = exp(eta)         ; etaminus         = exp(etaminus) 
D           = exp(D)           ; Dminus           = exp(Dminus) 
GG          = exp(GG)          ; GGminus          = exp(GGminus) 
iota        = exp(iota)        ; iotaminus        = exp(iotaminus) 
BB          = exp(BB)          ; BBminus          = exp(BBminus) 
PSI_W       = exp(PSI_W)       ; PSI_Wminus       = exp(PSI_Wminus) 
MP          = exp(MP)          ; MPminus          = exp(MPminus) 
p_mark      = exp(p_mark)      ; p_markminus      = exp(p_markminus) 


## === 1) Uncompress -> Δc on U-grid, then CDF (exactly like Lütticke) ===
nb_cop  = Int(grid["nb_copula"])
na_cop  = Int(grid["na_copula"])
nse_cop = Int(grid["nse_copula"])

thetaD    = uncompress(Float64.(grid["compressionIndexesCOP"]), cop_coefs, DCD, IDCD)
COP_Dev   = reshape(thetaD, (nb_cop, na_cop, nse_cop))
COP_Dev   = cumsum(cumsum(cumsum(COP_Dev; dims=1); dims=2); dims=3)

thetaD_m  = uncompress(Float64.(grid["compressionIndexesCOP"]), cop_coefs_m, DCD, IDCD)
COP_Dev_m = reshape(thetaD_m, (nb_cop, na_cop, nse_cop))
COP_Dev_m = cumsum(cumsum(cumsum(COP_Dev_m; dims=1); dims=2); dims=3)

## === 2) Today / SS marginals ⇒ raw cumsums for CDF axes (no pinning) ===
marginal_b      = Distribution[1:nb]
marginal_a      = Distribution[(1:na) .+ nb]
marginal_se     = Distribution[(1:nse) .+ nb .+ na]

CDF_b    = cumsum(marginal_b)
CDF_a    = cumsum(marginal_a)
CDF_se   = cumsum(marginal_se)

marginal_bminus  = Distributionminus[1:nb]
marginal_aminus  = Distributionminus[(1:na) .+ nb]
marginal_seminus = Distributionminus[(1:nse) .+ nb .+ na]

CDF_b_m  = cumsum(marginal_bminus)
CDF_a_m  = cumsum(marginal_aminus)
CDF_se_m = cumsum(marginal_seminus)

DistributionSS  = StateSS[1:nFullMarg]
marginal_b_SS   = DistributionSS[1:nb]
marginal_a_SS   = DistributionSS[(1:na) .+ nb]
marginal_se_SS  = DistributionSS[(1:nse) .+ nb .+ na]

CDF_b_SS  = cumsum(marginal_b_SS)
CDF_a_SS  = cumsum(marginal_a_SS)
CDF_se_SS = cumsum(marginal_se_SS) 

## === 3) SS copula: pdf (value grid) -> CDF on value-grid CDF axes ===
COP_SS = SS_stats["COP_SS"] 

## === 4) Copula U-grid axes (use your existing interior nodes) ===
s_m_b  = grid["copula_marginal_b"][:] 
s_m_a  = grid["copula_marginal_a"][:] 
s_m_se = grid["copula_marginal_se"][:] 

## === 5) Compose joint CDF at today’s U = (CDF_b_m, CDF_a_m, CDF_se_m) ===

Ubq = Vector{Float64}(CDF_b_m[:])
Uaq = Vector{Float64}(CDF_a_m[:])
Usq = Vector{Float64}(CDF_se_m[:])

C_SS_atU = myinterpolate3(Vector{Float64}(CDF_b_SS[:]), Vector{Float64}(CDF_a_SS[:]), Vector{Float64}(CDF_se_SS[:]), COP_SS,    Ubq, Uaq, Usq)
dC_atU   = myinterpolate3(Vector{Float64}(s_m_b[:]),    Vector{Float64}(s_m_a[:]),    Vector{Float64}(s_m_se[:]),    COP_Dev_m, Ubq, Uaq, Usq)

CDF_joint_m = C_SS_atU + dC_atU 

## === 6) CDF -> joint PDF (cell masses) via forward differences ===
PDF_joint_m = cdf_to_pdf3D_forwarddiff(CDF_joint_m) 

# ## === 7) Diagnostics ===
A_gnext = A_gauxnext-1 
A_g     = A_gaux-1 

MU = PDF_joint_m 

MU_b  = dropdims(sum(MU, dims=(2,3)), dims=(2,3))
MU_a  = dropdims(sum(MU, dims=(1,3)), dims=(1,3))'
MU_se = dropdims(sum(MU, dims=(1,2)), dims=(1,2)) 

# Eshare = MU_se(end) 

Eshare = param["Eshare"]

meshes = Dict{String, Any}()
meshes["b"], meshes["a"], meshes["se"] = ndgrid(grid["b"], grid["a"], grid["se"])

A = [
    1 - l_lambda + l_lambda*f    l_lambda*(1 - f)
    f                            1 - f
]
id = Matrix{Float64}(I, ns, ns)
P_SS3_aux   = kron(A, id)
P_SS3_aux   = [P_SS3_aux zeros(nse-1, 1)]
lastrow     = [zeros(1, 2*ns) 1]
P_SS3_aux   = [P_SS3_aux; lastrow]

H_tilde = kron(Matrix{Float64}(P_SS3_aux'), sparse(Matrix{Float64}(I, nb*na, nb*na)))


MU_tilde = reshape(reshape(MU, nb*na, nse) * P_SS3_aux, nb, na, nse) 

MU_tilde_b  = dropdims(sum(MU_tilde; dims=(2,3)); dims=(2,3)) 
MU_tilde_a  = dropdims(sum(MU_tilde; dims=(1,3)); dims=(1,3))' 
MU_tilde_se = dropdims(sum(MU_tilde; dims=(1,2)); dims=(1,2)) 

marginal_setilde_aux = MU_tilde_se[1:2*ns]

A_next = [
    1 - l_lambdanext + l_lambdanext*fnext    l_lambdanext*(1 - fnext)
    fnext                                    1 - fnext
]
P_SS3next_aux  = kron(A_next, Matrix{Float64}(I, ns, ns))
P_SS3next_aux  = [P_SS3next_aux zeros(nse-1, 1)]
lastrow        = [zeros(1, 2*ns) 1]
P_SS3next_aux  = [P_SS3next_aux; lastrow] 

P_transition_exp  = param["P_SS2"]*P_SS3next_aux 
P_transition_dist = param["P_SS2"]   # The new transition matrix

#  Income (grids)

auxWW                  = ones(nb, na, nse)
auxWW[:,:,ns+1:end]    = zeros(nb, na, nse - ns)
auxWW1                 = auxWW
auxWW                  = auxWW .* (meshes["se"] .^ (param["b_aux"] - 1))  # bonus

auxWW2                 = ones(nb, na, nse)
auxWW2[:,:,1:ns]       = zeros(nb, na, ns)
auxWW2[:,:,end]        = zeros(nb, na, 1) 

# param["beta_aux"] = param["beta"]*gamma_Q 

# param["beta_aux"] = param["beta"] 

# param["beta_aux"] = param["beta"]*D 

param["beta_aux"] = param["beta"] 

R_A_aux = R_A + (param["death_rate"] / (1 - param["death_rate"])) * Q  # annuity payments

# After-tax wage or unemployment benefit
NW = param["xi"] / (1 + param["xi"]) * nn * W
WW = (1 - tau_w) * (NW * auxWW1 + param["xi"] / (1 + param["xi"]) * param["w_bar"] .* auxWW2)
WW[:,:,end] = (1 - tau_a) * (param["Eratio"] * PROFIT) / Eshare * ones(nb, na)

inc = Dict{String, Any}()
inc["labor"]    = WW .* meshes["se"]
inc["dividend"] = meshes["a"] * R_A_aux
inc["capital"]  = meshes["a"] * Q
inc["bond"]     = (R_cbminus / PI) .* meshes["b"] +
                  (meshes["b"] .< 0) .* (param["Rprem"] / PI) .* meshes["b"]
inc["bond"]     = inc["bond"] / (1 - param["death_rate"]) * param["b_a_aux"]
inc["transfer"] = (LT + C_b) / (1 + param["frac_b"]) * ones(nb, na, nse) 

## First Set: Value functions, marginal values
## Update policies

EVa = reshape(reshape(Vanext, (nb*na, nse)) * P_transition_exp', (nb, na, nse))
R_tildeaux = R_tilde / PInext .+ (meshes["b"] .< 0) .* (param["Rprem"] / PInext)

EVb = reshape(reshape(R_tildeaux[:] / (1 - param["death_rate"]) * param["b_a_aux"] .* mutil_cnext, (nb*na, nse)) * P_transition_exp', (nb, na, nse)) 

# [c_a_star,b_a_star,a_a_star,c_n_star,b_n_star] = policies_update(EVb,EVa,Q,PI,R_tildeminus,1,1,inc,meshes,grid,param) 
c_a_star,b_a_star,a_a_star,c_n_star,b_n_star = policies_update(EVb,EVa,Q,PI,R_cbminus,1,1,inc,meshes,grid,param) 

## Update value functions

meshaux = copy(meshes)

_, _, meshaux["se_aux"] = ndgrid(grid["b"], grid["a"], 1:nse)

EV = reshape(VALUEnext, (nb*na, nse)) * P_transition_exp'

#itp = interpolate((meshaux["b"], meshaux["a"], meshaux["se_aux"]), reshape(EV, (nb, na, nse)), Gridded(Linear()))


b_grid  = vec(meshaux["b"][:, 1, 1])
a_grid  = vec(meshaux["a"][1, :, 1])
se_grid = vec(meshaux["se_aux"][1, 1, :])

EV3 = reshape(EV, nb, na, nse)

itp = interpolate((b_grid, a_grid, se_grid), EV3, Gridded(Linear()))


VALUE_next = extrapolate(itp, Line())
V_adjust   = util(c_a_star) .+ param["beta_aux"] * (1 - param["death_rate"]) .* VALUE_next.(b_a_star, a_a_star, meshaux["se_aux"])
V_noadjust = util(c_n_star) .+ param["beta_aux"] * (1 - param["death_rate"]) .* VALUE_next.(b_n_star, meshaux["a"], meshaux["se_aux"])

AProb = clamp.(1.0 ./ (1.0 .+ exp.(.-(V_adjust .- V_noadjust .- param["mu_chi"]) ./ param["sigma_chi"])), 1e-6, 1 - 1e-6)
AC = param["sigma_chi"] .* ((1.0 .- vec(AProb)) .* log.(1.0 .- vec(AProb)) + vec(AProb) 
                            .* log.(vec(AProb))) .+ param["mu_chi"] .* vec(AProb) .- (param["mu_chi"] 
                            - param["sigma_chi"] * log(1 + exp(param["mu_chi"] / param["sigma_chi"])))
AProb = AProb .* param["nu"]
AC = AC .* param["nu"]

VALUEaux = vec(AProb) .* vec(V_adjust) .+ (1.0 .- vec(AProb)) .* vec(V_noadjust) .- vec(AC) 

## Update Marginal Value of Bonds
# Ensure (nb, na, nse) shape so broadcast matches AProb; vec/reshape avoids dimension mismatch
mutil_c_n = reshape(mutil(vec(c_n_star)), (nb, na, nse))
mutil_c_a = reshape(mutil(vec(c_a_star)), (nb, na, nse))
mutil_c_aux = AProb .* mutil_c_a .+ (1.0 .- AProb) .* mutil_c_n  # Expected marginal utility

mutil_cnext_aux = reshape(reshape(mutil_cnext, nb * na, nse) * P_transition_exp', nb, na, nse)

b_grid  = vec(meshaux["b"][:, 1, 1])
a_grid  = vec(meshaux["a"][1, :, 1])
se_grid = vec(meshaux["se_aux"][1, 1, :])

# Ensure the values array matches (nb, na, nse)
@assert size(mutil_cnext_aux) == (length(b_grid), length(a_grid), length(se_grid))

itp = interpolate((b_grid, a_grid, se_grid), mutil_cnext_aux, Gridded(Linear()))

# MATLAB-like linear extrapolation outside the grid:
mutil_cnext_itp = extrapolate(itp, Line())


MRS_a_aux = param["beta_aux"] * (1 - param["death_rate"]) .* mutil_cnext_itp.(b_a_star, a_a_star, meshaux["se_aux"]) ./ mutil_c_a .* AProb .* MU_tilde
MRS_n_aux = param["beta_aux"] * (1 - param["death_rate"]) .* mutil_cnext_itp.(b_n_star, meshaux["a"], meshaux["se_aux"]) ./ mutil_c_n .* (1.0 .- AProb) .* MU_tilde


b_grid  = vec(meshaux["b"][:, 1, 1])
a_grid  = vec(meshaux["a"][1, :, 1])
se_grid = vec(meshaux["se_aux"][1, 1, :])
Va3 = reshape(EVa, nb, na, nse)
@assert size(Va3) == (length(b_grid), length(a_grid), length(se_grid))

itp     = interpolate((b_grid, a_grid, se_grid), Va3, Gridded(Linear()))
Va_next = extrapolate(itp, Line())

Va_aux = AProb .* (R_A_aux .+ Q) .* mutil_c_a .+ (1.0 .- AProb) .* R_A_aux .* mutil_c_n .+
         param["beta_aux"] * (1 - param["death_rate"]) .* (1.0 .- AProb) .* Va_next.(b_n_star, meshaux["a"], meshaux["se_aux"])


_, _, meshes["se3_aux"] = ndgrid(grid["b"], grid["a"], grid["se3_aux"])

MU_Ent = (meshes["se3_aux"] .== 1) .* MU
MU_tilde_Ent = reshape(reshape(MU_Ent, nb*na, nse) * P_SS3_aux, nb, na, nse) 

# RHS[nx+C_ind] = q_cons2(c_a_star,c_n_star,Wminus,nnminus,AProb,MU_tilde,meshes,param,grid,tau_wminus) 

# Differences for distributions
weight11 = zeros(nb * na, nse, nse)
weight12 = zeros(nb * na, nse, nse)
weight21 = zeros(nb * na, nse, nse)
weight22 = zeros(nb * na, nse, nse)

# Find next smallest on-grid value for money and capital choices
Dist_b_a, idb_a = genweight(b_a_star, grid["b"])
Dist_b_n, idb_n = genweight(b_n_star, grid["b"])
Dist_a_a, ida_a = genweight(a_a_star, grid["a"])
Dist_a_n, ida_n = genweight(meshaux["a"], grid["a"])

Dist_b_die, idb_die = genweight(zeros(nb, na, nse), grid["b"])
Dist_a_die, ida_die = genweight(zeros(nb, na, nse), grid["a"])

# Transition matrix for adjustment case
idb_a = repeat(vec(idb_a), 1, nse)
ida_a = repeat(vec(ida_a), 1, nse)
idse = vec(kron(1:nse, ones(Int, 1, nb * na * nse))')'

index11 = sub2ind((nb, na, nse), vec(idb_a), vec(ida_a), vec(idse))
index12 = sub2ind((nb, na, nse), vec(idb_a), vec(ida_a) .+ 1, vec(idse))
index21 = sub2ind((nb, na, nse), vec(idb_a) .+ 1, vec(ida_a), vec(idse))
index22 = sub2ind((nb, na, nse), vec(idb_a) .+ 1, vec(ida_a) .+ 1, vec(idse))

for hh in 1:nse
    weight11_aux = (1.0 .- Dist_b_a[:, :, hh]) .* (1.0 .- Dist_a_a[:, :, hh])
    weight12_aux = (1.0 .- Dist_b_a[:, :, hh]) .* Dist_a_a[:, :, hh]
    weight21_aux = Dist_b_a[:, :, hh] .* (1.0 .- Dist_a_a[:, :, hh])
    weight22_aux = Dist_b_a[:, :, hh] .* Dist_a_a[:, :, hh]

    weight11[:, :, hh] = vec(weight11_aux) * P_transition_dist[hh, :]'
    weight12[:, :, hh] = vec(weight12_aux) * P_transition_dist[hh, :]'
    weight21[:, :, hh] = vec(weight21_aux) * P_transition_dist[hh, :]'
    weight22[:, :, hh] = vec(weight22_aux) * P_transition_dist[hh, :]'
end

weight11 = permutedims(weight11, (1, 3, 2))
weight12 = permutedims(weight12, (1, 3, 2))
weight21 = permutedims(weight21, (1, 3, 2))
weight22 = permutedims(weight22, (1, 3, 2))
#rowindex = repeat(1:nb*na*nse, 1, 4 * nse)
rowindex = repeat(collect(1:nb*na*nse), 4*nse)


TT_a = sparse(vec(rowindex), [vec(index11); vec(index21); vec(index12); vec(index22)],
              [vec(weight11); vec(weight21); vec(weight12); vec(weight22)], nb * na * nse, nb * na * nse) 

weight11 = zeros(nb*na, nse, nse)
weight12 = zeros(nb*na, nse, nse)
weight21 = zeros(nb*na, nse, nse)
weight22 = zeros(nb*na, nse, nse)

# Transition matrix for non-adjustment case
idb_n = repeat(vec(idb_n), 1, nse)
ida_n = repeat(vec(ida_n), 1, nse)
idse  = vec(kron(1:nse, ones(1, nb*na*nse))')'

index11 = sub2ind((nb, na, nse), vec(idb_n), vec(ida_n), vec(idse))
index12 = sub2ind((nb, na, nse), vec(idb_n), vec(ida_n) .+ 1, vec(idse))
index21 = sub2ind((nb, na, nse), vec(idb_n) .+ 1, vec(ida_n), vec(idse))
index22 = sub2ind((nb, na, nse), vec(idb_n) .+ 1, vec(ida_n) .+ 1, vec(idse))

for hh in 1:nse
    # Corresponding weights
    weight11_aux = (1 .- Dist_b_n[:,:,hh]) .* (1 .- Dist_a_n[:,:,hh])
    weight12_aux = (1 .- Dist_b_n[:,:,hh]) .* Dist_a_n[:,:,hh]
    weight21_aux = Dist_b_n[:,:,hh] .* (1 .- Dist_a_n[:,:,hh])
    weight22_aux = Dist_b_n[:,:,hh] .* Dist_a_n[:,:,hh]

    # Dimensions (b x a, h', h)
    weight11[:,:,hh] = vec(weight11_aux) * P_transition_dist[hh,:]'
    weight12[:,:,hh] = vec(weight12_aux) * P_transition_dist[hh,:]'
    weight21[:,:,hh] = vec(weight21_aux) * P_transition_dist[hh,:]'
    weight22[:,:,hh] = vec(weight22_aux) * P_transition_dist[hh,:]'
end

# Dimensions (b x a, h, h')
weight11 = permutedims(weight11, (1, 3, 2))
weight12 = permutedims(weight12, (1, 3, 2))
weight21 = permutedims(weight21, (1, 3, 2))
weight22 = permutedims(weight22, (1, 3, 2))
rowindex = repeat(collect(1:nb*na*nse), 4*nse)


TT_n = sparse(vec(rowindex), [vec(index11); vec(index21); vec(index12); vec(index22)],
              [vec(weight11); vec(weight21); vec(weight12); vec(weight22)], nb*na*nse, nb*na*nse) 

weight11 = zeros(nb*na, nse, nse)
weight12 = zeros(nb*na, nse, nse)
weight21 = zeros(nb*na, nse, nse)
weight22 = zeros(nb*na, nse, nse)

## Transition matrix die
idb_die = repeat(vec(idb_die), 1, nse)
ida_die = repeat(vec(ida_die), 1, nse)
ids_die = vec(kron(1:nse, ones(1, nb*na*nse))')'

index11 = sub2ind((nb, na, nse), vec(idb_die), vec(ida_die), vec(ids_die))
index12 = sub2ind((nb, na, nse), vec(idb_die), vec(ida_die) .+ 1, vec(ids_die))
index21 = sub2ind((nb, na, nse), vec(idb_die) .+ 1, vec(ida_die), vec(ids_die))
index22 = sub2ind((nb, na, nse), vec(idb_die) .+ 1, vec(ida_die) .+ 1, vec(ids_die))

for hh in 1:nse
    # Corresponding weights
    weight11_aux = (1 .- Dist_b_die[:,:,hh]) .* (1 .- Dist_a_die[:,:,hh])
    weight12_aux = (1 .- Dist_b_die[:,:,hh]) .* Dist_a_die[:,:,hh]
    weight21_aux = Dist_b_die[:,:,hh] .* (1 .- Dist_a_die[:,:,hh])
    weight22_aux = Dist_b_die[:,:,hh] .* Dist_a_die[:,:,hh]

    # Dimensions (m x k, h', h)
    weight11[:,:,hh] = vec(weight11_aux) * P_transition_dist[hh,:]'
    weight12[:,:,hh] = vec(weight12_aux) * P_transition_dist[hh,:]'
    weight21[:,:,hh] = vec(weight21_aux) * P_transition_dist[hh,:]'
    weight22[:,:,hh] = vec(weight22_aux) * P_transition_dist[hh,:]'
end

# Dimensions (m x k, h, h')
weight11 = permutedims(weight11, (1, 3, 2))
weight12 = permutedims(weight12, (1, 3, 2))
weight21 = permutedims(weight21, (1, 3, 2))
weight22 = permutedims(weight22, (1, 3, 2))
# Match MATLAB repmat(1:N,[1 4*nse]) -> [1,2,...,N, 1,2,...,N, ...] when vectorized
rowindex = repeat(collect(1:nb*na*nse), 4*nse)

TT_die = sparse(vec(rowindex), [vec(index11); vec(index21); vec(index12); vec(index22)],
                [vec(weight11); vec(weight21); vec(weight12); vec(weight22)], nb*na*nse, nb*na*nse)

# Joint transition matrix and transition


n_aprob = length(vec(AProb))
APD_a = sparse(1:n_aprob, 1:n_aprob, vec(AProb))
APD_n = sparse(1:n_aprob, 1:n_aprob, 1 .- vec(AProb))
TT = (1 - param["death_rate"]) * (APD_a * TT_a + APD_n * TT_n) + param["death_rate"] * TT_die

TT = H_tilde' * TT

MUnext = vec(MU)' * TT
MUnext = reshape(vec(MUnext), (nb, na, nse)) 

death = param["death_rate"]

PDF_joint   = MUnext 
C_next_grid = cumsum(cumsum(cumsum(PDF_joint; dims=1); dims=2); dims=3) 

xb = collect(CDF_b)
xa = collect(CDF_a)
xs = collect(CDF_se)

b_knots  = vec(CDF_b)
a_knots  = vec(CDF_a)
se_knots = vec(CDF_se)
#=
F_next = extrapolate(
    interpolate(
        (b_knots, a_knots, se_knots),
        C_next_grid,
        Gridded(Cubic(Line(OnGrid())))
    ),
    Line()
)
=# 

# evaluates copula at a single point (m,k,y)
Copula(x::Real, y::Real, z::Real) =
    myinterpolate3(CDF_m_SS, CDF_k_SS, CDF_y_SS, COP_SS,
                   [x], [y], [z])[1,1,1] +
    myinterpolate3(s_m_m, s_m_k, s_m_y, COP_Dev,
                   [x], [y], [z])[1,1,1]

function Copula(x::AbstractVector, y::AbstractVector, z::AbstractVector)
    A = myinterpolate3(CDF_m_SS, CDF_k_SS, CDF_y_SS, COP_SS,  x, y, z)
    B = myinterpolate3(s_m_m,    s_m_k,    s_m_y,    COP_Dev, x, y, z)
    C = A .+ B

    return clamp.(C, 0.0, 1.0)
end






b_SS_knots = vec(CDF_b_SS)
a_SS_knots = vec(CDF_a_SS)
se_SS_knots = vec(CDF_se_SS)
#=
F_SS = extrapolate(
    interpolate(
        (b_SS_knots, a_SS_knots, se_SS_knots),
        COP_SS,
        Gridded(Cubic(Line(OnGrid())))
    ),
    Line()
)

C_next_onU = F_next[s_m_b[:], s_m_a[:], s_m_se[:]]
C_SS_onU   = F_SS[s_m_b[:], s_m_a[:], s_m_se[:]]

C_next_onU = myinterpolate3(CDF_b, CDF_a, CDF_se, C_next_grid, s_m_b, s_m_a, s_m_se)
  C_SS_onU = myinterpolate3(CDF_b_SS, CDF_a_SS, CDF_se_SS, COP_SS, s_m_b, s_m_a, s_m_se)
=#

# Compute CDF_Dev as difference of two interpolations (errors cancel out)

CDF_b_vec = Vector{Float64}(CDF_b)
CDF_a_vec = Vector{Float64}(CDF_a)
CDF_se_vec = Vector{Float64}(CDF_se)
CDF_b_SS_vec = Vector{Float64}(CDF_b_SS)
CDF_a_SS_vec = Vector{Float64}(CDF_a_SS)
CDF_se_SS_vec = Vector{Float64}(CDF_se_SS)
s_m_b_vec = Vector{Float64}(s_m_b)
s_m_a_vec = Vector{Float64}(s_m_a)
s_m_se_vec = Vector{Float64}(s_m_se)
CDF_Dev = myinterpolate3(CDF_b_vec, CDF_a_vec, CDF_se_vec, C_next_grid, s_m_b_vec, s_m_a_vec, s_m_se_vec) .-
          myinterpolate3(CDF_b_SS_vec, CDF_a_SS_vec, CDF_se_SS_vec, COP_SS, s_m_b_vec, s_m_a_vec, s_m_se_vec)

COP_dev_next = cdf_to_pdf3D_forwarddiff(CDF_Dev)

COP_thet     = compress(grid["compressionIndexesCOP"], COP_dev_next, DCD, IDCD)

RHS[COP_ind] = COP_thet 

aux_b                = dropdims(sum(sum(MUnext; dims=2); dims=3); dims=(2, 3)) 
aux_a     = dropdims(sum(sum(MUnext; dims=1); dims=3); dims=(1,3))' 
aux_se               = dropdims(sum(sum(MUnext; dims=1); dims=2); dims=(1,2)) 


# State variables

RHS[marginal_b_ind] = aux_b[1:end-1] 
RHS[marginal_a_ind] = aux_a[1:end-1] 
RHS[marginal_se_ind] = aux_se[1:end-1] 

RHS[R_star_ind] = log(param["R_cb"]) + (1-param["rho_R"])*(param["phi_pi"]*log(PI/param["pi_cb"])-param["phi_u"]*(unemp-SS_stats["u"])) + param["rho_R"]*(log(R_starminus/param["R_cb"])) + eps_R 
if string(param["regime"]) == "Taylor"
        RHS[R_cb_ind] = log(param["R_cb"]) + (1-param["rho_R"])*(param["phi_pi"]*log(PI/param["pi_cb"])-param["phi_u"]*(unemp-SS_stats["u"])) + param["rho_R"]*(log(R_cbminus/param["R_cb"])) + eps_R
elseif string(param["regime"]) == "QE"
        RHS[R_cb_ind] = log(param["R_cb"])
elseif string(param["regime"]) == "Partial QE"
        RHS[R_cb_ind] = log(param["R_cb"]) + 1*log(PI/param["pi_cb"])
end
RHS[w_ind] = log(param["w_bar"]) + param["rho_w"] * log(Wminus/param["w_bar"]) + param["rho_w"]*(param["d"]*log(param["pi_bar"]/PI)+(1-param["d"])*log(pastpiminus/PI)) + (1-param["rho_w"])*param["eps_w"]*log(PSI_W*H/SS_stats["r_l"]) 
RHS[A_b_ind] = log( lev*NW_b/Q ) 
RHS[B_b_ind] = log( T + B_gov_ncpnext - B_gov_ncp*R_cbminus/PI + (Q+R_A-R_cbminus/PI*Qminus)*A_g - UB - LT - param["tau_cp"]*Q*A_gnext - G + param["b_a_aux2"]*R_cbminus/PI*B_b - C_b ) 

RHS[A_g_ind] = log( x_cb*A_g + 1 )  
RHS[Q_ind] = log( iota2*(1+param["phi"]*log(x_k)+param["phi"]/2*(log(x_k))^2 ) - iota2next*LAMBDA/SS_stats["Lambda"]*param["phi"]*log(x_knext)*x_knext ) 
RHS[lev_ind] = log( ee/(param["DELTA"]-vv) ) 
RHS[NW_b_ind] = log( (param["theta_b"]*((RRa-param["b_a_aux2"]*R_cbminus/PI)*levminus+param["b_a_aux2"]*R_cbminus/PI)*NW_bminus + param["omega"]*Qminus*A_b) ) 
RHS[R_tilde_ind] = log( R_cb*D ) 

if string(param["regime"]) == "Taylor"
        RHS[x_cb_ind] = 0
elseif string(param["regime"]) == "QE"
        RHS[x_cb_ind] = log( 1 +param["rho_X_QE"]*(x_cbminus-1) + (1-param["rho_X_QE"])*(-param["phi_pi_QE"]*log(PI/param["pi_cb"])+param["phi_u_QE"]*(unemp-SS_stats["u"])) + eps_R )
elseif string(param["regime"]) == "Partial QE"
        RHS[x_cb_ind] = log( 1 +param["rho_X_QE"]*(x_cbminus-1) + (1-param["rho_X_QE"])*(-param["phi_pi_QE"]*log(PI/param["pi_cb"])+param["phi_u_QE"]*(unemp-SS_stats["u"])) + eps_R )
end
RHS[pastpi_ind] = log(PI) 

RHS[pastY_ind] = log(Y) 
RHS[pastC_ind] = log(C) 
RHS[pastI_ind] = log(Inv) 
RHS[pastPROFIT_ind] = log(PROFIT+param["fix2"]) 
RHS[pastunemp_ind] = log(unemp) 


RHS[pastG_ind] = log(G) 
RHS[pastLT_ind] = log(LT) 
RHS[B_F_ind] = param["rho_B_F"]*log(B_F) + (1-param["rho_B_F"])*log(SS_stats["Y"]/(SS_stats["Y"]-SS_stats["B_F"])) + eps_B_F 
RHS[Z_ind] = param["rho_Z"]        * log(Zminus)        + eps_Z 
RHS[PSI_RP_ind] = param["rho_PSI_RP"]   * log(PSI_RPminus)   + eps_RP 
RHS[eta_ind] = log(p_mark/(p_mark-1)) 
RHS[D_ind ] = param["rho_D"]        * log(Dminus)        + eps_D 

    if string(param["adjust"]) == "G"
        RHS[GG_ind] = param["rho_G"] * log(GGminus) + (1-param["rho_G"]) *
                       log(SS_stats["Y"]/(SS_stats["Y"]-SS_stats["LT"]-SS_stats["C_b"])) + eps_G
    elseif string(param["adjust"]) == "LT"
        RHS[GG_ind] = param["rho_G"]  * log(GGminus) + (1-param["rho_G"])  * log(SS_stats["Y"]/(SS_stats["Y"]-SS_stats["G"])) + eps_G
    end

RHS[iota_ind] = param["rho_iota"]  * log(iotaminus)  + eps_iota 
RHS[BB_ind] = param["rho_BB"]    * log(BBminus)    + eps_BB 
RHS[PSI_W_ind] = param["rho_PSI_W"] * log(PSI_Wminus) + eps_w 
RHS[MP_ind] = param["rho_MP"]    * log(MPminus)    + eps_R 
RHS[p_mark_ind] = param["rho_eta"]   * log(p_markminus) + (1-param["rho_eta"])*log(param["eta"]/(param["eta"]-1)) + eps_eta 


RHS[eps_QE_ind] = 0 
RHS[eps_B_F_ind] = 0 
RHS[eps_RP_ind] = 0 
RHS[eps_eta_ind] = 0 
RHS[eps_D_ind] = 0 
RHS[eps_R_ind] = 0 
RHS[eps_Z_ind] = 0 
RHS[eps_G_ind] = 0 
RHS[eps_iota_ind] = 0 
RHS[eps_w_ind] = 0 
RHS[eps_BB_ind] = 0 

# Control variables

RHS[nx .+ VALUE_ind] = invutil(VALUEaux[:])
RHS[nx .+ mutil_c_ind] = invmutil(mutil_c_aux[:])
RHS[nx .+ Va_ind] = invmutil(Va_aux[:]) 

RHS[nx+MRS_ind] = sum(sum(MRS_a_aux[:,:,end]+MRS_n_aux[:,:,end]))/param["Eshare"]*param["Lambda_b_aux"] 

RHS[nx+A_hh_ind] = sum(grid["a"].*marginal_aminus) 
RHS[nx+B_hh_ind] = sum(grid["b"].*marginal_bminus)/(1-param["death_rate"])*param["b_a_aux"] 
RHS[nx+C_ind] = q_cons2(c_a_star,c_n_star,W,nn,AProb,MU_tilde,meshes,param,grid,tau_w) 

RHS[nx+N_ind] = sum(marginal_seminus[1:ns])  # beginning of period aggregate employment
RHS[nx+L_ind] = nn * grid["s"]' * MU_tilde_se[1:ns]

RHS[nx+UB_ind] = param["w_bar"] * param["n"] * param["xi"] / (1 + param["xi"]) * grid["se"][ns+1:2*ns]' * MU_tilde_se[ns+1:2*ns] 

RHS[nx+K_ind] = A_hh + A_g + A_b + SS_stats["A_F"]       # beginning of period asset holding
RHS[nx+B_ind] = B_hh + B_b + (1 - 1/1) * SS_stats["Y"]  # beginning of period asset holding
RHS[nx+B_gov_ncp_ind] = B_b  + B_hh + (1-1/1)*SS_stats["Y"] - (Qminus*A_b- NW_bminus) - Qminus *(1+param["tau_cp"])* A_g 
RHS[nx+T_ind] = tau_w.*(W.*L + UB) + tau_a.*(PROFIT) 

if string(param["adjust"]) == "G"
           RHS[nx+LT_ind] = (1-1/GG)*SS_stats["Y"] - SS_stats["C_b"]
           RHS[nx+G_ind] = SS_stats["G"]
elseif string(param["adjust"]) == "LT"
        RHS[nx+G_ind] = (1-1/GG)*SS_stats["Y"]
        RHS[nx+LT_ind] = SS_stats["LT"]
end


RHS[nx+Lambda_ind] = MRS*param["Lambda_aux"]/param["Lambda_b_aux"] 
RHS[nx+pi_ind] = param["pi_cb"]*exp((param["rho_B"]/(param["rho_B"]+param["gamma_pi"]))*log(B_gov_ncp*R_cbminus/(SS_stats["B_gov_ncp"]*param["R_cb"]))-(param["gamma_T"]/(param["rho_B"]+param["gamma_pi"]))*log(T/SS_stats["T"])-1/(param["rho_B"]+param["gamma_pi"])*log(B_gov_ncpnext/SS_stats["B_gov_ncp"])) 

RHS[nx+V_ind] = M/param["iota"]*(Matrix(J')*grid["s_dist"])[1]

RHS[nx .+ J_ind] = ((H-param["fix_L"]-W).*(grid["s"]')*nn)' + LAMBDA*(1-param["death_rate"])*(1-l_lambdanext)*(1-param["in"])*param["P_SS"]*Jnext 
 RHS[nx+h_ind] = Z*MC*(1-param["theta"])*(v*K).^param["theta"].*(L).^(-param["theta"]) 
RHS[nx+h_ind] = Z*MC*(v*K).^param["theta"].*(L).^(param["theta_2"]-1)*(param["theta_2"]+param["alpha_lk"]*log(v*K/SS_stats["v"]/grid["K"])+param["alpha_ll"]*log(L/SS_stats["L"])) 

RHS[nx+v_ind] = min( (R_K/(param["delta_0"]*param["delta_1"]))^(1/(param["delta_1"]-1)), 1)  
RHS[nx+Y_ind] = Z*(v*K).^(param["theta"]).*(L).^param["theta_2"]*((L/SS_stats["L"])^(param["alpha_lk"]*log(v*K/SS_stats["v"]/grid["K"]))*(v*K/SS_stats["v"]/grid["K"])^(param["alpha_lk"]*log(L/SS_stats["L"]))) 

RHS[nx + Profit_ind] =
    (Y * (1 - eta / (2 * param["kappa"]) *
            (log(PI) - (1 - param["GAMMA"]) * log(param["pi_cb"]) - param["GAMMA"] * log(pastpiminus))^2)
     + Y * (-MC) - param["fix"] + (H - param["fix_L"] - W) * L - param["iota"] * V) +
    (R_K * v - param["delta_0"] * v^param["delta_1"]) * K +
    Q * (Knext - K) - (Knext - K) -
    (param["phi"] / 2) * (Knext / K - 1)^2 * K +
    Profit_FI - param["fix2"]



RHS[nx+r_k_ind] = Z*MC*(v*K).^(param["theta"]-1).*(L).^param["theta_2"]*(param["theta"]+param["alpha_lk"]*log(L/SS_stats["L"])+param["alpha_kk"]*log(v*K/SS_stats["v"]/grid["K"])) 
RHS[nx+r_a_ind] = (1-tau_a)*(1-param["Eratio"]-param["b_share"])*PROFIT/K 

RHS[nx+MC_ind] = 1-1/eta + (log(PI/(pastpiminus^param["GAMMA"]*param["pi_bar"]^(1-param["GAMMA"])))-LAMBDA*eta2next/eta2*Ynext/Y*log(PInext/(PI^param["GAMMA"]*param["pi_bar"]^(1-param["GAMMA"])) ))/param["kappa"] 

RHS[nx+zz_ind] = (RRa-param["b_a_aux2"]*R_cbminus/PI)*levminus + param["b_a_aux2"]*R_cbminus/PI 
# RHS[nx+zz_ind] = (RRa-R_cbminus/PI)*levminus + R_cbminus/PI 
RHS[nx+xx_ind] = (lev/levminus)*zz 
RHS[nx+Profit_FI_ind] = ((1-param["theta_b"])*((RRa - param["b_a_aux2"]*R_cbminus/PI)*levminus+param["b_a_aux2"]*R_cbminus/PI)*NW_bminus - param["omega"]*Qminus*A_b) 
RHS[nx+vv_ind] = ((1-param["theta_b"])*BB*LAMBDA*(RRanext-R_cb/PInext)+param["theta_b"]*BB*LAMBDA*xxnext*vvnext) 
RHS[nx+ee_ind] = ((1-param["theta_b"])*BB*LAMBDA*R_cb/PInext+param["theta_b"]*BB*LAMBDA*zznext*eenext) 
RHS[nx+C_b_ind] = ((1*param["beta_b"]*R_cb/PInext)/(1+param["phi_B"]*(B_bnext/B_b-1))*C_bnext^(-param["sigma2"]))^(-1/param["sigma2"]) 

RHS[nx+RRa_ind] = (Q+R_A)/Qminus 
RHS[nx+RR_ind] = R_cbminus/PI 
RHS[nx+I_ind] = iota*(1+param["phi"]/2*(log(x_k))^2)*(A_hhnext + A_gnext + A_bnext + SS_stats["A_F"]) - (1-param["delta_0"]*v^param["delta_1"])*K 
RHS[nx+unemp_ind] = (1-((1-l_lambda)*N + M)-param["Eshare"])/(1-param["Eshare"]) 
# RHS[nx+stock_ind] = Y + (1-param["delta_i"]*invenminus/SS_stats["inven"])*invenminus 
RHS[nx+x_k_ind] = Knext/K 
# RHS[nx+Q2_ind] = Q 

# RHS[nx+nn_ind] = ((1-tau_w)*W/(param["psi"]*BB))^(1/param["xi"]) 
RHS[nx+nn_ind] = ((1-tau_w)*W/(param["psi"]))^(1/param["xi"]) 
RHS[nx+M_ind] = ((1-N-Eshare)+l_lambda.*(N)).*V/((((1-N-Eshare)+l_lambda.*(N))^param["alpha"]+V^param["alpha"])^(1/param["alpha"])) 
RHS[nx+f_ind] = M/((1-N-Eshare)+l_lambda*(N)) 

RHS[nx+A_g_obs_ind] = (A_gauxnext-1) 
RHS[nx+Y_obs_ind] = Y 
RHS[nx+C_obs_ind] = C 
RHS[nx+I_obs_ind] = Inv 
RHS[nx+w_obs_ind] = W 
# RHS[nx+PROFIT_obs_ind] = PROFIT + Profit_FI 
RHS[nx+PROFIT_obs_ind] = PROFIT + param["fix2"] + log(B_Fnext) 
RHS[nx+unemp_obs_ind] = 100*unemp 
RHS[nx+inf_obs_ind] = PI 
RHS[nx+R_obs_ind] = R_cb 
RHS[nx+pastw_ind] = Wminus 

if string(param["adjust"]) == "G"
        # RHS[nx+G_obs_ind] = LT
        RHS[nx+pastGG_ind] = pastLTminus
        # RHS[nx+pastLT2_ind] = pastGminus
    elseif string(param["adjust"]) == "LT"
        # RHS[nx+G_obs_ind] = G
        RHS[nx+pastGG_ind] = pastGminus
        # RHS[nx+pastLT2_ind] = pastLTminus
end

RHS[nx+pastYY_ind] = pastYminus 
RHS[nx+pastCC_ind] = pastCminus 
RHS[nx+pastII_ind] = pastIminus 
RHS[nx+pastPPROFIT_ind] = pastPROFITminus 
RHS[nx+pastuu_ind] = pastunempminus 
RHS[nx+pastA_g_ind] = A_g 

RHS[nx+l_lambda_ind] = param["lambda"] 
RHS[nx+pastQ_ind] = Qminus 
RHS[nx+x_I_ind] = Inv/pastIminus 
# RHS[nx+eta2_ind] = eta/param["eta"] 
RHS[nx+eta2_ind] = eta 
RHS[nx+iota2_ind] = iota 

RHS[nx+G_obs_ind] = G 
RHS[nx+pastLT2_ind] = pastLTminus 
RHS[nx+pastG2_ind] = pastGminus 
RHS[nx+LT_obs_ind] = LT 

RHS[nx+B_gov_ncp2_ind] = B_b  + B_hh - (Qminus*A_b- NW_bminus) - Qminus *(1+param["tau_cp"])* A_g 



## Difference
Difference = InvGamma * ((LHS - RHS) ./ [ones(nx); ControlSS[1:end-oc]; ones(oc)])

    return Difference, LHS, RHS, MUnext, c_a_star, b_a_star, a_a_star, c_n_star, b_n_star, AC, AProb, P_transition_exp, P_transition_dist
end
