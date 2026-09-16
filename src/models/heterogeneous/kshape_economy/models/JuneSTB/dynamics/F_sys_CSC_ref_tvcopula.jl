using LinearAlgebra
using SparseArrays
using Interpolations
include(joinpath(@__DIR__, "../../../helpers/ndgrid.jl"))
include(joinpath(@__DIR__, "../../../helpers/genweight.jl"))
include(joinpath(@__DIR__, "../../../helpers/DCT/uncompress.jl"))
include(joinpath(@__DIR__, "../../../helpers/DCT/compress.jl"))
include(joinpath(@__DIR__, "../../../helpers/grids/myinterpolate3.jl"))
include(joinpath(@__DIR__, "../../../helpers/cdf_to_pdf3D_forwarddiff.jl"))
include(joinpath(@__DIR__, "../../../helpers/sub2ind.jl"))
include(joinpath(@__DIR__, "../steadystate/q_cons.jl"))
include(joinpath(@__DIR__, "../../../helpers/q_cons2.jl"))
include(joinpath(@__DIR__, "../../../helpers/Fastroot.jl"))
include(joinpath(@__DIR__, "../../../dynamics/policies_update.jl"))

"""
    F_sys_CSC_ref_tvcopula(State, Stateminus, Controlnext_sparse, Control_sparse,
                            StateSS, ControlSS, Gamma_state, Gamma_control, InvGamma,
                            param, grid, SS_stats, DCD, IDCD)

System of equations for the CSC model (two-sector labour market, financial
intermediary) in Schmitt-Grohé–Uribe generic form.

Compared to `F_sys_ref_tvcopula_QE`:
- No `P_SE` argument; the QE regime is unused (Taylor rule only, no x_cb active)
- Two skill groups (ns_1, ns_2): separate wages W_1/W_2, matching M_1/M_2,
  vacancy values V_1/V_2, job values J[1:ns], finding rates f_1/f_2
- Financial intermediary: lev, NW_b, zz, xx, vv, ee, C_b, Profit_FI
- Nested-CES production: unskilled L_1 vs capital-skill composite V_aux
- ZZ_1–ZZ_4 autoregressive labour-market wedge processes
"""
function F_sys_CSC_ref_tvcopula(State, Stateminus, Controlnext_sparse, Control_sparse,
                                  StateSS, ControlSS, Gamma_state, Gamma_control, InvGamma,
                                  param, grid, SS_stats, DCD, IDCD)

## Initializations

util     = c -> (c .^ (1 - param["sigma"])) ./ (1 - param["sigma"])
mutil    = c -> 1.0 ./ (c .^ param["sigma"])
invutil  = u -> ((1 - param["sigma"]) .* u) .^ (1 / (1 - param["sigma"]))
invmutil = mu -> (1.0 ./ mu) .^ (1 / param["sigma"])

# Grid sizes
nb  = Int(grid["nb"])
na  = Int(grid["na"])
nse = Int(grid["nse"])
ns  = Int(grid["ns"])
ns_1 = Int(grid["ns_1"])
ns_2 = Int(grid["ns_2"])
NN  = nb * na * nse
Ny  = 3 * NN

nx   = Int(grid["numstates"])
ny   = length(ControlSS)

nFullMarg = size(Gamma_state, 1)
nRedMarg  = size(Gamma_state, 2)
nCOP      = Int(grid["nCOP"])
NxNx      = nRedMarg + nCOP

oc = Int(grid["oc"])

## ---- Control indices ----

VALUE_ind   = 1:NN
mutil_c_ind = NN .+ (1:NN)
Va_ind      = 2*NN .+ (1:NN)

# Summary controls
MRS_ind       = Ny            + 1
A_hh_ind      = MRS_ind       + 1
B_hh_ind      = A_hh_ind      + 1
C_ind         = B_hh_ind      + 1
N_tilde_1_ind = C_ind         + 1
N_tilde_2_ind = N_tilde_1_ind + 1
L_1_ind       = N_tilde_2_ind + 1
L_2_ind       = L_1_ind       + 1
UB_ind        = L_2_ind       + 1
unemp_1_ind   = UB_ind        + 1
unemp_2_ind   = unemp_1_ind   + 1
unemp_ind     = unemp_2_ind   + 1
U_1_ind       = unemp_ind     + 1
U_2_ind       = U_1_ind       + 1
Income_Tax_ind = U_2_ind      + 1
J_bar_1_ind   = Income_Tax_ind + 1
J_bar_2_ind   = J_bar_1_ind   + 1

# Aggregate controls
K_ind         = J_bar_2_ind   + 1
B_ind         = K_ind         + 1
B_gov_ncp_ind = B_ind         + 1
T_ind         = B_gov_ncp_ind + 1
LT_ind        = T_ind         + 1
G_ind         = LT_ind        + 1
Lambda_ind    = G_ind         + 1
pi_ind        = Lambda_ind    + 1
V_1_ind       = pi_ind        + 1
V_2_ind       = V_1_ind       + 1
J_ind         = V_2_ind .+ (1:ns)
r_l_1_ind     = J_ind[end]    + 1
r_l_2_ind     = r_l_1_ind     + 1
v_ind         = r_l_2_ind     + 1
Y_ind         = v_ind         + 1
Profit_ind    = Y_ind         + 1
r_k_ind       = Profit_ind    + 1
r_a_ind       = r_k_ind       + 1
MC_ind        = r_a_ind       + 1
n_1_ind       = MC_ind        + 1
n_2_ind       = n_1_ind       + 1
M_1_ind       = n_2_ind       + 1
M_2_ind       = M_1_ind       + 1
f_1_ind       = M_2_ind       + 1
f_2_ind       = f_1_ind       + 1
zz_ind        = f_2_ind       + 1
xx_ind        = zz_ind        + 1
vv_ind        = xx_ind        + 1
ee_ind        = vv_ind        + 1
C_b_ind       = ee_ind        + 1
Profit_FI_ind = C_b_ind       + 1
RRa_ind       = Profit_FI_ind + 1
RR_ind        = RRa_ind       + 1
I_ind         = RR_ind        + 1
x_k_ind       = I_ind         + 1

# Observables
A_g_obs_ind      = x_k_ind         + 1
Y_obs_ind        = A_g_obs_ind     + 1
C_obs_ind        = Y_obs_ind       + 1
I_obs_ind        = C_obs_ind       + 1
w_1_obs_ind      = I_obs_ind       + 1
w_2_obs_ind      = w_1_obs_ind     + 1
w_obs_ind        = w_2_obs_ind     + 1
PROFIT_obs_ind   = w_obs_ind       + 1
unemp_1_obs_ind  = PROFIT_obs_ind  + 1
unemp_2_obs_ind  = unemp_1_obs_ind + 1
unemp_obs_ind    = unemp_2_obs_ind + 1
inf_obs_ind      = unemp_obs_ind   + 1
R_obs_ind        = inf_obs_ind     + 1
pastw_1_ind      = R_obs_ind       + 1
pastw_2_ind      = pastw_1_ind     + 1
pastw_ind        = pastw_2_ind     + 1
G_obs_ind        = pastw_ind       + 1
pastYY_ind       = G_obs_ind       + 1
pastCC_ind       = pastYY_ind      + 1
pastII_ind       = pastCC_ind      + 1
pastPPROFIT_ind  = pastII_ind      + 1
pastuu_1_ind     = pastPPROFIT_ind + 1
pastuu_2_ind     = pastuu_1_ind    + 1
pastuu_ind       = pastuu_2_ind    + 1
pastGG_ind       = pastuu_ind      + 1
pastA_g_ind      = pastGG_ind      + 1
l_lambda_1_ind   = pastA_g_ind     + 1
l_lambda_2_ind   = l_lambda_1_ind  + 1
pastQ_ind        = l_lambda_2_ind  + 1
x_I_ind          = pastQ_ind       + 1
eta2_ind         = x_I_ind         + 1
iota2_ind        = eta2_ind        + 1
pastLT2_ind      = iota2_ind       + 1
pastG2_ind       = pastLT2_ind     + 1
LT_obs_ind       = pastG2_ind      + 1
B_gov_ncp2_ind   = LT_obs_ind      + 1

## ---- State indices ----

marginal_b_ind  = 1:(nb-1)
marginal_a_ind  = (nb-1) .+ (1:(na-1))
marginal_se_ind = (nb+na-2) .+ (1:(nse-1))
COP_ind         = (nb+na+nse-3) .+ (1:nCOP)

R_cb_ind        = NxNx + 1
w_1_ind         = NxNx + 2
w_2_ind         = NxNx + 3
w_ind           = NxNx + 4
A_b_ind         = NxNx + 5
B_b_ind         = NxNx + 6
A_g_ind         = NxNx + 7
Q_ind           = NxNx + 8
lev_ind         = NxNx + 9
NW_b_ind        = NxNx + 10
R_tilde_ind     = NxNx + 11
x_cb_ind        = NxNx + 12
pastpi_ind      = NxNx + 13
pastY_ind       = NxNx + 14
pastC_ind       = NxNx + 15
pastI_ind       = NxNx + 16
pastPROFIT_ind  = NxNx + 17
pastunemp_1_ind = NxNx + 18
pastunemp_2_ind = NxNx + 19
pastunemp_ind   = NxNx + 20
pastG_ind       = NxNx + 21
pastLT_ind      = NxNx + 22
R_star_ind      = NxNx + 23
ZZ_1_ind        = NxNx + 24
ZZ_2_ind        = NxNx + 25
ZZ_3_ind        = NxNx + 26
ZZ_4_ind        = NxNx + 27
B_F_ind         = NxNx + 28
Z_ind           = NxNx + 29
PSI_RP_ind      = NxNx + 30
eta_ind         = NxNx + 31
D_ind           = NxNx + 32
GG_ind          = NxNx + 33
iota_ind        = NxNx + 34
BB_ind          = NxNx + 35
PSI_W_ind       = NxNx + 36
MP_ind          = NxNx + 37
p_mark_ind      = NxNx + 38
eps_1_ind       = NxNx + 39
eps_2_ind       = NxNx + 40
eps_3_ind       = NxNx + 41
eps_4_ind       = NxNx + 42
eps_QE_ind      = NxNx + 43
eps_RP_ind      = NxNx + 44
eps_B_F_ind     = NxNx + 45
eps_BB_ind      = NxNx + 46
eps_Z_ind       = NxNx + 47
eps_G_ind       = NxNx + 48
eps_D_ind       = NxNx + 49
eps_R_ind       = NxNx + 50
eps_iota_ind    = NxNx + 51
eps_eta_ind     = NxNx + 52
eps_w_ind       = NxNx + 53

NxNx_aux = 53

## ---- Expand compressed control vectors ----

Controlnext = ControlSS .* (1.0 .+ Gamma_control * Controlnext_sparse)
Control     = ControlSS .* (1.0 .+ Gamma_control * Control_sparse)
Controlnext[end-oc+1:end] = ControlSS[end-oc+1:end] + Gamma_control[end-oc+1:end, :] * Controlnext_sparse
Control[end-oc+1:end]     = ControlSS[end-oc+1:end] + Gamma_control[end-oc+1:end, :] * Control_sparse

## ---- Marginal distributions ----

Distribution      = StateSS[1:nFullMarg] + Gamma_state * State[1:nRedMarg]
Distributionminus = StateSS[1:nFullMarg] + Gamma_state * Stateminus[1:nRedMarg]

cop_coefs   = State[nRedMarg .+ (1:nCOP)]
cop_coefs_m = Stateminus[nRedMarg .+ (1:nCOP)]

## ---- Unpack aggregate state variables (log levels) ----

function _ss(i)
    StateSS[end-(NxNx_aux-i+NxNx)]
end
function _st(i)
    _ss(i) + State[end-(NxNx_aux-i+NxNx)]
end
function _stm(i)
    _ss(i) + Stateminus[end-(NxNx_aux-i+NxNx)]
end

R_cb     = _st(R_cb_ind);   R_cbminus     = _stm(R_cb_ind)
W_1      = _st(w_1_ind);    W_1minus      = _stm(w_1_ind)
W_2      = _st(w_2_ind);    W_2minus      = _stm(w_2_ind)
W        = _st(w_ind);      Wminus        = _stm(w_ind)
A_bnext  = _st(A_b_ind);    A_b           = _stm(A_b_ind)
B_bnext  = _st(B_b_ind);    B_b           = _stm(B_b_ind)
A_gauxnext = _st(A_g_ind);  A_gaux        = _stm(A_g_ind)
Q        = _st(Q_ind);      Qminus        = _stm(Q_ind)
lev      = _st(lev_ind);    levminus      = _stm(lev_ind)
NW_b     = _st(NW_b_ind);   NW_bminus     = _stm(NW_b_ind)
R_tilde  = _st(R_tilde_ind);R_tildeminus  = _stm(R_tilde_ind)
x_cb     = _st(x_cb_ind);   x_cbminus     = _stm(x_cb_ind)
pastpi   = _st(pastpi_ind); pastpiminus   = _stm(pastpi_ind)
pastY    = _st(pastY_ind);  pastYminus    = _stm(pastY_ind)
pastC    = _st(pastC_ind);  pastCminus    = _stm(pastC_ind)
pastI    = _st(pastI_ind);  pastIminus    = _stm(pastI_ind)
pastPROFIT   = _st(pastPROFIT_ind);   pastPROFITminus   = _stm(pastPROFIT_ind)
pastunemp_1  = _st(pastunemp_1_ind);  pastunemp_1minus  = _stm(pastunemp_1_ind)
pastunemp_2  = _st(pastunemp_2_ind);  pastunemp_2minus  = _stm(pastunemp_2_ind)
pastunemp    = _st(pastunemp_ind);    pastunempminus    = _stm(pastunemp_ind)
pastG    = _st(pastG_ind);  pastGminus    = _stm(pastG_ind)
pastLT   = _st(pastLT_ind); pastLTminus   = _stm(pastLT_ind)
R_star   = _st(R_star_ind); R_starminus   = _stm(R_star_ind)
ZZ_1     = _st(ZZ_1_ind);   ZZ_1minus     = _stm(ZZ_1_ind)
ZZ_2     = _st(ZZ_2_ind);   ZZ_2minus     = _stm(ZZ_2_ind)
ZZ_3     = _st(ZZ_3_ind);   ZZ_3minus     = _stm(ZZ_3_ind)
ZZ_4     = _st(ZZ_4_ind);   ZZ_4minus     = _stm(ZZ_4_ind)
B_Fnext  = _st(B_F_ind);    B_F           = _stm(B_F_ind)
Z        = _st(Z_ind);      Zminus        = _stm(Z_ind)
PSI_RP   = _st(PSI_RP_ind); PSI_RPminus   = _stm(PSI_RP_ind)
eta      = _st(eta_ind);    etaminus      = _stm(eta_ind)
D        = _st(D_ind);      Dminus        = _stm(D_ind)
GG       = _st(GG_ind);     GGminus       = _stm(GG_ind)
iota     = _st(iota_ind);   iotaminus     = _stm(iota_ind)
BB       = _st(BB_ind);     BBminus       = _stm(BB_ind)
PSI_W    = _st(PSI_W_ind);  PSI_Wminus    = _stm(PSI_W_ind)
MP       = _st(MP_ind);     MPminus       = _stm(MP_ind)
p_mark   = _st(p_mark_ind); p_markminus   = _stm(p_mark_ind)

# Shocks
eps_1next  = _st(eps_1_ind);  eps_1  = _stm(eps_1_ind)
eps_2next  = _st(eps_2_ind);  eps_2  = _stm(eps_2_ind)
eps_3next  = _st(eps_3_ind);  eps_3  = _stm(eps_3_ind)
eps_4next  = _st(eps_4_ind);  eps_4  = _stm(eps_4_ind)
eps_QEnext   = _st(eps_QE_ind);   eps_QE   = _stm(eps_QE_ind)
eps_RPnext   = _st(eps_RP_ind);   eps_RP   = _stm(eps_RP_ind)
eps_B_Fnext  = _st(eps_B_F_ind);  eps_B_F  = _stm(eps_B_F_ind)
eps_BBnext   = _st(eps_BB_ind);   eps_BB   = _stm(eps_BB_ind)
eps_Znext    = _st(eps_Z_ind);    eps_Z    = _stm(eps_Z_ind)
eps_Gnext    = _st(eps_G_ind);    eps_G    = _stm(eps_G_ind)
eps_Dnext    = _st(eps_D_ind);    eps_D    = _stm(eps_D_ind)
eps_Rnext    = _st(eps_R_ind);    eps_R    = _stm(eps_R_ind)
eps_iotanext = _st(eps_iota_ind); eps_iota = _stm(eps_iota_ind)
eps_etanext  = _st(eps_eta_ind);  eps_eta  = _stm(eps_eta_ind)
eps_wnext    = _st(eps_w_ind);    eps_w    = _stm(eps_w_ind)

## ---- Unpack control variables ----

VALUEnext   = util(Controlnext[VALUE_ind])
VALUE       = util(Control[VALUE_ind])
Vanext      = mutil(Controlnext[Va_ind])
Va          = mutil(Control[Va_ind])
mutil_cnext = mutil(Controlnext[mutil_c_ind])
mutil_c     = mutil(Control[mutil_c_ind])

# t+1 controls
A_hhnext      = exp(Controlnext[A_hh_ind])
B_hhnext      = exp(Controlnext[B_hh_ind])
Cnext         = exp(Controlnext[C_ind])
N_tilde_1next = exp(Controlnext[N_tilde_1_ind])
N_tilde_2next = exp(Controlnext[N_tilde_2_ind])
UBnext        = exp(Controlnext[UB_ind])
unemp_1next   = exp(Controlnext[unemp_1_ind])
unemp_2next   = exp(Controlnext[unemp_2_ind])
unempnext     = exp(Controlnext[unemp_ind])
Knext         = exp(Controlnext[K_ind])
Bnext         = exp(Controlnext[B_ind])
B_gov_ncpnext = exp(Controlnext[B_gov_ncp_ind])
LAMBDAnext    = exp(Controlnext[Lambda_ind])
PInext        = exp(Controlnext[pi_ind])
V_1next       = exp(Controlnext[V_1_ind])
V_2next       = exp(Controlnext[V_2_ind])
Jnext         = exp.(Controlnext[J_ind])
r_l_1next     = exp(Controlnext[r_l_1_ind])
r_l_2next     = exp(Controlnext[r_l_2_ind])
vnext         = exp(Controlnext[v_ind])
Ynext         = exp(Controlnext[Y_ind])
R_Knext       = exp(Controlnext[r_k_ind])
R_Anext       = exp(Controlnext[r_a_ind])
MCnext        = exp(Controlnext[MC_ind])
n_1next       = exp(Controlnext[n_1_ind])
n_2next       = exp(Controlnext[n_2_ind])
M_1next       = exp(Controlnext[M_1_ind])
M_2next       = exp(Controlnext[M_2_ind])
f_1next       = exp(Controlnext[f_1_ind])
f_2next       = exp(Controlnext[f_2_ind])
zznext        = exp(Controlnext[zz_ind])
xxnext        = exp(Controlnext[xx_ind])
vvnext        = exp(Controlnext[vv_ind])
eenext        = exp(Controlnext[ee_ind])
C_bnext       = exp(Controlnext[C_b_ind])
Profit_FInext = exp(Controlnext[Profit_FI_ind])
RRanext       = exp(Controlnext[RRa_ind])
RRnext        = exp(Controlnext[RR_ind])
Inext         = exp(Controlnext[I_ind])
x_knext       = exp(Controlnext[x_k_ind])
l_lambda_1next = exp(Controlnext[l_lambda_1_ind])
l_lambda_2next = exp(Controlnext[l_lambda_2_ind])
x_Inext       = exp(Controlnext[x_I_ind])
eta2next      = exp(Controlnext[eta2_ind])
iota2next     = exp(Controlnext[iota2_ind])
B_gov_ncp2next = exp(Controlnext[B_gov_ncp2_ind])

# t controls
MRS        = exp(Control[MRS_ind])
A_hh       = exp(Control[A_hh_ind])
B_hh       = exp(Control[B_hh_ind])
C          = exp(Control[C_ind])
N_tilde_1  = exp(Control[N_tilde_1_ind])
N_tilde_2  = exp(Control[N_tilde_2_ind])
L_1        = exp(Control[L_1_ind])
L_2        = exp(Control[L_2_ind])
UB         = exp(Control[UB_ind])
unemp_1    = exp(Control[unemp_1_ind])
unemp_2    = exp(Control[unemp_2_ind])
unemp      = exp(Control[unemp_ind])
U_1        = exp(Control[U_1_ind])
U_2        = exp(Control[U_2_ind])
Income_Tax = exp(Control[Income_Tax_ind])
J_bar_1    = exp(Control[J_bar_1_ind])
J_bar_2    = exp(Control[J_bar_2_ind])
K          = exp(Control[K_ind])
B          = exp(Control[B_ind])
B_gov_ncp  = exp(Control[B_gov_ncp_ind])
T          = exp(Control[T_ind])
LT         = exp(Control[LT_ind])
G          = exp(Control[G_ind])
LAMBDA     = exp(Control[Lambda_ind])
PI         = exp(Control[pi_ind])
V_1        = exp(Control[V_1_ind])
V_2        = exp(Control[V_2_ind])
J          = exp.(Control[J_ind])
r_l_1      = exp(Control[r_l_1_ind])
r_l_2      = exp(Control[r_l_2_ind])
v          = exp(Control[v_ind])
Y          = exp(Control[Y_ind])
PROFIT     = exp(Control[Profit_ind])
R_K        = exp(Control[r_k_ind])
R_A        = exp(Control[r_a_ind])
MC         = exp(Control[MC_ind])
n_1        = exp(Control[n_1_ind])
n_2        = exp(Control[n_2_ind])
M_1        = exp(Control[M_1_ind])
M_2        = exp(Control[M_2_ind])
f_1        = exp(Control[f_1_ind])
f_2        = exp(Control[f_2_ind])
zz         = exp(Control[zz_ind])
xx         = exp(Control[xx_ind])
vv         = exp(Control[vv_ind])
ee         = exp(Control[ee_ind])
C_b        = exp(Control[C_b_ind])
Profit_FI  = exp(Control[Profit_FI_ind])
RRa        = exp(Control[RRa_ind])
RR         = exp(Control[RR_ind])
Inv        = exp(Control[I_ind])
x_k        = exp(Control[x_k_ind])
l_lambda_1 = exp(Control[l_lambda_1_ind])
l_lambda_2 = exp(Control[l_lambda_2_ind])
pastQ      = exp(Control[pastQ_ind])
x_I        = exp(Control[x_I_ind])
eta2       = exp(Control[eta2_ind])
iota2      = exp(Control[iota2_ind])
pastLT2    = exp(Control[pastLT2_ind])
pastG2     = exp(Control[pastG2_ind])
LT_obs     = exp(Control[LT_obs_ind])
B_gov_ncp2 = exp(Control[B_gov_ncp2_ind])
A_g_obs    = exp(Control[A_g_obs_ind])

tau_L = param["tau_L"]
tau_P = param["tau_P"]
tau_a = param["tau_a"]

## ---- Initialise LHS / RHS ----

LHS = zeros(nx + ny)
RHS = zeros(nx + ny)

## ---- Write LHS (identity equations for control variables at t) ----

LHS[nx .+ VALUE_ind]   = Control[VALUE_ind]
LHS[nx .+ mutil_c_ind] = Control[mutil_c_ind]
LHS[nx .+ Va_ind]      = Control[Va_ind]

LHS[nx+MRS_ind]        = MRS
LHS[nx+A_hh_ind]       = A_hh
LHS[nx+B_hh_ind]       = B_hh
LHS[nx+C_ind]          = C
LHS[nx+N_tilde_1_ind]  = N_tilde_1
LHS[nx+N_tilde_2_ind]  = N_tilde_2
LHS[nx+L_1_ind]        = L_1
LHS[nx+L_2_ind]        = L_2
LHS[nx+UB_ind]         = UB
LHS[nx+unemp_1_ind]    = unemp_1
LHS[nx+unemp_2_ind]    = unemp_2
LHS[nx+unemp_ind]      = unemp
LHS[nx+U_1_ind]        = U_1
LHS[nx+U_2_ind]        = U_2
LHS[nx+Income_Tax_ind] = Income_Tax
LHS[nx+J_bar_1_ind]    = J_bar_1
LHS[nx+J_bar_2_ind]    = J_bar_2
LHS[nx+K_ind]          = K
LHS[nx+B_ind]          = B
LHS[nx+B_gov_ncp_ind]  = B_gov_ncp
LHS[nx+T_ind]          = T
LHS[nx+LT_ind]         = LT
LHS[nx+G_ind]          = G
LHS[nx+Lambda_ind]     = LAMBDA
LHS[nx+pi_ind]         = PI
LHS[nx+V_1_ind]        = V_1
LHS[nx+V_2_ind]        = V_2
LHS[nx .+ J_ind]       = J
LHS[nx+r_l_1_ind]      = r_l_1
LHS[nx+r_l_2_ind]      = r_l_2
LHS[nx+v_ind]          = v
LHS[nx+Y_ind]          = Y
LHS[nx+Profit_ind]     = PROFIT
LHS[nx+r_k_ind]        = R_K
LHS[nx+r_a_ind]        = R_A
LHS[nx+MC_ind]         = MC
LHS[nx+n_1_ind]        = n_1
LHS[nx+n_2_ind]        = n_2
LHS[nx+M_1_ind]        = M_1
LHS[nx+M_2_ind]        = M_2
LHS[nx+f_1_ind]        = f_1
LHS[nx+f_2_ind]        = f_2
LHS[nx+zz_ind]         = zz
LHS[nx+xx_ind]         = xx
LHS[nx+vv_ind]         = vv
LHS[nx+ee_ind]         = ee
LHS[nx+C_b_ind]        = C_b
LHS[nx+Profit_FI_ind]  = Profit_FI
LHS[nx+RRa_ind]        = RRa
LHS[nx+RR_ind]         = RR
LHS[nx+I_ind]          = Inv
LHS[nx+x_k_ind]        = x_k
LHS[nx+A_g_obs_ind]    = A_g_obs
LHS[nx+Y_obs_ind]      = exp(Control[Y_obs_ind])
LHS[nx+C_obs_ind]      = exp(Control[C_obs_ind])
LHS[nx+I_obs_ind]      = exp(Control[I_obs_ind])
LHS[nx+w_1_obs_ind]    = exp(Control[w_1_obs_ind])
LHS[nx+w_2_obs_ind]    = exp(Control[w_2_obs_ind])
LHS[nx+w_obs_ind]      = exp(Control[w_obs_ind])
LHS[nx+PROFIT_obs_ind] = exp(Control[PROFIT_obs_ind])
LHS[nx+unemp_1_obs_ind]= exp(Control[unemp_1_obs_ind])
LHS[nx+unemp_2_obs_ind]= exp(Control[unemp_2_obs_ind])
LHS[nx+unemp_obs_ind]  = exp(Control[unemp_obs_ind])
LHS[nx+inf_obs_ind]    = exp(Control[inf_obs_ind])
LHS[nx+R_obs_ind]      = exp(Control[R_obs_ind])
LHS[nx+pastw_1_ind]    = exp(Control[pastw_1_ind])
LHS[nx+pastw_2_ind]    = exp(Control[pastw_2_ind])
LHS[nx+pastw_ind]      = exp(Control[pastw_ind])
LHS[nx+G_obs_ind]      = exp(Control[G_obs_ind])
LHS[nx+pastYY_ind]     = exp(Control[pastYY_ind])
LHS[nx+pastCC_ind]     = exp(Control[pastCC_ind])
LHS[nx+pastII_ind]     = exp(Control[pastII_ind])
LHS[nx+pastPPROFIT_ind]= exp(Control[pastPPROFIT_ind])
LHS[nx+pastuu_1_ind]   = exp(Control[pastuu_1_ind])
LHS[nx+pastuu_2_ind]   = exp(Control[pastuu_2_ind])
LHS[nx+pastuu_ind]     = exp(Control[pastuu_ind])
LHS[nx+pastGG_ind]     = exp(Control[pastGG_ind])
LHS[nx+pastA_g_ind]    = exp(Control[pastA_g_ind])
LHS[nx+l_lambda_1_ind] = l_lambda_1
LHS[nx+l_lambda_2_ind] = l_lambda_2
LHS[nx+pastQ_ind]      = pastQ
LHS[nx+x_I_ind]        = x_I
LHS[nx+eta2_ind]       = eta2
LHS[nx+iota2_ind]      = iota2
LHS[nx+pastLT2_ind]    = pastLT2
LHS[nx+pastG2_ind]     = pastG2
LHS[nx+LT_obs_ind]     = LT_obs
LHS[nx+B_gov_ncp2_ind] = B_gov_ncp2

# LHS state entries (t-period values stored as state carry-forwards)
LHS[marginal_b_ind]  = Distribution[1:nb-1]
LHS[marginal_a_ind]  = Distribution[nb .+ (1:na-1)]
LHS[marginal_se_ind] = Distribution[(nb+na) .+ (1:nse-1)]
LHS[COP_ind]         = cop_coefs

LHS[R_cb_ind]        = R_cb
LHS[w_1_ind]         = W_1
LHS[w_2_ind]         = W_2
LHS[w_ind]           = W
LHS[A_b_ind]         = A_bnext
LHS[B_b_ind]         = B_bnext
LHS[A_g_ind]         = A_gauxnext
LHS[Q_ind]           = Q
LHS[lev_ind]         = lev
LHS[NW_b_ind]        = NW_b
LHS[R_tilde_ind]     = R_tilde
LHS[x_cb_ind]        = x_cb
LHS[pastpi_ind]      = pastpi
LHS[pastY_ind]       = pastY
LHS[pastC_ind]       = pastC
LHS[pastI_ind]       = pastI
LHS[pastPROFIT_ind]  = pastPROFIT
LHS[pastunemp_1_ind] = pastunemp_1
LHS[pastunemp_2_ind] = pastunemp_2
LHS[pastunemp_ind]   = pastunemp
LHS[pastG_ind]       = pastG
LHS[pastLT_ind]      = pastLT
LHS[R_star_ind]      = R_star
LHS[ZZ_1_ind]        = ZZ_1
LHS[ZZ_2_ind]        = ZZ_2
LHS[ZZ_3_ind]        = ZZ_3
LHS[ZZ_4_ind]        = ZZ_4
LHS[B_F_ind]         = B_Fnext
LHS[Z_ind]           = Z
LHS[PSI_RP_ind]      = PSI_RP
LHS[eta_ind]         = eta
LHS[D_ind]           = D
LHS[GG_ind]          = GG
LHS[iota_ind]        = iota
LHS[BB_ind]          = BB
LHS[PSI_W_ind]       = PSI_W
LHS[MP_ind]          = MP
LHS[p_mark_ind]      = p_mark
LHS[eps_1_ind]       = eps_1next
LHS[eps_2_ind]       = eps_2next
LHS[eps_3_ind]       = eps_3next
LHS[eps_4_ind]       = eps_4next
LHS[eps_QE_ind]      = eps_QEnext
LHS[eps_RP_ind]      = eps_RPnext
LHS[eps_B_F_ind]     = eps_B_Fnext
LHS[eps_BB_ind]      = eps_BBnext
LHS[eps_Z_ind]       = eps_Znext
LHS[eps_G_ind]       = eps_Gnext
LHS[eps_D_ind]       = eps_Dnext
LHS[eps_R_ind]       = eps_Rnext
LHS[eps_iota_ind]    = eps_iotanext
LHS[eps_eta_ind]     = eps_etanext
LHS[eps_w_ind]       = eps_wnext

## ---- Exponentiate log-level state variables ----

R_cb     = exp(R_cb);       R_cbminus     = exp(R_cbminus)
W_1      = exp(W_1);        W_1minus      = exp(W_1minus)
W_2      = exp(W_2);        W_2minus      = exp(W_2minus)
W        = exp(W);          Wminus        = exp(Wminus)
A_bnext  = exp(A_bnext);    A_b           = exp(A_b)
B_bnext  = exp(B_bnext);    B_b           = exp(B_b)
A_gaux   = exp(A_gaux);     A_gauxnext    = exp(A_gauxnext)
Q        = exp(Q);          Qminus        = exp(Qminus)
lev      = exp(lev);        levminus      = exp(levminus)
NW_b     = exp(NW_b);       NW_bminus     = exp(NW_bminus)
R_tilde  = exp(R_tilde);    R_tildeminus  = exp(R_tildeminus)
x_cb     = exp(x_cb);       x_cbminus     = exp(x_cbminus)
pastpi   = exp(pastpi);     pastpiminus   = exp(pastpiminus)
pastY    = exp(pastY);      pastYminus    = exp(pastYminus)
pastC    = exp(pastC);      pastCminus    = exp(pastCminus)
pastI    = exp(pastI);      pastIminus    = exp(pastIminus)
pastPROFIT  = exp(pastPROFIT);   pastPROFITminus  = exp(pastPROFITminus)
pastunemp_1 = exp(pastunemp_1);  pastunemp_1minus = exp(pastunemp_1minus)
pastunemp_2 = exp(pastunemp_2);  pastunemp_2minus = exp(pastunemp_2minus)
pastunemp   = exp(pastunemp);    pastunempminus   = exp(pastunempminus)
pastG    = exp(pastG);      pastGminus    = exp(pastGminus)
pastLT   = exp(pastLT);     pastLTminus   = exp(pastLTminus)
ZZ_1     = exp(ZZ_1);       ZZ_1minus     = exp(ZZ_1minus)
ZZ_2     = exp(ZZ_2);       ZZ_2minus     = exp(ZZ_2minus)
ZZ_3     = exp(ZZ_3);       ZZ_3minus     = exp(ZZ_3minus)
ZZ_4     = exp(ZZ_4);       ZZ_4minus     = exp(ZZ_4minus)
B_Fnext  = exp(B_Fnext);    B_F           = exp(B_F)
R_star   = exp(R_star);     R_starminus   = exp(R_starminus)
Z        = exp(Z);          Zminus        = exp(Zminus)
PSI_RP   = exp(PSI_RP);     PSI_RPminus   = exp(PSI_RPminus)
eta      = exp(eta);        etaminus      = exp(etaminus)
D        = exp(D);          Dminus        = exp(Dminus)
GG       = exp(GG);         GGminus       = exp(GGminus)
iota     = exp(iota);       iotaminus     = exp(iotaminus)
BB       = exp(BB);         BBminus       = exp(BBminus)
PSI_W    = exp(PSI_W);      PSI_Wminus    = exp(PSI_Wminus)
MP       = exp(MP);         MPminus       = exp(MPminus)
p_mark   = exp(p_mark);     p_markminus   = exp(p_markminus)

## ---- Copula reconstruction (identical to QE Julia) ----

nb_cop  = Int(grid["nb_copula"])
na_cop  = Int(grid["na_copula"])
nse_cop = Int(grid["nse_copula"])

thetaD    = uncompress(Float64.(grid["compressionIndexesCOP"]), cop_coefs,   DCD, IDCD)
COP_Dev   = reshape(thetaD,   (nb_cop, na_cop, nse_cop))
COP_Dev   = cumsum(cumsum(cumsum(COP_Dev; dims=1); dims=2); dims=3)

thetaD_m  = uncompress(Float64.(grid["compressionIndexesCOP"]), cop_coefs_m, DCD, IDCD)
COP_Dev_m = reshape(thetaD_m, (nb_cop, na_cop, nse_cop))
COP_Dev_m = cumsum(cumsum(cumsum(COP_Dev_m; dims=1); dims=2); dims=3)

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

COP_SS = SS_stats["COP_SS"]

s_m_b  = grid["copula_marginal_b"][:]
s_m_a  = grid["copula_marginal_a"][:]
s_m_se = grid["copula_marginal_se"][:]

Ubq = Vector{Float64}(CDF_b_m[:])
Uaq = Vector{Float64}(CDF_a_m[:])
Usq = Vector{Float64}(CDF_se_m[:])

C_SS_atU = myinterpolate3(Vector{Float64}(CDF_b_SS[:]), Vector{Float64}(CDF_a_SS[:]), Vector{Float64}(CDF_se_SS[:]), COP_SS,    Ubq, Uaq, Usq)
# dC_atU is the time-varying copula deviation and is intentionally unused in
# fixed-copula dynamics.
# dC_atU = myinterpolate3(Vector{Float64}(s_m_b[:]), Vector{Float64}(s_m_a[:]),
#                         Vector{Float64}(s_m_se[:]), COP_Dev_m, Ubq, Uaq, Usq)

# Fixed-copula dynamics: keep the steady-state copula and update only the
# marginal CDFs.  This is the fixed-copula path used for dynamic exercises;
# the time-varying copula deviation is deliberately not added here.
CDF_joint_m = C_SS_atU
PDF_joint_m = cdf_to_pdf3D_forwarddiff(CDF_joint_m)

A_gnext = A_gauxnext
A_g     = A_gaux

MU = PDF_joint_m

MU_b  = dropdims(sum(MU; dims=(2,3)); dims=(2,3))
MU_a  = dropdims(sum(MU; dims=(1,3)); dims=(1,3))'
MU_se = dropdims(sum(MU; dims=(1,2)); dims=(1,2))

Eshare = param["Eshare"]

meshes = Dict{String,Any}()
meshes["b"], meshes["a"], meshes["se"] = ndgrid(grid["b"], grid["a"], grid["se"])

## ---- Two-sector employment transition P_SS3_aux ----
# Layout: [emp_sec1 | emp_sec2 | unemp_sec1 | unemp_sec2 | entrepreneur]
# Rows: (ns_1 emp1, ns_2 emp2, ns_1 unemp1, ns_2 unemp2, 1 ent)

id1 = Matrix{Float64}(I, ns_1, ns_1)
id2 = Matrix{Float64}(I, ns_2, ns_2)
z12 = zeros(ns_1, ns_2)
z21 = zeros(ns_2, ns_1)

P_SS3_aux = [
    (1-l_lambda_1+l_lambda_1*f_1)*id1  z12                                  l_lambda_1*(1-f_1)*id1  zeros(ns_1,ns_2)                zeros(ns_1,1);
    z21                                  (1-l_lambda_2+l_lambda_2*f_2)*id2  zeros(ns_2,ns_1)        l_lambda_2*(1-f_2)*id2          zeros(ns_2,1);
    f_1*id1                             z12                                  (1-f_1)*id1             zeros(ns_1,ns_2)                zeros(ns_1,1);
    z21                                  f_2*id2                             zeros(ns_2,ns_1)        (1-f_2)*id2                     zeros(ns_2,1);
    zeros(1, 2*(ns_1+ns_2))                                                                                                            fill(1.0,1,1)
]

H_tilde = kron(Matrix{Float64}(P_SS3_aux'), sparse(Matrix{Float64}(I, nb*na, nb*na)))

MU_tilde    = reshape(reshape(MU, nb*na, nse) * P_SS3_aux, nb, na, nse)
MU_tilde_b  = dropdims(sum(MU_tilde; dims=(2,3)); dims=(2,3))
MU_tilde_a  = dropdims(sum(MU_tilde; dims=(1,3)); dims=(1,3))'
MU_tilde_se = dropdims(sum(MU_tilde; dims=(1,2)); dims=(1,2))

# P_SS3next_aux: transition using next-period finding rates
P_SS3next_aux = [
    (1-l_lambda_1next+l_lambda_1next*f_1next)*id1  z12                                       l_lambda_1next*(1-f_1next)*id1  zeros(ns_1,ns_2)                    zeros(ns_1,1);
    z21                                              (1-l_lambda_2next+l_lambda_2next*f_2next)*id2  zeros(ns_2,ns_1)        l_lambda_2next*(1-f_2next)*id2      zeros(ns_2,1);
    f_1next*id1                                    z12                                         (1-f_1next)*id1              zeros(ns_1,ns_2)                    zeros(ns_1,1);
    z21                                              f_2next*id2                               zeros(ns_2,ns_1)            (1-f_2next)*id2                     zeros(ns_2,1);
    zeros(1, 2*(ns_1+ns_2))                                                                                                                fill(1.0,1,1)
]

P_transition_exp  = param["P_SS2"] * P_SS3next_aux
P_transition_dist = param["P_SS2"]

## ---- MU_Ent: entrepreneur mass ----

_, _, meshes["se3_aux"] = ndgrid(grid["b"], grid["a"], grid["se3_aux"])
MU_Ent         = (meshes["se3_aux"] .== 1) .* MU
MU_tilde_Ent   = H_tilde * MU_Ent[:]
MU_tilde_Ent   = reshape(MU_tilde_Ent, nb, na, nse)

## ---- Income grids (two-sector) ----

param["beta_aux"] = param["beta"]

R_A_aux = R_A + param["a_a_aux"] *
          (param["death_rate"] / (1 - param["death_rate"])) * Q

auxWW1 = ones(nb, na, nse)
auxWW1[:, :, ns+1:end] .= 0.0
auxWW2 = ones(nb, na, nse)
auxWW2[:, :, 1:ns]  .= 0.0
auxWW2[:, :, end]   .= 0.0

# NW: net labour income per unit of idiosyncratic productivity
coeff = (1 / param["xi"] + tau_P) / (1 + 1 / param["xi"]) * param["GHH"]
NW_pretax = coeff * n_2 * W_2 .* ones(nb, na, nse)
NW_pretax[:, :, 1:ns_1] .= coeff * n_1 * W_1
NW_pretax[:, :, ns+1:ns+ns_1] .= coeff * n_1 * W_1
NW_pretax[:, :, ns+ns_1+1:ns+ns_1+ns_2] .= coeff * n_2 * W_2
NW_pretax[:, :, end] .= (param["Eratio"] * PROFIT) / Eshare

NW = coeff * (n_2 * W_2)^(1 - tau_P) .* ones(nb, na, nse)
NW[:, :, 1:ns_1] .= coeff * (n_1 * W_1)^(1 - tau_P)
NW[:, :, ns+1:ns+ns_1] .= coeff * (n_1 * W_1)^(1 - tau_P)
NW[:, :, ns+ns_1+1:ns+ns_1+ns_2] .= coeff * (n_2 * W_2)^(1 - tau_P)
WW = (1 - tau_L) .* (NW .* auxWW1 .+ NW .* auxWW2)
WW[:, :, end] = (1 - tau_L) .* ((param["Eratio"] * PROFIT) / Eshare)^(1 - tau_P) .* ones(nb, na)

inc = Dict{String,Any}()
inc["labor"]    = WW .* meshes["se"].^(1 - tau_P)
inc["dividend"] = meshes["a"] .* R_A_aux
inc["capital"]  = meshes["a"] .* Q
inc["bond"]     = (R_cbminus / PI) .* meshes["b"] .+
                  (meshes["b"] .< 0) .* (param["Rprem"] / PI) .* meshes["b"]
inc["bond"]     = inc["bond"] ./ (1 - param["death_rate"]) .* param["b_a_aux"]
inc["transfer"] = (LT + C_b) / (1 + param["frac_b"]) .* ones(nb, na, nse)

## ---- Household block (EGM policies + value functions) ----

EVa = reshape(reshape(Vanext, nb*na, nse) * P_transition_exp', nb, na, nse)
R_tildeaux = R_tilde / PInext .+ (meshes["b"] .< 0) .* (param["Rprem"] / PInext)
EVb = reshape(reshape(R_tildeaux[:] / (1 - param["death_rate"]) * param["b_a_aux"] .* mutil_cnext,
              nb*na, nse) * P_transition_exp', nb, na, nse)

c_a_star, b_a_star, a_a_star, c_n_star, b_n_star =
    policies_update(EVb, EVa, Q, PI, R_cbminus, 1, 1, inc, meshes, grid, param)

meshaux = copy(meshes)
_, _, meshaux["se_aux"] = ndgrid(grid["b"], grid["a"], 1:nse)

EV   = reshape(VALUEnext, nb*na, nse) * P_transition_exp'
EV3  = reshape(EV, nb, na, nse)
b_grid  = vec(meshaux["b"][:, 1, 1])
a_grid  = vec(meshaux["a"][1, :, 1])
se_grid = vec(meshaux["se_aux"][1, 1, :])
itp         = interpolate((b_grid, a_grid, se_grid), EV3, Gridded(Linear()))
VALUE_next  = extrapolate(itp, Line())

V_adjust   = util(c_a_star) .+ param["beta_aux"] * (1 - param["death_rate"]) .*
             VALUE_next.(b_a_star, a_a_star, meshaux["se_aux"])
V_noadjust = util(c_n_star) .+ param["beta_aux"] * (1 - param["death_rate"]) .*
             VALUE_next.(b_n_star, meshaux["a"], meshaux["se_aux"])

AProb = clamp.(1.0 ./ (1.0 .+ exp.(.-(V_adjust .- V_noadjust .- param["mu_chi"]) ./
        param["sigma_chi"])), 1e-6, 1-1e-6)
AC    = param["sigma_chi"] .* ((1.0 .- vec(AProb)) .* log.(1.0 .- vec(AProb)) .+
        vec(AProb) .* log.(vec(AProb))) .+
        param["mu_chi"] .* vec(AProb) .-
        (param["mu_chi"] - param["sigma_chi"] * log(1 + exp(param["mu_chi"] / param["sigma_chi"])))
AProb = AProb .* param["nu"]
AC    = AC    .* param["nu"]

VALUEaux = vec(AProb) .* vec(V_adjust) .+ (1.0 .- vec(AProb)) .* vec(V_noadjust) .- vec(AC)

mutil_c_n   = reshape(mutil(vec(c_n_star)), nb, na, nse)
mutil_c_a   = reshape(mutil(vec(c_a_star)), nb, na, nse)
mutil_c_aux = AProb .* mutil_c_a .+ (1.0 .- AProb) .* mutil_c_n

mutil_cnext_aux = reshape(reshape(mutil_cnext, nb*na, nse) * P_transition_exp', nb, na, nse)
itp2 = interpolate((b_grid, a_grid, se_grid), mutil_cnext_aux, Gridded(Linear()))
mutil_cnext_itp = extrapolate(itp2, Line())

MRS_a_aux = param["beta_aux"] * (1 - param["death_rate"]) .*
            mutil_cnext_itp.(b_a_star, a_a_star, meshaux["se_aux"]) ./ mutil_c_a .*
            AProb .* MU_tilde
MRS_n_aux = param["beta_aux"] * (1 - param["death_rate"]) .*
            mutil_cnext_itp.(b_n_star, meshaux["a"], meshaux["se_aux"]) ./ mutil_c_n .*
            (1.0 .- AProb) .* MU_tilde

EVa3 = reshape(EVa, nb, na, nse)
itp3 = interpolate((b_grid, a_grid, se_grid), EVa3, Gridded(Linear()))
Va_next = extrapolate(itp3, Line())

Va_aux = AProb .* (R_A_aux .+ Q) .* mutil_c_a .+
         (1.0 .- AProb) .* R_A_aux .* mutil_c_n .+
         param["beta_aux"] * (1 - param["death_rate"]) .* (1.0 .- AProb) .*
         Va_next.(b_n_star, meshaux["a"], meshaux["se_aux"])

## ---- Distribution transition matrices ----

weight11 = zeros(nb*na, nse, nse)
weight12 = zeros(nb*na, nse, nse)
weight21 = zeros(nb*na, nse, nse)
weight22 = zeros(nb*na, nse, nse)

Dist_b_a, idb_a = genweight(b_a_star, grid["b"])
Dist_b_n, idb_n = genweight(b_n_star, grid["b"])
Dist_a_a, ida_a = genweight(a_a_star, grid["a"])
Dist_a_n, ida_n = genweight(meshaux["a"], grid["a"])
Dist_b_die, idb_die = genweight(zeros(nb, na, nse), grid["b"])
Dist_a_die, ida_die = genweight(zeros(nb, na, nse), grid["a"])

idb_a_rep = repeat(vec(idb_a), 1, nse)
ida_a_rep = repeat(vec(ida_a), 1, nse)
idse_a    = vec(kron(1:nse, ones(Int, 1, nb*na*nse))')'

index11 = sub2ind((nb,na,nse), vec(idb_a_rep), vec(ida_a_rep),       vec(idse_a))
index12 = sub2ind((nb,na,nse), vec(idb_a_rep), vec(ida_a_rep) .+ 1,  vec(idse_a))
index21 = sub2ind((nb,na,nse), vec(idb_a_rep) .+ 1, vec(ida_a_rep),  vec(idse_a))
index22 = sub2ind((nb,na,nse), vec(idb_a_rep) .+ 1, vec(ida_a_rep) .+ 1, vec(idse_a))

for hh in 1:nse
    w11 = vec(1.0 .- Dist_b_a[:,:,hh]) .* vec(1.0 .- Dist_a_a[:,:,hh])
    w12 = vec(1.0 .- Dist_b_a[:,:,hh]) .* vec(Dist_a_a[:,:,hh])
    w21 = vec(Dist_b_a[:,:,hh])        .* vec(1.0 .- Dist_a_a[:,:,hh])
    w22 = vec(Dist_b_a[:,:,hh])        .* vec(Dist_a_a[:,:,hh])
    weight11[:,:,hh] = w11 * P_transition_dist[hh,:]'
    weight12[:,:,hh] = w12 * P_transition_dist[hh,:]'
    weight21[:,:,hh] = w21 * P_transition_dist[hh,:]'
    weight22[:,:,hh] = w22 * P_transition_dist[hh,:]'
end
weight11 = permutedims(weight11, (1,3,2))
weight12 = permutedims(weight12, (1,3,2))
weight21 = permutedims(weight21, (1,3,2))
weight22 = permutedims(weight22, (1,3,2))
rowindex = repeat(collect(1:nb*na*nse), 4*nse)
TT_a = sparse(vec(rowindex), [vec(index11); vec(index21); vec(index12); vec(index22)],
              [vec(weight11); vec(weight21); vec(weight12); vec(weight22)], nb*na*nse, nb*na*nse)

weight11 .= 0; weight12 .= 0; weight21 .= 0; weight22 .= 0

idb_n_rep = repeat(vec(idb_n), 1, nse)
ida_n_rep = repeat(vec(ida_n), 1, nse)
idse_n    = vec(kron(1:nse, ones(Int, 1, nb*na*nse))')'

index11 = sub2ind((nb,na,nse), vec(idb_n_rep), vec(ida_n_rep),       vec(idse_n))
index12 = sub2ind((nb,na,nse), vec(idb_n_rep), vec(ida_n_rep) .+ 1,  vec(idse_n))
index21 = sub2ind((nb,na,nse), vec(idb_n_rep) .+ 1, vec(ida_n_rep),  vec(idse_n))
index22 = sub2ind((nb,na,nse), vec(idb_n_rep) .+ 1, vec(ida_n_rep) .+ 1, vec(idse_n))

for hh in 1:nse
    w11 = vec(1.0 .- Dist_b_n[:,:,hh]) .* vec(1.0 .- Dist_a_n[:,:,hh])
    w12 = vec(1.0 .- Dist_b_n[:,:,hh]) .* vec(Dist_a_n[:,:,hh])
    w21 = vec(Dist_b_n[:,:,hh])        .* vec(1.0 .- Dist_a_n[:,:,hh])
    w22 = vec(Dist_b_n[:,:,hh])        .* vec(Dist_a_n[:,:,hh])
    weight11[:,:,hh] = w11 * P_transition_dist[hh,:]'
    weight12[:,:,hh] = w12 * P_transition_dist[hh,:]'
    weight21[:,:,hh] = w21 * P_transition_dist[hh,:]'
    weight22[:,:,hh] = w22 * P_transition_dist[hh,:]'
end
weight11 = permutedims(weight11, (1,3,2))
weight12 = permutedims(weight12, (1,3,2))
weight21 = permutedims(weight21, (1,3,2))
weight22 = permutedims(weight22, (1,3,2))
TT_n = sparse(vec(rowindex), [vec(index11); vec(index21); vec(index12); vec(index22)],
              [vec(weight11); vec(weight21); vec(weight12); vec(weight22)], nb*na*nse, nb*na*nse)

weight11 .= 0; weight12 .= 0; weight21 .= 0; weight22 .= 0

idb_d_rep = repeat(vec(idb_die), 1, nse)
ida_d_rep = repeat(vec(ida_die), 1, nse)
idse_d    = vec(kron(1:nse, ones(Int, 1, nb*na*nse))')'

index11 = sub2ind((nb,na,nse), vec(idb_d_rep), vec(ida_d_rep),       vec(idse_d))
index12 = sub2ind((nb,na,nse), vec(idb_d_rep), vec(ida_d_rep) .+ 1,  vec(idse_d))
index21 = sub2ind((nb,na,nse), vec(idb_d_rep) .+ 1, vec(ida_d_rep),  vec(idse_d))
index22 = sub2ind((nb,na,nse), vec(idb_d_rep) .+ 1, vec(ida_d_rep) .+ 1, vec(idse_d))

for hh in 1:nse
    w11 = vec(1.0 .- Dist_b_die[:,:,hh]) .* vec(1.0 .- Dist_a_die[:,:,hh])
    w12 = vec(1.0 .- Dist_b_die[:,:,hh]) .* vec(Dist_a_die[:,:,hh])
    w21 = vec(Dist_b_die[:,:,hh])         .* vec(1.0 .- Dist_a_die[:,:,hh])
    w22 = vec(Dist_b_die[:,:,hh])         .* vec(Dist_a_die[:,:,hh])
    weight11[:,:,hh] = w11 * P_transition_dist[hh,:]'
    weight12[:,:,hh] = w12 * P_transition_dist[hh,:]'
    weight21[:,:,hh] = w21 * P_transition_dist[hh,:]'
    weight22[:,:,hh] = w22 * P_transition_dist[hh,:]'
end
weight11 = permutedims(weight11, (1,3,2))
weight12 = permutedims(weight12, (1,3,2))
weight21 = permutedims(weight21, (1,3,2))
weight22 = permutedims(weight22, (1,3,2))
TT_die = sparse(vec(rowindex), [vec(index11); vec(index21); vec(index12); vec(index22)],
                [vec(weight11); vec(weight21); vec(weight12); vec(weight22)], nb*na*nse, nb*na*nse)

n_aprob = length(vec(AProb))
APD_a   = sparse(1:n_aprob, 1:n_aprob, vec(AProb))
APD_n   = sparse(1:n_aprob, 1:n_aprob, 1.0 .- vec(AProb))
TT = (1 - param["death_rate"]) * (APD_a * TT_a + APD_n * TT_n) +
      param["death_rate"] * TT_die
TT = H_tilde' * TT

MUnext = vec(MU)' * TT
MUnext = reshape(vec(MUnext), nb, na, nse)

## ---- Copula compression for next period ----

PDF_joint   = MUnext
C_next_grid = cumsum(cumsum(cumsum(PDF_joint; dims=1); dims=2); dims=3)

CDF_b_vec    = Vector{Float64}(CDF_b)
CDF_a_vec    = Vector{Float64}(CDF_a)
CDF_se_vec   = Vector{Float64}(CDF_se)
CDF_b_SS_vec = Vector{Float64}(CDF_b_SS)
CDF_a_SS_vec = Vector{Float64}(CDF_a_SS)
CDF_se_SS_vec= Vector{Float64}(CDF_se_SS)
s_m_b_vec    = Vector{Float64}(s_m_b)
s_m_a_vec    = Vector{Float64}(s_m_a)
s_m_se_vec   = Vector{Float64}(s_m_se)

CDF_Dev = myinterpolate3(CDF_b_vec, CDF_a_vec, CDF_se_vec, C_next_grid, s_m_b_vec, s_m_a_vec, s_m_se_vec) .-
          myinterpolate3(CDF_b_SS_vec, CDF_a_SS_vec, CDF_se_SS_vec, COP_SS, s_m_b_vec, s_m_a_vec, s_m_se_vec)
COP_dev_next = cdf_to_pdf3D_forwarddiff(CDF_Dev)
COP_thet     = compress(grid["compressionIndexesCOP"], COP_dev_next, DCD, IDCD)

RHS[COP_ind] = COP_thet

## ---- RHS: marginal distributions ----

aux_b  = dropdims(sum(sum(MUnext; dims=2); dims=3); dims=(2,3))
aux_a  = dropdims(sum(sum(MUnext; dims=1); dims=3); dims=(1,3))'
aux_se = dropdims(sum(sum(MUnext; dims=1); dims=2); dims=(1,2))

RHS[marginal_b_ind]  = aux_b[1:end-1]
RHS[marginal_a_ind]  = aux_a[1:end-1]
RHS[marginal_se_ind] = aux_se[1:end-1]

## ---- RHS: aggregate state equations ----

# Taylor rule (CSC uses only Taylor; no QE regime switch)
RHS[R_cb_ind]   = log(param["R_cb"]) + (1-param["rho_R"]) *
                  (param["phi_pi"]*log(PI/param["pi_cb"]) - param["phi_u"]*(unemp - SS_stats["u"])) +
                  param["rho_R"] * log(R_cbminus/param["R_cb"]) + log(MP)
RHS[R_star_ind] = log(param["R_cb"]) + (1-param["rho_R"]) *
                  (param["phi_pi"]*log(PI/param["pi_cb"]) - param["phi_u"]*(unemp - SS_stats["u"])) +
                  param["rho_R"] * log(R_starminus/param["R_cb"]) + log(MP)

# Sector-specific wage Phillips curves
RHS[w_1_ind] = log(param["w_bar_1"]) +
               param["rho_w_1"] * log(W_1minus/param["w_bar_1"]) +
               param["rho_w_1"] * (param["d_1"]*log(param["pi_bar"]/PI) + (1-param["d_1"])*log(pastpiminus/PI)) +
               (1-param["rho_w_1"]) * log(1/ZZ_1 * r_l_1/SS_stats["r_l_1"])
RHS[w_2_ind] = log(param["w_bar_2"]) +
               param["rho_w_2"] * log(W_2minus/param["w_bar_2"]) +
               param["rho_w_2"] * (param["d_2"]*log(param["pi_bar"]/PI) + (1-param["d_2"])*log(pastpiminus/PI)) +
               (1-param["rho_w_2"]) * log(1/PSI_W * r_l_2/SS_stats["r_l_2"])
RHS[w_ind]   = log((L_1*W_1 + L_2*W_2) / (L_1 + L_2))

# Financial intermediary / asset supply state equations
RHS[A_b_ind]    = log(lev * NW_b / Q)
RHS[B_b_ind]    = log(T + B_gov_ncpnext - B_gov_ncp*R_cbminus/PI +
                       (Q + R_A - R_cbminus/PI*Qminus)*A_g -
                       UB - LT - param["tau_cp"]*Q*A_gnext - G +
                       param["b_a_aux2"]*R_cbminus/PI*B_b - C_b)
RHS[A_g_ind]    = log(param["rho_passive_QE"]*A_g + (1-param["rho_passive_QE"])*SS_stats["A_g"])
RHS[Q_ind]      = log(iota2*(1 + param["phi"]*log(x_k) + param["phi"]/2*(log(x_k))^2) -
                       iota2next*LAMBDA/SS_stats["Lambda"]*param["phi"]*log(x_knext)*x_knext)
RHS[lev_ind]    = log(ee / (param["DELTA"] - vv))
RHS[NW_b_ind]   = log((param["theta_b"]*((RRa - param["b_a_aux2"]*R_cbminus/PI)*levminus +
                        param["b_a_aux2"]*R_cbminus/PI)*NW_bminus + param["omega"]*Qminus*A_b))
RHS[R_tilde_ind]= log(R_cb)
RHS[x_cb_ind]   = log(1 + param["rho_X_QE"]*(x_cbminus-1) +
                       (1-param["rho_X_QE"])*(-param["phi_pi_QE"]*log(PI/param["pi_cb"]) +
                       param["phi_u_QE"]*(unemp - SS_stats["u"])))

# Lagged observables carried as states
RHS[pastpi_ind]      = log(PI)
RHS[pastY_ind]       = log(Y)
RHS[pastC_ind]       = log(C)
RHS[pastI_ind]       = log(Inv)
RHS[pastPROFIT_ind]  = log(PROFIT + param["fix2"])
RHS[pastunemp_1_ind] = log(unemp_1)
RHS[pastunemp_2_ind] = log(unemp_2)
RHS[pastunemp_ind]   = log(unemp)
RHS[pastG_ind]       = log(G)
RHS[pastLT_ind]      = log(LT)

# Labour-market wedge shock processes (ZZ_1 mirrors rho_PSI_W; ZZ_2-4 are calibrated 0.9)
RHS[ZZ_1_ind] = param["rho_ZZ_1"] * log(ZZ_1minus) +
                (1 - param["rho_ZZ_1"]) * log(param["ZZ_1"]) + eps_1
RHS[ZZ_2_ind] = param["rho_ZZ_2"] * log(ZZ_2minus) +
                (1 - param["rho_ZZ_2"]) * log(param["ZZ_2"]) + eps_2
RHS[ZZ_3_ind] = param["rho_ZZ_3"] * log(ZZ_3minus) +
                (1 - param["rho_ZZ_3"]) * log(param["ZZ_3"]) + eps_3
RHS[ZZ_4_ind] = param["rho_ZZ_4"] * log(ZZ_4minus) + eps_4

# Exogenous processes
RHS[B_F_ind]    = param["rho_B_F"]   * log(B_F) +
                  (1-param["rho_B_F"]) * log(SS_stats["Y"]/(SS_stats["Y"]-SS_stats["B_F"])) + eps_B_F
RHS[Z_ind]      = param["rho_Z"]      * log(Zminus)      + (1-param["rho_Z"])*log(param["Z"]) + eps_Z
RHS[PSI_RP_ind] = param["rho_PSI_RP"] * log(PSI_RPminus) + eps_RP
RHS[eta_ind]    = log(p_mark / (p_mark - 1))
RHS[D_ind]      = param["rho_D"]      * log(Dminus)      + eps_D

## Code block is if we were using regime-switching
#if string(param["adjust"]) == "G"
    RHS[GG_ind] = param["rho_G"] * log(GGminus) +
                  (1-param["rho_G"]) * log(SS_stats["Y"]/(SS_stats["Y"]-SS_stats["LT"]-SS_stats["C_b"])) + eps_G
#= else
RHS[GG_ind] = param["rho_G"] * log(GGminus) +
              (1-param["rho_G"]) * log(SS_stats["Y"]/(SS_stats["Y"]-SS_stats["G"])) + eps_G
=#
RHS[iota_ind]   = param["rho_iota"]  * log(iotaminus)   + eps_iota
RHS[BB_ind]     = param["rho_BB"]    * log(BBminus)     + eps_BB
RHS[PSI_W_ind]  = param["rho_PSI_W"] * log(PSI_Wminus)  + eps_w
RHS[MP_ind]     = param["rho_MP"]    * log(MPminus)     + eps_R
RHS[p_mark_ind] = param["rho_eta"]   * log(p_markminus) +
                  (1-param["rho_eta"]) * log(param["eta"]/(param["eta"]-1)) + eps_eta

# Shock innovation equations (zero on RHS = innovations are i.i.d.)
RHS[eps_1_ind]     = 0;  RHS[eps_2_ind]    = 0;  RHS[eps_3_ind]  = 0;  RHS[eps_4_ind]  = 0
RHS[eps_QE_ind]    = 0;  RHS[eps_RP_ind]   = 0;  RHS[eps_B_F_ind]= 0;  RHS[eps_BB_ind] = 0
RHS[eps_Z_ind]     = 0;  RHS[eps_G_ind]    = 0;  RHS[eps_D_ind]  = 0;  RHS[eps_R_ind]  = 0
RHS[eps_iota_ind]  = 0;  RHS[eps_eta_ind]  = 0;  RHS[eps_w_ind]  = 0

## ---- RHS: control / aggregate equilibrium conditions ----

RHS[nx .+ VALUE_ind]   = invutil(VALUEaux[:])
RHS[nx .+ mutil_c_ind] = invmutil(mutil_c_aux[:])
RHS[nx .+ Va_ind]      = invmutil(Va_aux[:])

RHS[nx+MRS_ind] = sum(sum(MRS_a_aux[:,:,end] + MRS_n_aux[:,:,end])) / param["Eshare"] * param["Lambda_b_aux"]

RHS[nx+A_hh_ind] = sum(grid["a"] .* marginal_aminus)
RHS[nx+B_hh_ind] = sum(grid["b"] .* marginal_bminus) / (1 - param["death_rate"]) * param["b_a_aux"]
RHS[nx+C_ind]    = q_cons(c_a_star, c_n_star, param["w_bar_1"], param["w_bar_2"],
                          n_1, n_2, tau_L, tau_P, AProb, MU_tilde, meshes, param, grid)

RHS[nx+N_tilde_1_ind] = sum(MU_tilde_se[1:ns_1])
RHS[nx+N_tilde_2_ind] = sum(MU_tilde_se[ns_1+1:ns_1+ns_2])

RHS[nx+L_1_ind] = N_tilde_1 * grid["se_bar_1"] * n_1
RHS[nx+L_2_ind] = N_tilde_2 * grid["se_bar_2"] * n_2

RHS[nx+UB_ind] = sum(vec(NW_pretax[1, 1, ns+1:ns+ns]) .* grid["se"][ns+1:ns+ns] .* MU_tilde_se[ns+1:ns+ns])

# P_SS4_aux: transition with f=0 (for mass-of-seekers calculation)
P_SS4_aux = [
    (1-l_lambda_1)*id1  z12          l_lambda_1*id1    zeros(ns_1,ns_2)  zeros(ns_1,1);
    z21                  (1-l_lambda_2)*id2  zeros(ns_2,ns_1)  l_lambda_2*id2    zeros(ns_2,1);
    zeros(ns_1,ns_1)    z12          id1               zeros(ns_1,ns_2)  zeros(ns_1,1);
    z21                  zeros(ns_2,ns_2)  zeros(ns_2,ns_1)  id2               zeros(ns_2,1);
    zeros(1,2*(ns_1+ns_2))             fill(1.0,1,1)
]
MU_se_aux = P_SS4_aux' * MU_se

RHS[nx+unemp_1_ind] = sum(MU_tilde_se[ns+1:ns+ns_1]) /
                      (sum(MU_tilde_se[1:ns_1]) + sum(MU_tilde_se[ns+1:ns+ns_1]))
RHS[nx+unemp_2_ind] = sum(MU_tilde_se[ns+ns_1+1:ns+ns_1+ns_2]) /
                      (sum(MU_tilde_se[ns_1+1:ns_1+ns_2]) + sum(MU_tilde_se[ns+ns_1+1:ns+ns_1+ns_2]))
RHS[nx+unemp_ind]   = 1 - sum(MU_tilde_se[1:ns]) /
                      (sum(MU_tilde_se[1:ns]) + sum(MU_tilde_se[ns+1:ns+ns_1+ns_2]))

RHS[nx+U_1_ind] = sum(MU_se_aux[ns+1:ns+ns_1])
RHS[nx+U_2_ind] = sum(MU_se_aux[ns+ns_1+1:ns+ns_1+ns_2])

RHS[nx+K_ind]         = A_hh + A_g + A_b + SS_stats["A_F"]
RHS[nx+B_ind]         = B_hh + B_b + (1 - 1/1)*SS_stats["Y"]
RHS[nx+B_gov_ncp_ind] = B_b + B_hh + (1-1/1)*SS_stats["Y"] -
                        (Qminus*A_b - NW_bminus) - Qminus*(1+param["tau_cp"])*A_g

Lincome_pretax = sum(vec(NW_pretax[1, 1, 1:ns]) .* grid["se"][1:ns] .* MU_tilde_se[1:ns])
UB_pretax = RHS[nx+UB_ind]
Bincome_pretax = param["Eratio"] * PROFIT
Lincome_aftertax = sum(vec(WW[1, 1, 1:ns]) .* grid["se"][1:ns].^(1 - tau_P) .* MU_tilde_se[1:ns])
UB_aftertax = sum(vec(WW[1, 1, ns+1:ns+ns]) .* grid["se"][ns+1:ns+ns].^(1 - tau_P) .* MU_tilde_se[ns+1:ns+ns])
Bincome_aftertax = Eshare * WW[1, 1, end]
RHS[nx+Income_Tax_ind] = Lincome_pretax - Lincome_aftertax + UB_pretax - UB_aftertax + Bincome_pretax - Bincome_aftertax
RHS[nx+T_ind] = Income_Tax + tau_a * (1 - param["Eratio"]) * PROFIT

#if string(param["adjust"]) == "G"
    RHS[nx+LT_ind] = (1 - 1/GG)*SS_stats["Y"] - SS_stats["C_b"]
    RHS[nx+G_ind]  = SS_stats["G"]
#=else
    RHS[nx+G_ind]  = (1 - 1/GG)*SS_stats["Y"]
    RHS[nx+LT_ind] = SS_stats["LT"]
end=#

RHS[nx+Lambda_ind] = MRS * param["Lambda_aux"] / param["Lambda_b_aux"]
RHS[nx+pi_ind]     = param["pi_cb"] *
                     exp((param["rho_B"]/(param["rho_B"]+param["gamma_pi"])) *
                         log(B_gov_ncp*R_cbminus/(SS_stats["B_gov_ncp"]*param["R_cb"])) -
                         (param["gamma_T"]/(param["rho_B"]+param["gamma_pi"])) * log(T/SS_stats["T"]) -
                         1/(param["rho_B"]+param["gamma_pi"]) * log(B_gov_ncpnext/SS_stats["B_gov_ncp"]))

# Job value equations (one per skill type)
J_inst_aux     = vcat((r_l_1 - param["fix_L_1"] - W_1) * n_1 * grid["s"][1:ns_1],
                      (r_l_2 - param["fix_L_2"] - W_2) * n_2 * grid["s"][ns_1+1:ns_1+ns_2])
lambdanext_aux = [  (1-l_lambda_1next)*id1   zeros(ns_1,ns_2);
                     zeros(ns_2,ns_1)          (1-l_lambda_2next)*id2  ]

RHS[nx .+ J_ind] = J_inst_aux .+
                   LAMBDA * (1 - param["death_rate"]) * (1 - param["in"]) *
                   param["P_SS"] * lambdanext_aux * Jnext

# Vacancy posting values
RHS[nx+J_bar_1_ind] = dot(J[1:ns_1], MU_tilde_se[1:ns_1]) / sum(MU_tilde_se[1:ns_1])
RHS[nx+J_bar_2_ind] = dot(J[ns_1+1:ns_1+ns_2], MU_tilde_se[ns_1+1:ns_1+ns_2]) / sum(MU_tilde_se[ns_1+1:ns_1+ns_2])
RHS[nx+V_1_ind] = M_1 / param["iota_1"] * J_bar_1
RHS[nx+V_2_ind] = M_2 / param["iota_2"] * J_bar_2

# Production (nested CES: unskilled L_1 vs capital-skill composite)
K_tilde = v * K
V_aux   = (param["w"] * (ZZ_3 * K_tilde)^param["rho"] +
           (1 - param["w"]) * (ZZ_2 * L_2)^param["rho"])^(1/param["rho"])
F_aux   = (param["a"] * (ZZ_1 * L_1)^param["zeta"] +
           (1 - param["a"]) * V_aux^param["zeta"])^(1/param["zeta"])

RHS[nx+r_l_1_ind] = MC * Z * F_aux^(1-param["zeta"]) * param["a"] *
                    L_1^(param["zeta"]-1) * ZZ_1^param["zeta"]
RHS[nx+r_l_2_ind] = MC * Z * F_aux^(1-param["zeta"]) * (1-param["a"]) *
                    V_aux^(param["zeta"]-param["rho"]) * (1-param["w"]) *
                    L_2^(param["rho"]-1) * ZZ_2^param["rho"]
RHS[nx+v_ind]     = (R_K / (Q * param["delta_0"] * param["delta_1"]))^(1/(param["delta_1"]-1))
RHS[nx+Y_ind]     = Z * F_aux
RHS[nx+Profit_ind]= (Y * (1 - eta/(2*param["kappa"]) *
                          (log(PI) - (1-param["GAMMA"])*log(param["pi_cb"]) -
                           param["GAMMA"]*log(pastpiminus))^2) +
                     Y * (-MC) - param["fix"] - log(B_Fnext)*SS_stats["Y"] +
                     (r_l_1 - param["fix_L_1"] - W_1)*L_1 +
                     (r_l_2 - param["fix_L_2"] - W_2)*L_2 -
                     param["iota_1"]*V_1 - param["iota_2"]*V_2) +
                    (R_K*v - Q*(param["delta_00"] + param["delta_0"]*v^param["delta_1"]))*K -
                    Q*K + iota2*K + Q*Knext -
                    iota2*(Knext + param["phi"]/2*(log(Knext/K))^2*Knext) +
                    Profit_FI - param["fix2"]

RHS[nx+r_k_ind]   = MC * Z * F_aux^(1-param["zeta"]) * (1-param["a"]) *
                    V_aux^(param["zeta"]-param["rho"]) * param["w"] *
                    K_tilde^(param["rho"]-1) * ZZ_3^param["rho"]
RHS[nx+r_a_ind]   = (1 - tau_a) * (1 - param["Eratio"] - param["b_share"]) * PROFIT / K
RHS[nx+MC_ind]    = 1 - 1/eta +
                    (log(PI/(pastpiminus^param["GAMMA"]*param["pi_bar"]^(1-param["GAMMA"]))) -
                     LAMBDA * eta2next/eta2 * Ynext/Y *
                     log(PInext/(PI^param["GAMMA"]*param["pi_bar"]^(1-param["GAMMA"])))) / param["kappa"]

# Hours / effort per worker
if param["GHH"] == 1
    RHS[nx+n_1_ind] = ((1 - tau_L) * (1 - tau_P) * W_1^(1 - tau_P) / param["psi_1"])^(param["xi"] / (1 + param["xi"] * tau_P))
    RHS[nx+n_2_ind] = ((1 - tau_L) * (1 - tau_P) * W_2^(1 - tau_P) / param["psi_2"])^(param["xi"] / (1 + param["xi"] * tau_P))
else
    RHS[nx+n_1_ind] = 1.0
    RHS[nx+n_2_ind] = 1.0
end

# Matching functions
RHS[nx+M_1_ind] = U_1 * V_1 / ((U_1^param["alpha_1"] + V_1^param["alpha_1"])^(1/param["alpha_1"]))
RHS[nx+M_2_ind] = U_2 * V_2 / ((U_2^param["alpha_2"] + V_2^param["alpha_2"])^(1/param["alpha_2"]))
RHS[nx+f_1_ind] = M_1 / U_1
RHS[nx+f_2_ind] = M_2 / U_2

# Financial intermediary equations
RHS[nx+zz_ind]        = (RRa - param["b_a_aux2"]*R_cbminus/PI)*levminus + param["b_a_aux2"]*R_cbminus/PI
RHS[nx+xx_ind]        = (lev/levminus) * zz
RHS[nx+vv_ind]        = (1-param["theta_b"])*BB*LAMBDA*(RRanext - R_cb/PInext) +
                         param["theta_b"]*BB*LAMBDA*xxnext*vvnext
RHS[nx+ee_ind]        = (1-param["theta_b"])*BB*LAMBDA*(R_cb/PInext) +
                         param["theta_b"]*BB*LAMBDA*zznext*eenext
RHS[nx+C_b_ind]       = ((param["beta_b"]*D*R_cb/PInext /
                          (1 + param["phi_B"]*(B_bnext/B_b - 1)) *
                          C_bnext^(-param["sigma2"]))^(-1/param["sigma2"]))
RHS[nx+Profit_FI_ind] = (1-param["theta_b"]) *
                         ((RRa - param["b_a_aux2"]*R_cbminus/PI)*levminus +
                          param["b_a_aux2"]*R_cbminus/PI) * NW_bminus -
                         param["omega"]*Qminus*A_b
RHS[nx+RRa_ind]       = (Q + R_A) / Qminus
RHS[nx+RR_ind]        = R_cbminus / PI
RHS[nx+I_ind]         = iota*(1 + param["phi"]/2*(log(x_k))^2) *
                         (A_hhnext + A_gnext + A_bnext + SS_stats["A_F"]) -
                         Q * (1 - (param["delta_00"] + param["delta_0"]*v^param["delta_1"])) * K
RHS[nx+x_k_ind]       = Knext / K

# Observable identities
RHS[nx+A_g_obs_ind]      = A_gauxnext
RHS[nx+Y_obs_ind]        = Y
RHS[nx+C_obs_ind]        = C
RHS[nx+I_obs_ind]        = Inv
RHS[nx+w_1_obs_ind]      = W_1
RHS[nx+w_2_obs_ind]      = W_2
RHS[nx+w_obs_ind]        = W
RHS[nx+PROFIT_obs_ind]   = PROFIT + param["fix2"]
RHS[nx+unemp_1_obs_ind]  = 100 * unemp_1
RHS[nx+unemp_2_obs_ind]  = 100 * unemp_2
RHS[nx+unemp_obs_ind]    = 100 * unemp
RHS[nx+inf_obs_ind]      = PI
RHS[nx+R_obs_ind]        = R_cb
RHS[nx+pastw_1_ind]      = W_1minus
RHS[nx+pastw_2_ind]      = W_2minus
RHS[nx+pastw_ind]        = Wminus
RHS[nx+G_obs_ind]        = G
RHS[nx+pastYY_ind]       = pastYminus
RHS[nx+pastCC_ind]       = pastCminus
RHS[nx+pastII_ind]       = pastIminus
RHS[nx+pastPPROFIT_ind]  = pastPROFITminus
RHS[nx+pastuu_1_ind]     = pastunemp_1minus
RHS[nx+pastuu_2_ind]     = pastunemp_2minus
RHS[nx+pastuu_ind]       = pastunempminus
RHS[nx+pastGG_ind]       = pastLTminus
RHS[nx+pastA_g_ind]      = A_g
RHS[nx+l_lambda_1_ind]   = param["lambda_1"]
RHS[nx+l_lambda_2_ind]   = param["lambda_2"]
RHS[nx+pastQ_ind]        = Qminus
RHS[nx+x_I_ind]          = Q
RHS[nx+eta2_ind]         = eta
RHS[nx+iota2_ind]        = iota
RHS[nx+pastLT2_ind]      = pastLTminus
RHS[nx+pastG2_ind]       = pastGminus
RHS[nx+LT_obs_ind]       = LT
RHS[nx+B_gov_ncp2_ind]   = B_b + B_hh - (Qminus*A_b - NW_bminus) - Qminus*(1+param["tau_cp"])*A_g

## ---- Residuals ----

Difference = InvGamma * ((LHS - RHS) ./
             [ones(nx); ControlSS[1:end-oc]; ones(oc)])

MUnext_out = MUnext

return Difference, LHS, RHS, MUnext_out,
       c_a_star, b_a_star, a_a_star, c_n_star, b_n_star,
       AC, AProb, P_transition_exp, P_transition_dist

end
