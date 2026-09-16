function out = update_Jacob_CSC(param,grid,SS_stats,idx,StateSS,ControlSS)

% indexes

R_cb_ind        = idx.R_cb_ind;
w_1_ind         = idx.w_1_ind;
w_2_ind         = idx.w_2_ind;
w_ind           = idx.w_ind;
A_b_ind         = idx.A_b_ind;
B_b_ind         = idx.B_b_ind;
A_g_ind         = idx.A_g_ind;
Q_ind           = idx.Q_ind;
lev_ind         = idx.lev_ind;
NW_b_ind        = idx.NW_b_ind;
R_tilde_ind     = idx.R_tilde_ind;
x_cb_ind        = idx.x_cb_ind;
pastpi_ind      = idx.pastpi_ind;
pastY_ind       = idx.pastY_ind;
pastC_ind       = idx.pastC_ind;
pastI_ind       = idx.pastI_ind;
pastPROFIT_ind  = idx.pastPROFIT_ind;
pastunemp_1_ind = idx.pastunemp_1_ind;
pastunemp_2_ind = idx.pastunemp_2_ind;
pastunemp_ind   = idx.pastunemp_ind;
pastG_ind       = idx.pastG_ind;
pastLT_ind      = idx.pastLT_ind;
R_star_ind      = idx.R_star_ind;
ZZ_1_ind        = idx.ZZ_1_ind;
ZZ_2_ind        = idx.ZZ_2_ind;
ZZ_3_ind        = idx.ZZ_3_ind;
ZZ_4_ind        = idx.ZZ_4_ind;
B_F_ind         = idx.B_F_ind;
Z_ind           = idx.Z_ind;
PSI_RP_ind      = idx.PSI_RP_ind;
eta_ind         = idx.eta_ind;
D_ind           = idx.D_ind;
GG_ind          = idx.GG_ind;
iota_ind        = idx.iota_ind;
BB_ind          = idx.BB_ind;
PSI_W_ind       = idx.PSI_W_ind;
MP_ind          = idx.MP_ind;
p_mark_ind      = idx.p_mark_ind;
eps_1_ind       = idx.eps_1_ind;
eps_2_ind       = idx.eps_2_ind;
eps_3_ind       = idx.eps_3_ind;
eps_4_ind       = idx.eps_4_ind;
eps_QE_ind      = idx.eps_QE_ind;
eps_RP_ind      = idx.eps_RP_ind;
eps_B_F_ind     = idx.eps_B_F_ind;
eps_BB_ind      = idx.eps_BB_ind;
eps_Z_ind       = idx.eps_Z_ind;
eps_G_ind       = idx.eps_G_ind;
eps_D_ind       = idx.eps_D_ind;
eps_R_ind       = idx.eps_R_ind;
eps_iota_ind    = idx.eps_iota_ind;
eps_eta_ind     = idx.eps_eta_ind;
eps_w_ind       = idx.eps_w_ind;

% 1) Summary variables

MRS_ind          = idx.MRS_ind;
A_hh_ind         = idx.A_hh_ind;
B_hh_ind         = idx.B_hh_ind;
C_ind            = idx.C_ind;
N_tilde_1_ind    = idx.N_tilde_1_ind;
N_tilde_2_ind    = idx.N_tilde_2_ind;
L_1_ind          = idx.L_1_ind;
L_2_ind          = idx.L_2_ind;
UB_ind           = idx.UB_ind;
unemp_1_ind      = idx.unemp_1_ind;
unemp_2_ind      = idx.unemp_2_ind;
unemp_ind        = idx.unemp_ind;
U_1_ind          = idx.U_1_ind;
U_2_ind          = idx.U_2_ind;

% Other control variables

K_ind            = idx.K_ind;
B_ind            = idx.B_ind;
B_gov_ncp_ind    = idx.B_gov_ncp_ind;
T_ind            = idx.T_ind;
LT_ind 	         = idx.LT_ind;
G_ind            = idx.G_ind;
Lambda_ind       = idx.Lambda_ind;
pi_ind           = idx.pi_ind;
V_1_ind          = idx.V_1_ind;
V_2_ind          = idx.V_2_ind;
J_ind            = idx.J_ind;
r_l_1_ind        = idx.r_l_1_ind;
r_l_2_ind        = idx.r_l_2_ind;
v_ind            = idx.v_ind;
Y_ind            = idx.Y_ind;
Profit_ind       = idx.Profit_ind;
r_k_ind          = idx.r_k_ind;
r_a_ind          = idx.r_a_ind;
MC_ind           = idx.MC_ind;
n_1_ind          = idx.n_1_ind;
n_2_ind          = idx.n_2_ind;
M_1_ind          = idx.M_1_ind;
M_2_ind          = idx.M_2_ind;
f_1_ind          = idx.f_1_ind;
f_2_ind          = idx.f_2_ind;
zz_ind           = idx.zz_ind;
xx_ind           = idx.xx_ind;
vv_ind           = idx.vv_ind;
ee_ind           = idx.ee_ind;
C_b_ind          = idx.C_b_ind;
Profit_FI_ind    = idx.Profit_FI_ind;
RRa_ind          = idx.RRa_ind;
RR_ind           = idx.RR_ind;
I_ind            = idx.I_ind;
x_k_ind          = idx.x_k_ind;

% Observables

A_g_obs_ind      = idx.A_g_obs_ind;
Y_obs_ind        = idx.Y_obs_ind;
C_obs_ind        = idx.C_obs_ind;
I_obs_ind        = idx.I_obs_ind;
w_1_obs_ind      = idx.w_1_obs_ind;
w_2_obs_ind      = idx.w_2_obs_ind;
w_obs_ind        = idx.w_obs_ind;
PROFIT_obs_ind   = idx.PROFIT_obs_ind;
unemp_1_obs_ind  = idx.unemp_1_obs_ind;
unemp_2_obs_ind  = idx.unemp_2_obs_ind;
unemp_obs_ind    = idx.unemp_obs_ind;
inf_obs_ind      = idx.inf_obs_ind;
R_obs_ind        = idx.R_obs_ind;
pastw_1_ind      = idx.pastw_1_ind;
pastw_2_ind      = idx.pastw_2_ind;
pastw_ind        = idx.pastw_ind;
G_obs_ind        = idx.G_obs_ind;
pastYY_ind       = idx.pastYY_ind;
pastCC_ind       = idx.pastCC_ind;
pastII_ind       = idx.pastII_ind;
pastPPROFIT_ind  = idx.pastPPROFIT_ind;
pastuu_1_ind     = idx.pastuu_1_ind;
pastuu_2_ind     = idx.pastuu_2_ind;
pastuu_ind       = idx.pastuu_ind;
pastGG_ind       = idx.pastGG_ind;
pastA_g_ind      = idx.pastA_g_ind;
l_lambda_1_ind   = idx.l_lambda_1_ind;
l_lambda_2_ind   = idx.l_lambda_2_ind;
pastQ_ind        = idx.pastQ_ind;
x_I_ind          = idx.x_I_ind;
eta2_ind         = idx.eta2_ind;
iota2_ind        = idx.iota2_ind;
pastLT2_ind      = idx.pastLT2_ind;
pastG2_ind       = idx.pastG2_ind;
LT_obs_ind       = idx.LT_obs_ind;
B_gov_ncp2_ind   = idx.B_gov_ncp2_ind;

perturb_size  = 1e-5;

State_perturb   = zeros(grid.os,1) + perturb_size;
Control_perturb = zeros(grid.oc,1) + perturb_size;

State_aux       = exp( StateSS(end-grid.os+1:end)   + State_perturb );
Control_aux     = exp( ControlSS(end-grid.oc+1:end) + Control_perturb );

State_ss        = exp( StateSS(end-grid.os+1:end)    );
Control_ss      = exp( ControlSS(end-grid.oc+1:end)  );

R_cb_aux         = State_aux(R_cb_ind);
w_1_aux          = State_aux(w_1_ind);
w_2_aux          = State_aux(w_2_ind);
w_aux            = State_aux(w_ind);
A_b_aux          = State_aux(A_b_ind);
B_b_aux          = State_aux(B_b_ind);
A_g_aux          = State_aux(A_g_ind);
Q_aux            = State_aux(Q_ind);
lev_aux          = State_aux(lev_ind);
NW_b_aux         = State_aux(NW_b_ind);
R_tilde_aux      = State_aux(R_tilde_ind);
x_cb_aux         = State_aux(x_cb_ind);
pastpi_aux       = State_aux(pastpi_ind);
pastY_aux        = State_aux(pastY_ind);
pastC_aux        = State_aux(pastC_ind);
pastI_aux        = State_aux(pastI_ind);
pastPROFIT_aux   = State_aux(pastPROFIT_ind);
pastunemp_1_aux  = State_aux(pastunemp_1_ind);
pastunemp_2_aux  = State_aux(pastunemp_2_ind);
pastunemp_aux    = State_aux(pastunemp_ind);
pastG_aux        = State_aux(pastG_ind);
pastLT_aux       = State_aux(pastLT_ind);
R_star_aux       = State_aux(R_star_ind);

ZZ_1_aux         = State_aux(ZZ_1_ind);
ZZ_2_aux         = State_aux(ZZ_2_ind);
ZZ_3_aux         = State_aux(ZZ_3_ind);
ZZ_4_aux         = State_aux(ZZ_4_ind);
B_F_aux          = State_aux(B_F_ind);
Z_aux            = State_aux(Z_ind);
PSI_RP_aux       = State_aux(PSI_RP_ind);
eta_aux          = State_aux(eta_ind);
D_aux            = State_aux(D_ind);
GG_aux           = State_aux(GG_ind);
iota_aux         = State_aux(iota_ind);
BB_aux           = State_aux(BB_ind);
PSI_W_aux        = State_aux(PSI_W_ind); 
MP_aux           = State_aux(MP_ind); 
p_mark_aux       = State_aux(p_mark_ind); 

eps_1_aux        = perturb_size;
eps_2_aux        = perturb_size;
eps_3_aux        = perturb_size;
eps_4_aux        = perturb_size;
eps_QE_aux       = perturb_size;
eps_B_F_aux      = perturb_size;
eps_RP_aux       = perturb_size;
eps_eta_aux      = perturb_size;
eps_D_aux        = perturb_size;
eps_R_aux        = perturb_size;
eps_Z_aux        = perturb_size;
eps_G_aux        = perturb_size;
eps_iota_aux     = perturb_size;
eps_BB_aux       = perturb_size;
eps_w_aux        = perturb_size;

MRS_aux          = Control_aux(MRS_ind);
A_hh_aux         = Control_aux(A_hh_ind);
B_hh_aux         = Control_aux(B_hh_ind);
C_aux            = Control_aux(C_ind);
N_tilde_1_aux    = Control_aux(N_tilde_1_ind);
N_tilde_2_aux    = Control_aux(N_tilde_2_ind);
L_1_aux          = Control_aux(L_1_ind);
L_2_aux          = Control_aux(L_2_ind);
UB_aux           = Control_aux(UB_ind);
unemp_1_aux      = Control_aux(unemp_1_ind);
unemp_2_aux      = Control_aux(unemp_2_ind);
unemp_aux        = Control_aux(unemp_ind);
U_1_aux          = Control_aux(U_1_ind);
U_2_aux          = Control_aux(U_2_ind);

K_aux            = Control_aux(K_ind);
B_aux            = Control_aux(B_ind);
B_gov_ncp_aux    = Control_aux(B_gov_ncp_ind);
T_aux            = Control_aux(T_ind);
LT_aux           = Control_aux(LT_ind);
G_aux            = Control_aux(G_ind);
Lambda_aux       = Control_aux(Lambda_ind);
pi_aux           = Control_aux(pi_ind);
V_1_aux          = Control_aux(V_1_ind);
V_2_aux          = Control_aux(V_2_ind);
J_aux            = Control_aux(J_ind);
r_l_1_aux        = Control_aux(r_l_1_ind);
r_l_2_aux        = Control_aux(r_l_2_ind);
v_aux            = Control_aux(v_ind);
Y_aux            = Control_aux(Y_ind);
Profit_aux       = Control_aux(Profit_ind);
r_k_aux          = Control_aux(r_k_ind);
r_a_aux          = Control_aux(r_a_ind);
MC_aux           = Control_aux(MC_ind);

n_1_aux          = Control_aux(n_1_ind);
n_2_aux          = Control_aux(n_2_ind);
M_1_aux          = Control_aux(M_1_ind);
M_2_aux          = Control_aux(M_2_ind);
f_1_aux          = Control_aux(f_1_ind);
f_2_aux          = Control_aux(f_2_ind);
zz_aux           = Control_aux(zz_ind);
xx_aux           = Control_aux(xx_ind);
vv_aux           = Control_aux(vv_ind);
ee_aux           = Control_aux(ee_ind);
C_b_aux          = Control_aux(C_b_ind);
Profit_FI_aux    = Control_aux(Profit_FI_ind);
RRa_aux          = Control_aux(RRa_ind);
RR_aux           = Control_aux(RR_ind);
I_aux            = Control_aux(I_ind);
x_k_aux          = Control_aux(x_k_ind);

A_g_obs_aux       = Control_aux(A_g_obs_ind);
Y_obs_aux         = Control_aux(Y_obs_ind);
C_obs_aux         = Control_aux(C_obs_ind);
I_obs_aux         = Control_aux(I_obs_ind);
w_1_obs_aux       = Control_aux(w_1_obs_ind);
w_2_obs_aux       = Control_aux(w_2_obs_ind);
w_obs_aux         = Control_aux(w_obs_ind);
PROFIT_obs_aux    = Control_aux(PROFIT_obs_ind);
unemp_1_obs_aux   = Control_aux(unemp_1_obs_ind);
unemp_2_obs_aux   = Control_aux(unemp_2_obs_ind);
unemp_obs_aux     = Control_aux(unemp_obs_ind);
inf_obs_aux       = Control_aux(inf_obs_ind);
R_obs_aux         = Control_aux(R_obs_ind);
pastw_1_aux       = Control_aux(pastw_1_ind);
pastw_2_aux       = Control_aux(pastw_2_ind);
pastw_aux         = Control_aux(pastw_ind);
G_obs_aux         = Control_aux(G_obs_ind);

pastYY_aux        = Control_aux(pastYY_ind);
pastCC_aux        = Control_aux(pastCC_ind);
pastII_aux        = Control_aux(pastII_ind);
pastPPROFIT_aux   = Control_aux(pastPPROFIT_ind);
pastuu_1_aux      = Control_aux(pastuu_1_ind);
pastuu_2_aux      = Control_aux(pastuu_2_ind);
pastuu_aux        = Control_aux(pastuu_ind);
pastGG_aux        = Control_aux(pastGG_ind);
pastA_g_aux       = Control_aux(pastA_g_ind);

l_lambda_1_aux    = Control_aux(l_lambda_1_ind);
l_lambda_2_aux    = Control_aux(l_lambda_2_ind);
pastQ_aux         = Control_aux(pastQ_ind);
x_I_aux           = Control_aux(x_I_ind); 
eta2_aux          = Control_aux(eta2_ind); 
iota2_aux         = Control_aux(iota2_ind);
pastLT2_aux       = Control_aux(pastLT2_ind); 
pastG2_aux        = Control_aux(pastG2_ind);
LT_obs_aux        = Control_aux(LT_obs_ind);
B_gov_ncp2_aux    = Control_aux(B_gov_ncp2_ind);

tau_w_aux         = param.tau_w;
tau_a_aux         = param.tau_a;

R_cb_ss         = State_ss(R_cb_ind);
w_1_ss          = State_ss(w_1_ind);
w_2_ss          = State_ss(w_2_ind);
w_ss            = State_ss(w_ind);
A_b_ss          = State_ss(A_b_ind);
B_b_ss          = State_ss(B_b_ind);
A_g_ss          = State_ss(A_g_ind);
Q_ss            = State_ss(Q_ind);
lev_ss          = State_ss(lev_ind);
NW_b_ss         = State_ss(NW_b_ind);
R_tilde_ss      = State_ss(R_tilde_ind);
x_cb_ss         = State_ss(x_cb_ind);
pastpi_ss       = State_ss(pastpi_ind);
pastY_ss        = State_ss(pastY_ind);
pastC_ss        = State_ss(pastC_ind);
pastI_ss        = State_ss(pastI_ind);
pastPROFIT_ss   = State_ss(pastPROFIT_ind);
pastunemp_1_ss  = State_ss(pastunemp_1_ind);
pastunemp_2_ss  = State_ss(pastunemp_2_ind);
pastunemp_ss    = State_ss(pastunemp_ind);
pastG_ss        = State_ss(pastG_ind);
pastLT_ss       = State_ss(pastLT_ind);
R_star_ss       = State_ss(R_star_ind);

ZZ_1_ss         = State_ss(ZZ_1_ind);
ZZ_2_ss         = State_ss(ZZ_2_ind);
ZZ_3_ss         = State_ss(ZZ_3_ind);
ZZ_4_ss         = State_ss(ZZ_4_ind);
B_F_ss          = State_ss(B_F_ind);
Z_ss            = State_ss(Z_ind);
PSI_RP_ss       = State_ss(PSI_RP_ind);
eta_ss          = State_ss(eta_ind);
D_ss            = State_ss(D_ind);
GG_ss           = State_ss(GG_ind);
iota_ss         = State_ss(iota_ind);
BB_ss           = State_ss(BB_ind);
PSI_W_ss        = State_ss(PSI_W_ind); 
MP_ss           = State_ss(MP_ind); 
p_mark_ss       = State_ss(p_mark_ind); 

eps_1_ss        = 0;
eps_2_ss        = 0;
eps_3_ss        = 0;
eps_4_ss        = 0;
eps_QE_ss       = 0;
eps_B_F_ss      = 0;
eps_RP_ss       = 0;
eps_eta_ss      = 0;
eps_D_ss        = 0;
eps_R_ss        = 0;
eps_Z_ss        = 0;
eps_G_ss        = 0;
eps_iota_ss     = 0;
eps_BB_ss       = 0;
eps_w_ss        = 0;

MRS_ss          = Control_ss(MRS_ind);
A_hh_ss         = Control_ss(A_hh_ind);
B_hh_ss         = Control_ss(B_hh_ind);
C_ss            = Control_ss(C_ind);
N_tilde_1_ss    = Control_ss(N_tilde_1_ind);
N_tilde_2_ss    = Control_ss(N_tilde_2_ind);
L_1_ss          = Control_ss(L_1_ind);
L_2_ss          = Control_ss(L_2_ind);
UB_ss           = Control_ss(UB_ind);
unemp_1_ss      = Control_ss(unemp_1_ind);
unemp_2_ss      = Control_ss(unemp_2_ind);
unemp_ss        = Control_ss(unemp_ind);
U_1_ss          = Control_ss(U_1_ind);
U_2_ss          = Control_ss(U_2_ind);

K_ss            = Control_ss(K_ind);
B_ss            = Control_ss(B_ind);
B_gov_ncp_ss    = Control_ss(B_gov_ncp_ind);
T_ss            = Control_ss(T_ind);
LT_ss           = Control_ss(LT_ind);
G_ss            = Control_ss(G_ind);
Lambda_ss       = Control_ss(Lambda_ind);
pi_ss           = Control_ss(pi_ind);
V_1_ss          = Control_ss(V_1_ind);
V_2_ss          = Control_ss(V_2_ind);
J_ss            = Control_ss(J_ind);
r_l_1_ss        = Control_ss(r_l_1_ind);
r_l_2_ss        = Control_ss(r_l_2_ind);
v_ss            = Control_ss(v_ind);
Y_ss            = Control_ss(Y_ind);
Profit_ss       = Control_ss(Profit_ind);
r_k_ss          = Control_ss(r_k_ind);
r_a_ss          = Control_ss(r_a_ind);
MC_ss           = Control_ss(MC_ind);
n_1_ss          = Control_ss(n_1_ind);
n_2_ss          = Control_ss(n_2_ind);
M_1_ss          = Control_ss(M_1_ind);
M_2_ss          = Control_ss(M_2_ind);
f_1_ss          = Control_ss(f_1_ind);
f_2_ss          = Control_ss(f_2_ind);
zz_ss           = Control_ss(zz_ind);
xx_ss           = Control_ss(xx_ind);
vv_ss           = Control_ss(vv_ind);
ee_ss           = Control_ss(ee_ind);
C_b_ss          = Control_ss(C_b_ind);
Profit_FI_ss    = Control_ss(Profit_FI_ind);
RRa_ss          = Control_ss(RRa_ind);
RR_ss           = Control_ss(RR_ind);
I_ss            = Control_ss(I_ind);
x_k_ss          = Control_ss(x_k_ind);

A_g_obs_ss       = Control_ss(A_g_obs_ind);
Y_obs_ss         = Control_ss(Y_obs_ind);
C_obs_ss         = Control_ss(C_obs_ind);
I_obs_ss         = Control_ss(I_obs_ind);
w_1_obs_ss       = Control_ss(w_1_obs_ind);
w_2_obs_ss       = Control_ss(w_2_obs_ind);
w_obs_ss         = Control_ss(w_obs_ind);
PROFIT_obs_ss    = Control_ss(PROFIT_obs_ind);
unemp_1_obs_ss   = Control_ss(unemp_1_obs_ind);
unemp_2_obs_ss   = Control_ss(unemp_2_obs_ind);
unemp_obs_ss     = Control_ss(unemp_obs_ind);
inf_obs_ss       = Control_ss(inf_obs_ind);
R_obs_ss         = Control_ss(R_obs_ind);
pastw_1_ss       = Control_ss(pastw_1_ind);
pastw_2_ss       = Control_ss(pastw_2_ind);
pastw_ss         = Control_ss(pastw_ind);
G_obs_ss         = Control_ss(G_obs_ind);

pastYY_ss        = Control_ss(pastYY_ind);
pastCC_ss        = Control_ss(pastCC_ind);
pastII_ss        = Control_ss(pastII_ind);
pastPPROFIT_ss   = Control_ss(pastPPROFIT_ind);
pastuu_1_ss      = Control_ss(pastuu_1_ind);
pastuu_2_ss      = Control_ss(pastuu_2_ind);
pastuu_ss        = Control_ss(pastuu_ind);
pastGG_ss        = Control_ss(pastGG_ind);
pastA_g_ss       = Control_ss(pastA_g_ind);

l_lambda_1_ss    = Control_ss(l_lambda_1_ind);
l_lambda_2_ss    = Control_ss(l_lambda_2_ind);
pastQ_ss         = Control_ss(pastQ_ind);
x_I_ss           = Control_ss(x_I_ind); 
eta2_ss          = Control_ss(eta2_ind); 
iota2_ss         = Control_ss(iota2_ind);
pastLT2_ss       = Control_ss(pastLT2_ind); 
pastG2_ss        = Control_ss(pastG2_ind);
LT_obs_ss        = Control_ss(LT_obs_ind);
B_gov_ncp2_ss    = Control_ss(B_gov_ncp2_ind);


tau_w_ss = param.tau_w;
tau_a_ss = param.tau_a;

F21_aux = zeros(grid.os,grid.numstates);
F22_aux = zeros(grid.os,grid.numcontrols);
F23_aux = zeros(grid.os,grid.numstates);
F24_aux = zeros(grid.os,grid.numcontrols);

F41_aux = zeros(grid.oc,grid.numstates);
F42_aux = zeros(grid.oc,grid.numcontrols);
F43_aux = zeros(grid.oc,grid.numstates);
F44_aux = zeros(grid.oc,grid.numcontrols);



% State variables

% (1) R_cb

F21_aux(R_cb_ind,end-grid.os+R_cb_ind)   = (log(R_cb_aux)-log(R_cb_ss))/perturb_size;

F23_aux(R_cb_ind,end-grid.os+R_cb_ind)   = (-param.rho_R*(log(R_cb_aux/param.R_cb)-log(R_cb_ss/param.R_cb)))./perturb_size;
% F23_aux(R_cb_ind,end-grid.os+eps_R_ind)  = -(eps_R_aux-eps_R_ss)/perturb_size;
F21_aux(R_cb_ind,end-grid.os+MP_ind)     = -(log(MP_aux)-log(MP_ss))/perturb_size;
F24_aux(R_cb_ind,end-grid.oc+pi_ind)     = - ( (1-param.rho_R)*(param.phi_pi*(log(pi_aux/param.pi_cb) - log(pi_ss/param.pi_cb))) )/perturb_size;
F24_aux(R_cb_ind,end-grid.oc+unemp_ind)  = - (1-param.rho_R)*(-param.phi_u*((unemp_aux-unemp_ss)))./perturb_size;

% (2) w_1_t

F21_aux(w_1_ind,end-grid.os+w_1_ind)      = (log(w_1_aux)-log(w_1_ss))/perturb_size;

% F21_aux(w_1_ind,end-grid.os+ZZ_1_ind)     = - (1-param.rho_w_1)*(log(ZZ_1_aux)-log(ZZ_1_ss))/perturb_size;
F21_aux(w_1_ind,end-grid.os+ZZ_1_ind)     = - (1-param.rho_w_1)*(log(1/ZZ_1_aux)-log(1/ZZ_1_ss))/perturb_size;
F23_aux(w_1_ind,end-grid.os+w_1_ind)      = - param.rho_w_1*(log(w_1_aux/param.w_bar_1) - log(w_1_ss/param.w_bar_1))/perturb_size;
F23_aux(w_1_ind,end-grid.os+pastpi_ind)   = - param.rho_w_1*(1-param.d_1)*(log(pastpi_aux/param.pi_bar)-log(pastpi_ss/param.pi_bar))/perturb_size;
F24_aux(w_1_ind,end-grid.oc+pi_ind)       = - param.rho_w_1*(param.d_1*log(param.pi_bar/pi_aux)+(1-param.d_1)*log(param.pi_bar/pi_aux))/perturb_size;
F24_aux(w_1_ind,end-grid.oc+r_l_1_ind)    = - (1-param.rho_w_1)*param.eps_w_1*(log(r_l_1_aux/SS_stats.r_l_1)-log(r_l_1_ss/SS_stats.r_l_1))/perturb_size;

% (3) w_2_t

F21_aux(w_2_ind,end-grid.os+w_2_ind)      = (log(w_2_aux)-log(w_2_ss))/perturb_size;

% F21_aux(w_2_ind,end-grid.os+PSI_W_ind)    = - (1-param.rho_w_2)*(log(PSI_W_aux)-log(PSI_W_ss))/perturb_size;
F21_aux(w_2_ind,end-grid.os+PSI_W_ind)    = - (1-param.rho_w_2)*(log(1/PSI_W_aux)-log(1/PSI_W_ss))/perturb_size;
F23_aux(w_2_ind,end-grid.os+w_2_ind)      = - param.rho_w_2 * (log(w_2_aux/param.w_bar_2) - log(w_2_ss/param.w_bar_2))/perturb_size;
F23_aux(w_2_ind,end-grid.os+pastpi_ind)   = - param.rho_w_2*(1-param.d_2)*(log(pastpi_aux/param.pi_bar)-log(pastpi_ss/param.pi_bar))/perturb_size;
F24_aux(w_2_ind,end-grid.oc+pi_ind)       = - param.rho_w_2*(param.d_2*log(param.pi_bar/pi_aux)+(1-param.d_2)*log(param.pi_bar/pi_aux))/perturb_size;
F24_aux(w_2_ind,end-grid.oc+r_l_2_ind)    = - (1-param.rho_w_2)*param.eps_w_2*(log(r_l_2_aux/SS_stats.r_l_2)-log(r_l_2_ss/SS_stats.r_l_2))/perturb_size;

% (4) w_t

F21_aux(w_ind,end-grid.os+w_ind)          = (log(w_aux)-log(w_ss))/perturb_size;

F21_aux(w_ind,end-grid.os+w_1_ind)        = - ( log((L_1_ss *w_1_aux + L_2_ss*w_2_ss )/(L_1_ss +L_2_ss )) - log((L_1_ss*w_1_ss + L_2_ss*w_2_ss)/(L_1_ss+L_2_ss)) )/perturb_size;
F21_aux(w_ind,end-grid.os+w_2_ind)        = - ( log((L_1_ss *w_1_ss  + L_2_ss*w_2_aux)/(L_1_ss +L_2_ss )) - log((L_1_ss*w_1_ss + L_2_ss*w_2_ss)/(L_1_ss+L_2_ss)) )/perturb_size;
F24_aux(w_ind,end-grid.oc+L_1_ind)        = - ( log((L_1_aux*w_1_ss  + L_2_ss*w_2_ss )/(L_1_aux+L_2_ss )) - log((L_1_ss*w_1_ss + L_2_ss*w_2_ss)/(L_1_ss+L_2_ss)) )/perturb_size;
F24_aux(w_ind,end-grid.oc+L_2_ind)        = - ( log((L_1_ss *w_1_ss  + L_2_aux*w_2_ss)/(L_1_ss +L_2_aux)) - log((L_1_ss*w_1_ss + L_2_ss*w_2_ss)/(L_1_ss+L_2_ss)) )/perturb_size;

% (5) A_t+1^b

F21_aux(A_b_ind,end-grid.os+A_b_ind)    = (log(A_b_aux)-log(A_b_ss))/perturb_size;
F21_aux(A_b_ind,end-grid.os+Q_ind)      = -(log(lev_ss*NW_b_ss/Q_aux)-log(lev_ss*NW_b_ss/Q_ss))/perturb_size;
F21_aux(A_b_ind,end-grid.os+lev_ind)    = -(log(lev_aux*NW_b_ss/Q_ss)-log(lev_ss*NW_b_ss/Q_ss))/perturb_size;
F21_aux(A_b_ind,end-grid.os+NW_b_ind)   = -(log(lev_ss*NW_b_aux/Q_ss)-log(lev_ss*NW_b_ss/Q_ss))/perturb_size;

% (6) B_t+1^b

F21_aux(B_b_ind,end-grid.os+B_b_ind)       = (log(B_b_aux)-log(B_b_ss))/perturb_size;

F23_aux(B_b_ind,end-grid.os+B_b_ind)       = -( log( param.MMMF_cont_ratio*T_ss  + param.b_a_aux2*R_cb_ss/pi_ss*B_b_aux - C_b_ss )  - log( param.MMMF_cont_ratio*T_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss ))/perturb_size;
F24_aux(B_b_ind,end-grid.oc+C_b_ind)       = -( log( param.MMMF_cont_ratio*T_ss  + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss  - C_b_aux ) - log( param.MMMF_cont_ratio*T_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss ))/perturb_size;
F24_aux(B_b_ind,end-grid.oc+T_ind)         = -( log( T_aux + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss )  - log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss ))/perturb_size;
F23_aux(B_b_ind,end-grid.os+R_cb_ind)      = -( log( T_ss + B_gov_ncp_ss  - B_gov_ncp_ss*R_cb_aux/pi_ss + (Q_ss+r_a_ss-R_cb_aux/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_aux/pi_ss*B_b_ss - C_b_ss )  - log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss ))/perturb_size;
F24_aux(B_b_ind,end-grid.oc+pi_ind)        = -( log( T_ss + B_gov_ncp_ss  - B_gov_ncp_ss*R_cb_ss/pi_aux + (Q_ss+r_a_ss-R_cb_ss/pi_aux*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_aux*B_b_ss - C_b_ss )  - log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss ))/perturb_size;
F22_aux(B_b_ind,end-grid.oc+B_gov_ncp_ind) = -( log( T_ss + B_gov_ncp_aux - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss )  - log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss ))/perturb_size;
F24_aux(B_b_ind,end-grid.oc+B_gov_ncp_ind) = -( log( T_ss + B_gov_ncp_ss - B_gov_ncp_aux*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss )  - log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss ))/perturb_size;
F21_aux(B_b_ind,end-grid.os+Q_ind)         = -( log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_aux+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_aux*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss )  - log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss ))/perturb_size;
F23_aux(B_b_ind,end-grid.os+Q_ind)         = -( log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_aux)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss )  - log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss ))/perturb_size;
F24_aux(B_b_ind,end-grid.oc+r_a_ind)       = -( log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_aux-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss )  - log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss ))/perturb_size;
F23_aux(B_b_ind,end-grid.os+A_g_ind)       = -( log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_aux) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss )  - log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss ))/perturb_size;
F21_aux(B_b_ind,end-grid.os+A_g_ind)       = -( log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_aux) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss )  - log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss ))/perturb_size;
F24_aux(B_b_ind,end-grid.oc+UB_ind)        = -( log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_aux - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss )  - log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss ))/perturb_size;
F24_aux(B_b_ind,end-grid.oc+LT_ind)        = -( log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_aux - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss )  - log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss ))/perturb_size;
F24_aux(B_b_ind,end-grid.oc+G_ind)         = -( log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_aux + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss )  - log( T_ss + B_gov_ncp_ss - B_gov_ncp_ss*R_cb_ss/pi_ss + (Q_ss+r_a_ss-R_cb_ss/pi_ss*Q_ss)*(A_g_ss) - UB_ss - LT_ss - param.tau_cp*Q_ss*(A_g_ss) - G_ss + param.b_a_aux2*R_cb_ss/pi_ss*B_b_ss - C_b_ss ))/perturb_size;

% (7) A_t+1^g

F21_aux(A_g_ind,end-grid.os+A_g_ind)       = (log(A_g_aux)-log(A_g_ss))/perturb_size;

F23_aux(A_g_ind,end-grid.os+A_g_ind)      = -( log( ((param.rho_A_g)*(A_g_aux) + (1-param.rho_A_g)*SS_stats.A_g)                        ) - log( ((param.rho_A_g)*(A_g_ss) + (1-param.rho_A_g)*SS_stats.A_g) )  )/perturb_size;
% F21_aux(A_g_ind,end-grid.os+PSI_RP_ind)   = -( log( ((param.rho_A_g)*(A_g_ss)  + (1-param.rho_A_g)*SS_stats.A_g) + log(PSI_RP_aux)*Y_ss ) - log( ((param.rho_A_g)*(A_g_ss) + (1-param.rho_A_g)*SS_stats.A_g) )  )/perturb_size;

% (8) Q

F21_aux(Q_ind,end-grid.os+Q_ind)           = (log(Q_aux)-log(Q_ss))/perturb_size;

F22_aux(Q_ind,end-grid.oc+x_k_ind)         = - ( log( eta2_ss*(1+param.phi*log(x_k_ss)+param.phi/2*(log(x_k_ss))^2 ) - eta2_ss*Lambda_ss/SS_stats.Lambda*param.phi*log(x_k_aux)*x_k_aux ) - log( eta2_ss*(1+param.phi*log(x_k_ss)+param.phi/2*(log(x_k_ss))^2 ) - eta2_ss*Lambda_ss/SS_stats.Lambda*param.phi*log(x_k_ss)*x_k_ss ) )/perturb_size;
F22_aux(Q_ind,end-grid.oc+Lambda_ind)      = - ( log( eta2_ss*(1+param.phi*log(x_k_ss)+param.phi/2*(log(x_k_ss))^2 ) - eta2_ss*Lambda_aux/SS_stats.Lambda*param.phi*log(x_k_ss)*x_k_ss ) - log( eta2_ss*(1+param.phi*log(x_k_ss)+param.phi/2*(log(x_k_ss))^2 ) - eta2_ss*Lambda_ss/SS_stats.Lambda*param.phi*log(x_k_ss)*x_k_ss ) )/perturb_size;
F24_aux(Q_ind,end-grid.oc+x_k_ind)         = - ( log( eta2_ss*(1+param.phi*log(x_k_aux)+param.phi/2*(log(x_k_aux))^2 ) - eta2_ss*Lambda_ss/SS_stats.Lambda*param.phi*log(x_k_ss)*x_k_ss ) - log( eta2_ss*(1+param.phi*log(x_k_ss)+param.phi/2*(log(x_k_ss))^2 ) - eta2_ss*Lambda_ss/SS_stats.Lambda*param.phi*log(x_k_ss)*x_k_ss ) )/perturb_size;
F24_aux(Q_ind,end-grid.oc+iota2_ind)       = - ( log( iota2_aux*(1+param.phi*log(x_k_ss)+param.phi/2*(log(x_k_ss))^2 ) - iota2_ss*Lambda_ss/SS_stats.Lambda*param.phi*log(x_k_ss)*x_k_ss ) - log( iota2_ss*(1+param.phi*log(x_k_ss)+param.phi/2*(log(x_k_ss))^2 ) - iota2_ss*Lambda_ss/SS_stats.Lambda*param.phi*log(x_k_ss)*x_k_ss ) )/perturb_size;
F22_aux(Q_ind,end-grid.oc+iota2_ind)       = - ( log( iota2_ss*(1+param.phi*log(x_k_ss)+param.phi/2*(log(x_k_ss))^2 ) - iota2_aux*Lambda_ss/SS_stats.Lambda*param.phi*log(x_k_ss)*x_k_ss ) - log( iota2_ss*(1+param.phi*log(x_k_ss)+param.phi/2*(log(x_k_ss))^2 ) - iota2_ss*Lambda_ss/SS_stats.Lambda*param.phi*log(x_k_ss)*x_k_ss ) )/perturb_size;

% (9) Leverage (THETA)

F21_aux(lev_ind,end-grid.os+lev_ind)       = (log(lev_aux)-log(lev_ss))/perturb_size;

F24_aux(lev_ind,end-grid.oc+vv_ind)        = - (log( ee_ss/(param.DELTA-vv_aux))-log( ee_ss/(param.DELTA-vv_ss) ))/perturb_size;
F24_aux(lev_ind,end-grid.oc+ee_ind)        = - (log( ee_aux/(param.DELTA-vv_ss))-log( ee_ss/(param.DELTA-vv_ss) ))/perturb_size;

% (10) Net worth

F21_aux(NW_b_ind,end-grid.os+NW_b_ind)     = (log(NW_b_aux)-log(NW_b_ss))/perturb_size;

F23_aux(NW_b_ind,end-grid.os+R_cb_ind)     = - ( log( (param.theta_b*((RRa_ss -param.b_a_aux2*R_cb_aux/pi_ss)*lev_ss +param.b_a_aux2*R_cb_aux/pi_ss) *NW_b_ss  + param.omega*Q_ss *A_b_ss) ) - log( (param.theta_b*((RRa_ss-param.b_a_aux2*R_cb_ss/pi_ss)*lev_ss+param.b_a_aux2*R_cb_ss/pi_ss)*NW_b_ss + param.omega*Q_ss*A_b_ss) )   )/perturb_size;
F23_aux(NW_b_ind,end-grid.os+A_b_ind)      = - ( log( (param.theta_b*((RRa_ss -param.b_a_aux2*R_cb_ss/pi_ss) *lev_ss +param.b_a_aux2*R_cb_ss /pi_ss) *NW_b_ss  + param.omega*Q_ss *A_b_aux)) - log( (param.theta_b*((RRa_ss-param.b_a_aux2*R_cb_ss/pi_ss)*lev_ss+param.b_a_aux2*R_cb_ss/pi_ss)*NW_b_ss + param.omega*Q_ss*A_b_ss) )   )/perturb_size;
F23_aux(NW_b_ind,end-grid.os+Q_ind)        = - ( log( (param.theta_b*((RRa_ss -param.b_a_aux2*R_cb_ss/pi_ss) *lev_ss +param.b_a_aux2*R_cb_ss /pi_ss) *NW_b_ss  + param.omega*Q_aux*A_b_ss) ) - log( (param.theta_b*((RRa_ss-param.b_a_aux2*R_cb_ss/pi_ss)*lev_ss+param.b_a_aux2*R_cb_ss/pi_ss)*NW_b_ss + param.omega*Q_ss*A_b_ss) )   )/perturb_size;
F23_aux(NW_b_ind,end-grid.os+NW_b_ind)     = - ( log( (param.theta_b*((RRa_ss -param.b_a_aux2*R_cb_ss/pi_ss) *lev_ss +param.b_a_aux2*R_cb_ss /pi_ss) *NW_b_aux + param.omega*Q_ss *A_b_ss) ) - log( (param.theta_b*((RRa_ss-param.b_a_aux2*R_cb_ss/pi_ss)*lev_ss+param.b_a_aux2*R_cb_ss/pi_ss)*NW_b_ss + param.omega*Q_ss*A_b_ss) )   )/perturb_size;
F23_aux(NW_b_ind,end-grid.os+lev_ind)      = - ( log( (param.theta_b*((RRa_ss -param.b_a_aux2*R_cb_ss/pi_ss) *lev_aux+param.b_a_aux2*R_cb_ss /pi_ss) *NW_b_ss  + param.omega*Q_ss *A_b_ss) ) - log( (param.theta_b*((RRa_ss-param.b_a_aux2*R_cb_ss/pi_ss)*lev_ss+param.b_a_aux2*R_cb_ss/pi_ss)*NW_b_ss + param.omega*Q_ss*A_b_ss) )   )/perturb_size;
F24_aux(NW_b_ind,end-grid.oc+pi_ind)       = - ( log( (param.theta_b*((RRa_ss -param.b_a_aux2*R_cb_ss/pi_aux)*lev_ss +param.b_a_aux2*R_cb_ss /pi_aux)*NW_b_ss  + param.omega*Q_ss *A_b_ss) ) - log( (param.theta_b*((RRa_ss-param.b_a_aux2*R_cb_ss/pi_ss)*lev_ss+param.b_a_aux2*R_cb_ss/pi_ss)*NW_b_ss + param.omega*Q_ss*A_b_ss) )   )/perturb_size;
F24_aux(NW_b_ind,end-grid.oc+RRa_ind)      = - ( log( (param.theta_b*((RRa_aux-param.b_a_aux2*R_cb_ss/pi_ss) *lev_ss +param.b_a_aux2*R_cb_ss /pi_ss) *NW_b_ss  + param.omega*Q_ss *A_b_ss) ) - log( (param.theta_b*((RRa_ss-param.b_a_aux2*R_cb_ss/pi_ss)*lev_ss+param.b_a_aux2*R_cb_ss/pi_ss)*NW_b_ss + param.omega*Q_ss*A_b_ss) )   )/perturb_size;

% (11) R_tilde

F21_aux(R_tilde_ind,end-grid.os+R_tilde_ind) = (log(R_tilde_aux)-log(R_tilde_ss))/perturb_size;

F21_aux(R_tilde_ind,end-grid.os+R_cb_ind)    = - (log( R_cb_aux)-log(R_cb_ss))/perturb_size;

% (12) X_cb

F21_aux(x_cb_ind,end-grid.os+x_cb_ind)     = (log(x_cb_aux)-log(x_cb_ss))/perturb_size;

F23_aux(x_cb_ind,end-grid.os+x_cb_ind)     = - (log( 1 +param.rho_X_QE*(x_cb_aux-1) + (1-param.rho_X_QE)*(-param.phi_pi_QE*log(pi_ss/param.pi_cb) +param.phi_u_QE*(unemp_ss-SS_stats.u))) -  log( 1 +param.rho_X_QE*(x_cb_ss-1) + (1-param.rho_X_QE)*(-param.phi_pi_QE*log(pi_ss/param.pi_cb)+param.phi_u_QE*(unemp_ss-SS_stats.u))) )/perturb_size;
F24_aux(x_cb_ind,end-grid.oc+pi_ind)       = - (log( 1 +param.rho_X_QE*(x_cb_ss-1)  + (1-param.rho_X_QE)*(-param.phi_pi_QE*log(pi_aux/param.pi_cb)+param.phi_u_QE*(unemp_ss-SS_stats.u))) -  log( 1 +param.rho_X_QE*(x_cb_ss-1) + (1-param.rho_X_QE)*(-param.phi_pi_QE*log(pi_ss/param.pi_cb)+param.phi_u_QE*(unemp_ss-SS_stats.u))) )/perturb_size;
F24_aux(x_cb_ind,end-grid.oc+unemp_ind)    = - (log( 1 +param.rho_X_QE*(x_cb_ss-1)  + (1-param.rho_X_QE)*(-param.phi_pi_QE*log(pi_ss/param.pi_cb) +param.phi_u_QE*(unemp_aux-SS_stats.u))) -  log( 1 +param.rho_X_QE*(x_cb_ss-1) + (1-param.rho_X_QE)*(-param.phi_pi_QE*log(pi_ss/param.pi_cb)+param.phi_u_QE*(unemp_ss-SS_stats.u))) )/perturb_size;
% F21_aux(x_cb_ind,end-grid.os+PSI_RP_ind)   = - (log( 1 +param.rho_X_QE*(x_cb_ss-1)  + (1-param.rho_X_QE)*(-param.phi_pi_QE*log(pi_ss/param.pi_cb) +param.phi_u_QE*(unemp_ss-SS_stats.u))  + log(PSI_RP_aux) ) -  log( 1 +param.rho_X_QE*(x_cb_ss-1) + (1-param.rho_X_QE)*(-param.phi_pi_QE*log(pi_ss/param.pi_cb)+param.phi_u_QE*(unemp_ss-SS_stats.u))) )/perturb_size;

% (13) past pi

F21_aux(pastpi_ind,end-grid.os+pastpi_ind) = (log(pastpi_aux)-log(pastpi_ss))/perturb_size;

F24_aux(pastpi_ind,end-grid.oc+pi_ind)     = -(log(pi_aux)-log(pi_ss))/perturb_size;

% (14) past Y

F21_aux(pastY_ind,end-grid.os+pastY_ind) = (log(pastY_aux)-log(pastY_ss))/perturb_size;

F24_aux(pastY_ind,end-grid.oc+Y_ind)     = -(log(Y_aux)-log(Y_ss))/perturb_size;

% (15) past C

F21_aux(pastC_ind,end-grid.os+pastC_ind) = (log(pastC_aux)-log(pastC_ss))/perturb_size;

F24_aux(pastC_ind,end-grid.oc+C_ind)     = -(log(C_aux)-log(C_ss))/perturb_size;

% (16) past I

F21_aux(pastI_ind,end-grid.os+pastI_ind) = (log(pastI_aux)-log(pastI_ss))/perturb_size;

F24_aux(pastI_ind,end-grid.oc+I_ind)     = -(log(I_aux)-log(I_ss))/perturb_size;

% (17) past PROFIT

F21_aux(pastPROFIT_ind,end-grid.os+pastPROFIT_ind) = (log(pastPROFIT_aux)-log(pastPROFIT_ss))/perturb_size;

F24_aux(pastPROFIT_ind,end-grid.oc+Profit_ind)     = -(log(Profit_aux+Profit_FI_ss)-log(Profit_ss+Profit_FI_ss))/perturb_size;

% (18) past unemp 1

F21_aux(pastunemp_1_ind,end-grid.os+pastunemp_1_ind) = (log(pastunemp_1_aux)-log(pastunemp_1_ss))/perturb_size;

F24_aux(pastunemp_1_ind,end-grid.oc+unemp_1_ind)     = -(log(unemp_1_aux)-log(unemp_1_ss))/perturb_size;

% (19) past unemp 2

F21_aux(pastunemp_2_ind,end-grid.os+pastunemp_2_ind) = (log(pastunemp_2_aux)-log(pastunemp_2_ss))/perturb_size;

F24_aux(pastunemp_2_ind,end-grid.oc+unemp_2_ind)     = -(log(unemp_2_aux)-log(unemp_2_ss))/perturb_size;

% (20) past unemp

F21_aux(pastunemp_ind,end-grid.os+pastunemp_ind) = (log(pastunemp_aux)-log(pastunemp_ss))/perturb_size;

F24_aux(pastunemp_ind,end-grid.oc+unemp_ind)     = -(log(unemp_aux)-log(unemp_ss))/perturb_size;

% (21) past G

F21_aux(pastG_ind,end-grid.os+pastG_ind) = (log(pastG_aux)-log(pastG_ss))/perturb_size;

F24_aux(pastG_ind,end-grid.oc+G_ind)     = -(log(G_aux)-log(G_ss))/perturb_size;

% (22) past LT

F21_aux(pastLT_ind,end-grid.os+pastLT_ind) = (log(pastLT_aux)-log(pastLT_ss))/perturb_size;

F24_aux(pastLT_ind,end-grid.oc+LT_ind)     = -(log(LT_aux)-log(LT_ss))/perturb_size;

% (23) R_star

F21_aux(R_star_ind,end-grid.os+R_star_ind) = (log(R_star_aux)-log(R_star_ss))/perturb_size;

F23_aux(R_star_ind,end-grid.os+R_star_ind) = (-param.rho_R*(log(R_star_aux/param.R_cb)-log(R_star_ss/param.R_cb)))./perturb_size;
% F23_aux(R_star_ind,end-grid.os+eps_R_ind)  = -(eps_R_aux-eps_R_ss)/perturb_size;
F21_aux(R_star_ind,end-grid.os+MP_ind)     = -(log(MP_aux)-log(MP_ss))/perturb_size;
F24_aux(R_star_ind,end-grid.oc+pi_ind)     = - ( (1-param.rho_R)*(param.phi_pi*(log(pi_aux/param.pi_cb) - log(pi_ss/param.pi_cb))) )/perturb_size;
F24_aux(R_star_ind,end-grid.oc+unemp_ind)  = - (1-param.rho_R)*(-param.phi_u*((unemp_aux-unemp_ss)))./perturb_size;

% shock processes

% (24) ZZ 1

F21_aux(ZZ_1_ind,end-grid.os+ZZ_1_ind)     = 1;

F23_aux(ZZ_1_ind,end-grid.os+ZZ_1_ind)     = - param.rho_ZZ_1*(log(ZZ_1_aux)-log(ZZ_1_ss))/perturb_size;
F23_aux(ZZ_1_ind,end-grid.os+eps_1_ind)    = - (eps_1_aux-eps_1_ss)/perturb_size;

% (25) ZZ 2

F21_aux(ZZ_2_ind,end-grid.os+ZZ_2_ind)     = 1;

F23_aux(ZZ_2_ind,end-grid.os+ZZ_2_ind)     = - param.rho_ZZ_2*(log(ZZ_2_aux)-log(ZZ_2_ss))/perturb_size;
F23_aux(ZZ_2_ind,end-grid.os+eps_2_ind)    = - (eps_2_aux-eps_2_ss)/perturb_size;

% (26) ZZ 3

F21_aux(ZZ_3_ind,end-grid.os+ZZ_3_ind)     = 1;

F23_aux(ZZ_3_ind,end-grid.os+ZZ_3_ind)     = - param.rho_ZZ_3*(log(ZZ_3_aux)-log(ZZ_3_ss))/perturb_size;
F23_aux(ZZ_3_ind,end-grid.os+eps_3_ind)    = - (eps_3_aux-eps_3_ss)/perturb_size;

% (27) ZZ 4

F21_aux(ZZ_4_ind,end-grid.os+ZZ_4_ind)     = 1;

F23_aux(ZZ_4_ind,end-grid.os+ZZ_4_ind)     = - param.rho_ZZ_4*(log(ZZ_4_aux)-log(ZZ_4_ss))/perturb_size;
F23_aux(ZZ_4_ind,end-grid.os+eps_4_ind)    = - (eps_4_aux-eps_4_ss)/perturb_size;

% (28) B_F

F21_aux(B_F_ind,end-grid.os+B_F_ind)         = 1;

F23_aux(B_F_ind,end-grid.os+B_F_ind)         = - param.rho_B_F*(log(B_F_aux)-log(B_F_ss))/perturb_size;
F23_aux(B_F_ind,end-grid.os+eps_B_F_ind)     = - (eps_B_F_aux-eps_B_F_ss)/perturb_size;

% (29) Z 

F21_aux(Z_ind,end-grid.os+Z_ind)           = 1;

F23_aux(Z_ind,end-grid.os+Z_ind)           = - param.rho_Z*(log(Z_aux)-log(Z_ss))/perturb_size;
F23_aux(Z_ind,end-grid.os+eps_Z_ind)       = - (eps_Z_aux-eps_Z_ss)/perturb_size;

% (30) PSI RP

F21_aux(PSI_RP_ind,end-grid.os+PSI_RP_ind) = 1;

F23_aux(PSI_RP_ind,end-grid.os+PSI_RP_ind) = - param.rho_PSI_RP*(log(PSI_RP_aux)-log(PSI_RP_ss))/perturb_size;
F23_aux(PSI_RP_ind,end-grid.os+eps_RP_ind) = - (eps_RP_aux-eps_RP_ss)/perturb_size;

% (31) eta 

F21_aux(eta_ind,end-grid.os+eta_ind)       = 1;

F21_aux(eta_ind,end-grid.os+p_mark_ind)    = - ( log(p_mark_aux/(p_mark_aux-1)) - log(p_mark_ss/(p_mark_ss-1)) )/perturb_size;

% (32) D

F21_aux(D_ind,end-grid.os+D_ind)           = 1;

F23_aux(D_ind,end-grid.os+D_ind)           = - param.rho_D*(log(D_aux)-log(D_ss))/perturb_size;
F23_aux(D_ind,end-grid.os+eps_D_ind)        = - (eps_D_aux-eps_D_ss)/perturb_size;

% (33) G 

F21_aux(GG_ind,end-grid.os+GG_ind)           = 1;

F23_aux(GG_ind,end-grid.os+GG_ind)           = - param.rho_G *(log(GG_aux)-log(GG_ss))/perturb_size;
% F23_aux(GG_ind,end-grid.os+GG_ind)           = - param.rho_G *(log(GG_aux)-log(GG_ss))/perturb_size;

F23_aux(GG_ind,end-grid.os+eps_G_ind)        = - (eps_G_aux-eps_G_ss)/perturb_size;

% (34) iota 

F21_aux(iota_ind,end-grid.os+iota_ind)      = 1;

F23_aux(iota_ind,end-grid.os+iota_ind)      = - param.rho_iota*(log(iota_aux)-log(iota_ss))/perturb_size;
F23_aux(iota_ind,end-grid.os+eps_iota_ind)  = - (eps_iota_aux-eps_iota_ss)/perturb_size;

% (35) BB

F21_aux(BB_ind,end-grid.os+BB_ind)          = 1;

F23_aux(BB_ind,end-grid.os+BB_ind)          = - param.rho_BB*(log(BB_aux)-log(BB_ss))/perturb_size;
F23_aux(BB_ind,end-grid.os+eps_BB_ind)      = - (eps_BB_aux-eps_BB_ss)/perturb_size;

% (36) PSI W

F21_aux(PSI_W_ind,end-grid.os+PSI_W_ind)    = 1;

F23_aux(PSI_W_ind,end-grid.os+PSI_W_ind)    = - param.rho_PSI_W*(log(PSI_W_aux)-log(PSI_W_ss))/perturb_size;
F23_aux(PSI_W_ind,end-grid.os+eps_w_ind)    = - (eps_w_aux-eps_w_ss)/perturb_size;

% (37) MP 

F21_aux(MP_ind,end-grid.os+MP_ind)          = 1;

F23_aux(MP_ind,end-grid.os+MP_ind)          = - param.rho_MP*(log(MP_aux)-log(MP_ss))/perturb_size;
F23_aux(MP_ind,end-grid.os+eps_R_ind)       = - (eps_R_aux-eps_R_ss)/perturb_size;

% (38) p_mark

F21_aux(p_mark_ind,end-grid.os+p_mark_ind)      = 1;

F23_aux(p_mark_ind,end-grid.os+p_mark_ind)      = - param.rho_eta*(log(p_mark_aux)-log(p_mark_ss))/perturb_size;
F23_aux(p_mark_ind,end-grid.os+eps_eta_ind)     = - (eps_eta_aux-eps_eta_ss)/perturb_size;




% (39 - 53) Shocks

F21_aux(eps_1_ind    ,end-grid.os+eps_1_ind)       = 1; % 1
F21_aux(eps_2_ind    ,end-grid.os+eps_2_ind)       = 1; % 2
F21_aux(eps_3_ind    ,end-grid.os+eps_3_ind)       = 1; % 3
F21_aux(eps_4_ind    ,end-grid.os+eps_4_ind)       = 1; % 4
F21_aux(eps_QE_ind   ,end-grid.os+eps_QE_ind)      = 1; % 5
F21_aux(eps_RP_ind   ,end-grid.os+eps_RP_ind)      = 1; % 6
F21_aux(eps_B_F_ind  ,end-grid.os+eps_B_F_ind)     = 1; % 7
F21_aux(eps_BB_ind   ,end-grid.os+eps_BB_ind)      = 1; % 8
F21_aux(eps_Z_ind    ,end-grid.os+eps_Z_ind)       = 1; % 9
F21_aux(eps_G_ind    ,end-grid.os+eps_G_ind)       = 1; % 10
F21_aux(eps_D_ind    ,end-grid.os+eps_D_ind)       = 1; % 11
F21_aux(eps_R_ind    ,end-grid.os+eps_R_ind)       = 1; % 12
F21_aux(eps_iota_ind ,end-grid.os+eps_iota_ind)    = 1; % 13
F21_aux(eps_eta_ind  ,end-grid.os+eps_eta_ind)     = 1; % 14
F21_aux(eps_w_ind    ,end-grid.os+eps_w_ind)       = 1; % 15

% Control variables

% (1) K_t

F44_aux(K_ind,end-grid.oc+K_ind)    = (K_aux-K_ss)/perturb_size;

F43_aux(K_ind,end-grid.os+A_b_ind)  = -(A_b_aux-A_b_ss)/perturb_size;
F43_aux(K_ind,end-grid.os+A_g_ind)  = -(A_g_aux-A_g_ss)/perturb_size;
F44_aux(K_ind,end-grid.oc+A_hh_ind) = -(A_hh_aux-A_hh_ss)/perturb_size;


% (2) B_t

F44_aux(B_ind,end-grid.oc+B_ind)    = (B_aux-B_ss)/perturb_size;

F43_aux(B_ind,end-grid.os+B_b_ind)  = -(B_b_aux-B_b_ss)/perturb_size;
F44_aux(B_ind,end-grid.oc+B_hh_ind) = -(B_hh_aux-B_hh_ss)/perturb_size;

% (3) B_gov_ncp

F44_aux(B_gov_ncp_ind,end-grid.oc+B_gov_ncp_ind) = (B_gov_ncp_aux-B_gov_ncp_ss)/perturb_size;

F43_aux(B_gov_ncp_ind,end-grid.os+A_b_ind)       = - ( (B_b_ss + B_hh_ss + SS_stats.B_F  - (Q_ss*A_b_aux- NW_b_ss) - Q_ss *(1+param.tau_cp)* (A_g_ss)) - ( (B_b_ss + B_hh_ss + SS_stats.B_F - (Q_ss*A_b_ss- NW_b_ss) - Q_ss *(1+param.tau_cp)* (A_g_ss))  )  )/perturb_size; 
F43_aux(B_gov_ncp_ind,end-grid.os+B_b_ind)       = - ( (B_b_aux+ B_hh_ss + SS_stats.B_F - (Q_ss*A_b_ss- NW_b_ss)  - Q_ss *(1+param.tau_cp)* (A_g_ss)) - ( (B_b_ss + B_hh_ss + SS_stats.B_F - (Q_ss*A_b_ss- NW_b_ss) - Q_ss *(1+param.tau_cp)* (A_g_ss))  )  )/perturb_size;
F43_aux(B_gov_ncp_ind,end-grid.os+A_g_ind)       = - ( (B_b_ss + B_hh_ss + SS_stats.B_F  - (Q_ss*A_b_ss- NW_b_ss)  - Q_ss *(1+param.tau_cp)* (A_g_aux)) - ( (B_b_ss + B_hh_ss + SS_stats.B_F - (Q_ss*A_b_ss- NW_b_ss) - Q_ss *(1+param.tau_cp)* (A_g_ss))  )  )/perturb_size;
F43_aux(B_gov_ncp_ind,end-grid.os+Q_ind)         = - ( (B_b_ss + B_hh_ss + SS_stats.B_F  - (Q_aux*A_b_ss- NW_b_ss) - Q_aux*(1+param.tau_cp)* (A_g_ss)) - ( (B_b_ss + B_hh_ss + SS_stats.B_F - (Q_ss*A_b_ss- NW_b_ss) - Q_ss *(1+param.tau_cp)* (A_g_ss))  )  )/perturb_size;
F43_aux(B_gov_ncp_ind,end-grid.os+NW_b_ind)      = - ( (B_b_ss + B_hh_ss + SS_stats.B_F  - (Q_ss*A_b_ss- NW_b_aux) - Q_ss *(1+param.tau_cp)* (A_g_ss)) - ( (B_b_ss + B_hh_ss + SS_stats.B_F - (Q_ss*A_b_ss- NW_b_ss) - Q_ss *(1+param.tau_cp)* (A_g_ss))  )  )/perturb_size;
F44_aux(B_gov_ncp_ind,end-grid.oc+B_hh_ind)      = - (B_hh_aux-B_hh_ss)/perturb_size;

% (4) Tax

% T      = tau_w.*(L_1*W_1 + L_2*W_2 + UB) + tau_a.*(PROFIT);										

F44_aux(T_ind,end-grid.oc+T_ind)         = (T_aux-T_ss)/perturb_size;

F41_aux(T_ind,end-grid.os+w_1_ind)       = - ((tau_w_ss.*(w_1_aux.*L_1_ss  + w_2_ss.*L_2_ss  + UB_ss ) + tau_a_ss.*Profit_ss  + tau_a_ss.*(Profit_FI_ss-param.fix2))-(tau_w_ss.*(w_1_ss.*L_1_ss + w_2_ss.*L_2_ss + UB_ss) + tau_a_ss.*Profit_ss + tau_a_ss.*(Profit_FI_ss-param.fix2)))/perturb_size;
F41_aux(T_ind,end-grid.os+w_2_ind)       = - ((tau_w_ss.*(w_1_ss.*L_1_ss  + w_2_aux.*L_2_ss  + UB_ss ) + tau_a_ss.*Profit_ss  + tau_a_ss.*(Profit_FI_ss-param.fix2))-(tau_w_ss.*(w_1_ss.*L_1_ss + w_2_ss.*L_2_ss + UB_ss) + tau_a_ss.*Profit_ss + tau_a_ss.*(Profit_FI_ss-param.fix2)))/perturb_size;

F44_aux(T_ind,end-grid.oc+L_1_ind)       = - ((tau_w_ss.*(w_1_ss .*L_1_aux + w_2_ss.*L_2_ss  + UB_ss ) + tau_a_ss.*Profit_ss  + tau_a_ss.*(Profit_FI_ss-param.fix2))-(tau_w_ss.*(w_1_ss.*L_1_ss + w_2_ss.*L_2_ss + UB_ss) + tau_a_ss.*Profit_ss + tau_a_ss.*(Profit_FI_ss-param.fix2)))/perturb_size;
F44_aux(T_ind,end-grid.oc+L_2_ind)       = - ((tau_w_ss.*(w_1_ss .*L_1_ss + w_2_ss.*L_2_aux  + UB_ss ) + tau_a_ss.*Profit_ss  + tau_a_ss.*(Profit_FI_ss-param.fix2))-(tau_w_ss.*(w_1_ss.*L_1_ss + w_2_ss.*L_2_ss + UB_ss) + tau_a_ss.*Profit_ss + tau_a_ss.*(Profit_FI_ss-param.fix2)))/perturb_size;

F44_aux(T_ind,end-grid.oc+UB_ind)        = - ((tau_w_ss.*(w_1_ss .*L_1_ss  + w_2_ss.*L_2_ss  + UB_aux) + tau_a_ss.*Profit_ss  + tau_a_ss.*(Profit_FI_ss-param.fix2))-(tau_w_ss.*(w_1_ss.*L_1_ss + w_2_ss.*L_2_ss + UB_ss) + tau_a_ss.*Profit_ss + tau_a_ss.*(Profit_FI_ss-param.fix2)))/perturb_size;
F44_aux(T_ind,end-grid.oc+Profit_ind)    = - ((tau_w_ss.*(w_1_ss .*L_1_ss  + w_2_ss.*L_2_ss  + UB_ss ) + tau_a_ss.*Profit_aux + tau_a_ss.*(Profit_FI_ss-param.fix2))-(tau_w_ss.*(w_1_ss.*L_1_ss + w_2_ss.*L_2_ss + UB_ss) + tau_a_ss.*Profit_ss + tau_a_ss.*(Profit_FI_ss-param.fix2)))/perturb_size;

% (5) LT

F44_aux(LT_ind,end-grid.oc+LT_ind)       = (LT_aux-LT_ss) /perturb_size;

F41_aux(LT_ind,end-grid.os+GG_ind)       = -(-1/GG_aux+1/GG_ss)*Y_ss/perturb_size;

% (6) G

F44_aux(G_ind,end-grid.oc+G_ind)         = (G_aux-G_ss)/perturb_size;

% F41_aux(G_ind,end-grid.os+GG_ind)        = -(-1/GG_aux+1/GG_ss)*Y_ss/perturb_size;

% (7) Lambda

F44_aux(Lambda_ind,end-grid.oc+Lambda_ind) = (Lambda_aux - Lambda_ss)/perturb_size;

F44_aux(Lambda_ind,end-grid.oc+MRS_ind)    = -(MRS_aux*param.Lambda_aux/param.Lambda_b_aux-MRS_ss*param.Lambda_aux/param.Lambda_b_aux)/perturb_size;

% (8) pi

F44_aux(pi_ind,end-grid.oc+pi_ind)          = (pi_aux-pi_ss)/perturb_size;

F42_aux(pi_ind,end-grid.oc+B_gov_ncp_ind)   = - (param.pi_cb*exp((param.rho_B/(param.rho_B+param.gamma_pi))*log(B_gov_ncp_ss*R_cb_ss/(SS_stats.B_gov_ncp*param.R_cb))-(param.gamma_T/(param.rho_B+param.gamma_pi))*log(T_ss/SS_stats.T)-1/(param.rho_B+param.gamma_pi)*log(B_gov_ncp_aux/SS_stats.B_gov_ncp)) - ...
											     param.pi_cb*exp((param.rho_B/(param.rho_B+param.gamma_pi))*log(B_gov_ncp_ss*R_cb_ss/(SS_stats.B_gov_ncp*param.R_cb))-(param.gamma_T/(param.rho_B+param.gamma_pi))*log(T_ss/SS_stats.T)-1/(param.rho_B+param.gamma_pi)*log(B_gov_ncp_ss/SS_stats.B_gov_ncp)))/perturb_size;
F43_aux(pi_ind,end-grid.os+R_cb_ind)        = - (param.pi_cb*exp((param.rho_B/(param.rho_B+param.gamma_pi))*log(B_gov_ncp_ss*R_cb_aux/(SS_stats.B_gov_ncp*param.R_cb))-(param.gamma_T/(param.rho_B+param.gamma_pi))*log(T_ss/SS_stats.T)-1/(param.rho_B+param.gamma_pi)*log(B_gov_ncp_ss/SS_stats.B_gov_ncp)) - ...
											     param.pi_cb*exp((param.rho_B/(param.rho_B+param.gamma_pi))*log(B_gov_ncp_ss*R_cb_ss/(SS_stats.B_gov_ncp*param.R_cb))-(param.gamma_T/(param.rho_B+param.gamma_pi))*log(T_ss/SS_stats.T)-1/(param.rho_B+param.gamma_pi)*log(B_gov_ncp_ss/SS_stats.B_gov_ncp)))/perturb_size;
F44_aux(pi_ind,end-grid.oc+B_gov_ncp_ind)   = - (param.pi_cb*exp((param.rho_B/(param.rho_B+param.gamma_pi))*log(B_gov_ncp_aux*R_cb_ss/(SS_stats.B_gov_ncp*param.R_cb))-(param.gamma_T/(param.rho_B+param.gamma_pi))*log(T_ss/SS_stats.T)-1/(param.rho_B+param.gamma_pi)*log(B_gov_ncp_ss/SS_stats.B_gov_ncp)) - ...
											     param.pi_cb*exp((param.rho_B/(param.rho_B+param.gamma_pi))*log(B_gov_ncp_ss*R_cb_ss/(SS_stats.B_gov_ncp*param.R_cb))-(param.gamma_T/(param.rho_B+param.gamma_pi))*log(T_ss/SS_stats.T)-1/(param.rho_B+param.gamma_pi)*log(B_gov_ncp_ss/SS_stats.B_gov_ncp)))/perturb_size;
F44_aux(pi_ind,end-grid.oc+T_ind)           = - (param.pi_cb*exp((param.rho_B/(param.rho_B+param.gamma_pi))*log(B_gov_ncp_ss*R_cb_ss/(SS_stats.B_gov_ncp*param.R_cb))-(param.gamma_T/(param.rho_B+param.gamma_pi))*log(T_aux/SS_stats.T)-1/(param.rho_B+param.gamma_pi)*log(B_gov_ncp_ss/SS_stats.B_gov_ncp)) - ...
											     param.pi_cb*exp((param.rho_B/(param.rho_B+param.gamma_pi))*log(B_gov_ncp_ss*R_cb_ss/(SS_stats.B_gov_ncp*param.R_cb))-(param.gamma_T/(param.rho_B+param.gamma_pi))*log(T_ss/SS_stats.T)-1/(param.rho_B+param.gamma_pi)*log(B_gov_ncp2_ss/SS_stats.B_gov_ncp2)))/perturb_size;

% (9-10) Vacancies

F44_aux(V_1_ind,end-grid.oc+V_1_ind)        = (V_1_aux-V_1_ss)/perturb_size;

F44_aux(V_1_ind,end-grid.oc+M_1_ind)        = - ( M_1_aux/param.iota_1*(J_ss(1:grid.ns_1)'*grid.se_dist(1:grid.ns_1))./sum(grid.se_dist(1:grid.ns_1)) - M_1_ss/param.iota_1*(J_ss(1:grid.ns_1)'*grid.se_dist(1:grid.ns_1))./sum(grid.se_dist(1:grid.ns_1)) )/perturb_size;

% for iii = 1:grid.ns_1
for iii = 1:grid.ns	

	J_auxs{iii}      = J_ss;
	J_auxs{iii}(iii) = J_aux(iii);

	F44_aux(V_1_ind,end-grid.oc+J_ind(iii)) = - ( M_1_ss/param.iota_1*(J_auxs{iii}(1:grid.ns_1)'*grid.se_dist(1:grid.ns_1))./sum(grid.se_dist(1:grid.ns_1)) - M_1_ss/param.iota_1*(J_ss(1:grid.ns_1)'*grid.se_dist(1:grid.ns_1))./sum(grid.se_dist(1:grid.ns_1)) )/perturb_size;

end

F44_aux(V_2_ind,end-grid.oc+V_2_ind)        = (V_2_aux-V_2_ss)/perturb_size;

F44_aux(V_2_ind,end-grid.oc+M_2_ind)        = - ( M_2_aux/param.iota_2*(J_ss(grid.ns_1+1:grid.ns_1+grid.ns_2)'*grid.se_dist(grid.ns_1+1:grid.ns_1+grid.ns_2))./sum(grid.se_dist(grid.ns_1+1:grid.ns_1+grid.ns_2)) - M_2_ss/param.iota_2*(J_ss(grid.ns_1+1:grid.ns_1+grid.ns_2)'*grid.se_dist(grid.ns_1+1:grid.ns_1+grid.ns_2))./sum(grid.se_dist(grid.ns_1+1:grid.ns_1+grid.ns_2)) )/perturb_size;

% for iii = 1:grid.ns_2
for iii = 1:grid.ns

	% J_auxs{grid.ns_1+iii}      = J_ss;
	% J_auxs{grid.ns_1+iii}(iii) = J_aux(grid.ns_1+iii);

	J_auxs{iii}      = J_ss;
	J_auxs{iii}(iii) = J_aux(iii);

	F44_aux(V_2_ind,end-grid.oc+J_ind(iii)) = - ( M_2_ss/param.iota_2*(J_auxs{iii}(grid.ns_1+1:grid.ns_1+grid.ns_2)'*grid.se_dist(grid.ns_1+1:grid.ns_1+grid.ns_2))./sum(grid.se_dist(grid.ns_1+1:grid.ns_1+grid.ns_2)) - M_2_ss/param.iota_2*(J_ss(grid.ns_1+1:grid.ns_1+grid.ns_2)'*grid.se_dist(grid.ns_1+1:grid.ns_1+grid.ns_2))./sum(grid.se_dist(grid.ns_1+1:grid.ns_1+grid.ns_2)) )/perturb_size;

end


% (11-...) J

% J_inst_aux        = [(r_l_1-param.fix_L_1-W_1)*n_1*grid.s(1:grid.ns_1)';(r_l_2-param.fix_L_2-W_2)*n_2*grid.s(grid.ns_1+1:grid.ns_1+grid.ns_2)'];

lambdanext_ss     = [(1-l_lambda_1_ss) *eye(grid.ns_1),zeros(grid.ns_1,grid.ns_2);zeros(grid.ns_2,grid.ns_1),(1-l_lambda_2_ss) *eye(grid.ns_2)];
lambdanext_1_aux  = [(1-l_lambda_1_aux)*eye(grid.ns_1),zeros(grid.ns_1,grid.ns_2);zeros(grid.ns_2,grid.ns_1),(1-l_lambda_2_ss) *eye(grid.ns_2)];
lambdanext_2_aux  = [(1-l_lambda_1_ss) *eye(grid.ns_1),zeros(grid.ns_1,grid.ns_2);zeros(grid.ns_2,grid.ns_1),(1-l_lambda_2_aux)*eye(grid.ns_2)];

for iii = 1:grid.ns

	F44_aux(J_ind(iii),end-grid.oc+J_ind(iii))     = (J_auxs{iii}(iii)-J_ss(iii))/perturb_size;

	if iii <= grid.ns_1

		F41_aux(J_ind(iii),end-grid.os+w_1_ind)     = (w_1_aux-w_1_ss)*n_1_ss*grid.s(iii)/perturb_size;
		F44_aux(J_ind(iii),end-grid.oc+r_l_1_ind)   = - (r_l_1_aux-r_l_1_ss)*n_1_ss*grid.s(iii)/perturb_size;
		F44_aux(J_ind(iii),end-grid.oc+n_1_ind)     = - (r_l_1_ss-param.fix_L_1-w_1_ss).*grid.s(iii)*(n_1_aux-n_1_ss)/perturb_size;

	elseif iii > grid.ns_1

		F41_aux(J_ind(iii),end-grid.os+w_2_ind)     = (w_2_aux-w_2_ss)*n_2_ss*grid.s(iii)/perturb_size;
		F44_aux(J_ind(iii),end-grid.oc+r_l_2_ind)   = - (r_l_2_aux-r_l_2_ss)*n_2_ss*grid.s(iii)/perturb_size;
		F44_aux(J_ind(iii),end-grid.oc+n_2_ind)     = - (r_l_2_ss-param.fix_L_2-w_2_ss).*grid.s(iii)*(n_2_aux-n_2_ss)/perturb_size;

	end

	F44_aux(J_ind(iii),end-grid.oc+Lambda_ind)     = - (Lambda_aux*(1-param.death_rate)*(1-param.in)*param.P_SS(iii,:)*lambdanext_ss   *J_ss    - Lambda_ss*(1-param.death_rate)*(1-param.in)*param.P_SS(iii,:)*lambdanext_ss*J_ss )/perturb_size;
	F42_aux(J_ind(iii),end-grid.oc+l_lambda_1_ind) = - (Lambda_ss *(1-param.death_rate)*(1-param.in)*param.P_SS(iii,:)*lambdanext_1_aux*J_ss    - Lambda_ss*(1-param.death_rate)*(1-param.in)*param.P_SS(iii,:)*lambdanext_ss*J_ss )/perturb_size;
	F42_aux(J_ind(iii),end-grid.oc+l_lambda_2_ind) = - (Lambda_ss *(1-param.death_rate)*(1-param.in)*param.P_SS(iii,:)*lambdanext_2_aux*J_ss    - Lambda_ss*(1-param.death_rate)*(1-param.in)*param.P_SS(iii,:)*lambdanext_ss*J_ss )/perturb_size;

	for jjj  = 1:grid.ns

		F42_aux(J_ind(iii),end-grid.oc+J_ind(jjj)) = - (Lambda_ss *(1-param.death_rate)*(1-param.in)*param.P_SS(iii,:)*lambdanext_ss*J_auxs{jjj} - Lambda_ss*(1-param.death_rate)*(1-param.in)*param.P_SS(iii,:)*lambdanext_ss*J_ss )/perturb_size;

	end

end

V_aux_ss      = (param.w*(v_ss *K_ss )^param.rho+(1-param.w)*L_2_ss ^param.rho)^(1/param.rho);
V_aux_K_aux   = (param.w*(v_ss *K_aux)^param.rho+(1-param.w)*L_2_ss ^param.rho)^(1/param.rho);
V_aux_v_aux   = (param.w*(v_aux*K_ss )^param.rho+(1-param.w)*L_2_ss ^param.rho)^(1/param.rho);
V_aux_L_2_aux = (param.w*(v_ss *K_ss )^param.rho+(1-param.w)*L_2_aux^param.rho)^(1/param.rho);

F_aux_ss      = (param.a*L_1_ss ^param.zeta+(1-param.a)*V_aux_ss     ^param.zeta)^(1/param.zeta);
F_aux_K_aux   = (param.a*L_1_ss ^param.zeta+(1-param.a)*V_aux_K_aux  ^param.zeta)^(1/param.zeta);
F_aux_v_aux   = (param.a*L_1_ss ^param.zeta+(1-param.a)*V_aux_v_aux  ^param.zeta)^(1/param.zeta);
F_aux_L_1_aux = (param.a*L_1_aux^param.zeta+(1-param.a)*V_aux_ss     ^param.zeta)^(1/param.zeta);
F_aux_L_2_aux = (param.a*L_1_ss ^param.zeta+(1-param.a)*V_aux_L_2_aux^param.zeta)^(1/param.zeta);

% (20) r_l_1

F44_aux(r_l_1_ind,end-grid.oc+r_l_1_ind) = (r_l_1_aux-r_l_1_ss)/perturb_size;

F41_aux(r_l_1_ind,end-grid.os+Z_ind)     = - ( MC_ss *Z_aux*F_aux_ss     ^(1-param.zeta)*param.a*L_1_ss ^(param.zeta-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*param.a*L_1_ss^(param.zeta-1) )/perturb_size;
F44_aux(r_l_1_ind,end-grid.oc+MC_ind)    = - ( MC_aux*Z_ss *F_aux_ss     ^(1-param.zeta)*param.a*L_1_ss ^(param.zeta-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*param.a*L_1_ss^(param.zeta-1) )/perturb_size;
F44_aux(r_l_1_ind,end-grid.oc+v_ind)     = - ( MC_ss *Z_ss *F_aux_v_aux  ^(1-param.zeta)*param.a*L_1_ss ^(param.zeta-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*param.a*L_1_ss^(param.zeta-1) )/perturb_size;
F44_aux(r_l_1_ind,end-grid.oc+K_ind)     = - ( MC_ss *Z_ss *F_aux_K_aux  ^(1-param.zeta)*param.a*L_1_ss ^(param.zeta-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*param.a*L_1_ss^(param.zeta-1) )/perturb_size;
F44_aux(r_l_1_ind,end-grid.oc+L_1_ind)   = - ( MC_ss *Z_ss *F_aux_L_1_aux^(1-param.zeta)*param.a*L_1_aux^(param.zeta-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*param.a*L_1_ss^(param.zeta-1) )/perturb_size;
F44_aux(r_l_1_ind,end-grid.oc+L_2_ind)   = - ( MC_ss *Z_ss *F_aux_L_2_aux^(1-param.zeta)*param.a*L_1_ss ^(param.zeta-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*param.a*L_1_ss^(param.zeta-1) )/perturb_size;

% (21) r_l_1

F44_aux(r_l_2_ind,end-grid.oc+r_l_2_ind) = (r_l_2_aux-r_l_2_ss)/perturb_size;

F41_aux(r_l_2_ind,end-grid.os+Z_ind)     = - ( MC_ss *Z_aux*F_aux_ss     ^(1-param.zeta)*(1-param.a)*V_aux_ss     ^(param.zeta-param.rho)*(1-param.w)*L_2_ss ^(param.rho-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*(1-param.a)*V_aux_ss^(param.zeta-param.rho)*(1-param.w)*L_2_ss^(param.rho-1) )/perturb_size;
F44_aux(r_l_2_ind,end-grid.oc+MC_ind)    = - ( MC_aux*Z_ss *F_aux_ss     ^(1-param.zeta)*(1-param.a)*V_aux_ss     ^(param.zeta-param.rho)*(1-param.w)*L_2_ss ^(param.rho-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*(1-param.a)*V_aux_ss^(param.zeta-param.rho)*(1-param.w)*L_2_ss^(param.rho-1) )/perturb_size;
F44_aux(r_l_2_ind,end-grid.oc+v_ind)     = - ( MC_ss *Z_ss *F_aux_v_aux  ^(1-param.zeta)*(1-param.a)*V_aux_v_aux  ^(param.zeta-param.rho)*(1-param.w)*L_2_ss ^(param.rho-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*(1-param.a)*V_aux_ss^(param.zeta-param.rho)*(1-param.w)*L_2_ss^(param.rho-1) )/perturb_size;
F44_aux(r_l_2_ind,end-grid.oc+K_ind)     = - ( MC_ss *Z_ss *F_aux_K_aux  ^(1-param.zeta)*(1-param.a)*V_aux_K_aux  ^(param.zeta-param.rho)*(1-param.w)*L_2_ss ^(param.rho-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*(1-param.a)*V_aux_ss^(param.zeta-param.rho)*(1-param.w)*L_2_ss^(param.rho-1) )/perturb_size;
F44_aux(r_l_2_ind,end-grid.oc+L_1_ind)   = - ( MC_ss *Z_ss *F_aux_L_1_aux^(1-param.zeta)*(1-param.a)*V_aux_ss     ^(param.zeta-param.rho)*(1-param.w)*L_2_ss ^(param.rho-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*(1-param.a)*V_aux_ss^(param.zeta-param.rho)*(1-param.w)*L_2_ss^(param.rho-1) )/perturb_size;
F44_aux(r_l_2_ind,end-grid.oc+L_2_ind)   = - ( MC_ss *Z_ss *F_aux_L_2_aux^(1-param.zeta)*(1-param.a)*V_aux_L_2_aux^(param.zeta-param.rho)*(1-param.w)*L_2_aux^(param.rho-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*(1-param.a)*V_aux_ss^(param.zeta-param.rho)*(1-param.w)*L_2_ss^(param.rho-1) )/perturb_size;

% (22) v

F44_aux(v_ind,end-grid.oc+v_ind)         = (v_aux-v_ss)/perturb_size;

% F44_aux(v_ind,end-grid.oc+r_k_ind)       = - ((r_k_aux/(param.delta_0*param.delta_1))^(1/(param.delta_1-1)) - (r_k_ss/(param.delta_0*param.delta_1))^(1/(param.delta_1-1)) )/perturb_size;

% (23) Y

% RHS(nx+Y_ind)      = Z*F_aux;

F44_aux(Y_ind,end-grid.oc+Y_ind)        = (Y_aux-Y_ss)/perturb_size;

F41_aux(Y_ind,end-grid.os+Z_ind)        = - ( Z_aux*F_aux_ss      - Z_ss*F_aux_ss )/perturb_size;
F44_aux(Y_ind,end-grid.oc+v_ind)        = - ( Z_ss *F_aux_v_aux   - Z_ss*F_aux_ss )/perturb_size;
F44_aux(Y_ind,end-grid.oc+K_ind)        = - ( Z_ss *F_aux_K_aux   - Z_ss*F_aux_ss )/perturb_size;
F44_aux(Y_ind,end-grid.oc+L_1_ind)      = - ( Z_ss *F_aux_L_1_aux - Z_ss*F_aux_ss )/perturb_size;
F44_aux(Y_ind,end-grid.oc+L_2_ind)      = - ( Z_ss *F_aux_L_2_aux - Z_ss*F_aux_ss )/perturb_size;

% (24) Profit

F44_aux(Profit_ind,end-grid.oc+Profit_ind)    = (Profit_aux-Profit_ss)/perturb_size;

F41_aux(Profit_ind,end-grid.os+eta_ind)       = - ( ((Y_ss *(1-eta_aux/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F41_aux(Profit_ind,end-grid.os+w_1_ind)       = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_aux).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F41_aux(Profit_ind,end-grid.os+w_2_ind)       = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_aux).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss ).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F41_aux(Profit_ind,end-grid.os+B_F_ind)       = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - (param.fix_ratio + log(B_F_aux))*Y_ss   ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F41_aux(Profit_ind,end-grid.os+Q_ind)         = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_aux*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F42_aux(Profit_ind,end-grid.oc+K_ind)         = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_aux-K_ss)-(K_aux-K_ss) - param.phi/2*(K_aux/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F43_aux(Profit_ind,end-grid.os+pastpi_ind)    = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_aux)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F44_aux(Profit_ind,end-grid.oc+pi_ind)        = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_aux)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F44_aux(Profit_ind,end-grid.oc+Profit_FI_ind) = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) + Profit_FI_aux - param.fix2 - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F44_aux(Profit_ind,end-grid.oc+MC_ind)        = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_aux) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F44_aux(Profit_ind,end-grid.oc+Y_ind)         = - ( ((Y_aux *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_aux.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F44_aux(Profit_ind,end-grid.oc+r_l_1_ind)     = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_aux-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F44_aux(Profit_ind,end-grid.oc+r_l_2_ind)     = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_aux-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F44_aux(Profit_ind,end-grid.oc+L_1_ind)       = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_aux + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss  + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F44_aux(Profit_ind,end-grid.oc+L_2_ind)       = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss  + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_aux - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss  + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss  - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F44_aux(Profit_ind,end-grid.oc+V_1_ind)       = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_aux - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F44_aux(Profit_ind,end-grid.oc+V_2_ind)       = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_aux)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F44_aux(Profit_ind,end-grid.oc+v_ind)         = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_aux - param.delta_0*v_aux^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F44_aux(Profit_ind,end-grid.oc+r_k_ind)       = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_aux*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

F44_aux(Profit_ind,end-grid.oc+K_ind)         = - ( ((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_aux +  Q_ss*(K_ss-K_aux)-(K_ss-K_aux) - param.phi/2*(K_ss/K_aux-1)^2*K_aux ) - ...
													((Y_ss *(1-param.eta/(2*param.kappa)*(log(pi_ss)-(1-param.GAMMA)*log(param.pi_cb)-param.GAMMA*log(pastpi_ss)).^2)+ Y_ss.*(-MC_ss) - param.fix  ...
		                						    + (r_l_1_ss-param.fix_L_1-w_1_ss).*L_1_ss + (r_l_2_ss-param.fix_L_2-w_2_ss).*L_2_ss - param.iota_1.*V_1_ss - param.iota_2*V_2_ss)  ...
		                							+ (r_k_ss*v_ss - param.delta_0*v_ss^param.delta_1).*K_ss +  Q_ss*(K_ss-K_ss)-(K_ss-K_ss) - param.phi/2*(K_ss/K_ss-1)^2*K_ss ) )/perturb_size;

% (25) r^k

F44_aux(r_k_ind,end-grid.oc+r_k_ind)          = (r_k_aux-r_k_ss)/perturb_size;

F41_aux(r_k_ind,end-grid.os+Z_ind)            = -( MC_ss *Z_aux*F_aux_ss     ^(1-param.zeta)*(1-param.a)*V_aux_ss     ^(param.zeta-param.rho)*param.w*(v_ss *K_ss )^(param.rho-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*(1-param.a)*V_aux_ss^(param.zeta-param.rho)*param.w*(v_ss*K_ss)^(param.rho-1) )/perturb_size;
F44_aux(r_k_ind,end-grid.oc+MC_ind)           = -( MC_aux*Z_ss *F_aux_ss     ^(1-param.zeta)*(1-param.a)*V_aux_ss     ^(param.zeta-param.rho)*param.w*(v_ss *K_ss )^(param.rho-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*(1-param.a)*V_aux_ss^(param.zeta-param.rho)*param.w*(v_ss*K_ss)^(param.rho-1) )/perturb_size;
F44_aux(r_k_ind,end-grid.oc+v_ind)            = -( MC_ss *Z_ss *F_aux_v_aux  ^(1-param.zeta)*(1-param.a)*V_aux_v_aux  ^(param.zeta-param.rho)*param.w*(v_ss *K_ss )^(param.rho-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*(1-param.a)*V_aux_ss^(param.zeta-param.rho)*param.w*(v_ss*K_ss)^(param.rho-1) )/perturb_size;
% F44_aux(r_k_ind,end-grid.oc+v_ind)            = -( MC_ss *Z_ss *F_aux_v_aux  ^(1-param.zeta)*(1-param.a)*V_aux_v_aux  ^(param.zeta-param.rho)*param.w*(v_aux *K_ss )^(param.rho-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*(1-param.a)*V_aux_ss^(param.zeta-param.rho)*param.w*(v_ss*K_ss)^(param.rho-1) )/perturb_size;
F44_aux(r_k_ind,end-grid.oc+K_ind)            = -( MC_ss *Z_ss *F_aux_K_aux  ^(1-param.zeta)*(1-param.a)*V_aux_K_aux  ^(param.zeta-param.rho)*param.w*(v_ss *K_aux)^(param.rho-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*(1-param.a)*V_aux_ss^(param.zeta-param.rho)*param.w*(v_ss*K_ss)^(param.rho-1) )/perturb_size;
F44_aux(r_k_ind,end-grid.oc+L_1_ind)          = -( MC_ss *Z_ss *F_aux_L_1_aux^(1-param.zeta)*(1-param.a)*V_aux_ss     ^(param.zeta-param.rho)*param.w*(v_ss *K_ss )^(param.rho-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*(1-param.a)*V_aux_ss^(param.zeta-param.rho)*param.w*(v_ss*K_ss)^(param.rho-1) )/perturb_size;
F44_aux(r_k_ind,end-grid.oc+L_2_ind)          = -( MC_ss *Z_ss *F_aux_L_2_aux^(1-param.zeta)*(1-param.a)*V_aux_L_2_aux^(param.zeta-param.rho)*param.w*(v_ss *K_ss )^(param.rho-1) - MC_ss*Z_ss*F_aux_ss^(1-param.zeta)*(1-param.a)*V_aux_ss^(param.zeta-param.rho)*param.w*(v_ss*K_ss)^(param.rho-1) )/perturb_size;

% (26) r^a

F44_aux(r_a_ind,end-grid.oc+r_a_ind)          = (r_a_aux-r_a_ss)/perturb_size;

F44_aux(r_a_ind,end-grid.oc+Profit_ind)       = - ((1-tau_a_ss)*(1-param.Eratio-param.b_share)*Profit_aux/K_ss - (1-tau_a_ss)*(1-param.Eratio-param.b_share)*Profit_ss/K_ss )/perturb_size;
F44_aux(r_a_ind,end-grid.oc+K_ind)            = - ((1-tau_a_ss)*(1-param.Eratio-param.b_share)*Profit_ss/K_aux - (1-tau_a_ss)*(1-param.Eratio-param.b_share)*Profit_ss/K_ss )/perturb_size;

% (27) MC

F44_aux(MC_ind,end-grid.oc+MC_ind)            = (MC_aux-MC_ss)/perturb_size;

F41_aux(MC_ind,end-grid.os+eta_ind)           = - ( (1-1/eta_aux)-(1-1/eta_ss) )/perturb_size;
F44_aux(MC_ind,end-grid.oc+pi_ind)            = - ( ( 1-1/eta_ss + (log(pi_aux/(pastpi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)))-Lambda_ss*eta2_ss /eta2_ss *Y_ss/Y_ss *log(pi_ss /(pi_aux^param.GAMMA*param.pi_bar^(1-param.GAMMA)) ))/param.kappa  ) - ...
													( 1-1/eta_ss + (log(pi_ss /(pastpi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)))-Lambda_ss*eta2_ss /eta2_ss *Y_ss/Y_ss *log(pi_ss /(pi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)) ))/param.kappa  ) )/perturb_size;
F42_aux(MC_ind,end-grid.oc+pi_ind)            = - ( ( 1-1/eta_ss + (log(pi_ss /(pastpi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)))-Lambda_ss*eta2_ss /eta2_ss *Y_ss/Y_ss *log(pi_aux/(pi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)) ))/param.kappa  ) - ...
													( 1-1/eta_ss + (log(pi_ss /(pastpi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)))-Lambda_ss*eta2_ss /eta2_ss *Y_ss/Y_ss *log(pi_ss /(pi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)) ))/param.kappa  ) )/perturb_size;
F43_aux(MC_ind,end-grid.os+pastpi_ind)        = - ( ( 1-1/eta_ss + (log(pi_ss /(pastpi_aux^param.GAMMA*param.pi_bar^(1-param.GAMMA)))-Lambda_ss*eta2_ss /eta2_ss *Y_ss/Y_ss *log(pi_ss /(pi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)) ))/param.kappa  ) - ...
													( 1-1/eta_ss + (log(pi_ss /(pastpi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)))-Lambda_ss*eta2_ss /eta2_ss *Y_ss/Y_ss *log(pi_ss /(pi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)) ))/param.kappa  ) )/perturb_size;
F44_aux(MC_ind,end-grid.oc+Y_ind)             = - ( ( 1-1/eta_ss + (log(pi_ss /(pastpi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)))-Lambda_ss*eta2_ss /eta2_ss *Y_ss/Y_aux*log(pi_ss /(pi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)) ))/param.kappa  ) - ...
													( 1-1/eta_ss + (log(pi_ss /(pastpi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)))-Lambda_ss*eta2_ss /eta2_ss *Y_ss/Y_ss *log(pi_ss /(pi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)) ))/param.kappa  ) )/perturb_size;
F42_aux(MC_ind,end-grid.oc+Y_ind)             = - ( ( 1-1/eta_ss + (log(pi_ss /(pastpi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)))-Lambda_ss*eta2_ss /eta2_ss *Y_aux/Y_ss*log(pi_ss /(pi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)) ))/param.kappa  ) - ...
													( 1-1/eta_ss + (log(pi_ss /(pastpi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)))-Lambda_ss*eta2_ss /eta2_ss *Y_ss/Y_ss *log(pi_ss /(pi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)) ))/param.kappa  ) )/perturb_size;
F42_aux(MC_ind,end-grid.oc+eta2_ind)          = - ( ( 1-1/eta_ss + (log(pi_ss /(pastpi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)))-Lambda_ss*eta2_aux/eta2_ss *Y_ss/Y_ss *log(pi_ss /(pi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)) ))/param.kappa  ) - ...
													( 1-1/eta_ss + (log(pi_ss /(pastpi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)))-Lambda_ss*eta2_ss /eta2_ss *Y_ss/Y_ss *log(pi_ss /(pi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)) ))/param.kappa  ) )/perturb_size;
F44_aux(MC_ind,end-grid.oc+eta2_ind)          = - ( ( 1-1/eta_ss + (log(pi_ss /(pastpi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)))-Lambda_ss*eta2_ss /eta2_aux*Y_ss/Y_ss *log(pi_ss /(pi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)) ))/param.kappa  ) - ...
													( 1-1/eta_ss + (log(pi_ss /(pastpi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)))-Lambda_ss*eta2_ss /eta2_ss *Y_ss/Y_ss *log(pi_ss /(pi_ss ^param.GAMMA*param.pi_bar^(1-param.GAMMA)) ))/param.kappa  ) )/perturb_size;


% (28) n_1

F44_aux(n_1_ind,end-grid.oc+n_1_ind)           = (n_1_aux-n_1_ss)/perturb_size;

if param.GHH == 1

F41_aux(n_1_ind,end-grid.os+w_1_ind)           = - ( ((1-tau_w_ss)*w_1_aux/param.psi_1)^(1/param.xi)- ((1-tau_w_ss)*w_1_ss/param.psi_1)^(1/param.xi))/perturb_size;

end

% (29) n_2

F44_aux(n_2_ind,end-grid.oc+n_2_ind)           = (n_2_aux-n_2_ss)/perturb_size;

if param.GHH == 1

F41_aux(n_2_ind,end-grid.os+w_2_ind)           = - ( ((1-tau_w_ss)*w_2_aux/param.psi_2)^(1/param.xi)- ((1-tau_w_ss)*w_2_ss/param.psi_2)^(1/param.xi))/perturb_size;

end


% (30) M_1

F44_aux(M_1_ind,end-grid.oc+M_1_ind)    = (M_1_aux-M_1_ss)/perturb_size;

F44_aux(M_1_ind,end-grid.oc+U_1_ind)    = - ( U_1_aux*V_1_ss /((U_1_aux^param.alpha_1+V_1_ss ^param.alpha_1)^(1/param.alpha_1)) - U_1_ss*V_1_ss/((U_1_ss^param.alpha_1+V_1_ss^param.alpha_1)^(1/param.alpha_1)) )/perturb_size;
F44_aux(M_1_ind,end-grid.oc+V_1_ind)    = - ( U_1_ss *V_1_aux/((U_1_ss ^param.alpha_1+V_1_aux^param.alpha_1)^(1/param.alpha_1)) - U_1_ss*V_1_ss/((U_1_ss^param.alpha_1+V_1_ss^param.alpha_1)^(1/param.alpha_1)) )/perturb_size;

% (31) M_2

F44_aux(M_2_ind,end-grid.oc+M_2_ind)    = (M_2_aux-M_2_ss)/perturb_size;

F44_aux(M_2_ind,end-grid.oc+U_2_ind)    = - ( U_2_aux*V_2_ss /((U_2_aux^param.alpha_2+V_2_ss ^param.alpha_2)^(1/param.alpha_2)) - U_2_ss*V_2_ss/((U_2_ss^param.alpha_2+V_2_ss^param.alpha_2)^(1/param.alpha_2)) )/perturb_size;
F44_aux(M_2_ind,end-grid.oc+V_2_ind)    = - ( U_2_ss *V_2_aux/((U_2_ss ^param.alpha_2+V_2_aux^param.alpha_2)^(1/param.alpha_2)) - U_2_ss*V_2_ss/((U_2_ss^param.alpha_2+V_2_ss^param.alpha_2)^(1/param.alpha_2)) )/perturb_size;


% (32) f_1

F44_aux(f_1_ind,end-grid.oc+f_1_ind)     = (f_1_aux-f_1_ss)/perturb_size;

F44_aux(f_1_ind,end-grid.oc+M_1_ind)     = - (M_1_aux/U_1_ss  - M_1_ss/U_1_ss)/perturb_size;
F44_aux(f_1_ind,end-grid.oc+U_1_ind)     = - (M_1_ss /U_1_aux - M_1_ss/U_1_ss)/perturb_size;

% (33) f_2

F44_aux(f_2_ind,end-grid.oc+f_2_ind)     = (f_2_aux-f_2_ss)/perturb_size;

F44_aux(f_2_ind,end-grid.oc+M_2_ind)     = - (M_2_aux/U_2_ss  - M_2_ss/U_2_ss)/perturb_size;
F44_aux(f_2_ind,end-grid.oc+U_2_ind)     = - (M_2_ss /U_2_aux - M_2_ss/U_2_ss)/perturb_size;


% (34) zz

F44_aux(zz_ind,end-grid.oc+zz_ind)        = (zz_aux-zz_ss)/perturb_size;

F44_aux(zz_ind,end-grid.oc+RRa_ind)       = - (((RRa_aux-param.b_a_aux2*R_cb_ss /pi_ss) *lev_ss  + param.b_a_aux2*R_cb_ss /pi_ss)  - ((RRa_ss-R_cb_ss/pi_ss)*lev_ss + R_cb_ss/pi_ss)  )/perturb_size;
F43_aux(zz_ind,end-grid.os+R_cb_ind)      = - (((RRa_ss- param.b_a_aux2*R_cb_aux/pi_ss) *lev_ss  + param.b_a_aux2*R_cb_aux/pi_ss)  - ((RRa_ss-R_cb_ss/pi_ss)*lev_ss + R_cb_ss/pi_ss)  )/perturb_size;
F43_aux(zz_ind,end-grid.os+lev_ind)       = - (((RRa_ss- param.b_a_aux2*R_cb_ss /pi_ss) *lev_aux + param.b_a_aux2*R_cb_ss /pi_ss)  - ((RRa_ss-R_cb_ss/pi_ss)*lev_ss + R_cb_ss/pi_ss)  )/perturb_size;
F44_aux(zz_ind,end-grid.oc+pi_ind)        = - (((RRa_ss- param.b_a_aux2*R_cb_ss /pi_aux)*lev_ss  + param.b_a_aux2*R_cb_ss /pi_aux) - ((RRa_ss-R_cb_ss/pi_ss)*lev_ss + R_cb_ss/pi_ss)  )/perturb_size;

% (35) xx

F44_aux(xx_ind,end-grid.oc+xx_ind)        = (xx_aux-xx_ss)/perturb_size;

F44_aux(xx_ind,end-grid.oc+zz_ind)        = - ((lev_ss/lev_ss)*zz_aux-(lev_ss/lev_ss)*zz_ss)/perturb_size;
F41_aux(xx_ind,end-grid.os+lev_ind)       = - ((lev_aux/lev_ss)*zz_ss-(lev_ss/lev_ss)*zz_ss)/perturb_size;
F43_aux(xx_ind,end-grid.os+lev_ind)       = - ((lev_ss/lev_aux)*zz_ss-(lev_ss/lev_ss)*zz_ss)/perturb_size;

% (36) vv

F44_aux(vv_ind,end-grid.oc+vv_ind)        = (vv_aux-vv_ss)/perturb_size;

F41_aux(vv_ind,end-grid.os+R_cb_ind)      = - ( ((1-param.theta_b)*Lambda_ss       *(RRa_ss -param.b_a_aux2*R_cb_aux/pi_ss)+param.theta_b*Lambda_ss       *xx_ss*vv_ss)  - ((1-param.theta_b)*Lambda_ss*(RRa_ss-R_cb_ss/pi_ss)+param.theta_b*Lambda_ss*xx_ss*vv_ss)  )/perturb_size;
F41_aux(vv_ind,end-grid.os+BB_ind)        = - ( ((1-param.theta_b)*BB_aux*Lambda_ss*(RRa_ss -param.b_a_aux2*R_cb_ss/pi_ss) +param.theta_b*BB_aux*Lambda_ss*xx_ss*vv_ss)  - ((1-param.theta_b)*Lambda_ss*(RRa_ss-R_cb_ss/pi_ss)+param.theta_b*Lambda_ss*xx_ss*vv_ss)  )/perturb_size;
F42_aux(vv_ind,end-grid.oc+RRa_ind)       = - ( ((1-param.theta_b)*Lambda_ss       *(RRa_aux-param.b_a_aux2*R_cb_ss/pi_ss) +param.theta_b*Lambda_ss       *xx_ss*vv_ss)  - ((1-param.theta_b)*Lambda_ss*(RRa_ss-R_cb_ss/pi_ss)+param.theta_b*Lambda_ss*xx_ss*vv_ss)  )/perturb_size;
F42_aux(vv_ind,end-grid.oc+pi_ind)        = - ( ((1-param.theta_b)*Lambda_ss       *(RRa_ss -param.b_a_aux2*R_cb_ss/pi_aux)+param.theta_b*Lambda_ss       *xx_ss*vv_ss)  - ((1-param.theta_b)*Lambda_ss*(RRa_ss-R_cb_ss/pi_ss)+param.theta_b*Lambda_ss*xx_ss*vv_ss)  )/perturb_size;
F42_aux(vv_ind,end-grid.oc+xx_ind)        = - ( ((1-param.theta_b)*Lambda_ss       *(RRa_ss -param.b_a_aux2*R_cb_ss/pi_ss) +param.theta_b*Lambda_ss       *xx_aux*vv_ss) - ((1-param.theta_b)*Lambda_ss*(RRa_ss-R_cb_ss/pi_ss)+param.theta_b*Lambda_ss*xx_ss*vv_ss)  )/perturb_size;
F42_aux(vv_ind,end-grid.oc+vv_ind)        = - ( ((1-param.theta_b)*Lambda_ss       *(RRa_ss -param.b_a_aux2*R_cb_ss/pi_ss) +param.theta_b*Lambda_ss       *xx_ss*vv_aux) - ((1-param.theta_b)*Lambda_ss*(RRa_ss-R_cb_ss/pi_ss)+param.theta_b*Lambda_ss*xx_ss*vv_ss)  )/perturb_size;
F44_aux(vv_ind,end-grid.oc+Lambda_ind)    = - ( ((1-param.theta_b)*Lambda_aux      *(RRa_ss -param.b_a_aux2*R_cb_ss/pi_ss) +param.theta_b*Lambda_aux      *xx_ss*vv_ss)  - ((1-param.theta_b)*Lambda_ss*(RRa_ss-R_cb_ss/pi_ss)+param.theta_b*Lambda_ss*xx_ss*vv_ss)  )/perturb_size;

% (37) ee

F44_aux(ee_ind,end-grid.oc+ee_ind)            = (ee_aux-ee_ss)/perturb_size;

F41_aux(ee_ind,end-grid.os+R_cb_ind)          = -((1-param.theta_b)*Lambda_ss*(R_cb_aux-param.b_a_aux2*R_cb_ss)/pi_ss)/perturb_size;
F41_aux(ee_ind,end-grid.os+BB_ind)            = -(((1-param.theta_b)*BB_aux*Lambda_ss*param.b_a_aux2*R_cb_ss/pi_ss+param.theta_b*BB_aux*Lambda_ss*zz_ss*ee_ss)-((1-param.theta_b)*Lambda_ss*R_cb_ss/pi_ss+param.theta_b*Lambda_ss*zz_ss*ee_ss))/perturb_size;
F44_aux(ee_ind,end-grid.oc+Lambda_ind)        = -(((1-param.theta_b)*Lambda_aux*param.b_a_aux2*R_cb_ss/pi_ss +param.theta_b*Lambda_aux*zz_ss *ee_ss) -((1-param.theta_b)*Lambda_ss*R_cb_ss/pi_ss+param.theta_b*Lambda_ss*zz_ss*ee_ss))/perturb_size;
F42_aux(ee_ind,end-grid.oc+pi_ind)            = -(((1-param.theta_b)*Lambda_ss *param.b_a_aux2*R_cb_ss/pi_aux+param.theta_b*Lambda_ss *zz_ss *ee_ss) -((1-param.theta_b)*Lambda_ss*R_cb_ss/pi_ss+param.theta_b*Lambda_ss*zz_ss*ee_ss))/perturb_size;
F42_aux(ee_ind,end-grid.oc+zz_ind)            = -(((1-param.theta_b)*Lambda_ss *param.b_a_aux2*R_cb_ss/pi_ss +param.theta_b*Lambda_ss *zz_aux*ee_ss) -((1-param.theta_b)*Lambda_ss*R_cb_ss/pi_ss+param.theta_b*Lambda_ss*zz_ss*ee_ss))/perturb_size;
F42_aux(ee_ind,end-grid.oc+ee_ind)            = -(((1-param.theta_b)*Lambda_ss *param.b_a_aux2*R_cb_ss/pi_ss +param.theta_b*Lambda_ss *zz_ss *ee_aux)-((1-param.theta_b)*Lambda_ss*R_cb_ss/pi_ss+param.theta_b*Lambda_ss*zz_ss*ee_ss))/perturb_size;

% (38) C_b

F44_aux(C_b_ind,end-grid.oc+C_b_ind)          = (C_b_aux-C_b_ss)/perturb_size;

F41_aux(C_b_ind,end-grid.os+R_cb_ind)         = - ( ((1*param.beta_b*R_cb_aux/pi_ss) /(1+param.phi_B*(B_b_ss/B_b_ss-1))*C_b_ss^(-param.sigma2))^(-1/param.sigma2) -  ((1*param.beta_b*R_cb_ss/pi_ss)/(1+param.phi_B*(B_b_ss/B_b_ss-1))*C_b_ss^(-param.sigma2))^(-1/param.sigma2) )/perturb_size;
F41_aux(C_b_ind,end-grid.os+D_ind)            = - ( ((1*param.beta_b*D_aux*R_cb_ss/pi_ss) /(1+param.phi_B*(B_b_ss/B_b_ss-1))*C_b_ss^(-param.sigma2))^(-1/param.sigma2) -  ((1*param.beta_b*R_cb_ss/pi_ss)/(1+param.phi_B*(B_b_ss/B_b_ss-1))*C_b_ss^(-param.sigma2))^(-1/param.sigma2) )/perturb_size;
F42_aux(C_b_ind,end-grid.oc+pi_ind)           = - ( ((1*param.beta_b*R_cb_ss /pi_aux)/(1+param.phi_B*(B_b_ss/B_b_ss-1))*C_b_ss^(-param.sigma2))^(-1/param.sigma2) -  ((1*param.beta_b*R_cb_ss/pi_ss)/(1+param.phi_B*(B_b_ss/B_b_ss-1))*C_b_ss^(-param.sigma2))^(-1/param.sigma2) )/perturb_size;
F42_aux(C_b_ind,end-grid.oc+C_b_ind)          = - ( ((1*param.beta_b*R_cb_ss /pi_ss) /(1+param.phi_B*(B_b_ss/B_b_ss-1))*C_b_aux^(-param.sigma2))^(-1/param.sigma2) - ((1*param.beta_b*R_cb_ss/pi_ss)/(1+param.phi_B*(B_b_ss/B_b_ss-1))*C_b_ss^(-param.sigma2))^(-1/param.sigma2) )/perturb_size;

% (39) Profit_FI

F44_aux(Profit_FI_ind,end-grid.oc+Profit_FI_ind) = (Profit_FI_aux-Profit_FI_ss)/perturb_size;

F44_aux(Profit_FI_ind,end-grid.oc+RRa_ind)       = - ( ((1-param.theta_b)*((RRa_aux - param.b_a_aux2*R_cb_ss/pi_ss) *lev_ss +param.b_a_aux2*R_cb_ss/pi_ss) *NW_b_ss  - param.omega*Q_ss*A_b_ss)  - ((1-param.theta_b)*((RRa_ss - param.b_a_aux2*R_cb_ss/pi_ss)*lev_ss+R_cb_ss/pi_ss)*NW_b_ss - param.omega*Q_ss*A_b_ss)     )/perturb_size;
F44_aux(Profit_FI_ind,end-grid.oc+pi_ind)        = - ( ((1-param.theta_b)*((RRa_ss  - param.b_a_aux2*R_cb_ss/pi_aux)*lev_ss +param.b_a_aux2*R_cb_ss/pi_aux)*NW_b_ss  - param.omega*Q_ss*A_b_ss)  - ((1-param.theta_b)*((RRa_ss - param.b_a_aux2*R_cb_ss/pi_ss)*lev_ss+R_cb_ss/pi_ss)*NW_b_ss - param.omega*Q_ss*A_b_ss)     )/perturb_size;
F43_aux(Profit_FI_ind,end-grid.os+lev_ind)       = - ( ((1-param.theta_b)*((RRa_ss  - param.b_a_aux2*R_cb_ss/pi_ss) *lev_aux+param.b_a_aux2*R_cb_ss/pi_ss) *NW_b_ss  - param.omega*Q_ss*A_b_ss)  - ((1-param.theta_b)*((RRa_ss - param.b_a_aux2*R_cb_ss/pi_ss)*lev_ss+R_cb_ss/pi_ss)*NW_b_ss - param.omega*Q_ss*A_b_ss)     )/perturb_size;
F43_aux(Profit_FI_ind,end-grid.os+R_cb_ind)      = - ( ((1-param.theta_b)*((RRa_ss  - param.b_a_aux2*R_cb_aux/pi_ss)*lev_ss +param.b_a_aux2*R_cb_aux/pi_ss)*NW_b_ss  - param.omega*Q_ss*A_b_ss)  - ((1-param.theta_b)*((RRa_ss - param.b_a_aux2*R_cb_ss/pi_ss)*lev_ss+R_cb_ss/pi_ss)*NW_b_ss - param.omega*Q_ss*A_b_ss)     )/perturb_size;
F43_aux(Profit_FI_ind,end-grid.os+NW_b_ind)      = - ( ((1-param.theta_b)*((RRa_ss  - param.b_a_aux2*R_cb_ss/pi_ss) *lev_ss +param.b_a_aux2*R_cb_ss/pi_ss) *NW_b_aux - param.omega*Q_ss*A_b_ss)  - ((1-param.theta_b)*((RRa_ss - param.b_a_aux2*R_cb_ss/pi_ss)*lev_ss+R_cb_ss/pi_ss)*NW_b_ss - param.omega*Q_ss*A_b_ss)     )/perturb_size;
F43_aux(Profit_FI_ind,end-grid.os+Q_ind)         = - ( ((1-param.theta_b)*((RRa_ss  - param.b_a_aux2*R_cb_ss/pi_ss) *lev_ss +param.b_a_aux2*R_cb_ss/pi_ss) *NW_b_ss  - param.omega*Q_aux*A_b_ss) - ((1-param.theta_b)*((RRa_ss - param.b_a_aux2*R_cb_ss/pi_ss)*lev_ss+R_cb_ss/pi_ss)*NW_b_ss - param.omega*Q_ss*A_b_ss)     )/perturb_size;
F43_aux(Profit_FI_ind,end-grid.os+A_b_ind)       = - ( ((1-param.theta_b)*((RRa_ss  - param.b_a_aux2*R_cb_ss/pi_ss) *lev_ss +param.b_a_aux2*R_cb_ss/pi_ss) *NW_b_ss  - param.omega*Q_ss*A_b_aux) - ((1-param.theta_b)*((RRa_ss - param.b_a_aux2*R_cb_ss/pi_ss)*lev_ss+R_cb_ss/pi_ss)*NW_b_ss - param.omega*Q_ss*A_b_ss)     )/perturb_size;

% (40) RRa

F44_aux(RRa_ind,end-grid.oc+RRa_ind)             = (RRa_aux-RRa_ss)/perturb_size;

F41_aux(RRa_ind,end-grid.os+Q_ind)               = - ((Q_aux+r_a_ss)/Q_ss-(Q_ss+r_a_ss)/Q_ss)/perturb_size;
F43_aux(RRa_ind,end-grid.os+Q_ind)               = - ((Q_ss+r_a_ss)/Q_aux-(Q_ss+r_a_ss)/Q_ss)/perturb_size;
F44_aux(RRa_ind,end-grid.oc+r_a_ind)             = - ((Q_ss+r_a_aux)/Q_ss-(Q_ss+r_a_ss)/Q_ss)/perturb_size;

% (41) RR

F44_aux(RR_ind,end-grid.oc+RR_ind)               = (RR_aux-RR_ss)/perturb_size;

F43_aux(RR_ind,end-grid.os+R_cb_ind)             = - (R_cb_aux/pi_ss-R_cb_ss/pi_ss)/perturb_size;
F44_aux(RR_ind,end-grid.oc+pi_ind)               = - (R_cb_ss/pi_aux-R_cb_ss/pi_ss)/perturb_size;

% (42) I

F44_aux(I_ind,end-grid.oc+I_ind)    = (I_aux-I_ss)/perturb_size;

F41_aux(I_ind,end-grid.os+iota_ind) = - (iota_aux*K_ss-K_ss)/perturb_size; 
F41_aux(I_ind,end-grid.os+A_b_ind)  = - (A_b_aux-A_b_ss)/perturb_size; 
F41_aux(I_ind,end-grid.os+A_g_ind)  = - (A_g_aux-A_g_ss)/perturb_size; 
F42_aux(I_ind,end-grid.oc+A_hh_ind) = - (A_hh_aux-A_hh_ss)/perturb_size; 
F44_aux(I_ind,end-grid.oc+K_ind)    = (1-param.delta_0*v_ss^param.delta_1)*(K_aux-K_ss)/perturb_size;
F44_aux(I_ind,end-grid.oc+v_ind)    = - param.delta_0*(v_aux^param.delta_1-v_ss^param.delta_1)*K_ss/perturb_size;
F44_aux(I_ind,end-grid.oc+x_k_ind)  = - (K_ss*(1+param.phi/2*(log(x_k_aux))^2)-K_ss)/perturb_size; 

% (43) x_k

F44_aux(x_k_ind,end-grid.oc+x_k_ind)  = (x_k_aux-x_k_ss)/perturb_size;

F42_aux(x_k_ind,end-grid.oc+K_ind)    = - (K_aux/K_ss-K_ss/K_ss)/perturb_size;            
F44_aux(x_k_ind,end-grid.oc+K_ind)    = - (K_ss/K_aux-K_ss/K_ss)/perturb_size;                          


% (44) A_g_obs

F44_aux(A_g_obs_ind,end-grid.oc+A_g_obs_ind)     = (A_g_obs_aux - A_g_obs_ss)/perturb_size;

F41_aux(A_g_obs_ind,end-grid.os+A_g_ind)         = - ( (A_g_aux) - (A_g_ss) )/perturb_size;

% (45) Y_obs

F44_aux(Y_obs_ind,end-grid.oc+Y_obs_ind)     = (Y_obs_aux-Y_obs_ss)/perturb_size;

F44_aux(Y_obs_ind,end-grid.oc+Y_ind)         = - (Y_aux-Y_ss)/perturb_size;


% (46) C_obs

F44_aux(C_obs_ind,end-grid.oc+C_obs_ind)     = (C_obs_aux-C_obs_ss)/perturb_size;

F44_aux(C_obs_ind,end-grid.oc+C_ind)         = - (C_aux-C_ss)/perturb_size;

% (47) I_obs

F44_aux(I_obs_ind,end-grid.oc+I_obs_ind)     = (I_obs_aux-I_obs_ss)/perturb_size;

F44_aux(I_obs_ind,end-grid.oc+I_ind)         = - (I_aux-I_ss)/perturb_size;

% (48) w_1_obs

F44_aux(w_1_obs_ind,end-grid.oc+w_1_obs_ind)  = (w_1_obs_aux-w_1_obs_ss)/perturb_size;

F41_aux(w_1_obs_ind,end-grid.os+w_1_ind)      = - (w_1_aux -  w_1_ss )/perturb_size;

% (49) w_2_obs 

F44_aux(w_2_obs_ind,end-grid.oc+w_2_obs_ind)  = (w_2_obs_aux-w_2_obs_ss)/perturb_size;

F41_aux(w_2_obs_ind,end-grid.os+w_2_ind)      = - (w_2_aux -  w_2_ss )/perturb_size;

% (50) w_obs 

F44_aux(w_obs_ind,end-grid.oc+w_obs_ind)  = (w_obs_aux-w_obs_ss)/perturb_size;

F41_aux(w_obs_ind,end-grid.os+w_ind)      = - (w_aux -  w_ss )/perturb_size;

% (51) Profit obs

F44_aux(PROFIT_obs_ind,end-grid.oc+PROFIT_obs_ind)  = ( PROFIT_obs_aux - PROFIT_obs_ss )/perturb_size;

F44_aux(PROFIT_obs_ind,end-grid.oc+Profit_ind)      = - ( Profit_aux - Profit_ss )/perturb_size;

% F41_aux(PROFIT_obs_ind,end-grid.os+B_F_ind)         = - ( log(B_F_aux) - log(B_F_ss) )/perturb_size;

% (52) unemp_1_obs

F44_aux(unemp_1_obs_ind,end-grid.oc+unemp_1_obs_ind)  = ( unemp_1_obs_aux - unemp_1_obs_ss )/perturb_size;

F44_aux(unemp_1_obs_ind,end-grid.oc+unemp_1_ind)      = - ( 100*unemp_1_aux - 100*unemp_1_ss )/perturb_size;

% (53) unemp_2_obs

F44_aux(unemp_2_obs_ind,end-grid.oc+unemp_2_obs_ind)  = ( unemp_2_obs_aux - unemp_2_obs_ss )/perturb_size;

F44_aux(unemp_2_obs_ind,end-grid.oc+unemp_2_ind)      = - ( 100*unemp_2_aux - 100*unemp_2_ss )/perturb_size;

% (54) unemp_obs

F44_aux(unemp_obs_ind,end-grid.oc+unemp_obs_ind)  = ( unemp_obs_aux - unemp_obs_ss )/perturb_size;

F44_aux(unemp_obs_ind,end-grid.oc+unemp_ind)      = - ( 100*unemp_aux - 100*unemp_ss )/perturb_size;

% (55) PI_obs

F44_aux(inf_obs_ind,end-grid.oc+inf_obs_ind)  = ( inf_obs_aux - inf_obs_ss )/perturb_size;

F44_aux(inf_obs_ind,end-grid.oc+pi_ind)       = - ( pi_aux - pi_ss )/perturb_size;

% (56) R_obs

F44_aux(R_obs_ind,end-grid.oc+R_obs_ind)  = ( R_obs_aux - R_obs_ss )/perturb_size;

F41_aux(R_obs_ind,end-grid.os+R_cb_ind)   = - ( R_cb_aux - R_cb_ss )/perturb_size;

% (57) pastw_1

F44_aux(pastw_1_ind,end-grid.oc+pastw_1_ind)  = ( pastw_1_aux - pastw_1_ss )/perturb_size;

F43_aux(pastw_1_ind,end-grid.os+w_1_ind)      = - ( w_1_aux - w_1_ss )/perturb_size;

% (58) pastw_2

F44_aux(pastw_2_ind,end-grid.oc+pastw_2_ind)  = ( pastw_2_aux - pastw_2_ss )/perturb_size;

F43_aux(pastw_2_ind,end-grid.os+w_2_ind)      = - ( w_2_aux - w_2_ss )/perturb_size;

% (59) pastw

F44_aux(pastw_ind,end-grid.oc+pastw_ind)  = ( pastw_aux - pastw_ss )/perturb_size;

F43_aux(pastw_ind,end-grid.os+w_ind)      = - ( w_aux - w_ss )/perturb_size;

% (60) G obs

F44_aux(G_obs_ind,end-grid.oc+G_obs_ind)     = (G_obs_aux-G_obs_ss)/perturb_size;

F44_aux(G_obs_ind,end-grid.oc+G_ind)         = - (G_aux-G_ss)/perturb_size;

% (61) pastYY 

F43_aux(pastYY_ind,end-grid.os+pastY_ind)   = - ( pastY_aux - pastY_ss )/perturb_size;
F44_aux(pastYY_ind,end-grid.oc+pastYY_ind)  = ( pastYY_aux - pastYY_ss )/perturb_size;

% (62) pastCC

F43_aux(pastCC_ind,end-grid.os+pastC_ind)   = - ( pastC_aux - pastC_ss )/perturb_size;
F44_aux(pastCC_ind,end-grid.oc+pastCC_ind)  = ( pastCC_aux - pastCC_ss )/perturb_size;

% (63) pastII

F43_aux(pastII_ind,end-grid.os+pastI_ind)   = - ( pastI_aux - pastI_ss )/perturb_size;
F44_aux(pastII_ind,end-grid.oc+pastII_ind)  = ( pastII_aux - pastII_ss )/perturb_size;

% (64) Past profit

F43_aux(pastPPROFIT_ind,end-grid.os+pastPROFIT_ind)   = - ( pastPROFIT_aux - pastPROFIT_ss )/perturb_size;
F44_aux(pastPPROFIT_ind,end-grid.oc+pastPPROFIT_ind)  = ( pastPPROFIT_aux - pastPPROFIT_ss )/perturb_size;

% (65) Past uu_1

F43_aux(pastuu_1_ind,end-grid.os+pastunemp_1_ind) = - ( pastunemp_1_aux - pastunemp_1_ss )/perturb_size;
F44_aux(pastuu_1_ind,end-grid.oc+pastuu_1_ind)    = ( pastuu_1_aux - pastuu_1_ss )/perturb_size;

% (66) Past uu_2

F43_aux(pastuu_2_ind,end-grid.os+pastunemp_2_ind) = - ( pastunemp_2_aux - pastunemp_2_ss )/perturb_size;
F44_aux(pastuu_2_ind,end-grid.oc+pastuu_2_ind)    = ( pastuu_2_aux - pastuu_2_ss )/perturb_size;

% (67) Past uu

F43_aux(pastuu_ind,end-grid.os+pastunemp_ind) = - ( pastunemp_aux - pastunemp_ss )/perturb_size;
F44_aux(pastuu_ind,end-grid.oc+pastuu_ind)    = ( pastuu_aux - pastuu_ss )/perturb_size;

% (68) Past GG

F44_aux(pastGG_ind,end-grid.oc+pastGG_ind) = ( pastGG_aux - pastGG_ss )/perturb_size;

F43_aux(pastGG_ind,end-grid.os+pastLT_ind) = - ( pastLT_aux - pastLT_ss )/perturb_size;
% F43_aux(pastGG_ind,end-grid.os+pastG_ind)  = - ( pastG_aux - pastG_ss )/perturb_size;

% (69) Past A_G

F43_aux(pastA_g_ind,end-grid.os+A_g_ind)     = - ( A_g_aux - A_g_ss )/perturb_size;
F44_aux(pastA_g_ind,end-grid.oc+pastA_g_ind) = ( pastA_g_aux - pastA_g_ss )/perturb_size;

% (70) l_lambda_1

% F41_aux(l_lambda_1_ind,end-grid.os+iota_ind)     = - ( log(iota_aux) - log(iota_ss) )/perturb_size;
F44_aux(l_lambda_1_ind,end-grid.oc+l_lambda_1_ind) = ( l_lambda_1_aux - l_lambda_1_ss )/perturb_size;

% (71) l_lambda_2

% F41_aux(l_lambda_2_ind,end-grid.os+iota_ind)     = - ( log(iota_aux) - log(iota_ss) )/perturb_size;
F44_aux(l_lambda_2_ind,end-grid.oc+l_lambda_2_ind) = ( l_lambda_2_aux - l_lambda_2_ss )/perturb_size;

% (72) past Q

F43_aux(pastQ_ind,end-grid.os+Q_ind)     = - ( Q_aux - Q_ss )/perturb_size;
F44_aux(pastQ_ind,end-grid.oc+pastQ_ind) = ( pastQ_aux - pastQ_ss )/perturb_size;

% (73) x_I

F43_aux(x_I_ind,end-grid.os+pastI_ind) = - ( I_ss/pastI_aux - I_ss/pastI_ss )/perturb_size;
F44_aux(x_I_ind,end-grid.oc+I_ind)     = - ( I_aux/pastI_ss - I_ss/pastI_ss )/perturb_size;
F44_aux(x_I_ind,end-grid.oc+x_I_ind)   = ( x_I_aux - x_I_ss )/perturb_size;

% (74) eta2

% F41_aux(eta2_ind,end-grid.os+eta_ind)     = - ( eta_aux/param.eta - eta_ss/param.eta )/perturb_size;
F41_aux(eta2_ind,end-grid.os+eta_ind)   = - ( eta_aux - eta_ss )/perturb_size;
F44_aux(eta2_ind,end-grid.oc+eta2_ind)  = ( eta2_aux - eta2_ss )/perturb_size;

% (75) iota2

F41_aux(iota2_ind,end-grid.os+iota_ind)  = - ( iota_aux - iota_ss )/perturb_size;
F44_aux(iota2_ind,end-grid.oc+iota2_ind) = ( iota2_aux - iota2_ss )/perturb_size;

% (76) pastLT2

F44_aux(pastLT2_ind,end-grid.oc+pastLT2_ind)  = ( pastLT2_aux - pastLT2_ss )/perturb_size;
F43_aux(pastLT2_ind,end-grid.os+pastLT_ind)   = - ( pastLT_aux - pastLT_ss )/perturb_size;

% (77) pastG2

F44_aux(pastG2_ind,end-grid.oc+pastG2_ind)  = ( pastG2_aux - pastG2_ss )/perturb_size;
F43_aux(pastG2_ind,end-grid.os+pastG_ind)   = - ( pastG_aux - pastG_ss )/perturb_size;

% (78) LT_obs

F44_aux(LT_obs_ind,end-grid.oc+LT_obs_ind)  = ( LT_obs_aux - LT_obs_ss )/perturb_size;
F44_aux(LT_obs_ind,end-grid.oc+LT_ind)      = - ( LT_aux - LT_ss )/perturb_size;

% (79) B_gov_ncp2

F43_aux(B_gov_ncp2_ind,end-grid.os+A_b_ind)       = - ( (B_b_ss + B_hh_ss + SS_stats.B_F - (Q_ss*A_b_aux- NW_b_ss) - Q_ss *(1+param.tau_cp)* (A_g_ss)) - ( (B_b_ss + B_hh_ss + SS_stats.B_F - (Q_ss*A_b_ss- NW_b_ss) - Q_ss *(1+param.tau_cp)* (A_g_ss))  )  )/perturb_size; 
F43_aux(B_gov_ncp2_ind,end-grid.os+B_b_ind)       = - ( (B_b_aux + B_hh_ss + SS_stats.B_F - (Q_ss*A_b_ss- NW_b_ss) - Q_ss *(1+param.tau_cp)* (A_g_ss)) - ( (B_b_ss + B_hh_ss + SS_stats.B_F - (Q_ss*A_b_ss- NW_b_ss) - Q_ss *(1+param.tau_cp)* (A_g_ss))  )  )/perturb_size;
F43_aux(B_gov_ncp2_ind,end-grid.os+A_g_ind)       = - ( (B_b_ss + B_hh_ss + SS_stats.B_F - (Q_ss*A_b_ss- NW_b_ss) - Q_ss *(1+param.tau_cp)* (A_g_aux)) - ( (B_b_ss + B_hh_ss + SS_stats.B_F - (Q_ss*A_b_ss- NW_b_ss) - Q_ss *(1+param.tau_cp)* (A_g_ss))  )  )/perturb_size;
F43_aux(B_gov_ncp2_ind,end-grid.os+Q_ind)         = - ( (B_b_ss + B_hh_ss + SS_stats.B_F - (Q_aux*A_b_ss- NW_b_ss) - Q_aux *(1+param.tau_cp)* (A_g_ss)) - ( (B_b_ss + B_hh_ss + SS_stats.B_F - (Q_ss*A_b_ss- NW_b_ss) - Q_ss *(1+param.tau_cp)* (A_g_ss))  )  )/perturb_size;
F43_aux(B_gov_ncp2_ind,end-grid.os+NW_b_ind)      = - ( (B_b_ss + B_hh_ss + SS_stats.B_F - (Q_ss*A_b_ss- NW_b_aux) - Q_ss *(1+param.tau_cp)* (A_g_ss)) - ( (B_b_ss + B_hh_ss + SS_stats.B_F - (Q_ss*A_b_ss- NW_b_ss) - Q_ss *(1+param.tau_cp)* (A_g_ss))  )  )/perturb_size;
F44_aux(B_gov_ncp2_ind,end-grid.oc+B_hh_ind)      = - (B_hh_aux-B_hh_ss)/perturb_size;
F44_aux(B_gov_ncp2_ind,end-grid.oc+B_gov_ncp2_ind) = (B_gov_ncp2_aux-B_gov_ncp2_ss)/perturb_size;






out.F21_aux = F21_aux;
out.F22_aux = F22_aux;
out.F23_aux = F23_aux;
out.F24_aux = F24_aux;
out.F41_aux = F41_aux;
out.F42_aux = F42_aux;
out.F43_aux = F43_aux;
out.F44_aux = F44_aux;