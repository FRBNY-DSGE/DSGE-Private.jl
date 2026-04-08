## ============================================================================
## Heterogeneous-Agent Block for the kshape Economy
##
## Included inside Fsys(F, X, XPrime, θ, grids, id, ss, eqconds,
##   dct_compression_indices, Γ, DC, IDC, DCD, IDCD, m)
##
## In scope from Fsys preamble:
##   b_ndgrid, a_ndgrid, se_ndgrid, b_grid, a_grid, se_grid
##   nb, na, nse  (= size(b_ndgrid))
##   F, X, XPrime, θ, grids, id, ss, eqconds, dct_compression_indices
##   Γ, DC, IDC, DCD, IDCD, m
##   All aggregate variables from @sslogdeviations2levels
## ============================================================================
using Revise, SparseArrays
using DSGE, ModelConstructors
using JLD2
using OrderedCollections: OrderedDict
using DSGE: @sslogdeviations2levels, @sslogdeviations2levels_unprimekeys, @unpack_and_first
include("helpers/DCT/uncompress.jl")
include("helpers/DCT/compress.jl")
include("helpers/grids/myinterpolate3.jl")
include("helpers/cdf_to_pdf3D_forwarddiff.jl")
include("helpers/genweight.jl")
include("helpers/sub2ind.jl")
include("steadystate/Fastroot.jl")
include("dynamics/policies_update.jl")
include("util.jl")
F = OrderedDict{Symbol, Any}()
m = mBBQ()
filepath = joinpath(pwd(), "data/SSInfo.jld2")
m <= Setting(:ss_file_path, filepath)
DSGE.populate_from_jld2!(m)
DSGE.setup_indices!(m)

@load "data/XssYss.jld2" StateSS ControlSS SS_stats grid param
grid = Dict{Symbol, Any}(Symbol(k) => grid[k] for k in ["nb", "na", "nse", "ns", "nCOP", "oc", "os", "numstates", "s_dist", "s", "K"])
state_id, control_id = DSGE.build_indices(grid, length(StateSS), length(ControlSS))
ss = DSGE.build_ss(param, SS_stats)
m <= Setting(:COP_SS, SS_stats["COP_SS"])

# Utility functions (CRRA with σ = θ[:σ_2])
#=
util     = c -> (c .^ (1 - θ[:σ_2])) ./ (1 - θ[:σ_2])
mutil    = c -> 1.0 ./ (c .^ θ[:σ_2])
invmutil = mu -> (1.0 ./ mu) .^ (1 / θ[:σ_2])
=#
# Grid dimensions from settings (already set by model_settings! / populate_from_jld2!)
ns        = get_setting(m, :ns)
nb_cop    = get_setting(m, :nb_copula)
na_cop    = get_setting(m, :na_copula)
nse_cop   = get_setting(m, :nse_copula)
nb        = get_setting(m, :nb)
na        = get_setting(m, :na)
nse       = get_setting(m, :nse)

DC = get_setting(m, :DC)
IDC = get_setting(m, :IDC)
DCD = get_setting(m, :DCD)
IDCD = get_setting(m, :IDCD)
nState   = length(StateSS)
nCtrl    = length(ControlSS)
nRedCtrl = get_setting(m, :nRedCtrl)
dct_compression_indices = get_setting(m, :dct_compression_indices)
θ = parameters2namedtuple(m)

# At steady state: zero log-deviations for all vars
# To perturb: e.g. Xt[state_id[:R_cb_t]] .= 0.01
oc  = get_setting(m, :oc)
Xt  = zeros(nState)
Xt1 = zeros(nState)
Yt  = zeros(nRedCtrl)
Yt1 = zeros(nRedCtrl)
############################################################################
# I. Recover heterogeneous-agent controls via Gamma_control
############################################################################

Γ_ctrl = get_setting(m, :Γ_control)
NN     = nb * na * nse

# Next period (from Yt1)
Controlnext_full = ControlSS .* (1 .+ Γ_ctrl * Yt1)
Control_full = ControlSS .* (1 .+ Γ_ctrl * Yt)

Control_full[end-oc+1:end] = ControlSS[end-oc+1:end] .+ Γ_ctrl[end-oc+1:end, :] * Yt
Controlnext_full[end-oc+1:end] = ControlSS[end-oc+1:end] .+ Γ_ctrl[end-oc+1:end, :] * Yt1

Controlnext_VALUE = util(Controlnext_full[1:NN])
mutil_cnext       = mutil(Controlnext_full[NN+1:2*NN])
Vanext            = mutil(Controlnext_full[2*NN+1:3*NN])
# Current period (from Yt)
VALUE   = util(Control_full[1:NN])
mutil_c = mutil(Control_full[NN+1:2*NN])
Va      = mutil(Control_full[2*NN+1:3*NN])

############################################################################
# II. Reconstruct marginal distributions via Gamma_state
############################################################################

nFullMarg = get_setting(m, :nFullMarg)
nRedMarg  = get_setting(m, :nRedMarg)
Γ_state   = get_setting(m, :Γ_state)

# Current period (Stateminus in MATLAB)
Distributionminus = StateSS[1:nFullMarg] .+ Γ_state * Xt[1:nRedMarg]
distr_b   = Distributionminus[1:nb]
distr_a   = Distributionminus[nb+1:nb+na]
distr_se  = Distributionminus[nb+na+1:end]

# Next period (State in MATLAB)
Distribution = StateSS[1:nFullMarg] .+ Γ_state * Xt1[1:nRedMarg]
distr_b′  = Distribution[1:nb]
distr_a′  = Distribution[nb+1:nb+na]
distr_se′ = Distribution[nb+na+1:end]

############################################################################
# iv. copula decompression
############################################################################
compressionIndexesCOP = Float64.(get_setting(m, :compressionIndexesCOP))
nCOP = get_setting(m, :nCOP)

# next-period copula state (State in MATLAB)
cop_coefs   = Xt1[nRedMarg+1:nRedMarg+nCOP]
thetad      = uncompress(compressionIndexesCOP, cop_coefs, DCD, IDCD)
cop_dev     = reshape(thetad, (nb_cop, na_cop, nse_cop))
cop_dev     = cumsum(cumsum(cumsum(cop_dev; dims=1); dims=2); dims=3)

# current-period copula state (Stateminus in MATLAB)
cop_coefs_m = Xt[nRedMarg+1:nRedMarg+nCOP]
thetad_m    = uncompress(compressionIndexesCOP, cop_coefs_m, DCD, IDCD)
cop_dev_m   = reshape(thetad_m, (nb_cop, na_cop, nse_cop))
cop_dev_m   = cumsum(cumsum(cumsum(cop_dev_m; dims=1); dims=2); dims=3)

############################################################################
# v. cdfs from current-period marginals
############################################################################

cdf_b   = cumsum(distr_b)
cdf_a   = cumsum(distr_a)
cdf_se  = cumsum(distr_se)

cdf_b_ss  = cumsum(StateSS[1:nb])
cdf_a_ss  = cumsum(StateSS[nb+1:nb+na])
cdf_se_ss = cumsum(StateSS[nb+na+1:nFullMarg])

############################################################################
# vi. reconstruct joint pdf via copula
############################################################################
grids = copy(m.grids)
s_m_b  = grids[:copula_marginal_b][:]
s_m_a  = grids[:copula_marginal_a][:]
s_m_se = grids[:copula_marginal_se][:]

ubq = Vector{Float64}(cdf_b[:])
uaq = Vector{Float64}(cdf_a[:])
usq = Vector{Float64}(cdf_se[:])

c_ss_atu    = myinterpolate3(Vector{Float64}(cdf_b_ss[:]), 
                             Vector{Float64}(cdf_a_ss[:]), 
                             Vector{Float64}(cdf_se_ss[:]), 
                             get_setting(m, :COP_SS),ubq, uaq, usq)
dc_atu      = myinterpolate3(Vector{Float64}(s_m_b[:]),    
                             Vector{Float64}(s_m_a[:]),    
                             Vector{Float64}(s_m_se[:]),    
                             cop_dev_m, ubq, uaq, usq)
cdf_joint_m = c_ss_atu + dc_atu
pdf_joint_m = cdf_to_pdf3D_forwarddiff(cdf_joint_m)

############################################################################
# VII. Aggregate variables
############################################################################

# States (from Xt = period t, Xt1 = period t+1)
# Aggregate state variables — direct extraction matching archive pattern:
#   SS[idx] + deviation, then exp() to get levels
#   current period (Xt = Stateminus in archive), next period (Xt1 = State in archive)
R_cb_t    = exp(StateSS[state_id[:R_cb_t]]   + Xt[state_id[:R_cb_t]])
w_t       = exp(StateSS[state_id[:w_t]]       + Xt[state_id[:w_t]])
w′_t      = exp(StateSS[state_id[:w_t]]       + Xt1[state_id[:w_t]])
Q_t       = exp(StateSS[state_id[:Q_t]]       + Xt[state_id[:Q_t]])
Q′_t      = exp(StateSS[state_id[:Q_t]]       + Xt1[state_id[:Q_t]])
A_gaux_t  = exp(StateSS[state_id[:A_gaux_t]]  + Xt[state_id[:A_gaux_t]])
A_gaux′_t = exp(StateSS[state_id[:A_gaux_t]]  + Xt1[state_id[:A_gaux_t]])
R_tilde_t  = exp(StateSS[state_id[:R_tilde_t]] + Xt[state_id[:R_tilde_t]])
R_tilde′_t = exp(StateSS[state_id[:R_tilde_t]] + Xt1[state_id[:R_tilde_t]])

# Controls current period: exp(Control_full[idx]) — matches original exp(Control[pi_ind])
# (oc-override already stores log(SS) + deviation, so no need to add ControlSS again)
pi_t     = exp(Control_full[control_id[:pi_t]])
r_a_t    = exp(Control_full[control_id[:r_a_t]])
nn_t     = exp(Control_full[control_id[:nn_t]])
l_λ_t      = exp(Control_full[control_id[:l_λ_t]])
LT_t     = exp(Control_full[control_id[:LT_t]])
C_b_t    = exp(Control_full[control_id[:C_b_t]])
Profit_t = exp(Control_full[control_id[:Profit_t]])
f_t      = exp(Control_full[control_id[:f_t]])

# Controls next period (unprimed key for index lookup, Controlnext_full for value)
pi′_t = exp(Controlnext_full[control_id[:pi_t]])
l_λ′_t  = exp(Controlnext_full[control_id[:l_λ_t]])
f′_t  = exp(Controlnext_full[control_id[:f_t]])

MU = pdf_joint_m

############################################################################
# VIII. Time-varying employment transition matrices
############################################################################

A_emp = [
    1 - l_λ_t + l_λ_t * f_t    l_λ_t * (1 - f_t)
    f_t                      1 - f_t
]
I_ns      = Matrix{Float64}(I, ns, ns)   # local identity (avoid shadowing `id` dict)
P_SS3_aux = kron(A_emp, I_ns)
P_SS3_aux = [P_SS3_aux zeros(nse-1, 1)]
P_SS3_aux = [P_SS3_aux; [zeros(1, 2*ns) 1]]

H_tilde  = kron(Matrix{Float64}(P_SS3_aux'), sparse(Matrix{Float64}(I, nb*na, nb*na)))
MU_tilde = reshape(reshape(MU, nb*na, nse) * P_SS3_aux, nb, na, nse)

A_emp_next    = [
    1 - l_λ′_t + l_λ′_t * f′_t    l_λ′_t * (1 - f′_t)
    f′_t                        1 - f′_t
]
P_SS3next_aux = kron(A_emp_next, Matrix{Float64}(I, ns, ns))
P_SS3next_aux = [P_SS3next_aux zeros(nse-1, 1)]
P_SS3next_aux = [P_SS3next_aux; [zeros(1, 2*ns) 1]]

P_SS2             = grids[:P_SS2]
P_transition_exp  = P_SS2 * P_SS3next_aux
P_transition_dist = P_SS2

############################################################################
# IX. Income
############################################################################
b_ndgrid = grids[:b_ndgrid]
a_ndgrid = grids[:a_ndgrid]
se_ndgrid = grids[:se_ndgrid]

auxWW               = ones(nb, na, nse)
auxWW[:,:,ns+1:end] = zeros(nb, na, nse - ns)
auxWW1              = copy(auxWW)
auxWW               = auxWW .* (se_ndgrid .^ (θ[:b_aux] - 1))

auxWW2               = ones(nb, na, nse)
auxWW2[:,:,1:ns]     = zeros(nb, na, ns)
auxWW2[:,:,end]      = zeros(nb, na, 1)

R_A_aux = r_a_t + (θ[:dr] / (1 - θ[:dr])) * Q′_t

NW = θ[:ξ] / (1 + θ[:ξ]) * nn_t * w′_t
WW = (1 - θ[:τ_w]) * (NW .* auxWW1 .+ θ[:ξ] / (1 + θ[:ξ]) * θ[:w_bar] .* auxWW2)
WW[:,:,end] .= (1 - θ[:τ_a]) * (θ[:Eratio] * Profit_t) / θ[:Eshare]

inc = Dict{Symbol, Any}()
inc[:labor]    = WW .* se_ndgrid
inc[:dividend] = a_ndgrid .* R_A_aux
inc[:capital]  = a_ndgrid .* Q′_t
inc[:bond]     = (R_cb_t / pi_t) .* b_ndgrid .+
               (b_ndgrid .< 0) .* (θ[:Rprem] / pi_t) .* b_ndgrid
inc[:bond]     = inc[:bond] ./ (1 - θ[:dr]) .* θ[:b_a_aux]
inc[:transfer] = (LT_t + C_b_t) / (1 + θ[:ϕ_b]) .* ones(nb, na, nse)

############################################################################
# X. Policy update
############################################################################

EVa = reshape(reshape(Vanext, (nb*na, nse)) * P_transition_exp', (nb, na, nse))

R_tildeaux = R_tilde′_t / pi′_t .+ (b_ndgrid .< 0) .* (θ[:Rprem] / pi′_t)
EVb = reshape(reshape(R_tildeaux[:] ./ (1 - θ[:dr]) .* θ[:b_a_aux] .* mutil_cnext, (nb*na, nse)) * P_transition_exp', (nb, na, nse))

c_a_star, b_a_star, a_a_star, c_n_star, b_n_star = policies_update(EVb, EVa, Q′_t, pi_t, R_cb_t, 1, 1, inc, grids, θ)
############################################################################
# XI. Value function update
############################################################################
b_grid = grids[:b_grid].points
a_grid = grids[:a_grid].points
se_grid = grids[:se_grid].points

se_idx = repeat(reshape(collect(1:nse), 1, 1, nse), outer=(nb, na, 1))

EV3 = reshape(reshape(Controlnext_VALUE, (nb*na, nse)) * P_transition_exp', nb, na, nse)
itp_v     = interpolate((b_grid, a_grid, collect(1:nse)), EV3, Gridded(Linear()))
Controlnext_VALUE_itp = extrapolate(itp_v, Line())

V_adjust   = util(c_a_star) .+ θ[:β] * (1 - θ[:dr]) .* Controlnext_VALUE_itp.(b_a_star, a_a_star, se_idx)
V_noadjust = util(c_n_star) .+ θ[:β] * (1 - θ[:dr]) .* Controlnext_VALUE_itp.(b_n_star, a_ndgrid,  se_idx)

AProb = clamp.(1.0 ./ (1.0 .+ exp.(.-(V_adjust .- V_noadjust .- θ[:μ_χ]) ./ θ[:σ_χ])), 1e-6, 1 - 1e-6)
AC    = θ[:σ_χ] .* ((1.0 .- vec(AProb)) .* log.(1.0 .- vec(AProb)) .+
                       vec(AProb) .* log.(vec(AProb))) .+
        θ[:μ_χ] .* vec(AProb) .- (θ[:μ_χ] - θ[:σ_χ] * log(1 + exp(θ[:μ_χ] / θ[:σ_χ])))
AProb = AProb .* θ[:ψ]
AC    = AC    .* θ[:ψ]

VALUEaux = vec(AProb) .* vec(V_adjust) .+ (1.0 .- vec(AProb)) .* vec(V_noadjust) .- vec(AC)

############################################################################
# XII. Marginal utility of consumption update
############################################################################

mutil_c_n   = reshape(mutil(vec(c_n_star)), (nb, na, nse))
mutil_c_a   = reshape(mutil(vec(c_a_star)), (nb, na, nse))
mutil_c_aux = AProb .* mutil_c_a .+ (1.0 .- AProb) .* mutil_c_n

mutil_cnext_aux = reshape(reshape(mutil_cnext, nb*na, nse) * P_transition_exp', nb, na, nse)
itp_mu          = interpolate((b_grid, a_grid, collect(1:nse)), mutil_cnext_aux, Gridded(Linear()))
mutil_cnext_itp = extrapolate(itp_mu, Line())

############################################################################
# XIII. Marginal value of illiquid assets update
############################################################################

Va3     = reshape(EVa, nb, na, nse)
itp_va  = interpolate((b_grid, a_grid, collect(1:nse)), Va3, Gridded(Linear()))
Va_itp  = extrapolate(itp_va, Line())

Va_aux = AProb .* (R_A_aux .+ Q′_t) .* mutil_c_a .+ (1.0 .- AProb) .* R_A_aux .* mutil_c_n .+
         θ[:β] * (1 - θ[:dr]) .* (1.0 .- AProb) .* Va_itp.(b_n_star, a_ndgrid, se_idx)

############################################################################
# XIV. Residuals for jump variables (value functions at t)
############################################################################

F[:eq_Value]      = Control_full[1:NN]        .- invutil(vec(VALUEaux))
F[:eq_mutil_cons] = Control_full[NN+1:2*NN]   .- invmutil(vec(mutil_c_aux))
F[:eq_Va]         = Control_full[2*NN+1:3*NN] .- invmutil(vec(Va_aux))

############################################################################
# XV. Distribution transition
############################################################################

Dist_b_a, idb_a = genweight(b_a_star, b_grid)
Dist_b_n, idb_n = genweight(b_n_star, b_grid)
Dist_a_a, ida_a = genweight(a_a_star, a_grid)
Dist_a_n, ida_n = genweight(a_ndgrid,  a_grid)

Dist_b_die, idb_die = genweight(zeros(nb, na, nse), b_grid)
Dist_a_die, ida_die = genweight(zeros(nb, na, nse), a_grid)

# Adjustment transition matrix
weight11 = zeros(nb*na, nse, nse)
weight12 = zeros(nb*na, nse, nse)
weight21 = zeros(nb*na, nse, nse)
weight22 = zeros(nb*na, nse, nse)

idb_a_r = repeat(vec(idb_a), 1, nse)
ida_a_r = repeat(vec(ida_a), 1, nse)
idse    = vec(kron(1:nse, ones(Int, 1, nb*na*nse))')'

index11 = sub2ind((nb, na, nse), vec(idb_a_r),     vec(ida_a_r),     vec(idse))
index12 = sub2ind((nb, na, nse), vec(idb_a_r),     vec(ida_a_r) .+ 1, vec(idse))
index21 = sub2ind((nb, na, nse), vec(idb_a_r) .+ 1, vec(ida_a_r),    vec(idse))
index22 = sub2ind((nb, na, nse), vec(idb_a_r) .+ 1, vec(ida_a_r) .+ 1, vec(idse))

for hh in 1:nse
    weight11[:,:,hh] = vec((1 .- Dist_b_a[:,:,hh]) .* (1 .- Dist_a_a[:,:,hh])) * P_transition_dist[hh,:]'
    weight12[:,:,hh] = vec((1 .- Dist_b_a[:,:,hh]) .* Dist_a_a[:,:,hh])         * P_transition_dist[hh,:]'
    weight21[:,:,hh] = vec(Dist_b_a[:,:,hh] .* (1 .- Dist_a_a[:,:,hh]))         * P_transition_dist[hh,:]'
    weight22[:,:,hh] = vec(Dist_b_a[:,:,hh] .* Dist_a_a[:,:,hh])                * P_transition_dist[hh,:]'
end

weight11 = permutedims(weight11, (1, 3, 2))
weight12 = permutedims(weight12, (1, 3, 2))
weight21 = permutedims(weight21, (1, 3, 2))
weight22 = permutedims(weight22, (1, 3, 2))
rowindex = repeat(collect(1:nb*na*nse), 4*nse)

TT_a = sparse(vec(rowindex), [vec(index11); vec(index21); vec(index12); vec(index22)],
              [vec(weight11); vec(weight21); vec(weight12); vec(weight22)], nb*na*nse, nb*na*nse)

# Non-adjustment transition matrix
weight11 = zeros(nb*na, nse, nse)
weight12 = zeros(nb*na, nse, nse)
weight21 = zeros(nb*na, nse, nse)
weight22 = zeros(nb*na, nse, nse)

idb_n_r = repeat(vec(idb_n), 1, nse)
ida_n_r = repeat(vec(ida_n), 1, nse)
idse    = vec(kron(1:nse, ones(1, nb*na*nse))')'

index11 = sub2ind((nb, na, nse), vec(idb_n_r),     vec(ida_n_r),      vec(idse))
index12 = sub2ind((nb, na, nse), vec(idb_n_r),     vec(ida_n_r) .+ 1, vec(idse))
index21 = sub2ind((nb, na, nse), vec(idb_n_r) .+ 1, vec(ida_n_r),     vec(idse))
index22 = sub2ind((nb, na, nse), vec(idb_n_r) .+ 1, vec(ida_n_r) .+ 1, vec(idse))

for hh in 1:nse
    weight11[:,:,hh] = vec((1 .- Dist_b_n[:,:,hh]) .* (1 .- Dist_a_n[:,:,hh])) * P_transition_dist[hh,:]'
    weight12[:,:,hh] = vec((1 .- Dist_b_n[:,:,hh]) .* Dist_a_n[:,:,hh])         * P_transition_dist[hh,:]'
    weight21[:,:,hh] = vec(Dist_b_n[:,:,hh] .* (1 .- Dist_a_n[:,:,hh]))         * P_transition_dist[hh,:]'
    weight22[:,:,hh] = vec(Dist_b_n[:,:,hh] .* Dist_a_n[:,:,hh])                * P_transition_dist[hh,:]'
end

weight11 = permutedims(weight11, (1, 3, 2))
weight12 = permutedims(weight12, (1, 3, 2))
weight21 = permutedims(weight21, (1, 3, 2))
weight22 = permutedims(weight22, (1, 3, 2))
rowindex = repeat(collect(1:nb*na*nse), 4*nse)

TT_n = sparse(vec(rowindex), [vec(index11); vec(index21); vec(index12); vec(index22)],
              [vec(weight11); vec(weight21); vec(weight12); vec(weight22)], nb*na*nse, nb*na*nse)

# Death/newborn transition matrix
weight11 = zeros(nb*na, nse, nse)
weight12 = zeros(nb*na, nse, nse)
weight21 = zeros(nb*na, nse, nse)
weight22 = zeros(nb*na, nse, nse)

idb_die_r = repeat(vec(idb_die), 1, nse)
ida_die_r = repeat(vec(ida_die), 1, nse)
ids_die   = vec(kron(1:nse, ones(1, nb*na*nse))')'

index11 = sub2ind((nb, na, nse), vec(idb_die_r),     vec(ida_die_r),      vec(ids_die))
index12 = sub2ind((nb, na, nse), vec(idb_die_r),     vec(ida_die_r) .+ 1, vec(ids_die))
index21 = sub2ind((nb, na, nse), vec(idb_die_r) .+ 1, vec(ida_die_r),     vec(ids_die))
index22 = sub2ind((nb, na, nse), vec(idb_die_r) .+ 1, vec(ida_die_r) .+ 1, vec(ids_die))

for hh in 1:nse
    weight11[:,:,hh] = vec((1 .- Dist_b_die[:,:,hh]) .* (1 .- Dist_a_die[:,:,hh])) * P_transition_dist[hh,:]'
    weight12[:,:,hh] = vec((1 .- Dist_b_die[:,:,hh]) .* Dist_a_die[:,:,hh])         * P_transition_dist[hh,:]'
    weight21[:,:,hh] = vec(Dist_b_die[:,:,hh] .* (1 .- Dist_a_die[:,:,hh]))         * P_transition_dist[hh,:]'
    weight22[:,:,hh] = vec(Dist_b_die[:,:,hh] .* Dist_a_die[:,:,hh])                * P_transition_dist[hh,:]'
end

weight11 = permutedims(weight11, (1, 3, 2))
weight12 = permutedims(weight12, (1, 3, 2))
weight21 = permutedims(weight21, (1, 3, 2))
weight22 = permutedims(weight22, (1, 3, 2))
rowindex = repeat(collect(1:nb*na*nse), 4*nse)

TT_die = sparse(vec(rowindex), [vec(index11); vec(index21); vec(index12); vec(index22)],
                [vec(weight11); vec(weight21); vec(weight12); vec(weight22)], nb*na*nse, nb*na*nse)

# Combine and evolve distribution
n_aprob = length(vec(AProb))
APD_a   = sparse(1:n_aprob, 1:n_aprob, vec(AProb))
APD_n   = sparse(1:n_aprob, 1:n_aprob, 1.0 .- vec(AProb))
TT      = (1 - θ[:dr]) * (APD_a * TT_a + APD_n * TT_n) + θ[:dr] * TT_die
TT      = H_tilde' * TT

MUnext = reshape(vec(vec(MU)' * TT), nb, na, nse)

############################################################################
# XVI. Compress predicted copula and write residuals
############################################################################

C_next_grid = cumsum(cumsum(cumsum(MUnext; dims=1); dims=2); dims=3)

cdf_b_next  = cumsum(distr_b′)
cdf_a_next  = cumsum(distr_a′)
cdf_se_next = cumsum(distr_se′)

CDF_Dev = myinterpolate3(Vector{Float64}(cdf_b_next), Vector{Float64}(cdf_a_next), Vector{Float64}(cdf_se_next),
                         C_next_grid,
                         Vector{Float64}(s_m_b), Vector{Float64}(s_m_a), Vector{Float64}(s_m_se)) .-
          myinterpolate3(Vector{Float64}(cdf_b_ss), Vector{Float64}(cdf_a_ss), Vector{Float64}(cdf_se_ss),
                         get_setting(m, :COP_SS),
                         Vector{Float64}(s_m_b), Vector{Float64}(s_m_a), Vector{Float64}(s_m_se))

COP_thet = compress(dct_compression_indices[:copula], cdf_to_pdf3D_forwarddiff(CDF_Dev), DCD, IDCD)

F[:eq_copula] = COP_thet - Xt1[nRedMarg+1:nRedMarg+nCOP]

# Marginal distribution residuals
aux_b  = dropdims(sum(MUnext; dims=(2,3)), dims=(2,3))
aux_a  = dropdims(sum(MUnext; dims=(1,3)), dims=(1,3))'
aux_se = dropdims(sum(MUnext; dims=(1,2)), dims=(1,2))

F[:eq_marginal_pdf_b]  = aux_b[1:nb-1]   - distr_b′[1:nb-1]
F[:eq_marginal_pdf_a]  = aux_a[1:na-1]   - distr_a′[1:na-1]
F[:eq_marginal_pdf_se] = aux_se[1:nse-1] - distr_se′[1:nse-1]


