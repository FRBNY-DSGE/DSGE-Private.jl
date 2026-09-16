

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

marginal_b_ind  = 1:(nb-1)
marginal_a_ind  = (nb-1) .+ (1:(na-1))
marginal_se_ind = (nb + na - 2) .+ (1:(nse-1))

COP_ind         = (nb + na + nse - 3) .+ (1:nCOP) 

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

VALUEnext     = util(Controlnext[VALUE_ind]) 
VALUE         = util(Control[VALUE_ind]) 
Vanext        = mutil(Controlnext[Va_ind]) 
Va            = mutil(Control[Va_ind]) 
mutil_cnext   = mutil(Controlnext[mutil_c_ind]) 
mutil_c       = mutil(Control[mutil_c_ind]) 
LHS[nx .+ VALUE_ind] = Control[VALUE_ind]
LHS[nx .+ mutil_c_ind] = Control[mutil_c_ind]
LHS[nx .+ Va_ind] = Control[Va_ind] 

LHS[marginal_b_ind] = Distribution[1:nb-1]
LHS[marginal_a_ind] = Distribution[nb .+ (1:na-1)]
LHS[marginal_se_ind] = Distribution[(nb + na) .+ (1:nse-1)] 

LHS[COP_ind] = cop_coefs

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


