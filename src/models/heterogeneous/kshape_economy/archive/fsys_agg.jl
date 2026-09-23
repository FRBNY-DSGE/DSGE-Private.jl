using DSGE: @sslogdeviations2levels, @sslogdeviations2levels_unprimekeys, @unpack_and_first
include("helpers/q_cons2.jl")

"""
```
Fsys_agg(F, X, XPrime, θ, grids, id, nt, eq)
```
Return deviations from aggregate equilibrium conditions for simplified 6-variable HANK model.

### Inputs
- `F::AbstractVector`: vector holding deviations for aggregate equilibrium conditions
- `X::AbstractArray`,`XPrime::AbstractArray`: deviations from steady state in periods t [`X`] and t+1 [`XPrime`]
- `θ::NamedTuple`: maps parameter name to value
- `grids::OrderedDict`: maps names of quantities related to the idiosyncratic state space to their values
- `id::OrderedDict{Symbol, Int or UnitRange{Int}}`: maps variable name (e.g. `MRS_t` and `MRS′_t`)
    to its index/indices in `X` and `XPrime`.
- `nt::NamedTuple`: maps variable name (e.g. `MRS_t` and `MRS′_t`) to steady-state log level
- `eq::OrderedDict{Symbol, Int or UnitRange{Int}}`: maps name of equilibrium conditions
    to its index/indices in `F`

"""
function Fsys_agg(F::AbstractVector, X::AbstractArray, XPrime::AbstractArray,
                  θ::NamedTuple, grids::OrderedDict, id::OrderedDict{Symbol, I1}, nt::NamedTuple,
                  eq::OrderedDict{Symbol, I2}) where {I1 <: Union{Int, UnitRange}, I2 <: Union{Int, UnitRange}}

    ############################################################################
    #            I. Read out argument values                                   #
    ############################################################################

    # Read out equation scalar indices from equilibrium conditions dictionary
    # @unpack_and_first extracts the value and takes first() if it's a range
    @unpack_and_first eq_MRS = eq
    @unpack_and_first eq_agg_illiquid_assets = eq
    @unpack_and_first eq_agg_liquid_assets = eq
    @unpack_and_first eq_agg_consumption = eq
    @unpack_and_first eq_agg_employment = eq
    @unpack_and_first eq_agg_labor = eq

    # Extract current period variables from log deviations
    # @sslogdeviations2levels converts log deviations to levels: var = exp(nt[:var] + X[id[:var]])
    @sslogdeviations2levels MRS_t, A_hh_t, B_hh_t, C_t, N_t, L_t = X, id, nt

    # Extract next period variables from log deviations
    # @sslogdeviations2levels_unprimekeys uses unprimed keys for nt lookup
    @sslogdeviations2levels_unprimekeys MRS′_t, A_hh′_t, B_hh′_t, C′_t, N′_t, L′_t = XPrime, id, nt

    ############################################################################
    #           Error term calculations (i.e. model starts here)          #
    ############################################################################

    # Extract distributional results from grids (these should be precomputed and stored)
    # These are typically computed from the heterogeneous agent problem
    MRS_a_aux = grids[:MRS_a_aux]::Array{Float64, 3}
    MRS_n_aux = grids[:MRS_n_aux]::Array{Float64, 3}
    marginal_aminus = grids[:marginal_aminus]::Vector{Float64}
    marginal_bminus = grids[:marginal_bminus]::Vector{Float64}
    marginal_seminus = grids[:marginal_seminus]::Vector{Float64}
    c_a_star = grids[:c_a_star]::Array{Float64, 3}
    c_n_star = grids[:c_n_star]::Array{Float64, 3}
    AProb = grids[:AProb]::Array{Float64, 3}
    MU_tilde = grids[:MU_tilde]::Array{Float64, 3}
    MU_tilde_se = grids[:MU_tilde_se]::Vector{Float64}
    
    # Extract grids
    a_grid = grids[:a]::Vector{Float64}
    b_grid = grids[:b]::Vector{Float64}
    s_grid = grids[:s]::Vector{Float64}
    
    # Extract additional variables needed for computation
    W = grids[:W]::Float64
    nn = grids[:nn]::Float64
    tau_w = grids[:tau_w]::Float64
    
    # Grid dimensions
    ns = Int(grids[:ns])
    
    # Create meshes dictionary (if needed by q_cons2)
    meshes = Dict(:b => grids[:b_mesh], :a => grids[:a_mesh], :se => grids[:se_mesh])
    
    ############################################################################
    # Compute RHS of equilibrium conditions
    ############################################################################
    
    # eq_MRS: Marginal Rate of Substitution equation
    # RHS[nx+MRS_ind] = sum(sum(MRS_a_aux[:,:,end]+MRS_n_aux[:,:,end]))/param["Eshare"]*param["Lambda_b_aux"]
    MRS_RHS = sum(MRS_a_aux[:, :, end] .+ MRS_n_aux[:, :, end]) / θ[:Eshare] * θ[:Lambda_b_aux]
    F[eq_MRS] = MRS_t - MRS_RHS
    
    # eq_agg_illiquid_assets: Aggregate illiquid assets market clearing
    # RHS[nx+A_hh_ind] = sum(grid["a"].*marginal_aminus)
    A_hh_RHS = sum(a_grid .* marginal_aminus)
    F[eq_agg_illiquid_assets] = A_hh_t - A_hh_RHS
    
    # eq_agg_liquid_assets: Aggregate liquid assets market clearing
    # RHS[nx+B_hh_ind] = sum(grid["b"].*marginal_bminus)/(1-param["death_rate"])*param["b_a_aux"]
    B_hh_RHS = sum(b_grid .* marginal_bminus) / (1 - θ[:death_rate]) * θ[:b_a_aux]
    F[eq_agg_liquid_assets] = B_hh_t - B_hh_RHS
    
    # eq_agg_consumption: Aggregate consumption equation
    # RHS[nx+C_ind] = q_cons2(c_a_star,c_n_star,W,nn,AProb,MU_tilde,meshes,param,grid,tau_w)
    C_RHS = q_cons2(c_a_star, c_n_star, W, nn, AProb, MU_tilde, meshes, θ, grids, tau_w)
    F[eq_agg_consumption] = C_t - C_RHS
    
    # eq_agg_employment: Aggregate employment equation (beginning of period)
    # RHS[nx+N_ind] = sum(marginal_seminus[1:ns])
    N_RHS = sum(marginal_seminus[1:ns])
    F[eq_agg_employment] = N_t - N_RHS
    
    # eq_agg_labor: Aggregate labor equation
    # RHS[nx+L_ind] = nn * grid["s"]' * MU_tilde_se[1:ns]
    L_RHS = nn * (s_grid' * MU_tilde_se[1:ns])
    F[eq_agg_labor] = L_t - L_RHS

    return F
end
