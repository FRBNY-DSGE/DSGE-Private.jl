
# This code works under presumption that shocks are orthogonal in QQQ matrix:
"""
'''
function compute_dsge_irfs(dsge::Dict{Symbol, Array{S}}, 
                           horizon::Int, 
                           n_obs::Int; 
                           flip_shocks::Bool = false, 
                           use_coint::Bool = false, 
                           n_coint::Int = n_coint, 
                           cum_sum = 1:n_obs) where {S <: Real}

computes the impulse responses implied by the linearized state-space representation of a DSGE model assuming that the covariance matrix of structural QQQ is diagonal (i.e. shocks are orthogonal)

The state-space system is assumed to take the form
s_t = TTT × s_{t -1} + RR × ϵ_{t}
y_t = ZZ × s_t + DD + MM × ϵ
ϵ ∼ Normal(0, QQQ)

where 's_t' are state variables, 'y_t' are observalbes, and 'ϵ_t' are orthogonal structural shocks. The function computes period-by-period responses of both states and observables to a one-standard-deviation
innovation in each element of 'ϵ_t'.

If 'use_coint = true', the function also includes cointegrating components of the measurement system by
extending the 'ZZ', 'DD', 'EE', and 'MM' matrices to include additional rows corresponding to
cointegrating relationships.



### Inputs
* 'dsge::Dict{Symbol,Any}': container holding the system matrices of the solved DSGE model:
  - ':TTT' — transition matrix for state variables  
  - ':RRR' — matrix mapping shocks into states  
  - ':ZZ', ':DD', ':EE', ':MM' — measurement system  
  - ':QQQ' — covariance matrix of structural shocks
* 'horizon::Int': number of periods over which to compute IRFs.
* 'n_obs::Int': number of observables (rows of 'ZZ').

### Keyword Arguments
* 'flip_shocks::Bool': sign convention for shocks; default produces *negative* impact responses.
  Set to 'true' to obtain positive shocks.
* 'use_coint::Bool': include cointegration components in measurement system (default 'false').
* 'n_coint::Int': number of cointegrating relationships, used only if 'use_coint = true'.
* 'cum_sum': indices of observables whose IRFs are reported in cumulative form
  ('cumsum' across the horizon).

### Algorithm
1. Select the appropriate block of the measurement matrices depending on 'use_coint'.
2. For each structural shock 'i':
   * Create a one-standard-deviation shock vector 'shock' with magnitude '√QQQ[i,i]'
     (sign controlled by 'flip_shocks').
   * Compute the initial state response 's₁ = RRR × shock' and its measurement implication
     'y₁ = ZZ × s₁ + MM × shock'.
   * Propagate forward for 't = 2, …, horizon' using the transition law
     'sₜ = TTT × sₜ₋₁' and corresponding observables.
   * Apply a cumulative sum for selected observables.
   * Store responses for each shock in the output array.
3. Return a 3-dimensional tensor of IRFs with dimensions '(n_shocks × n_obs × horizon)'.

### Returns
A 3-D array 'fin_matrix' where 'fin_matrix[i, j, t]' gives the response of observable 'j'
at horizon 't' to a one-standard-deviation innovation in structural shock 'i'.

### Notes
* Assumes shocks are orthogonal ('QQQ' diagonal).  
* Used internally for aligning VECM impulse responses with DSGE-based identification.

'''
"""

function compute_dsge_irfs(dsge::Dict{Symbol, Array{S}}, 
                           horizon::Int, 
                           n_obs::Int; 
                           flip_shocks::Bool = false, 
                           use_coint::Bool = false, 
                           n_coint::Int = n_coint, 
                           cum_sum = 1:n_obs) where {S <: Real}
    # Option to remove cointegration from measurement equation matrices
    n_sh = size(dsge[:QQQ], 1)
    if !use_coint
        ZZ = dsge[:ZZ][1:n_obs, :]
        DD = dsge[:DD][1:n_obs]
        EE = dsge[:EE][1:n_obs, 1:n_obs]
        MM = dsge[:MM][1:n_obs, :]

        obs_irf = zeros(n_obs, horizon)
        fin_matrix = zeros(n_sh, n_obs, horizon)
    else
        ZZ = dsge[:ZZ]
        DD = dsge[:DD]
        EE = dsge[:EE]
        MM = dsge[:MM]

        obs_irf = zeros(n_obs + n_coint, horizon) # Manually add coint dims (make automatic)
        fin_matrix = zeros(n_sh, n_obs + n_coint, horizon)
        cum_sum = 1:n_obs + n_coint
    end

    n_states = size(dsge[:TTT], 1)
    states_irf = zeros(n_states, horizon)

    for i in 1:n_sh
        # Collapse QQ matrix into n_sh * 1 vector and take sqrt of that specific shock
        shock = zeros(n_sh)
        shock[i] = flip_shocks ? sqrt(dsge[:QQQ][i, i]) : -sqrt(dsge[:QQQ][i, i])

        # Initial impact of shock on all states (n_states x shocks) * (shocks x 1)
        states_irf[:, 1] .= dsge[:RRR] * shock
        obs_irf[:, 1] .= ZZ * states_irf[:, 1] + MM * shock

        # For all other horizons
        for t in 2:horizon
            states_irf[:, t] .= dsge[:TTT] * states_irf[:, t-1]
            obs_irf[:, t] .= ZZ * states_irf[:, t]
        end
        # Take culumative sum
        obs_irf[cum_sum, :] = cumsum(obs_irf[cum_sum, :], dims=2)

        # Add obs x horizon to shock specific index
        fin_matrix = hcat(fin_matrix, obs_irf')
    end

    return fin_matrix
end

"""
```
function impulse_responses_vecm(dsge::Dict{Symbol,Any}, YYYY::Matrix{Float64},
    XXXX::Matrix{Float64}, XXYY::Matrix{Float64},
YYYYD::Matrix{Float64}, XXYYD::Matrix{Float64},
data::Matrix{Float64}, λ::Real, lags::Int,
horizon::Int, coint_vec::Matrix{Float64},
n_coint::Int; cointadd::Int = 0,
cum_irf::Vector{Int} = [1,2,3])

computes the VECM impulse responses identified by the DSGE model. The procedure follows the Del Negro, Schorfheide (2004, 2006, 2009) methodology, where a VECM is estimated or regularized to approximate the DSGE's reduced-form dynamics and then identified using the DSGE's implied impact matrix. 

The underlying VECM is
Δy_{t + 1} = e_t β_e + X_{t+1} β_ν + u_{t + 1}
u_{t + 1} ∼ N(0, Σ)

where e_t are the cointigration error terms ("e_t = coint_vec × y_t"), "β_e" are error-correction coefficients, and β_ν are lag coefficients.
The DSGE identification enters through the rotation implied by the QR decomposition of the DSGE's impact matrix on observables.
### Inputs
* 'dsge::Dict{Symbol,Any}': solved DSGE system (':TTT', ':RRR', ':ZZ', ':MM', ':QQQ', etc.).
* 'YYYY', 'XXXX', 'XXYY', 'YYYYD', 'XXYYD': regression matrices entering the VECM estimation
  (possibly pre-demeaned or shrinkage-adjusted versions).
* 'data::Matrix{Float64}': matrix of observed data ('n_obs × T').
* 'λ::Real': shrinkage weight controlling the DSGE prior; larger 'λ' → tighter DSGE weighting.
* 'lags::Int': number of VAR lags in the VECM.
* 'horizon::Int': number of periods for IRFs.
* 'coint_vec::Matrix{Float64}': cointegration matrix such that 'coint_vec * yₜ' yields 'n_coint × T' cointegration errors.
* 'n_coint::Int': number of cointegrating relationships.

### Keyword Arguments
* 'cointadd::Int': toggle for inclusion of additional cointegration adjustments (default 0).
* 'cum_irf::Vector{Int}': indices of variables for which IRFs are cumulated.

### Algorithm
1. **Estimate or regularize the VECM system**
   using shrinkage 'λ' to obtain coefficient matrix 'β_hat' and innovation covariance 'Σ_hat'.
2. **Partition coefficients** into error-correction ('β_coint') and VAR dynamics ('β_tilde') blocks.
3. **Form the companion representation** for the VECM and initialize selection matrices.
4. **Orthogonalize shocks** using Cholesky decomposition 'Σ_chol = chol(Σ_hat).L'.
5. **Iterate IRFs** forward for 't = 1,…,horizon':
   propagate 'cc_dy' and 'cc_y', store responses 'bbirf[t,:]'.
   Apply cumulative sum for selected observables.
6. **Align with DSGE identification**:
   compute DSGE IRFs via ['compute_dsge_irfs'](@ref),
   take first-period impact matrix 'a0', perform QR decomposition
   F = qr(a0)'
   Ω = sign.diag(F.R).*(F.Q')
   and rotate the vecm shocks accordingly: Σ_chol × Ω

   7. Return the rotated, DSGE-identified impulse responses.

### Returns
A matrix 'shock_matrix' of dimension '(horizon × n_obs²)' where each block of
columns corresponds to the IRFs of observables to one structural shock.

### Notes
* Combines statistical (VECM) dynamics with DSGE structural identification.
* Follows the procedure of Del Negro & Schorfheide (2004, 2006, 2009).

```
"""
function impulse_responses_vecm(dsge, YYYY, XXXX, XXYY, YYYYD, XXYYD, data, λ, lags, horizon, coint_vec, n_coint; cointadd = 0, cum_irf = [1 2 3]) #equal to [1 2 3] for no real reason
    T = size(data, 1)
    n_obs = size(coint_vec, 1)
    if !isinf(λ)
        LAMinv = 1.0I(size(XXXXD, 1))
        yyyyc = Laminv * XXYY + λ +XXXD
        invxxxxc = inv(xxxxc)
        β_hat = (invxxxxc * xxyyc)' * LAMinv
        Σ_hat = (yyyyc - xxyyc' * invxxxxc*xxyyc) / (T - lags + λ)

        β_tilde = [β_hat[:, (n_coint + 2):end];
                    [1.0I(n_obs*(lags - 1)) zeros(n_obs * (lags - 1), n_obs)]]

        if n_coint > 0
           β_coint  = β_hat[:, 1:n_coint]
           β_coint_tilde = [β_coint; zeros(n_obs*(lags - 1), n_coint)]
        else
           β_coint  = 0
           β_coint_tilde = zeros(n_obs*lags, n_coint)
        end
        β_lagcum = 1.0I(n_obs)
                                                               for i ∈ 1:lags
           β_lagcum = β_lagcum - β_hat[:, (n_coint + 1 + (i - 1)*n_obs +1: n_coint + 1 + i*n_obs)]
       end
    end
    if cointadd == 0
        β_cointc = 1.0I(n_obs)
        β_coint = β_hat[:, 1:n_coint]
        β_coint_tilde = [β_coint; zeros(n_obs * (lags - 1), n_coint)]
        β_coint_corth = β_cointc[:, n_obs - size(β_coint, 2):n_obs]
        β_cointc = c_cointc[:, 1:size(c_coint, 2)]
        β_coint_orth = (1.0I(n_obs) - β_cointc*inv(β_coint'*β_cointc)*β_coint')*β_coint_corth
        coint_vec_orth = ((1.0I(n_obs)  - β_cointc*inv(coint_vec*β_cointc)*coint_vec)*β_coint_corth)'

        mudy = coint_vec_orth' * inv(β_coint_orth' * β_lagcum * coint_vec_orth') * β_coint_orth' * β_hat[:, n_coint +1]
    end
    Σ_chol = cholesky(Matrix(Symmetric(Σ_hat))).L
    cc_dy = 1.0I(n_obs * lags)
    cc_y = 1.0I(n_obs * lags)

    msel = [1.0I(n_obs); zeros((lags-1)*n_obs, n_obs)]
    bbirf = zeros(horizon, n_obs^2)
    out = msel' * cc_dy*msel * Σ_chol
    bbirf[1, :] = vec(out)'

    for t ∈ 2:horizon
        cc_dy = β_tilde * cc_dy
        if n_coint > 0
            cc_dy = cc_dy + (β_coint_tilde * coint_vec *msel')*cc_y
            cc_y = cc_y + cc_dy
        end
        out = msel'*cc_dy*msel*Σ_chol
        bbirf[t, :] = vec(out)'
    end
    ccum_irf = []
    ccum_irf = reduce(vcat, [collect(i:n_obs:n_obs^2) for i in cum_irf])'
    bbirf[:, ccum_irf] = cumsum(bbirf[:, ccum_irf]; dims=1)

    aairf = compute_dsge_irfs(dsge, horizon, n_obs)

    a0 = Array{Float64, 2}(undef, n_obs, 0)
    a0 = hcat(a0, [permutedims(aairf[1, n_obs*(i - 1)+1:n_obs*i])' for i in 1:n_obs]...)

    F = qr(a0)'
    phimat = Matrix(F.Q)'
    r_a = Matrix(F.R)

    r = [sign(x) for x ∈ diag(r_a)]
    phimat = (r .* ones(1, n_obs)) .* phimat
    shock_matrix  = bbirf * kron(phimat, 1.0I(n_obs))

    return shock_matrix
end

