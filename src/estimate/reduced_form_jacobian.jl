"""
```
function reduced_form_jacobian(m, θ, T, R, Γ0, Γ1, Γ2, Γ3;
                               check_valid_measurement_eqn = true)
```
computes the Jacobian of the reduced form state space matrices T, R, Z, and D with respect
to the parameters θ for the state space system

```
sₜ = Tsₜ₋₁ + Rϵₜ
yₜ = Zsₜ + D
```
where ϵₜ ∼ N(0, I). The function assumes that the reduced form matrices are solved
using the Klein method or any other method where we represent
the structural model as

```
Γ₀sₜ = Γ₁Eₜsₜ₊₁ + Γ₂sₜ₋₁ + Γ₃ϵₜ.
```

### Arguments:
- `m::AbstractDSGEModel`: model object
- `θ::ParameterVector`: vector of model parameters
- `T::AbstractMatrix`: transition matrix from time t-1 to t
- `R::AbstractMatrix`: shock loading on exogenous shocks
- `Γ0::AbstractMatrix`: structural matrix applying to time t states
- `Γ1::AbstractMatrix`: structural matrix applying to forward-looking variables
- `Γ2::AbstractMatrix`: structural matrix applying to backward-looking variables
- `Γ3::AbstractMatrix`: structural matrix applying to time t shocks

### Keyword Arguments
- `check_valid_measurement_eqn::Bool`: Throws an error if the measurement equation does not
    satisfy the required assumptions for this method. An entry of the matrix Z and vector D cannot
    depend on more than one of the parameters θ, transition matrix T, or shock loading R
    because we do not have a mapping

### Outputs
- The Jacobians ∂T∂θ, ∂R∂θ, ∂Z∂θ, and ∂D∂θ

Remaining things to do:
* Speed up auto-differentiation by allowing the user to tell the function
  whether or not the measurement matrices ever depend on T or R (to avoid excess differentiation)
* Add in differentiation for cases when you have nonzero CCC and nonzero EE
* Handle case of not differentiating w.r.t. fixed parameters to reduce state space size
* Extend method to allow Z and D to depend on θ, T, and R
"""
function reduced_form_jacobian(m::AbstractDSGEModel, θ::ParameterVector,
                               T::AbstractMatrix{S}, R::AbstractMatrix{S},
                               Γ0::AbstractMatrix{S}, Γ1::AbstractMatrix{S},
                               Γ2::AbstractMatrix{S}, Γ3::AbstractMatrix{S};
                               check_valid_measurement_eqn::Bool = true) where {S<:Real}
    # Apply implicit function theorem to compute Jacobians of
    # reduced form matrices
    ∂T∂θ, ∂R∂θ = transition_matrices_jacobian(m, θ, T, R, Γ0, Γ1, Γ2, Γ3)
    ∂Z∂θ, ∂D∂θ = measurement_matrices_jacobian(m, θ, T, R, ∂T∂θ, ∂R∂θ;
                                               check_valid_measurement_eqn = check_valid_measurement_eqn)
    return ∂T∂θ, ∂R∂θ, ∂Z∂θ, ∂D∂θ
end

"""
```
structural_matrices_jacobian(m, θ)
```
computes the Jacobian of the structural matrices in the structural system

```
Γ₀sₜ = Γ₁Eₜsₜ₊₁ + Γ₂sₜ₋₁ + Γ₃ϵₜ.
```
when evaluated at θ.

### Arguments
- `m::AbstractDSGEModel`: model object
- `θ::ParameterVector`: vector of model parameters

### Outputs
- The Jacobians ∂Γ0∂θ, ∂Γ1∂θ, ∂Γ2∂θ, and ∂Γ3∂θ

"""
function structural_matrices_jacobian(m::AbstractDSGEModel, θ::ParameterVector) where {S<:Real}

    @inline function diff_struct_obj_fnct(var)
        DSGE.update!(m, var)
        Γ0, Γ1, Γ2, Γ3 = eqcond(m; method = :klein)
        return [vec(Γ0); vec(Γ1); vec(Γ2); vec(Γ3)]
    end
    derivs = ForwardDiff.sparse_jacobian(diff_struct_obj_fnct, map(α -> α.value, θ))

    # derivs is a matrix whose rows are gradients of a single entry
    # of a structural matrix w.r.t. the vector θ,
    # so to retrieve ∂Γ0∂θ, we want the first n_states(m)^2 rows
    Γ0_dim = n_states(m)^2
    Γ1_dim = Γ0_dim # copies b/c primitive
    Γ2_dim = Γ0_dim
    Γ3_dim = n_states(m) * n_shocks_exogenous(m)
    nstates = n_states(m)
    ∂Γ0∂θ = derivs[1:Γ0_dim,:]
    ∂Γ1∂θ = derivs[(Γ0_dim+1):(Γ1_dim+Γ0_dim),:]
    ∂Γ2∂θ = derivs[(Γ1_dim+Γ0_dim+1):(Γ2_dim+Γ1_dim+Γ0_dim),:]
    ∂Γ3∂θ = derivs[(Γ2_dim+Γ1_dim+Γ0_dim+1):end,:]

    return ∂Γ0∂θ, ∂Γ1∂θ, ∂Γ2∂θ, ∂Γ3∂θ
end

"""
```
transition_matrices_jacobian(m, θ, T, Γ0, Γ1, Γ2, Γ3)
```
computes the Jacobian of the transition matrices T and R
in the state space transition equation

```
sₜ = Tsₜ₋₁ + Rϵₜ
```
when evaluated at θ using the implicit function theorem.

### Arguments
- `m::AbstractDSGEModel`: model object
- `θ::ParameterVector`: vector of model parameters
- `T::AbstractMatrix`: transition matrix from time t-1 to t
- `R::AbstractMatrix`: shock loading on exogenous shocks
- `Γ0::AbstractMatrix`: structural matrix applying to time t states
- `Γ1::AbstractMatrix`: structural matrix applying to forward-looking variables
- `Γ2::AbstractMatrix`: structural matrix applying to backward-looking variables
- `Γ3::AbstractMatrix`: structural matrix applying to time t shocks

### Outputs
- The Jacobians ∂T∂θ and ∂R∂θ

"""
function transition_matrices_jacobian(m::AbstractDSGEModel, θ::ParameterVector,
                                      T::AbstractMatrix{S}, Γ0::AbstractMatrix{S},
                                      Γ1::AbstractMatrix{S},
                                      Γ2::AbstractMatrix{S}, Γ3::AbstractMatrix{S}) #;
                                  # zero_entries::BitArray = BitArray{undef,0,0}) where {S<:Real}
    # Compute derivatives of structural model matrices
    ∂Γ0∂θ, ∂Γ1∂θ, ∂Γ2∂θ, ∂Γ3∂θ = diff_structural_matrices(m, θ)

    # Compute ∂F / ∂TA' and ∂F / ∂θ' to get ∂T∂θ
    iden = Matrix{S}(I, size(T,1), size(T,1))
    kron_T_iden = kron(T_transpose, iden) # used for multiple computations
    T_transpose = T';
    ∂F∂T = kron(iden, Γ0) - kron(T_tranpose, Γ1) - kron(iden, Γ1 * T)
    ∂F∂θ = kron_T_iden * ∂vecΓ0∂θ -
        kron(T_transpose^2, iden) * ∂vecΓ1∂θ - ∂vecΓ2∂θ
    ∂T∂θ = -inv(∂F∂T) * ∂F∂θ

    # if !isempty(zero_entries)
    #     ∂A∂θ[zero_entries] .= zero(S)
    # end

    # Compute ∂R∂θ
    inv_Γ0_min_Γ1T = inv(Γ0 - Γ1 * T)
    n_exo_sh = n_shocks_exogenous(m)
    ∂R∂θ = -kron((inv_Γ0_min_Γ1T * Γ3)', inv_Γ0_min_Γ1T) *
             (∂Γ0∂θ - kron_T_iden * ∂Γ1∂θ - kron(iden,Γ1) * ∂T∂θ) +
             kron(Matrix{S}(I, n_exo_sh , n_exo_sh), inv_Γ0_min_Γ1T) * ∂Γ3∂θ

    return ∂T∂θ, ∂R∂θ
end

"""
```
measurement_matrices_jacobian(m, θ, T, Γ0, Γ1, Γ2, Γ3)
```
computes the Jacobian of the matrix Z and vector D in the measurement equation

```
yₜ = Zsₜ + D
```
when evaluated at θ.

### Arguments
- `m::AbstractDSGEModel`: model object
- `θ::ParameterVector`: vector of model parameters
- `T::AbstractMatrix`: transition matrix from time t-1 to t
- `R::AbstractMatrix`: shock loading on exogenous shocks
- `∂T∂θ::AbstractMatrix`: Jacobian of T
- `∂T∂R::AbstractMatrix`: Jacobian of R

### Outputs
- The Jacobians ∂Z∂θ and ∂D∂θ

"""
function measurement_matrices_jacobian(m::AbstractDSGEModel, θ::ParameterVector,
                                   T::AbstractMatrix{S}, R::AbstractMatrix{S},
                                   ∂T∂θ::AbstractMatrix{S}, ∂R∂θ::AbstractMatrix{S};
                                   check_valid_measurement_eqn::Bool = true) where {S<:Real}
    # We assume T, R are evaluated at θ
    Nθ = length(θ)
    NT = length(T)
    NR = length(R)

    # Differentiate Z, D as functions of theta, T, and R
    @inline function diff_meas_obj_fnct(var)
        θvar = var[1:Nθ]
        Tvar = var[Nθ+1:Nθ+NT]
        Rvar = var[Nθ+NT+1:end]
        DSGE.update!(m, var)
        measure_mat = measurement(m, Tvar, Rvar, zeros(sqrt(NT)))
        Zvar = measure_mat.ZZ
        Dvar = measure_mat.DD
        return vcat(vec(Zvar), vec(Dvar))
    end

    derivs = ForwardDiff.sparse_jacobian(diff_meas_obj_fnct, map(α -> α.value, θ))

    # Extract individual jacobians of Z w.r.t. θ, T, and R
    nobs = n_observables(m)
    nstates = n_states(m)
    Z_dim = nobs * nstates
    D_dim = nstates
    ∂Z∂θ = derivs[1:Z_dim,1:Nθ]
    ∂Z∂T = derivs[1:Z_dim,(Nθ+1):(Nθ+NT)]
    ∂Z∂R = derivs[1:Z_dim,(Nθ+NT+1):end]
    ∂D∂θ = derivs[1+Z_dim:end,1:Nθ]
    ∂D∂T = derivs[1+Z_dim:end,(Nθ+1):(Nθ+NT)]
    ∂D∂R = derivs[1+Z_dim:end,(Nθ+NT+1):end]

    # Determine if entries of ZZ ever depends on more than one of θ, T, or R
    if check_valid_measurement_eqn
        ∂Z∂θ_nonzero = sum(∂Z∂θ, dims = 2) .> 0 # Nonzero row sum -> Z depends on at least one θ
        ∂Z∂T_nonzero = sum(∂Z∂T, dims = 2) .> 0
        ∂Z∂R_nonzero = sum(∂Z∂R, dims = 2) .> 0
        ∂D∂θ_nonzero = sum(∂D∂θ, dims = 2) .> 0
        ∂D∂T_nonzero = sum(∂D∂T, dims = 2) .> 0
        ∂D∂R_nonzero = sum(∂D∂R, dims = 2) .> 0
        Z_dep_θ_T = sum(abs(∂Z∂θ_nonzero .* ∂Z∂T_nonzero) .> 0) # product is nonzero
        Z_dep_T_R = sum(abs(∂Z∂T_nonzero .* ∂Z∂R_nonzero) .> 0) # if depends on at least
        Z_dep_θ_R = sum(abs(∂Z∂θ_nonzero .* ∂Z∂R_nonzero) .> 0) # two of θ, T, and R
        D_dep_θ_T = sum(abs(∂D∂θ_nonzero .* ∂D∂T_nonzero) .> 0) # product is nonzero
        D_dep_T_R = sum(abs(∂D∂T_nonzero .* ∂D∂R_nonzero) .> 0) # if depends on at least
        D_dep_θ_R = sum(abs(∂D∂θ_nonzero .* ∂D∂R_nonzero) .> 0) # two of θ, T, and R
        if Z_dep_θ_T + Z_dep_T_R + Z_dep_θ_R + D_dep_θ_T + D_dep_T_R + D_dep_θ_R > 0
            error("Measurement equation invalid for applying the implicit function theorem to compute the Jacobian of reduced form matrices with respect to parameters.")
        end
    end

    # Compute derivatives via chain rule
    ∂Z∂θ += ∂Z∂T * ∂T∂θ + ∂Z∂R * ∂R∂θ
    ∂D∂θ += ∂D∂T * ∂T∂θ + ∂D∂R * ∂R∂θ
    return ∂Z∂θ
end
