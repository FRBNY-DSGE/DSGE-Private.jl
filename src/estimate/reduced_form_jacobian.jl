"""
```
function reduced_form_jacobian(m, θ, T, R, Γ0, Γ1, Γ2, Γ3)
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
- `θ::AbstractVector`  or `ParameterVector`: vector of model parameters
- `T::AbstractMatrix`: transition matrix from time t-1 to t
- `R::AbstractMatrix`: shock loading on exogenous shocks
- `Γ0::AbstractMatrix`: structural matrix applying to time t states
- `Γ1::AbstractMatrix`: structural matrix applying to forward-looking variables
- `Γ2::AbstractMatrix`: structural matrix applying to backward-looking variables
- `Γ3::AbstractMatrix`: structural matrix applying to time t shocks

### Outputs
- The Jacobians ∂T∂θ, ∂R∂θ, ∂Z∂θ, and ∂D∂θ

Remaining things to do:
* Speed up auto-differentiation by allowing the user to tell the function
  whether or not the measurement matrices ever depend on T or R (to avoid excess differentiation)
* Add in differentiation for cases when you have nonzero CCC and nonzero EE

Notes on type choices
* We currently require T, R, and Klein metrices to be subtypes of Real, and the same subtypes,
  so that when we call measurement with inputs m, T, R, and C, the Measurement object
  will be created, and it is auto-differentiable
"""
function reduced_form_jacobian(m::AbstractDSGEModel, θ::AbstractVector{U},
                               T::AbstractMatrix{X}, R::AbstractMatrix{X},
                               Γ0::AbstractMatrix{Y}, Γ1::AbstractMatrix{Y},
                               Γ2::AbstractMatrix{Y},
                               Γ3::AbstractMatrix{Y}) where {U<:Real, X<:Real, Y<:Real}

    # Apply implicit function theorem to compute Jacobians of
    # reduced form matrices
    ∂T∂θ, ∂R∂θ = transition_matrices_jacobian(m, θ, T, Γ0, Γ1, Γ2, Γ3)
    ∂Z∂θ, ∂D∂θ = measurement_matrices_jacobian(m, θ, T, R, ∂T∂θ, ∂R∂θ)

    return ∂T∂θ, ∂R∂θ, ∂Z∂θ, ∂D∂θ
end

function reduced_form_jacobian(m::AbstractDSGEModel, θ::ParameterVector,
                               T::AbstractMatrix{X}, R::AbstractMatrix{X},
                               Γ0::AbstractMatrix{Y}, Γ1::AbstractMatrix{Y},
                               Γ2::AbstractMatrix{Y}, Γ3::AbstractMatrix{Y}) where {X<:Real, Y<:Real}
    reduced_form_jacobian(m, map(x -> x.value, θ), T, R, Γ0, Γ1, Γ2, Γ3)
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
- `θ::AbstractVector`  or `ParameterVector`: vector of model parameters

### Outputs
- The Jacobians ∂Γ0∂θ, ∂Γ1∂θ, ∂Γ2∂θ, and ∂Γ3∂θ

"""
function structural_matrices_jacobian(m::AbstractDSGEModel, θ::AbstractVector{S}) where {S<:Real}

    θold = copy(θ) # To ensure m's parameters are the same after this function ends

    # Check for fixed parameters
    if length(θ) < length(m.parameters)
        unfixed_inds = .!(get_fixed_parameter_indices(m))
        update_wrapper! = x -> ModelConstructors.update!(m.parameters,
                                                      x, unfixed_inds;
                                                      change_value_type = true)
    else
        update_wrapper! = x -> ModelConstructors.update!(m.parameters,
                                                      x; change_value_type = true)
    end

    # Write structural matrices as function of unfixed parameters.
    @inline function diff_struct_obj_fnct(var)
        update_wrapper!(var)
        Γ0, Γ1, Γ2, Γ3 = eqcond(m; method = :klein)
        return vcat(vec(Γ0), vec(Γ1), vec(Γ2), vec(Γ3))
    end

    derivs = ForwardDiff.jacobian(diff_struct_obj_fnct, θ)

    # derivs is a matrix whose rows are gradients of a single entry
    # of a structural matrix w.r.t. the vector θ,
    # so to retrieve ∂Γ0∂θ, we want the first n_endogenous_states_klein^2 rows
    n_endo = get_setting(m, :n_endogenous_states_klein)
    Γ0_dim = n_endo^2
    Γ1_dim = Γ0_dim # copies b/c primitive
    Γ2_dim = Γ0_dim
    Γ3_dim = n_endo * n_shocks_exogenous(m)
    model_type = findparam(m)
    ∂Γ0∂θ = Matrix{model_type}(derivs[1:Γ0_dim,:])
    ∂Γ1∂θ = Matrix{model_type}(derivs[(Γ0_dim+1):(Γ1_dim+Γ0_dim),:])
    ∂Γ2∂θ = Matrix{model_type}(derivs[(Γ1_dim+Γ0_dim+1):(Γ2_dim+Γ1_dim+Γ0_dim),:])
    ∂Γ3∂θ = Matrix{model_type}(derivs[(Γ2_dim+Γ1_dim+Γ0_dim+1):end,:])

    update_wrapper!(θold)

    return ∂Γ0∂θ, ∂Γ1∂θ, ∂Γ2∂θ, ∂Γ3∂θ
end
function structural_matrices_jacobian(m::AbstractDSGEModel, θ::ParameterVector)
    structural_matrices_jacobian(m, map(x -> x.value, θ))
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
- `θ::AbstractVector`  or `ParameterVector`: vector of model parameters
- `T::AbstractMatrix`: transition matrix from time t-1 to t
- `R::AbstractMatrix`: shock loading on exogenous shocks
- `Γ0::AbstractMatrix`: structural matrix applying to time t states
- `Γ1::AbstractMatrix`: structural matrix applying to forward-looking variables
- `Γ2::AbstractMatrix`: structural matrix applying to backward-looking variables
- `Γ3::AbstractMatrix`: structural matrix applying to time t shocks

### Outputs
- The Jacobians ∂T∂θ and ∂R∂θ

"""
function transition_matrices_jacobian(m::AbstractDSGEModel, θ::AbstractVector{U},
                                      T::AbstractMatrix{X}, Γ0::AbstractMatrix{Y},
                                      Γ1::AbstractMatrix{Y}, Γ2::AbstractMatrix{Y},
                                      Γ3::AbstractMatrix{Y}) where {U<:Real, X<:Real, Y<:Real}
    θold = copy(θ)

    # Compute derivatives of structural model matrices
    ∂Γ0∂θ, ∂Γ1∂θ, ∂Γ2∂θ, ∂Γ3∂θ = structural_matrices_jacobian(m, θ)

    # Compute ∂F / ∂T' and ∂F / ∂θ' to get ∂T∂θ
    iden = Matrix(I, size(T,1), size(T,1))
    T_transpose = T';
    kron_T_iden = kron(T_transpose, iden) # used for multiple computations
    ∂F∂T = kron(iden, Γ0) - kron(T_transpose, Γ1) - kron(iden, Γ1 * T)
    ∂F∂θ = kron_T_iden * ∂Γ0∂θ -
        kron(T_transpose^2, iden) * ∂Γ1∂θ - ∂Γ2∂θ
    ∂T∂θ = -inv(∂F∂T) * ∂F∂θ

    # Compute ∂R∂θ
    inv_Γ0_min_Γ1T = inv(Γ0 - Γ1 * T)
    n_exo_sh = n_shocks_exogenous(m)
    ∂R∂θ = -kron((inv_Γ0_min_Γ1T * Γ3)', inv_Γ0_min_Γ1T) *
             (∂Γ0∂θ - kron_T_iden * ∂Γ1∂θ - kron(iden,Γ1) * ∂T∂θ) +
             kron(Matrix(I, n_exo_sh , n_exo_sh), inv_Γ0_min_Γ1T) * ∂Γ3∂θ

    if length(θ) < length(m.parameters)
        unfixed_inds = .!(get_fixed_parameter_indices(m))
        ModelConstructors.update!(m.parameters, θold, unfixed_inds; change_value_type = true)
    else
        ModelConstructors.update!(m.parameters, θold; change_value_type = true)
    end

    # Convert to type S to make sure partial derivatives are the same type as T and R
    return ∂T∂θ, ∂R∂θ
end
function transition_matrices_jacobian(m::AbstractDSGEModel, θ::ParameterVector,
                                      T::AbstractMatrix{X}, Γ0::AbstractMatrix{Y},
                                      Γ1::AbstractMatrix{Y},
                                      Γ2::AbstractMatrix{Y},
                                      Γ3::AbstractMatrix{Y}) where {X<:Real, Y<:Real}
    transition_matrices_jacobian(m, map(x -> x.value, θ), T, Γ0, Γ1, Γ2, Γ3)
end

"""
```
measurement_matrices_jacobian(m, θ, T, R, ∂T∂θ, ∂R∂θ)
```
computes the Jacobian of the matrix Z and vector D in the measurement equation

```
yₜ = Zsₜ + D
```
when evaluated at θ.

### Arguments
- `m::AbstractDSGEModel`: model object
- `θ::AbstractVector`  or `ParameterVector`: vector of model parameters
- `T::AbstractMatrix`: transition matrix from time t-1 to t
- `R::AbstractMatrix`: shock loading on exogenous shocks
- `∂T∂θ::AbstractMatrix`: Jacobian of T
- `∂T∂R::AbstractMatrix`: Jacobian of R

### Outputs
- The Jacobians ∂Z∂θ and ∂D∂θ

"""
function measurement_matrices_jacobian(m::AbstractDSGEModel, θ::AbstractVector{U},
                                       T::AbstractMatrix{X}, R::AbstractMatrix{X},
                                       ∂T∂θ::AbstractMatrix{Y},
                                       ∂R∂θ::AbstractMatrix{Y}) where {U<:Real, X<:Real, Y<:Real}
    θold = copy(θ)

    # We assume T, R are evaluated at θ
    Nθ = length(θ)
    NT = length(T)
    NR = length(R)

    # Check if there are fixed parameters and accordingly
    # define appropriate function to differentiate
    # Z and D as functions of theta, T, and R
    if length(θ) < length(m.parameters)
        unfixed_inds = .!(get_fixed_parameter_indices(m))
        update_wrapper! = x -> ModelConstructors.update!(m.parameters,
                                                      x, unfixed_inds;
                                                      change_value_type = true)
    else
        update_wrapper! = x -> ModelConstructors.update!(m.parameters, x,
                                                      change_value_type = true)
    end

    @inline function diff_meas_obj_fnct(var)
        θvar = var[1:Nθ]
        Tvar = var[Nθ+1:Nθ+NT]
        Rvar = var[Nθ+NT+1:end]
        update_wrapper!(θvar)
        measure_mat = measurement(m, reshape(Tvar, size(T)...), reshape(Rvar, size(R)...),
                                  zeros(eltype(Tvar), size(T,1)))
        Zvar = measure_mat.ZZ
        Dvar = measure_mat.DD
        return vcat(vec(Zvar), vec(Dvar))
    end

    derivs = ForwardDiff.jacobian(diff_meas_obj_fnct, Vector{Real}(vcat(θ, vec(T), vec(R))))

    # Extract individual jacobians of Z w.r.t. θ, T, and R
    nobs = n_observables(m)
    nstates = get_setting(m, :n_endogenous_states_klein)
    Z_dim = nobs * nstates
    D_dim = nstates
    ∂Z∂θ = derivs[1:Z_dim,1:Nθ]
    ∂Z∂T = derivs[1:Z_dim,(Nθ+1):(Nθ+NT)]
    ∂Z∂R = derivs[1:Z_dim,(Nθ+NT+1):end]
    ∂D∂θ = derivs[1+Z_dim:end,1:Nθ]
    ∂D∂T = derivs[1+Z_dim:end,(Nθ+1):(Nθ+NT)]
    ∂D∂R = derivs[1+Z_dim:end,(Nθ+NT+1):end]

    # Set parameters of m back to the old ones
    update_wrapper!(θold)

    # Compute derivatives via chain rule
    ∂Z∂θ += ∂Z∂T * ∂T∂θ + ∂Z∂R * ∂R∂θ
    ∂D∂θ += ∂D∂T * ∂T∂θ + ∂D∂R * ∂R∂θ

    # Make sure partials are same type as T and R
    return ∂Z∂θ, ∂D∂θ
end
function measurement_matrices_jacobian(m::AbstractDSGEModel, θ::ParameterVector,
                                       T::AbstractMatrix{X}, R::AbstractMatrix{X},
                                       ∂T∂θ::AbstractMatrix{Y},
                                       ∂R∂θ::AbstractMatrix{Y}) where {X<:Real, Y<:Real}
    measurement_matrices_jacobian(m, map(x -> x.value, θ), T, R, ∂T∂θ, ∂R∂θ)
end
