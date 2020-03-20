"""
```
function reduced_form_jacobian(m, θ, T, R, Γ0, Γ1, Γ2, Γ3;
    measurement_jacobians = [:ZZ, :DD], use_sparse = true)
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

### Keywords
- `measurement_jacobians::Vector{Symbol}`: specifies which Jacobians of the
    measurement matrices to return. We always return the Jacobians of the
    state transition matrices. Note that even if
    the requested jacobians are not ordered in the same way as the
    the default, we will still return them in the same order as the default.
    For example, `jacobians = [:DD, :ZZ]` returns `∂T∂θ, ∂R∂θ, ∂Z∂θ, ∂D∂θ`.
    Also, note that `Z` => `ZZ` and `D` => `DD`.
- `use_sparse::Bool`: when true, sparse matrices are used throughout the computations.

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
                               Γ2::AbstractMatrix{Y}, Γ3::AbstractMatrix{Y};
                               measurement_jacobians::Vector{Symbol} =
                               [:ZZ, :DD], use_sparse::Bool = true) where {U<:Real, X<:Real, Y<:Real}

    # Figure out what to compute and return
    do_ZZ  = :ZZ in measurement_jacobians
    do_DD  = :DD in measurement_jacobians

    # Apply implicit function theorem to compute Jacobians of
    # reduced form matrices
    ∂T∂θ, ∂R∂θ = transition_matrices_jacobian(m, θ, T, Γ0, Γ1, Γ2, Γ3;
                                              use_sparse = use_sparse) # this must always be computed
    out = measurement_matrices_jacobian(m, θ, T, R, ∂T∂θ, ∂R∂θ;
                                        compute_Z = do_ZZ, # these may be optional
                                        compute_D = do_DD,
                                        use_sparse = use_sparse)

    if do_ZZ && do_DD
        return ∂T∂θ, ∂R∂θ, out[1], out[2]
    else
        return ∂T∂θ, ∂R∂θ, out
    end
end

function reduced_form_jacobian(m::AbstractDSGEModel, θ::ParameterVector,
                               T::AbstractMatrix{X}, R::AbstractMatrix{X},
                               Γ0::AbstractMatrix{Y}, Γ1::AbstractMatrix{Y},
                               Γ2::AbstractMatrix{Y}, Γ3::AbstractMatrix{Y};
                               use_sparse::Bool = true) where {X<:Real, Y<:Real}
    reduced_form_jacobian(m, map(x -> x.value, θ), T, R, Γ0, Γ1, Γ2, Γ3;
                          use_sparse = use_sparse)
end

"""
```
structural_matrices_jacobian(m, θ; use_sparse = true)
```
computes the Jacobian of the structural matrices in the structural system

```
Γ₀sₜ = Γ₁Eₜsₜ₊₁ + Γ₂sₜ₋₁ + Γ₃ϵₜ.
```
when evaluated at θ.

### Arguments
- `m::AbstractDSGEModel`: model object
- `θ::AbstractVector`  or `ParameterVector`: vector of model parameters

### Keywords
- `use_sparse::Bool`: when true, sparse matrices are used throughout the computations.

### Outputs
- The Jacobians ∂Γ0∂θ, ∂Γ1∂θ, ∂Γ2∂θ, and ∂Γ3∂θ

"""
function structural_matrices_jacobian(m::AbstractDSGEModel, θ::AbstractVector{S};
                                      use_sparse::Bool = true) where {S<:Real}

    θold = copy(θ) # To ensure m's parameters are the same after this function ends

    # Check for fixed parameters
    if length(θ) < length(m.parameters)
        unfixed = get_unfixed_parameter_indices(m)
        update_wrapper! = x -> ModelConstructors.update!(m.parameters,
                                                      x, unfixed;
                                                      change_value_type = true)
    else
        update_wrapper! = x -> ModelConstructors.update!(m.parameters,
                                                      x; change_value_type = true)
    end

    # Write structural matrices as function of unfixed parameters.
    # SPEED THIS UP USING SPARSEDIFFTOOLS
    # AND IGNORING ENTRIES WHICH DON'T DEPEND ON ANYTHING
    @inline function diff_struct_obj_fnct(var)
        update_wrapper!(var)
        steadystate!(m)
        Γ0, Γ1, Γ2, Γ3 = eqcond(m; method = :klein,
                                matrix_type = Real)
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
    ∂Γ0∂θ = sparse(convert(Matrix{model_type}, derivs[1:Γ0_dim,:]))
    ∂Γ1∂θ = sparse(convert(Matrix{model_type}, derivs[(Γ0_dim+1):(Γ1_dim+Γ0_dim),:]))
    ∂Γ2∂θ = sparse(convert(Matrix{model_type}, derivs[(Γ1_dim+Γ0_dim+1):(Γ2_dim+Γ1_dim+Γ0_dim),:]))
    ∂Γ3∂θ = sparse(convert(Matrix{model_type}, derivs[(Γ2_dim+Γ1_dim+Γ0_dim+1):end,:]))

    update_wrapper!(θold)
    steadystate!(m)

    return ∂Γ0∂θ, ∂Γ1∂θ, ∂Γ2∂θ, ∂Γ3∂θ
end

function structural_matrices_jacobian(m::AbstractDSGEModel, θ::ParameterVector)
    structural_matrices_jacobian(m, map(x -> x.value, θ))
end

"""
```
transition_matrices_jacobian(m, θ, T, Γ0, Γ1, Γ2, Γ3; use_sparse = true)
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

### Keywords
- `use_sparse::Bool`: when true, sparse matrices are used throughout the computations.

### Outputs
- The Jacobians ∂T∂θ and ∂R∂θ

"""
function transition_matrices_jacobian(m::AbstractDSGEModel, θ::AbstractVector{U},
                                      T::AbstractMatrix{X}, Γ0::AbstractMatrix{Y},
                                      Γ1::AbstractMatrix{Y}, Γ2::AbstractMatrix{Y},
                                      Γ3::AbstractMatrix{Y}; use_sparse::Bool = true) where {U<:Real, X<:Real, Y<:Real}
    θold = copy(θ)

    # Compute derivatives of structural model matrices
    ∂Γ0∂θ, ∂Γ1∂θ, ∂Γ2∂θ, ∂Γ3∂θ = structural_matrices_jacobian(m, θ; use_sparse = use_sparse)

    # Compute ∂F / ∂T' and ∂F / ∂θ' to get ∂T∂θ
    iden = SparseMatrixCSC{eltype(T)}(I, size(T,1), size(T,1))
    T_transpose = T'
    kron_T_iden = kron(T_transpose, iden) # used for multiple computations
    if use_sparse
        ∂F∂T = kron(iden, Γ0) - kron(T_transpose, Γ1) - kron(iden, sparse(Γ1 * T))
        ∂F∂θ = kron_T_iden * ∂Γ0∂θ -
            kron(T_transpose^2, iden) * ∂Γ1∂θ - ∂Γ2∂θ
        ∂T∂θ = -sparse(∂F∂T \ ∂F∂θ)
    else
        ∂F∂T = kron(iden, Γ0) - kron(T_transpose, Γ1) - kron(iden, Γ1 * T)
        ∂F∂θ = kron_T_iden * ∂Γ0∂θ -
                   kron(T_transpose^2, iden) * ∂Γ1∂θ - ∂Γ2∂θ
        ∂T∂θ = -(∂F∂T \ ∂F∂θ) # Equivalent to ∂T∂θ = -inv(convert(Matrix{eltype(iden)}, ∂F∂T)) * ∂F∂θ
    end

    # Compute ∂R∂θ
    inv_Γ0_min_Γ1T = inv(convert(Matrix{eltype(iden)}, Γ0 - Γ1 * T))
    n_exo_sh = n_shocks_exogenous(m)
    ∂R∂θ = -kron((inv_Γ0_min_Γ1T * Γ3)', inv_Γ0_min_Γ1T) *
             (∂Γ0∂θ - kron_T_iden * ∂Γ1∂θ - kron(iden,Γ1) * ∂T∂θ) +
             kron(SparseMatrixCSC{eltype(T)}(I, n_exo_sh , n_exo_sh), inv_Γ0_min_Γ1T) * ∂Γ3∂θ
    if use_sparse
        ∂R∂θ = sparse(∂R∂θ)
    end

    # Make sure types are the same as they used to be (may change due to use of ForwardDiff)
    if length(θ) < length(m.parameters)
        unfixed = get_unfixed_parameter_indices(m)
        ModelConstructors.update!(m.parameters, θold, unfixed; change_value_type = true)
    else
        ModelConstructors.update!(m.parameters, θold; change_value_type = true)
    end
    steadystate!(m)

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

### Keywords
- `compute_Z::Bool`: true if we want to compute the Jacobian of Z
- `compute_D::Bool`: true if we want to compute the Jacobian of D
- `use_sparse::Bool`: when true, we use sparse matrices throughout the computations

### Outputs
- The Jacobians ∂Z∂θ and ∂D∂θ

"""
function measurement_matrices_jacobian(m::AbstractDSGEModel, θ::AbstractVector{U},
                                       T::AbstractMatrix{X}, R::AbstractMatrix{X},
                                       ∂T∂θ::AbstractMatrix{Y},
                                       ∂R∂θ::AbstractMatrix{Y};
                                       compute_Z::Bool = true,
                                       compute_D::Bool = true,
                                       use_sparse::Bool = true) where {U<:Real, X<:Real, Y<:Real}
    θold = copy(θ)

    # We assume T, R are evaluated at θ
    Nθ = length(θ)
    NT = length(T)
    NR = length(R)

    # Check if there are fixed parameters and accordingly
    # define appropriate function to differentiate
    # Z and D as functions of theta, T, and R
    if length(θ) < length(m.parameters)
        unfixed = get_unfixed_parameter_indices(m)
        update_wrapper! = x -> ModelConstructors.update!(m.parameters,
                                                      x, unfixed;
                                                      change_value_type = true)
    else
        update_wrapper! = x -> ModelConstructors.update!(m.parameters, x,
                                                      change_value_type = true)
    end

    # SPEED THIS UP WITH SPARSEDIFFTOOLS
    # CAN MANUALLY FIGURE OUT THE SPARSITY PATTERN FOR Z AND D TOO
    # TTT AND RRR MAY BE APPROXIMATED AS SPARSE POTENTIALLY FOR THINGS SMALLER THAN MACHINE ERROR
    # ALSO ADD AN OPTION TO AVOID COMPUTING THE DERIVATIVE W.R.T. TTT AND RRR
    diff_meas_obj_fnct = if compute_Z && compute_D
        function _diff_meas_obj_fnct1(var)
            θvar = var[1:Nθ]
            Tvar = var[Nθ+1:Nθ+NT]
            Rvar = var[Nθ+NT+1:end]
            update_wrapper!(θvar)
            steadystate!(m)
            measure_mat = measurement(m, reshape(Tvar, size(T)...), reshape(Rvar, size(R)...),
                                      zeros(eltype(Tvar), size(T,1)))
            return vcat(vec(measure_mat.ZZ), vec(measure_mat.DD))
        end
    elseif compute_Z
        function _diff_meas_obj_fnct2(var)
            θvar = var[1:Nθ]
            Tvar = var[Nθ+1:Nθ+NT]
            Rvar = var[Nθ+NT+1:end]
            update_wrapper!(θvar)
            steadystate!(m)
            measure_mat = measurement(m, reshape(Tvar, size(T)...), reshape(Rvar, size(R)...),
                                      zeros(eltype(Tvar), size(T,1)))
            return vec(measure_mat.ZZ)
        end
    elseif compute_D
        function _diff_meas_obj_fnct3(var)
            θvar = var[1:Nθ]
            Tvar = var[Nθ+1:Nθ+NT]
            Rvar = var[Nθ+NT+1:end]
            update_wrapper!(θvar)
            steadystate!(m)
            measure_mat = measurement(m, reshape(Tvar, size(T)...), reshape(Rvar, size(R)...),
                                      zeros(eltype(Tvar), size(T,1)))
            return vec(measure_mat.DD)
        end
    else
        error("Both `compute_Z` and `compute_D` cannot be false.")
    end
    derivs = ForwardDiff.jacobian(diff_meas_obj_fnct, Vector{Float64}(vcat(θ, vec(T), vec(R))))

    # Extract individual jacobians of Z, D w.r.t. θ, T, and R.
    # Compute derivatives via chain rule afterward.
    nobs = n_observables(m)
    nstates = get_setting(m, :n_endogenous_states_klein)
    Z_dim = nobs * nstates
    D_dim = nobs
    model_type = findparam(m)

    if compute_Z
        if use_sparse
            ∂Z∂θ = sparse(convert(Matrix{model_type}, derivs[1:Z_dim, 1:Nθ]))
            ∂Z∂T = sparse(convert(Matrix{model_type}, derivs[1:Z_dim, (Nθ+1):(Nθ+NT)]))
            ∂Z∂R = sparse(convert(Matrix{model_type}, derivs[1:Z_dim, (Nθ+NT+1):end]))
        else
            ∂Z∂θ = convert(Matrix{model_type}, derivs[1:Z_dim, 1:Nθ])
            ∂Z∂T = convert(Matrix{model_type}, derivs[1:Z_dim, (Nθ+1):(Nθ+NT)])
            ∂Z∂R = convert(Matrix{model_type}, derivs[1:Z_dim, (Nθ+NT+1):end])
        end
        ∂Z∂θ += ∂Z∂T * ∂T∂θ + ∂Z∂R * ∂R∂θ
    end

    if compute_D
        inds = compute_Z ? (1 + Z_dim:Z_dim + D_dim) : (1:D_dim) # need to determine the correct indices
        if use_sparse
            ∂D∂θ = sparse(convert(Matrix{model_type}, derivs[inds, 1:Nθ]))
            ∂D∂T = sparse(convert(Matrix{model_type}, derivs[inds, (Nθ+1):(Nθ+NT)]))
            ∂D∂R = sparse(convert(Matrix{model_type}, derivs[inds, (Nθ+NT+1):end]))
        else
            ∂D∂θ = convert(Matrix{model_type}, derivs[inds, 1:Nθ])
            ∂D∂T = convert(Matrix{model_type}, derivs[inds, (Nθ+1):(Nθ+NT)])
            ∂D∂R = convert(Matrix{model_type}, derivs[inds, (Nθ+NT+1):end])
        end
        ∂D∂θ += ∂D∂T * ∂T∂θ + ∂D∂R * ∂R∂θ
    end

    # Set parameters of m back to the old ones
    update_wrapper!(θold)
    steadystate!(m)

    # Make sure partials are same type as T and R
    if compute_Z && compute_D
        return ∂Z∂θ, ∂D∂θ
    elseif compute_Z
        return ∂Z∂θ
    else
        return ∂D∂θ
    end
end

function measurement_matrices_jacobian(m::AbstractDSGEModel, θ::ParameterVector,
                                       T::AbstractMatrix{X}, R::AbstractMatrix{X},
                                       ∂T∂θ::AbstractMatrix{Y}, ∂R∂θ::AbstractMatrix{Y};
                                       use_sparse::Bool = true) where {X<:Real, Y<:Real}
    measurement_matrices_jacobian(m, map(x -> x.value, θ), T, R, ∂T∂θ, ∂R∂θ;
                                  use_sparse = use_sparse)
end
