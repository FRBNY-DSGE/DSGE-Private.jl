function hmc_posterior(m::AbstractDSGEModel{S}, x::AbstractVector{T}, data::AbstractArray{S}) where {T <: Real, S <: Real}
    return hmc_likelihood(m, x, data) +
        ModelConstructors.prior(m.parameters[.!get_fixed_parameter_indices(m)], x; isfree = true)
end


function hmc_likelihood(m::AbstractDSGEModel{S}, x::AbstractVector{T}, data::AbstractArray{S}) where {T <: Real, S <: Real}

    # Transform x from ℝⁿ to model space and update parameters in m
    unfixed = .!(convert(BitArray{1}, get_fixed_parameter_indices(m)))
    θ = transform_to_model_space(m.parameters[unfixed], x)
    update!(m.parameters, θ, unfixed) # Update the values of only unfixed parameters
    steadystate!(m) # need to recompute steadystate after updating parameters

    # Compute likelihood
    TTT, RRR, CCC = solve(m)
    if get_setting(m, :solution_method) == :gensys
        inds = non_expectational_error_terms(m)
        TTT = TTT[inds, inds]
        RRR = RRR[inds, :]
        CCC = CCC[inds]
    end
    meas = measurement(m, TTT, RRR, CCC)
    s0, P0 = DSGE.init_stationary_states(TTT, RRR, CCC, meas[:QQ];
                                         check_is_stationary = false)
    loglhs = DSGE.kalman_likelihood(data, TTT, RRR, CCC,
                                    meas[:QQ], meas[:ZZ], meas[:DD], meas[:EE],
                                    s0, P0)
    return sum(loglhs)
end

# KF as a function of the reduced for matrices
# REWRITE as a generated function so we automatically know
# which matrices are part of RF or not.
# assumes matrices are in the pre-specified order [:TTT, :RRR, :CCC, :ZZ, :DD, :QQ, :EE]
function kalman_likelihood(m::AbstractDSGEModel, RF::AbstractVector{T},
                           matrices::Vector{Symbol},
                           mat_dict::Dict{Symbol, AbstractArray{S}}) where {T <: Real, S <: Real}
    # Get dimension sizes
    if get_setting(m, :solution_method) == :gensys
        n_states = non_expectational_error_terms(m)
    end
    n_obs = n_observables(m)
    n_shocks = n_shocks_exogenous(m)

    # Figure out which matrices are in RF and which are fixed
    fixed_matrices = setdiff([:TTT, :RRR, :CCC, :ZZ, :DD, :QQ, :EE], matrices)
    in_RF = Dict{Symbol, Bool}()
    for mat in matrices
        in_RF[mat] = true
    end
    for mat in fixed_matrices
        in_RF[mat] = false
    end

    # Instantiate state space system matrices
    N = 0
    if in_RF[:TTT]
        TTT = reshape(RF[1 + N:n_states^2 + N], n_states, n_states)
        N   += n_states^2
    else
        TTT = mat_dict[:TTT]
    end
    if in_RF[:RRR]
        RRR = reshape(RF[1 + N:n_states * n_shocks + N], n_states, n_shocks)
        N   += n_states * n_shocks
    else
        RRR = mat_dict[:RRR]
    end
    if in_RF[:CCC]
        CCC = RF[1 + N:n_states + N]
        N   += n_states
    else
        CCC = mat_dict[:CCC]
    end
    if in_RF[:ZZ]
        ZZ = reshape(RF[1 + N:n_obs * n_states + N], n_obs, n_states)
        N  += n_obs * n_states
    else
        ZZ = mat_dict[:ZZ]
    end
    if in_RF[:DD]
        DD = RF[1 + N:n_obs + N]
        N  += n_obs
    else
        DD = mat_dict[:DD]
    end
    if in_RF[:QQ]
        QQ = reshape(RF[1 + N:n_shocks^2 + N], n_shocks, n_shocks)
        N  += n_shocks^2
    else
        QQ = mat_dict[:QQ]
    end
    if in_RF[:EE]
        EE = reshape(RF[1 + N:n_obs^2 + N], n_obs, n_obs)
    else
        EE = mat_dict[:EE]
    end

    s0, P0 = init_stationary_states(TTT, RRR, CCC, QQ; check_is_stationary = false)
    return sum(DSGE.kalman_likelihood(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, s0, P0))
end

function hmc_gradient(m::AbstractDSGEModel{S}, x::AbstractVector{T}, data::AbstractArray{S};
                      matrices::Vector{Symbol} = [:TTT, :RRR, :ZZ, :DD],
                      ordered::Bool = false) where {T <: Real, S <: Real}

    # We need the matrices to satisfy a pre-specified order
    if !ordered
        sort_order = Dict{Symbol, Int}()
        sort_order[:TTT] = 1
        sort_order[:RRR] = 1
        sort_order[:CCC] = 1
        sort_order[:ZZ]  = 1
        sort_order[:DD]  = 1
        sort_order[:QQ]  = 1
        sort_order[:EE]  = 1
        matrices = sort(matrices, by = mat -> sort_order[mat])
    end

    # Transform x from ℝⁿ to model space θ and update parameters in m
    unfixed   = convert(BitArray{1}, get_unfixed_parameter_indices(m))
    fixed     = convert(BitArray{1}, get_fixed_parameter_indices(m))
    # unfixed_θ = transform_to_model_space(m.parameters[unfixed], x)
    # update!(m.parameters, unfixed_θ, unfixed) # this only updates unfixed parameters
    # steadystate!(m) # need to recompute steadystate after updating parameters

    # Faster to do the following, fewer allocations
    full_x = Vector{eltype(x)}(undef, length(m.parameters))
    full_x[unfixed] .= x
    full_x[fixed]   .= map(y -> y.value, m.parameters[fixed])
    transform_to_model_space!(m, full_x)

    # account for change of variables
    ∂θ∂x = differentiate_transform_to_model_space(m.parameters[unfixed], x)

    # Compute Jacobians of RF with respect to θ
    Γ0, Γ1, Γ2, Γ3 = eqcond(m; method = :klein)
    mat_dict = Dict{Symbol, AbstractArray{eltype(x)}}()        # Use a Dict as a way to avoid copying matrices
    mat_dict[:TTT], mat_dict[:RRR], mat_dict[:CCC] = solve(m)  # we want to avoid an additional allocation
    if get_setting(m, :solution_method) == :gensys             # so we avoid doing a loop over [TTT, RRR, CCC, ...]
        inds = non_expectational_error_terms(m)
        mat_dict[:TTT] = mat_dict[:TTT][inds, inds]
        mat_dict[:RRR] = mat_dict[:RRR][inds, :]
        mat_dict[:CCC] = mat_dict[:CCC][inds]
    end
    ∂RF∂θ = Dict{Symbol, AbstractArray{S}}() # This is to avoid another allocation
    ∂RF∂θ[:TTT], ∂RF∂θ[:RRR], ∂RF∂θ[:ZZ], ∂RF∂θ[:DD] = reduced_form_jacobian(m, unfixed_θ, mat_dict[:TTT], mat_dict[:RRR],
                                                                             Γ0, Γ1, Γ2, Γ3; matrices = matrices)

    # Compute gradient of log likelihood (from Kalman filter)
    # to RF state space system matrices
    meas = measurement(m, mat_dict[:TTT], mat_dict[:RRR], mat_dict[:CCC])
    mat_dict[:ZZ]  = meas[:ZZ]
    mat_dict[:DD]  = meas[:DD]
    mat_dict[:QQ]  = meas[:QQ]
    mat_dict[:EE]  = meas[:EE]
    RF = param(mapreduce(x -> vec(mat_dict[x]), vcat, matrices))
    kf_loglh(RF) = kalman_likelihood(m, RF, matrices, mat_dict)
    Tracker.back!(kf_loglh(RF))
    grad_RF = Tracker.grad(RF)

    # Compute gradient of log likelihood with respect to ℝⁿ
    N = zero(Int)
    grad = Vector{S}(undef, length(x))
    for matrix in matrices
        grad += ∂RF∂θ[matrix]' * grad_RF[1 + N:length(mat_dict[matrix]) + N]
        N += length(mat_dict[matrix])
    end
    grad .*= ∂θ∂x

    # Compute gradient of prior to ℝⁿ
    # Note that the `prior(⋅)` used here is the prior over ℝⁿ, i.e.
    # f_Y(y) = f_X(v(y)) × |v'(y)| where f(⋅) is the prior pdf, Y = ℝⁿ, X = model space,
    #                              and v is the transformation from to ℝⁿ to model space
    nonuniform_prior = map(x -> !(typeof(get(x.prior)) <: Uniform), m.parameters[unfixed])
    reverse_mode_x = param(x[nonuniform_prior])
    nonuniform_parameters = m.parameters[unfixed][nonuniform_prior]
    # change_type_vals = param(map(x -> x.value, nonuniform_parameters))
    # update!(nonuniform_parameters, change_type_vals; change_value_type = true)
    Tracker.back!(ModelConstructors.prior(nonuniform_parameters, reverse_mode_x; isfree = true))
    grad_prior = zeros(length(unfixed_θ))
    grad_prior[nonuniform_priors] = vec(Tracker.grad(reverse_mode_x))
    grad += grad_prior

    # Convert types of parameters away from ForwardDiff.Dual or Tracker
    update!(m.parameters, unfixed_θ, unfixed; change_value_type = true)

    return grad
end
