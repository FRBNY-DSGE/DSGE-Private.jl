# TODO: add an option that constructs nt and id only once rather than repeatedly
"""
```
jacobian(m::BayerBornLuetticke)
```
compute the Jacobians of the non-linear difference equations defined
by the functions `Fsys` and `Fsys_agg` using the package `ForwardDiff`.

If the user only wants to update the aggregate blocks of the Jacobians
without changing the heterogeneous blocks, then the user
should set `m <= Setting(:differentiate_heterogeneous_blocks, false)`.
In this case, the Jacobians are already stored in `m` and
are updated in place.

# Returns
- `A`,`B`: first derivatives of `Fsys` with respect to arguments `X` [`B`] and
    `XPrime` [`A`]
"""
@inline function jacobian(m::BayerBornLuetticke{T}) where {T <: Real}
    if get_setting(m, :linearize_heterogeneous_blocks)::Bool
        # Compute the Jacobian from scratch (assumes steadystate!(m) has already been called)
        return _jacobian!(m)
    else
        # Note that since get_untransformed_values(m[:A]) = m[:A].value,
        # the A matrix in m[:A] is being directly updated
        return _update_aggregate_jacobian!(m, get_untransformed_values(m[:A])::Matrix{T},
                                           get_untransformed_values(m[:B])::Matrix{T})
    end
end

function _jacobian!(m::BayerBornLuetticke)

    # Information needed from m for set up
    θ = parameters2namedtuple(m)
    nt = construct_steadystate_namedtuple(m) # see helper_functions/steady_state/prepare_linearization.jl
    id = construct_prime_and_noprime_indices(m; only_aggregate = false)
    nm, nk, ny = get_idiosyncratic_dims(m)

    ############################################################################
    # Prepare elements used for uncompression
    ############################################################################
    # Matrices to take care of reduced degree of freedom in marginal distributions
    Γ  = shuffle_matrix(m[:distr_star])

    # Matrices for discrete cosine transforms
    DC = Vector{Array{Float64, 2}}(undef, 3)
    DC[1]  = mydctmx(nm)
    DC[2]  = mydctmx(nk)
    DC[3]  = mydctmx(ny)
    IDC    = [DC[1]', DC[2]', DC[3]'] # TODO: why do we need to take the transpose?

    DCD = Vector{Array{Float64, 2}}(undef, 3)
    DCD[1]  = mydctmx(nm-1)
    DCD[2]  = mydctmx(nk-1)
    DCD[3]  = mydctmx(ny-1)
    IDCD    = [DCD[1]', DCD[2]', DCD[3]']

    ############################################################################
    # Check whether steady state solves the difference equation
    # (left here in case the user ever wants to check)
    ############################################################################
    # X0 = zeros(get_setting(m, :n_model_states)) .+ ForwardDiff.Dual(0.0,tuple(zeros(5)...))
    # F  = Fsys(X0, X0, θ, m.grids, id, nt, m.equilibrium_conditions,
    #           get_setting(m, :dct_compression_indices), Γ, DC, IDC, DCD, IDCD)
    # if maximum(abs.(F)) / 10 > get_setting(m, :ϵ)
    #     @warn  "F = 0 is not at required precision"
    # end

    ############################################################################
    # Calculate Jacobians of the Difference equation F
    ############################################################################
    length_X0   = get_setting(m, :n_model_states)::Int
    n_dct_Vm    = length(get_setting(m, :dct_compression_indices)[:Vm]::Vector{Int})
    n_dct_Vk    = length(get_setting(m, :dct_compression_indices)[:Vk]::Vector{Int})
    n_marginals = length(id[:marginal_pdf_y_t]) + length(id[:marginal_pdf_m_t]) + length(id[:marginal_pdf_k_t])
    nxB         = length_X0 - n_dct_Vm - n_dct_Vk
    nxA         = length_X0 - n_marginals

    # The objective function omits the Vm, Vk in the X vector, but we left off at index id[:Vm_t][1] - 1,
    # so after we use zeros for the Vm, Vk parts of X, we need to start from id[:Vm_t][1]. We then go
    # to nxB since that is the total number of X elements we want to perturb.
    # For XPrime, we ignore the marginal perturbations and then have to start at nxB + 1 since
    # x is a vector of length nxB + nxA.
    obj_fnct    = x -> Fsys([x[1:id[:Vm_t][1]-1]; Zeros(n_dct_Vm + n_dct_Vk); x[id[:Vm_t][1]:nxB]],
                            [Zeros(n_marginals); x[nxB+1:end]],
                            θ, m.grids, id, nt, m.equilibrium_conditions,
                            get_setting(m, :dct_compression_indices), Γ, DC, IDC, DCD, IDCD)

    # TODO: make Fsys in place? Would it make sense to make Fsys in place in general, particularly w.r.t Fsys_agg?
    # Relatedly, could we use a sparsity pattern to do the autodiffing so we don't need to do a bunch of copying
    # for SGU_estim and can just directly differentiate into the Jacobian by using the correct sparsity matrix?
    BA          = ForwardDiff.jacobian(obj_fnct, zeros(nxB+nxA))

    n_vars = n_model_states(m)
    A      = zeros(n_vars, n_vars)
    B      = zeros(n_vars, n_vars)

    B[:,1:id[:Vm_t][1]-1]                 = BA[:,1:id[:Vm_t][1]-1]
    B[:,id[:Vk_t][end]+1:end]             = BA[:,id[:Vm_t][1]:nxB]
    A[:,id[:marginal_pdf_y_t][end]+1:end] = BA[:,nxB+1:end]

    # Make use of the fact that Vk/Vm has no influence on any variable in
    # the system, thus derivative is 1
    for (i, j) in zip(m.equilibrium_conditions[:eq_marginal_value_bonds], id[:Vm_t])
        B[i, j] = 1.0
    end
    for (i, j) in zip(m.equilibrium_conditions[:eq_marginal_value_capital], id[:Vk_t])
        B[i, j] = 1.0
    end

    # Make use of the fact that future distribution has no influence on any variable in
    # the system, thus derivative is Γ
    for (count, i) in enumerate(id[:marginal_pdf_m_t])
        A[id[:marginal_pdf_m_t],i] = -Γ[1][1:end-1,count]
    end
    for (count, i) in enumerate(id[:marginal_pdf_k_t])
        A[id[:marginal_pdf_k_t],i] = -Γ[2][1:end-1,count]
    end
    for (count, i) in enumerate(id[:marginal_pdf_y_t])
        A[id[:marginal_pdf_y_t],i] = -Γ[3][1:end-1,count]
    end

    # Store A and B Jacobians
    m[:A] = A
    m[:B] = B

    return A, B
end

function _update_aggregate_jacobian!(m::BayerBornLuetticke{T}, A::Matrix{T}, B::Matrix{T}) where {T <: Real}

    # Information needed from m for set up
    θ = parameters2namedtuple(m)
    nt = construct_steadystate_namedtuple(m) # see helper_functions/steady_state/prepare_linearization.jl
    id = construct_prime_and_noprime_indices(m; only_aggregate = true)

    eqconds          = m.equilibrium_conditions
    endo_states      = m.endogenous_states
    aggr_eqconds     = get_aggregate_equilibrium_conditions(m)
    aggr_endo_states = get_aggregate_endogenous_states(m)

    ############################################################################
    # Calculate derivatives of non-lineear difference equation
    ############################################################################

    length_X0   = length(aggr_eqconds) # num. aggregate variables = num. equilibrium conditions
    BA          = ForwardDiff.jacobian(x -> Fsys_agg(x[1:length_X0], x[length_X0+1:end], θ,
                                                     m.grids, id, nt, aggr_eqconds),
                                       zeros(2 * length_X0))
    Aa          = BA[:, length_X0+1:end] # aggregate A
    Ba          = BA[:, 1:length_X0]     # aggregate B

    for (aggr_endo_state_name, aggr_endo_state_i) in aggr_endo_states # endo states are the columns
        for (aggr_eqcond_name, aggr_eqcond_i) in aggr_eqconds # eqconds are the rows
            # So the following line populates A in column major order
            A[first(eqconds[aggr_eqcond_name]), first(endo_states[aggr_endo_state_name])] = Aa[aggr_eqcond_i, aggr_endo_state_i]
        end
    end

    return A, B
end
