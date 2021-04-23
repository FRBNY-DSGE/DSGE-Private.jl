@doc raw"""
    SGU(XSS,A,B,m_par,n_par,indexes,Copula,compressionIndexes,distrSS;estim)

Calculate the linearized solution to the non-linear difference equations defined
by function [`Fsys()`](@ref), using Schmitt-Grohé & Uribe (JEDC 2004) style linearization
(apply the implicit function theorem to obtain linear observation and
state transition equations).

The Jacobian is calculated using the package `ForwardDiff`

# Arguments
- `XSS`: steady state around which the system is linearized
- `A`,`B`: matrices to be filled with first derivatives (see `Returns`)
- `m_par::ModelParameters`, `n_par::NumericalParameters`: `n_par.sol_algo` determines
    the solution algorithm
- `Copula::Function`,`distrSS::Array{Float64,3}`: `Copula` maps marginals to
    linearized approximation of joint distribution around `distrSS`
- `indexes::IndexStruct`,`compressionIndexes`: access states and controls by name
    (DCT coefficients of compressed ``V_m`` and ``V_k`` in case of
    `compressionIndexes`)

# Returns
- `gx`,`hx`: observation equations [`gx`] and state transition equations [`hx`]
- `alarm_sgu`,`nk`: `alarm_sgu=true` when solving algorithm fails, `nk` number of
    predetermined variables
- `A`,`B`: first derivatives of [`Fsys()`](@ref) with respect to arguments `X` [`B`] and
    `XPrime` [`A`]
"""
function SGU(m::BayerBornLuetticke, A::Array, B::Array; estim::Bool = false)

    # TODO: maybe also just pass θ, nt, id to SGU rather than do these steps every time
    θ = parameters2namedtuple(m)
    nt = DSGE.construct_steadystate_namedtuple(m)
    id = DSGE.construct_prime_and_noprime_indices(m; only_aggregate = false)
    nm, nk, ny = DSGE.get_idiosyncratic_dims(m)

    ############################################################################
    # Prepare elements used for uncompression
    ############################################################################
    # Matrices to take care of reduced degree of freedom in marginal distributions
    Γ  = DSGE.shuffleMatrix(m[:distr_star])

    # Matrices for discrete cosine transforms
    DC = Array{Array{Float64,2},1}(undef,3)
    DC[1]  = DSGE.mydctmx(nm)
    DC[2]  = DSGE.mydctmx(nk)
    DC[3]  = DSGE.mydctmx(ny)
    IDC    = [DC[1]', DC[2]', DC[3]']

    DCD = Array{Array{Float64,2},1}(undef,3)
    DCD[1]  = DSGE.mydctmx(nm-1)
    DCD[2]  = DSGE.mydctmx(nk-1)
    DCD[3]  = DSGE.mydctmx(ny-1)
    IDCD    = [DCD[1]', DCD[2]', DCD[3]']

    ############################################################################
    # Check whether Steady state solves the difference equation
    ############################################################################
    # length_X0 = indexes.profits # Convention is that profits is the last control
    length_X0 = get_setting(m, :n_vars)
    # X0 = zeros(length_X0) .+ ForwardDiff.Dual(0.0,tuple(zeros(5)...))
    # F  = Fsys(X0,X0,XSS,m_par,n_par,indexes,Γ,compressionIndexes,DC, IDC, DCD, IDCD)
    # if maximum(abs.(F))/10>n_par.ϵ
    #     @warn  "F=0 is not at required precision"
    # end

    ############################################################################
    # Calculate Jacobians of the Difference equation F
    ############################################################################

    n_dct_Vm    = length(get_setting(m, :dct_compression_indices)[:Vm])
    n_dct_Vk    = length(get_setting(m, :dct_compression_indices)[:Vk])
    nxB         = length_X0 - n_dct_Vm - n_dct_Vk
    nxA         = length_X0 - length(id[:marginal_pdf_y_t]) - length(id[:marginal_pdf_m_t]) - length(id[:marginal_pdf_k_t])

    # The objective function omits the Vm, Vk in the X vector, but we left off at index id[:Vm_t][1] - 1,
    # so after we use zeros for the Vm, Vk parts of X, we need to start from id[:Vm_t][1]. We then go
    # to nxB since that is the total number of X elements we want to perturb.
    # For XPrime, we ignore the marginal perturbations and then have to start at nxB + 1 since
    # x is a vector of length nxB + nxA.
    obj_fnct    = x -> Fsys([x[1:id[:Vm_t][1]-1]; zeros(n_dct_Vm + n_dct_Vk); x[id[:Vm_t][1]:nxB]],
                            [zeros(length(id[:marginal_pdf_y_t]) + length(id[:marginal_pdf_m_t]) +
                                   length(id[:marginal_pdf_k_t])); x[nxB+1:end]],
                            θ, m.grids, id, nt, m.equilibrium_conditions,
                            get_setting(m, :dct_compression_indices), Γ, DC, IDC, DCD, IDCD)

    # TODO: make Fsys in place? Would it make sense to make Fsys in place in general, particularly w.r.t Fsys_agg?
    # Relatedly, could we use a sparsity pattern to do the autodiffing so we don't need to do a bunch of copying
    # for SGU_estim and can just directly differentiate into the Jacobian by using the correct sparsity matrix?
    BA          = ForwardDiff.jacobian(obj_fnct, zeros(nxB+nxA))

    B[:,1:id[:Vm_t][1]-1]                 = BA[:,1:id[:Vm_t][1]-1]
    B[:,id[:Vk_t][end]+1:end]             = BA[:,id[:Vm_t][1]:nxB]
    A[:,id[:marginal_pdf_y_t][end]+1:end] = BA[:,nxB+1:end]

    # Make use of the fact that Vk/Vm has no influence on any variable in
    # the system, thus derivative is 1
    for (i, j) in zip(m.equilibrium_conditions[:eq_marginal_value_bonds], id[:Vm_t])
        B[i, j] = 1.0
    end
    for i in zip(m.equilibrium_conditions[:eq_marginal_value_capital], id[:Vk_t])
        B[i, j] = 1.0
    end

    # Make use of the fact that future distribution has no influence on any variable in
    # the system, thus derivative is Γ
    for count = 1:nm-1 #in eachcol(A)
        i = id[:marginal_pdf_m_t][count]
        A[id[:marginal_pdf_m_t],i] = -Γ[1][1:end-1,count]
    end
    for count = 1:nk-1 #in eachcol(A)
        i = id[:marginal_pdf_k_t][count]
        A[id[:marginal_pdf_k_t],i] = -Γ[2][1:end-1,count]
    end
    for count = 1:ny-1 #in eachcol(A)
        i = id[:marginal_pdf_y_t][count]
        A[id[:marginal_pdf_y_t],i] = -Γ[3][1:end-1,count]
    end

    ############################################################################
    # Solve the linearized model: Policy Functions and LOMs
    ############################################################################
    gx, hx, alarm_sgu, nk = SolveDiffEq(m, A, B, estim)

    return gx, hx, alarm_sgu, nk, A, B
end
