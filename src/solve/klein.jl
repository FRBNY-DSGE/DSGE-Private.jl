function klein(m::AbstractModel{T}; minimum_inversion_tol::Float64 = 1e-4, verbose::Symbol = :none) where {T <: Real}

    #################
    # Linearization
    #################
    jac_out = jacobian(m)

    # Get A and B matrices for Klein
    if isa(jac_out, Tuple)
        A::Matrix{T}, B::Matrix{T} = jac_out # Need to get dense matrices out for schur

        # n is total number of variables (predet + jump)
	    n = size(A, 1)
    else
        Jac1         = jacobian(m)
	    A::Matrix{T} = Jac1[:, 1:n]
	    B::Matrix{T} = -Jac1[:, n+1:2*n]
    end

    # NK is number of predetermined variables
    # NK = get_setting(m, :n_predetermined_variables)
    NK = n_backward_looking_states(m)::Int

    ##################################################################################
    # Klein Solution Method---apply generalized Schur decomposition a la Klein (2000)
    ##################################################################################

    # Apply generlaized Schur decomposition
    # A ≈ QZ[:Q]*QZ[:S]*QZ[:Z]'
    # B ≈ QZ[:Q]*QZ[:T]*QZ[:Z]'
	QZ = schur(A, B)

    # Reorder so that stable comes first
    alpha::Vector{complex(promote_type(eltype(A), eltype(B)))} = QZ.α
    beta::Vector{complex(promote_type(eltype(A), eltype(B)))} = QZ.β
	eigs = QZ.β ./ QZ.α #real(QZ.β ./ QZ.α)
	eigselect::AbstractArray{Bool} = abs.(eigs) .< 1 # returns false for NaN gen. eigenvalue which is correct here bc they are > 1
	ordschur!(QZ, eigselect)

    # Check that number of stable eigenvalues equals the number of predetermined state variables
	nk = sum(eigselect)
	if nk>NK
	    @warn "Equilibrium is locally indeterminate"
	elseif nk<NK
	    @warn "No local equilibrium exists"
	end

    inv_method = haskey(get_settings(m), :klein_inversion_method) ?
        get_setting(m, :klein_inversion_method) : :minimum_norm

    if inv_method == :minimum_norm
        gx_coef, hx_coef = klein_minimum_norm_inversion(QZ, n, NK; tol = minimum_inversion_tol)
    elseif
        gx_coef, hx_coef = klein_direct_inversion(QZ, n, NK)
    else
        throw(ArgumentError("Inversion method $(inv_method) is not recognized. Available ones are [:minimum_norm, :direct]"))
    end

	# next, want to represent policy functions in terms of meaningful things
	# gx_fval = Qy'*gx_coef*Qx
	# hx_fval = Qx'*hx_coef*Qx
    return gx_coef, hx_coef, 0 # TODO: return an actual eu variable with info, like in gensys
end

# Need an additional transition_equation function to properly stack the
# individual state and jump transition matrices/shock mapping matrices to
# a single state space for all of the model_states
function klein_transition_matrices(m::AbstractModel{T}, TTT_state::Matrix{T}, TTT_jump::Matrix{T}) where {T <: Real}

    TTT = Matrix{T}(undef, n_model_states(m), n_model_states(m))

    n_states = n_backward_looking_states(m)::Int

    # Loading mapping time t states to time t+1 states
    TTT[1:n_states, 1:n_states] = TTT_state

    # Loading mapping time t jumps to time t+1 states
    TTT[1:n_states, n_states+1:end] .= 0.

    # Loading mapping time t states to time t+1 jumps
    TTT[n_states+1:end, 1:n_states] = TTT_jump * TTT_state

    # Loading mapping time t jumps to time t+1 jumps
    TTT[n_states+1:end, n_states+1:end] .= 0.

    RRR = shock_loading(m, TTT_jump)

    return TTT, RRR
end

function klein_minimum_norm_inversion(QZ::GeneralizedSchur, n::Int, NK::Int; tol::Float64 = 1e-4)
	U::Matrix{Float64} = QZ.Z'
	T::Matrix{Float64} = QZ.T
	S::Matrix{Float64} = QZ.S

#=    U11 = Matrix{Float64}(undef, NK, NK)
    U12 = Matrix{Float64}(undef, NK, NK-2)
    U21 = Matrix{Float64}(undef, NK-2, NK)
    U22 = Matrix{Float64}(undef, NK-2, NK-2)
    S11 = Matrix{Float64}(undef, NK, NK)
    T11 = Matrix{Float64}(undef, NK, NK)=#

    U_sz = size(U)
    U11 = view(U, 1:NK,1:NK)
	U12 = view(U, 1:NK, NK+1:U_sz[2])
	U21 = view(U, NK+1:U_sz[1], 1:NK)
	U22 = view(U, NK+1:U_sz[1], NK+1:U_sz[2])

	S11 = view(S, 1:NK, 1:NK)
	T11 = view(T, 1:NK, 1:NK)

    # Find minimum norm solution to U₂₁ + U₂₂*g_x = 0 (more numerically stable than -U₂₂⁻¹*U₂₁)
    gx_coef = Matrix{Float64}(undef, n-NK, NK)
	gx_coef = try
        -U22' * pinv(U22 * U22') * U21
    catch ex
        if isa(ex, LinearAlgebra.LAPACKException)
            # @info "LAPACK exception thrown while computing pseudo inverse of U22*U22'"
            return gx_coef, Array{Float64, 2}(undef, NK, NK), -1
        else
            rethrow(ex)
        end
    end

    # Solve for h_x (in a more numerically stable way)
	S11invT11 = S11 \ T11
	Ustuff = U11 + U12 * gx_coef
	invterm = try
        # pinv(eye(NK) + gx_coef' * gx_coef)
        pinv(I + gx_coef' * gx_coef)
    catch ex
        if isa(ex, LinearAlgebra.LAPACKException)
            # @info "LAPACK exception thrown while computing pseudo inverse of eye(NK) + gx_coef'*gx_+coef"
            return gx_coef, Array{Float64, 2}(undef, NK, NK), -1
        else
            rethrow(ex)
        end
    end
    # hx_coef = Array{Float64, 2}(NK, NK)
	hx_coef = invterm * Ustuff' * S11invT11 * Ustuff

	# Ensure that hx and S11invT11 should have same eigenvalues
	# (eigst,valst) = eig(S11invT11);
	# (eighx,valhx) = eig(hx_coef);
	eigst = eigvals(S11invT11)
    eighx = eigvals(hx_coef)
    if abs(norm(eighx, Inf) - norm(eigst, Inf)) > tol
		@warn "max abs eigenvalue of S11invT11 and hx are different!"
	end

    gx_coef, hx_coef
end

# Direct inversion method copied from SolveDiffEq in https://github.com/BenjaminBorn/HANK_BusinessCycleAndInequality
function klein_direct_inversion(Schur_decomp::GeneralizedSchur, n::Int, nk::Int)
    z21 = view(Schur_decomp.Z, (nk+1):n, 1:nk)
    z11 = view(Schur_decomp.Z, 1:nk, 1:nk)
    s11 = view(Schur_decomp.S, 1:nk, 1:nk)
    t11 = view(Schur_decomp.T, 1:nk, 1:nk)

    if rank(z11) < nk # rank(z11) = 211, nk = 212 => somewhere there's something just slightly off
        @warn "invertibility condition violated"
        hx = Array{Float64}(undef, nk, nk) # change: original code use n_states, but nk = n_states when saddle-path stability satisfied
        gx = Array{Float64}(undef, n-nk, n-nk) # change: original code uses n_jumps, but n_jumps = n-nk when saddle-path stability satisfied
        # alarm_sgu = true # not used by dSGE.jl
        return gx, hx
    end
    z11i = z11 \ I # I is the identity matrix -> doesn't allocate an array!
    gx = real(z21 * z11i)
    hx = real(z11 * (s11 \ t11) * z11i)

    return gx, hx
end
