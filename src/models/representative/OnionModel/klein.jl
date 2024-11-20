

@inline eye(x::Integer) = Matrix{Float64}(I,x,x)

function klein_BP(m::AbstractModel, A::AbstractMatrix, B::AbstractMatrix)

    #################
    # Linearization:
    #################
    #Jac1 = Matrix{Float64}(jacobian(m))
    ##################################################################################
    # Klein Solution Method---apply generalized Schur decomposition a la Klein (2000)
    ##################################################################################

    # NK is number of predetermined variables
    NK = get_setting(m, :n_back_states)
    n = get_setting(m, :n_model_states) #2 * get_setting(m, :n_sectors) + 6
    @show NK, n

    # n is number of variables (predet + non-predet)
	#n = size(Jac1, 1)

    # A and B matrices
	#A::Matrix{Float64} = Jac1[:, 1:n]
	#B = -Jac1[:, n+1:2*n]

    # Apply generlaized Schur decomposition
    # A ≈ QZ[:Q]*QZ[:S]*QZ[:Z]'
    # B ≈ QZ[:Q]*QZ[:T]*QZ[:Z]'
	QZ = schur(complex(A),complex(B))

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


	U = deepcopy(QZ.Z')
    Z = QZ.Z
	T = QZ.T
	S = QZ.S

    @show n, NK
#=
    U11 = Matrix{Float64}(undef, NK, NK)
    U12 = Matrix{Float64}(undef, NK, NK-2)
    U21 = Matrix{Float64}(undef, NK-2, NK)
    U22 = Matrix{Float64}(undef, NK-2, NK-2)
    S11 = Matrix{Float64}(undef, NK, NK)
    T11 = Matrix{Float64}(undef, NK, NK)
=#

    U11 = U[1:NK,1:NK]
	U12 = U[1:NK,NK+1:end]
	U21 = U[NK+1:end,1:NK]
	U22 = U[NK+1:end,NK+1:end]

	S11 = S[1:NK,1:NK]
	T11 = T[1:NK,1:NK]

    MDN_gx = real(Z[NK+1:end, 1:NK] * (Z[1:NK, 1:NK] \ eye(NK)))

    # Find minimum norm solution to U₂₁ + U₂₂*g_x = 0 (more numerically stable than -U₂₂⁻¹*U₂₁)
    gx_coef = Matrix{Float64}(undef, n-NK, NK)
	gx_coef = try
        -U22'*pinv(U22*U22')*U21
    catch ex
        if isa(ex, LinearAlgebra.LAPACKException)
            @warn "Lapack exception thrown"
            #@info "LAPACK exception thrown while computing pseudo inverse of U22*U22'"
            return gx_coef, Array{Float64, 2}(undef, NK, NK), -1
        else
            rethrow(ex)
        end
    end


    # Solve for h_x (in a more numerically stable way)
	S11invT11 = S11\T11;
	Ustuff = (U11 + U12*gx_coef);
	invterm = try
        pinv(eye(NK) + gx_coef' * gx_coef)
    catch ex
        if isa(ex, LinearAlgebra.LAPACKException)
            @warn "Lapack exception thrown"
            #@info "LAPACK exception thrown while computing pseudo inverse of eye(NK) + gx_coef'*gx_+coef"
            return gx_coef, Array{Float64, 2}(undef, NK, NK), -1
        else
            rethrow(ex)
        end
    end
    # hx_coef = Array{Float64, 2}(NK, NK)
	hx_coef = invterm*Ustuff'*S11invT11*Ustuff;

	# Ensure that hx and S11invT11 should have same eigenvalues
	# (eigst,valst) = eig(S11invT11);
	# (eighx,valhx) = eig(hx_coef);
	eigst = eigvals(S11invT11)
    eighx = eigvals(hx_coef)
    if abs(norm(eighx, Inf) - norm(eigst, Inf)) > 1e-4
		@warn "max abs eigenvalue of S11invT11 and hx are different!"
	end

	# next, want to represent policy functions in terms of meaningful things
	# gx_fval = Qy'*gx_coef*Qx
	# hx_fval = Qx'*hx_coef*Qx
    return gx_coef, hx_coef, U, Z, MDN_gx #, 0
end


@inline n_model_states(m::OnionModel) = get_setting(m, :n_model_states)
@inline n_backward_looking_states(m::OnionModel) = get_setting(m, :n_back_states)

# Need an additional transition_equation function to properly stack the
# individual state and jump transition matrices/shock mapping matrices to
# a single state space for all of the model_states
function klein_transition_matrices(m::OnionModel,
                                   TTT_state::Matrix{C}, TTT_jump::Matrix{C}) where C <: Float64
    TTT = zeros(n_model_states(m),n_model_states(m))

    # Loading mapping time t states to time t+1 states
    TTT[1:n_backward_looking_states(m), 1:n_backward_looking_states(m)] = TTT_state

    # Loading mapping time t jumps to time t+1 states
    TTT[1:n_backward_looking_states(m), n_backward_looking_states(m)+1:end] .= 0.

    # Loading mapping time t states to time t+1 jumps
    TTT[n_backward_looking_states(m)+1:end, 1:n_backward_looking_states(m)] = TTT_jump*TTT_state

    # Loading mapping time t jumps to time t+1 jumps
    TTT[n_backward_looking_states(m)+1:end, n_backward_looking_states(m)+1:end] .= 0.

    RRR = shock_loading_BP(m, TTT_jump)

    return TTT , RRR
end



function shock_loading_BP(m::OnionModel, TTT_jump::AbstractMatrix)
    exo = m.exogenous_shocks
    endo = m.endogenous_states


    _RRR = zeros(399, 1) #(n_backward_looking_states, n_shocks_exogenous)
    RRR = zeros(798, 1) #(n_model_states, n_shocks_exogenous)

    _RRR[endo[:τ], exo[:τ_sh]] = 1.

    RRR[1:399, :] = _RRR
    RRR[400:end, :] = TTT_jump * _RRR

    return RRR

end
