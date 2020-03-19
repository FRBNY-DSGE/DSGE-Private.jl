function klein(m::AbstractModel)

    #################
    # Linearization:
    #################
    if get_setting(m, :autodiff)
        C = 1.0
        MargUtil = 1.0
        aaa = 1.0
        bbb = 1.0

        x = repeat(vcat(m[:μstar].value, # kf
                        m[:kstar].value, # k
                        (1+m[:r].value)*m[:π_star].value - 1, # i
                        m[:ystar].value, # y
                        m[:ωstar].value, # w
                        m[:xstar].value, # I
                        m[:bg].value,  #bg
                        1.0, # b
                        m[:g].value, # g
                        m[:γ].value, # z
                        1.0, # μ
                        aaa, # λ_w
                        bbb, # λ_f
                        0.0, #rm shock
                        m[:lstar].value, # ell
                        C, # C
                        1 + m[:r].value, # R
                        (1+m[:r])*m[:π_star].value - 1, # i
                        m[:Tstar].value, # t
                        m[:ωstar].value, # w
                        1.0, # L
                        m[:π_star].value, #π
                        m[:π_star].value*exp(m[:γ].value), # π_w
                        MargUtil, # marg utility
                        m[:ystar].value, #y
                        m[:xstar].value, #I
                        1.0, #marginal costs
                        1.0, # Q
                        m[:Rkstar].value, #cap return
                        m[:Tg].value), 2) #Tg

        Jac1 = Matrix{Float64}(jacobian_ad(m, x))
    else
        Jac1 = Matrix{Float64}(jacobian(m))
    end
    ##################################################################################
    # Klein Solution Method---apply generalized Schur decomposition a la Klein (2000)
    ##################################################################################

    # NK is number of predetermined variables
    NK = get_setting(m, :n_predetermined_variables)
    # n is number of variables (predet + non-predet)
	n = size(Jac1, 1)

    # A and B matrices
	A::Matrix{Float64} = Jac1[:, 1:n]
	B = -Jac1[:, n+1:2*n]

    # Apply generlaized Schur decomposition
    # A ≈ QZ[:Q]*QZ[:S]*QZ[:Z]'
    # B ≈ QZ[:Q]*QZ[:T]*QZ[:Z]'
	QZ = schur(A,B)

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

	U::Matrix{Float64} = QZ.Z'
	T::Matrix{Float64} = QZ.T
	S::Matrix{Float64} = QZ.S

    U11 = Matrix{Float64}(undef, NK, NK)
    U12 = Matrix{Float64}(undef, NK, NK-2)
    U21 = Matrix{Float64}(undef, NK-2, NK)
    U22 = Matrix{Float64}(undef, NK-2, NK-2)
    S11 = Matrix{Float64}(undef, NK, NK)
    T11 = Matrix{Float64}(undef, NK, NK)

    U11 = U[1:NK,1:NK]
	U12 = U[1:NK,NK+1:end]
	U21 = U[NK+1:end,1:NK]
	U22 = U[NK+1:end,NK+1:end]

	S11 = S[1:NK,1:NK]
	T11 = T[1:NK,1:NK]

    # Find minimum norm solution to U₂₁ + U₂₂*g_x = 0 (more numerically stable than -U₂₂⁻¹*U₂₁)
    gx_coef = Matrix{Float64}(undef,size(U22,2) , NK)#n-NK, NK)
	gx_coef = try
        -U22'*pinv(U22*U22')*U21
    catch ex
        if isa(ex, LinearAlgebra.LAPACKException)
            #@info "LAPACK exception thrown while computing pseudo inverse of U22*U22'"
            return gx_coef, Array{Float64, 2}(undef, NK, NK), -1
        else
            rethrow(ex)
        end
    end
    @show size(gx_coef)


    # Solve for h_x (in a more numerically stable way)
	S11invT11 = S11\T11;
	Ustuff = (U11 + U12*gx_coef);
	invterm = try
        pinv(eye(NK) + gx_coef' * gx_coef)
    catch ex
        if isa(ex, LinearAlgebra.LAPACKException)
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
    return gx_coef, hx_coef, 0
end

# Need an additional transition_equation function to properly stack the
# individual state and jump transition matrices/shock mapping matrices to
# a single state space for all of the model_states
function klein_transition_matrices(m::AbstractModel,
                                   TTT_state::Matrix{Float64}, TTT_jump::Matrix{Float64})
    TTT = zeros(n_model_states(m), n_model_states(m))

    # Loading mapping time t states to time t+1 states
    TTT[1:n_backward_looking_states(m), 1:n_backward_looking_states(m)] = TTT_state

    # Loading mapping time t jumps to time t+1 states
    TTT[1:n_backward_looking_states(m), n_backward_looking_states(m)+1:end] .= 0.

    # Loading mapping time t states to time t+1 jumps
    TTT[n_backward_looking_states(m)+1:end, 1:n_backward_looking_states(m)] = TTT_jump*TTT_state

    # Loading mapping time t jumps to time t+1 jumps
    TTT[n_backward_looking_states(m)+1:end, n_backward_looking_states(m)+1:end] .= 0.

    RRR = shock_loading(m, TTT_jump)

    return TTT, RRR
end
