function stateSpaceReduction(g0, g1, n_v, n_g, n_p, n_Z, reduceDist_hor)
    # Reduce Dimensionality of State Space
    #
    # REQUIRES: <block_arnoldi_func.m>
    #
    # INPUTS:
    #   g0 = LHS matrix, only used to check it satisfies require form
    #   g1 = Dynamics matrix
    #   n_v = number of jump variables
    #   n_g = number of state variables
    #   n_p = number of static constraints
    #   n_Z = number of exogenous variables
    #
    # OUTPUTS:
    #   state_red = tranformation to get from full grid to reduced states
    #   inv_state_red = inverse transform
    #   n_g = number of state variables after reduction
    #

    ## Check to make sure that LHS satisfies the required form

    loc = [1:n_v+n_g, n_v+n_g+n_p+1:n_v+n_g+n_p+n_Z]
#    if (maximum(abs.(g0[loc,loc] .- my_speye(n_v+n_g+n_Z)))≈0)
#        error("Make sure that g0 is normalized.")
#    end

    n_total = n_v + n_g

    ## Slice Dynamics Equation into Different Parts
    B_pv = g1[n_v+n_g+1:n_v+n_g+n_p,n_v+n_g+1:n_v+n_g+n_p]\g1[n_v+n_g+1:n_v+n_g+n_p,1:n_v]
    B_pZ = g1[n_v+n_g+1:n_v+n_g+n_p,n_v+n_g+1:n_v+n_g+n_p]\g1[n_v+n_g+1:n_v+n_g+n_p,n_v+n_g+n_p+1:n_v+n_g+n_p+n_Z]
    B_pg = g1[n_v+n_g+1:n_v+n_g+n_p,n_v+n_g+1:n_v+n_g+n_p]\g1[n_v+n_g+1:n_v+n_g+n_p,n_v+1:n_v+n_g]
    B_gg = g1[n_v+1:n_v+n_g,n_v+1:n_v+n_g]
    B_gp = g1[n_v+1:n_v+n_g,n_v+n_g+1:n_v+n_g+n_p]

    ## Orthonormalize B_pg as the direction to keep
    # Drop redundant Directions
    ~, d0, V_g = svd(B_pg)#,'econ')
    aux = diagm(0 => d0)
    n_Bpg = Int64(sum(aux .> 10 * eps() * aux[1]))
    V_g = V_g[:,1:n_Bpg]

    ## Arnoldi Iteration
    hor = reduceDist_hor
    A(x) = B_gg'*x - B_pg'*(B_gp'*x)
    # [V_g,~] = block_arnoldi_func(A,V_g,hor)
    V_g, ~ = deflated_block_arnoldi(A, V_g, hor)
    n_g = size(V_g,2)

    ## Build State-Space Reduction transform
    state_red = spzeros(Float64, n_v+n_g+n_Z,n_total+n_p+n_Z)
    state_red[1:n_v,1:n_v] = my_speye(n_v)
    state_red[n_v+1:n_v+n_g,n_v+1:n_total] = V_g'
    state_red[:,n_total+1:n_total+n_p] = 0
    state_red[n_v+n_g+1:n_v+n_g+n_Z,n_total+n_p+1:n_total+n_p+n_Z] = my_speye(n_Z)

    ## Build inverse transform
    inv_state_red = spzeros(Float64, n_total+n_p+n_Z,n_v+n_g+n_Z)
    inv_state_red[1:n_v,1:n_v] = my_speye(n_v)
    inv_state_red[n_v+1:n_total,n_v+1:n_g+n_v] = V_g
    inv_state_red[n_total+1:n_total+n_p,n_v+1:n_v+n_g] = -B_pg*V_g
    inv_state_red[n_total+1:n_total+n_p,1:n_v] = -B_pv
    inv_state_red[n_total+1:n_total+n_p,n_v+n_g+1:n_v+n_g+n_Z] = -B_pZ
    inv_state_red[n_total+n_p+1:n_total+n_p+n_Z,n_v+n_g+1:n_v+n_g+n_Z] = my_speye(n_Z)

    return state_red, inv_state_red, n_g
end

function deflated_block_arnoldi(A::Function, B::Matrix{Float64}, m::Int64)

    F = qr(B)
    Q = Matrix{Float64}(F.Q * Matrix{Float64}(LinearAlgebra.I, size(B)))
    basis = Matrix{Float64}(undef, size(Q,1), 0)
    realsmall = sqrt(eps())

    if m == 1
        basis = Q
    else
        for i in 1:m-1
            # Manual Gram-Schmidt
            basis = hcat(basis, Q)
            aux = A(Q)::Matrix{Float64}
            for j in 1:size(basis,2)
                aux -= basis[:,j] .* basis[:,j]' * aux
            end

            # Check for potential deflation
            Q = Matrix{Float64}(undef, size(Q,1), 0)
            for j in 1:size(aux,2)
                weight = sqrt(sum(aux[:,j].^2))
                if weight > realsmall
                    Q = hcat(Q, aux[:,j] / weight)
                    for k in j+1:size(aux,2)
                        aux[:,k] -= Q[:,end] * (Q[:,end]' * aux[:,k])
                    end
                end
            end

            # More Gram-Schmidt
            for j in 1:size(basis,2)
                Q -= basis[:,j] .* basis[:,j]' * Q
            end
            Q = Q ./ sqrt.(sum(Q.^2, dims=1))
        end
    end
    err = qr(A(Q) - basis * basis' * A(Q)).R::Matrix{Float64}
    return basis, Q, err
end
