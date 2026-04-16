using LinearAlgebra

"""
    SGU_solver(F1, F2, F3, F4, grid; overrideEigen=false)

Solve the Schmitt-Grohe-Uribe linear system from precomputed Jacobians.

This is the Jacobian-only version of `SGU_solver`: it skips numerical differentiation
and starts from the four Jacobian blocks

- `F1 = dF/dX'`
- `F2 = dF/dY'`
- `F3 = dF/dX`
- `F4 = dF/dY`

It then performs the generalized Schur (QZ) step, selects the stable generalized
 eigenvalues, reorders the decomposition, and returns the linear policy rules

- `hx`: law of motion for states
- `gx`: policy rule for controls

# Arguments
- `F1`, `F2`, `F3`, `F4`: Jacobian blocks from the SGU system.
- `grid`: dictionary containing at least `"numstates"`.
- `overrideEigen`: if `true`, shift the stability threshold exactly as in the main solver
  when the stable eigenvalue count does not equal `numstates`.

# Returns
Tuple `(hx, gx, F1, F2, F3, F4)`.
"""
function SGU_solver(F1, F2, F3, F4, grid; overrideEigen=false)
    numstates = Int(grid["numstates"])

    # QZ decomposition: [F1, F2] * E[x', u'] = -[F3, F4] * [x, u]
    A = [F1 F2]
    B = -[F3 F4]
    schur_result = schur(A, B)
    s = schur_result.S
    t = schur_result.T
    #Main.xx[][:schur_result] = schur_result
    #Main.xx[][:S] = s
    #Main.xx[][:T] = t
    # Use alpha/beta from the generalized Schur decomposition rather than diag(s)./diag(t)
    # so complex conjugate pairs are handled correctly.
    relev = abs.(schur_result.alpha) ./ abs.(schur_result.beta)
    #Main.xx[][:relev] = relev
    #@assert false
    ll = sort(relev)
    slt = relev .>= 1.0
    nk = sum(slt)

    if nk > numstates
        if overrideEigen
            @warn "The Equilibrium is Locally Indeterminate, critical eigenvalue shifted to: $(ll[end - numstates])"
            slt = relev .> ll[end - numstates]
            nk = sum(slt)
        else
            error("No Local Equilibrium Exists, last eigenvalue: $(ll[end - numstates])")
        end
    elseif nk < numstates
        if overrideEigen
            threshold = ll[end - numstates]
            @warn "No Local Equilibrium Exists, critical eigenvalue shifted to: $threshold"
            slt = relev .> threshold
            nk = sum(slt)
        else
            error("No Local Equilibrium Exists, last eigenvalue: $(ll[end - numstates])")
        end
    end

    # Reorder the generalized Schur form so selected eigenvalues move to the top-left block.
    S = copy(Matrix(schur_result.S))
    T = copy(Matrix(schur_result.T))
    Qmat = copy(Matrix(schur_result.left))
    Zmat = copy(Matrix(schur_result.right))
    selectInt = LinearAlgebra.LAPACK.BlasInt.(slt)

    @info "SGU_solver: Before tgsen: rank(Zmat)=$(rank(Zmat)), sum(selectInt)=$(sum(selectInt))"
    LinearAlgebra.LAPACK.tgsen!(selectInt, S, T, Qmat, Zmat)
    @info "SGU_solver: After tgsen: rank(Zmat)=$(rank(Zmat))"

    z11 = Zmat[1:nk, 1:nk]
    z21 = Zmat[(nk + 1):end, 1:nk]
    s11 = S[1:nk, 1:nk]
    t11 = T[1:nk, 1:nk]

    z11_rank = rank(z11)
    @info "SGU_solver: z11 size=$(size(z11)), rank=$(z11_rank), nk=$nk"
    if z11_rank < nk
        @warn "invertibility condition violated: rank(z11)=$(z11_rank) < nk=$nk"
    end

    z11i = z11 \ Matrix{Float64}(I, nk, nk)
    gx = real(z21 * z11i)
    hx = real(z11 * (s11 \ t11) * z11i)

    @info "SGU_solver: done."
    return hx, gx, F1, F2, F3, F4
end
