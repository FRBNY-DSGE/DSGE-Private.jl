using LinearAlgebra

"""
    SGU_solver(param, grid, Jacob_base, idx, F21_ad, F22_ad, F23_ad, F24_ad, F41_ad, F42_ad, F43_ad, F44_ad)

Jacobian-only SGU solve translated from `SGU_EST_ref_v2.m`.
The aggregate Jacobian update blocks (`F21_ad`...`F44_ad`) are expected to be already trimmed.
"""
function SGU_solver(param,grid,Jacob_base,F21_ad, F22_ad, F23_ad, F24_ad, F41_ad, F42_ad, F43_ad, F44_ad)
    numstates = Int(grid["numstates"])
    numcontrols = Int(grid["numcontrols"])

    #TODO: need change this, is currently matlab, future will be in model object
    getjb(x, k::Symbol) = x isa AbstractDict ? (haskey(x, k) ? x[k] : x[String(k)]) : getproperty(x, k)
    getparam(x, k::AbstractString, default) = x isa AbstractDict ? get(x, k, default) : default

    F1Xnext = getjb(Jacob_base, :F1Xnext)
    F1Ynext = getjb(Jacob_base, :F1Ynext)
    F1X = getjb(Jacob_base, :F1X)
    F1Y = getjb(Jacob_base, :F1Y)
    F4Xnext = getjb(Jacob_base, :F4Xnext)
    F4Ynext = getjb(Jacob_base, :F4Ynext)
    F4X = getjb(Jacob_base, :F4X)
    F4Y = getjb(Jacob_base, :F4Y)
    F5Xnext = getjb(Jacob_base, :F5Xnext)
    F5Ynext = getjb(Jacob_base, :F5Ynext)
    F5X = getjb(Jacob_base, :F5X)
    F5Y = getjb(Jacob_base, :F5Y)

    # Translation of SGU_EST_ref_v2.m with already-trimmed Jacobian update blocks.
    #TODO: CHECK THIS INDEXING
    F1 = vcat(F1Xnext, F21_ad, F4Xnext, F5Xnext, F41_ad)
    F2 = vcat(F1Ynext, F22_ad, F4Ynext, F5Ynext, F42_ad)
    F3 = vcat(F1X, F23_ad, F4X, F5X, F43_ad)
    F4 = vcat(F1Y, F24_ad, F4Y, F5Y, F44_ad)

    # QZ decomposition: [F1, F2] * E[x', u'] = -[F3, F4] * [x, u]
    A = [F1 F2]
    B = -[F3 F4]
    schur_result = schur(A, B)
    relev = abs.(schur_result.alpha) ./ abs.(schur_result.beta)
    ll = sort(relev)
    slt = relev .>= 1.0
    nk = sum(slt)
    indicator = 1
    overrideEigen = getparam(param, "overrideEigen", false)

    if nk > numstates
        if overrideEigen
            @warn "The Equilibrium is Locally Indeterminate, critical eigenvalue shifted to: $(ll[end - numstates])"
            slt = relev .> ll[end - numstates]
            nk = sum(slt)
        else
            error("No Local Equilibrium Exists, last eigenvalue: $(ll[end - numstates])")
        end
        indicator = 0
    elseif nk < numstates
        if overrideEigen
            threshold = ll[end - numstates]
            @warn "No Local Equilibrium Exists, critical eigenvalue shifted to: $threshold"
            slt = relev .> threshold
            nk = sum(slt)
        else
            error("No Local Equilibrium Exists, last eigenvalue: $(ll[end - numstates])")
        end
        indicator = 0
    end

    reordered = ordschur(schur_result, slt)
    S = Matrix(reordered.S)
    T = Matrix(reordered.T)
    Zmat = Matrix(reordered.right)

    z11 = Zmat[1:nk, 1:nk]
    z21 = Zmat[(nk + 1):end, 1:nk]
    s11 = S[1:nk, 1:nk]
    t11 = T[1:nk, 1:nk]

    if indicator > 0 && rank(z11) < nk
        @warn "invertibility condition violated"
        indicator = 0
    end

    if indicator > 0
        z11i = z11 \ Matrix{Float64}(I, nk, nk)
        gx = real(z21 * z11i)
        hx = real(z11 * (s11 \ t11) * z11i)
    else
        gx = zeros(numcontrols, numstates)
        hx = zeros(numstates, numstates)
    end

    return hx, gx, F1, F2, F3, F4, param, indicator
end
