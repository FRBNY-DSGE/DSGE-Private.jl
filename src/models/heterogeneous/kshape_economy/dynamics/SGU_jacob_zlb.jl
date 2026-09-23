using LinearAlgebra

"""
    SGU_solver(m.dicts[:param], m.dicts[:grid], Jacob_base, idx, F21_ad, F22_ad, F23_ad, F24_ad, F41_ad, F42_ad, F43_ad, F44_ad)

Jacobian-only SGU solve translated from `SGU_EST_ref_v2.m`.
The aggregate Jacobian update blocks (`F21_ad`...`F44_ad`) are expected to be already trimmed.
"""
function SGU_solver_zlb(m, F21_ad, F22_ad, F23_ad, F24_ad, F41_ad, F42_ad, F43_ad, F44_ad)
    numstates = Int(m.dicts[:grid]["numstates"])
    numcontrols = Int(m.dicts[:grid]["numcontrols"])

    #TODO: need change this, is currently matlab, future will be in model object
    getjb(x, k::Symbol) = x isa AbstractDict ? (haskey(x, k) ? x[k] : x[String(k)]) : getproperty(x, k)
    getparam(x, k::AbstractString, default) = x isa AbstractDict ? get(x, k, default) : default

    Jacob_base = m.dicts[:Jacob_base]
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

    #add padding to new jacobians
    padding_state = size(F1Xnext,2) - m.dicts[:grid]["os"]
    padding_control = size(F1Ynext,2) - m.dicts[:grid]["oc"]
    # Helper function to add columns of zeros to the left
    function pad_left(mat, ncols)
        ncols <= 0 && return mat
        hcat(zeros(eltype(mat), size(mat, 1), ncols), mat)
    end

    F21_ad = pad_left(F21_ad, padding_state)
    println("F21_ad shape: ", size(F21_ad))
    F23_ad = pad_left(F23_ad, padding_state)
    println("F23_ad shape: ", size(F23_ad))
    F41_ad = pad_left(F41_ad, padding_state)
    println("F41_ad shape: ", size(F41_ad))
    F43_ad = pad_left(F43_ad, padding_state)
    println("F43_ad shape: ", size(F43_ad))

    F22_ad = pad_left(F22_ad, padding_control)
    println("F22_ad shape: ", size(F22_ad))
    F24_ad = pad_left(F24_ad, padding_control)
    println("F24_ad shape: ", size(F24_ad))
    F42_ad = pad_left(F42_ad, padding_control)
    println("F42_ad shape: ", size(F42_ad))
    F44_ad = pad_left(F44_ad, padding_control)
    println("F44_ad shape: ", size(F44_ad))

    # Translation of SGU_EST_ref_v2.m with already-trimmed Jacobian update blocks.
    #TODO: CHECK THIS INDEXING
    F1 = vcat(F1Xnext, F21_ad, F4Xnext, F5Xnext, F41_ad[end-m.dicts[:grid]["oc_agg"]+1:end,:])
    println("F1 shape: ", size(F1))
    F2 = vcat(F1Ynext, F22_ad, F4Ynext, F5Ynext, F42_ad[end-m.dicts[:grid]["oc_agg"]+1:end,:])
    println("F2 shape: ", size(F2))
    F3 = vcat(F1X, F23_ad, F4X, F5X, F43_ad[end-m.dicts[:grid]["oc_agg"]+1:end,:])
    println("F3 shape: ", size(F3))
    F4 = vcat(F1Y, F24_ad, F4Y, F5Y, F44_ad[end-m.dicts[:grid]["oc_agg"]+1:end,:])
    println("F4 shape: ", size(F4))

    # QZ decomposition: [F1, F2] * E[x', u'] = -[F3, F4] * [x, u]
    A = [F1 F2]
    println("A shape: ", size(A))
    B = -[F3 F4]
    println("B shape: ", size(B))

    return F1, F2, F3, F4
end
