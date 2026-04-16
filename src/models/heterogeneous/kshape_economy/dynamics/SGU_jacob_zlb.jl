using LinearAlgebra

"""
    SGU_solver_zlb(param, grid, Jacob_base, F21_ad, F22_ad, F23_ad, F24_ad, F41_ad, F42_ad, F43_ad, F44_ad)

Build the ZLB Jacobian system matrices translated from `SGU_EST_ZLB_v2.m`.
The aggregate Jacobian update blocks (`F21_ad`...`F44_ad`) are expected to be already trimmed.

Returns `(F1, F2, F3, F4, param)`.
"""
function SGU_solver_zlb(param, grid, Jacob_base, F21_ad, F22_ad, F23_ad, F24_ad, F41_ad, F42_ad, F43_ad, F44_ad)
    getjb(x, k::Symbol) = x isa AbstractDict ? (haskey(x, k) ? x[k] : x[String(k)]) : getproperty(x, k)

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

    # Translation of SGU_EST_ZLB_v2.m with already-trimmed Jacobian update blocks.
    F1 = vcat(F1Xnext, F21_ad, F4Xnext, F5Xnext, F41_ad)
    F2 = vcat(F1Ynext, F22_ad, F4Ynext, F5Ynext, F42_ad)
    F3 = vcat(F1X, F23_ad, F4X, F5X, F43_ad)
    F4 = vcat(F1Y, F24_ad, F4Y, F5Y, F44_ad)

    return F1, F2, F3, F4, param
end
