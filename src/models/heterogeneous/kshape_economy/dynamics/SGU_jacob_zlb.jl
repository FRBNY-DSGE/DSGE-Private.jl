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

    function untrim_left(mat::AbstractMatrix, full_cols::Int)
        nrows, ncols = size(mat)
        ncols > full_cols && error("Trimmed block has more columns than target width.")
        ncols == full_cols && return mat
        return hcat(zeros(eltype(mat), nrows, full_cols - ncols), mat)
    end

    F21_full = untrim_left(F21_ad, size(F1Xnext, 2))
    F23_full = untrim_left(F23_ad, size(F1X, 2))
    F41_full = untrim_left(F41_ad, size(F4Xnext, 2))
    F43_full = untrim_left(F43_ad, size(F4X, 2))
    F22_full = untrim_left(F22_ad, size(F1Ynext, 2))
    F24_full = untrim_left(F24_ad, size(F1Y, 2))
    F42_full = untrim_left(F42_ad, size(F4Ynext, 2))
    F44_full = untrim_left(F44_ad, size(F4Y, 2))

    # Translation of SGU_EST_ZLB_v2.m with reconstructed full-width update blocks.
    F1 = vcat(F1Xnext, F21_full, F4Xnext, F5Xnext, F41_full)
    F2 = vcat(F1Ynext, F22_full, F4Ynext, F5Ynext, F42_full)
    F3 = vcat(F1X, F23_full, F4X, F5X, F43_full)
    F4 = vcat(F1Y, F24_full, F4Y, F5Y, F44_full)

    return F1, F2, F3, F4, param
end
