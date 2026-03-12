# ------------------------------------------------------------
# 1D cubic spline with not-a-knot BCs and linear extrapolation
# Matches MATLAB's griddedInterpolant(..., 'spline', 'linear')
# ------------------------------------------------------------

"""
    _solve_spline_coeffs(x, y) -> Vector

Solve for second derivatives of a not-a-knot cubic spline.
"""
function _solve_spline_coeffs(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:Real}
    n = length(x)

    h = Vector{T}(undef, n-1)
    @inbounds for i in 1:n-1
        h[i] = x[i+1] - x[i]
    end

    delta = Vector{T}(undef, n-1)
    @inbounds for i in 1:n-1
        delta[i] = (y[i+1] - y[i]) / h[i]
    end

    # Build n×n system for second derivatives with not-a-knot BCs
    A = zeros(T, n, n)
    rr = zeros(T, n)

    # Not-a-knot left: h[2]*d[1] - (h[1]+h[2])*d[2] + h[1]*d[3] = 0
    @inbounds begin
        A[1, 1] = h[2]
        A[1, 2] = -(h[1] + h[2])
        A[1, 3] = h[1]

        for i in 2:n-1
            A[i, i-1] = h[i-1]
            A[i, i]   = 2.0 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            rr[i] = 6.0 * (delta[i] - delta[i-1])
        end

        # Not-a-knot right
        A[n, n-2] = h[n-1]
        A[n, n-1] = -(h[n-2] + h[n-1])
        A[n, n]   = h[n-2]
    end

    return A \ rr
end

"""
    _eval_spline1d(x, y, d, xq) -> T

Evaluate cubic spline at a single query point xq, with linear extrapolation.
"""
@inline function _eval_spline1d(x::AbstractVector{T}, y::AbstractVector{T},
                                 d::AbstractVector{T}, xq::T) where {T<:Real}
    n = length(x)
    @inbounds begin
        if xq <= x[1]
            # Linear extrapolation using spline derivative at left boundary
            h1 = x[2] - x[1]
            slope = (y[2] - y[1]) / h1 - h1 / 6.0 * (d[2] + 2.0 * d[1])
            return y[1] + (xq - x[1]) * slope
        elseif xq >= x[n]
            # Linear extrapolation using spline derivative at right boundary
            hn = x[n] - x[n-1]
            slope = (y[n] - y[n-1]) / hn + hn / 6.0 * (2.0 * d[n] + d[n-1])
            return y[n] + (xq - x[n]) * slope
        else
            i = searchsortedlast(x, xq)
            i = max(1, min(i, n-1))
            hi = x[i+1] - x[i]
            dx1 = x[i+1] - xq
            dx0 = xq - x[i]
            return (d[i] * dx1^3 + d[i+1] * dx0^3) / (6.0 * hi) +
                   (y[i] / hi - d[i] * hi / 6.0) * dx1 +
                   (y[i+1] / hi - d[i+1] * hi / 6.0) * dx0
        end
    end
end

"""
    _interp1d_spline_scalar(x, y, xq) -> T

Solve for spline coefficients and evaluate at a single point.
"""
@inline function _interp1d_spline_scalar(x::AbstractVector{T}, y::AbstractVector{T}, xq::T) where {T<:Real}
    d = _solve_spline_coeffs(x, y)
    return _eval_spline1d(x, y, d, xq)
end

# ------------------------------------------------------------
# 3D tensor-product cubic spline interpolant
# Evaluates dim3 first (using precomputed coefficients on original
# data) to ensure extrapolation matches MATLAB's tensor-product
# B-spline behavior.
# ------------------------------------------------------------

struct Interp3NonuniformCubic{T}
    x1::Vector{T}
    x2::Vector{T}
    x3::Vector{T}
    V::Array{T,3}
    # Precomputed second derivatives along each dimension
    d1::Array{T,3}  # along dim1 for each (j,k)
    d3::Array{T,3}  # along dim3 for each (i,j)
    buf1::Vector{T}
    buf2::Vector{T}
end

function Interp3NonuniformCubic(x1::AbstractVector{T},
                                x2::AbstractVector{T},
                                x3::AbstractVector{T},
                                V::AbstractArray{T,3}) where {T<:Real}

    n1, n2, n3 = size(V)
    @assert n1 == length(x1)
    @assert n2 == length(x2)
    @assert n3 == length(x3)

    x1v = collect(x1)
    x2v = collect(x2)
    x3v = collect(x3)
    Vv = Array(V)

    # Precompute spline coefficients along dim1 for each (j,k) slice
    d1 = Array{T,3}(undef, n1, n2, n3)
    for k in 1:n3
        for j in 1:n2
            d1[:, j, k] = _solve_spline_coeffs(x1v, Vv[:, j, k])
        end
    end

    # Precompute spline coefficients along dim3 for each (i,j) slice
    d3 = Array{T,3}(undef, n1, n2, n3)
    for j in 1:n2
        for i in 1:n1
            d3[i, j, :] = _solve_spline_coeffs(x3v, Vv[i, j, :])
        end
    end

    return Interp3NonuniformCubic(
        x1v, x2v, x3v, Vv, d1, d3,
        Vector{T}(undef, n1),
        Vector{T}(undef, n2)
    )
end

# Evaluate at a single point (xq1, xq2, xq3)
function (F::Interp3NonuniformCubic{T})(xq1::T, xq2::T, xq3::T) where {T<:Real}

    nb, na, ns = size(F.V)

    # Step 1: Interpolate along dim3 (z) for all (i,j) using precomputed d3
    #         This ensures extrapolation uses coefficients from the original data
    @inbounds for j in 1:na
        for i in 1:nb
            F.buf1[i] = _eval_spline1d(F.x3,
                                         view(F.V, i, j, :),
                                         view(F.d3, i, j, :),
                                         xq3)
        end
        # Step 2: Interpolate along dim1 (x) using fresh spline on dim3-interpolated data
        F.buf2[j] = _interp1d_spline_scalar(F.x1, F.buf1, xq1)
    end

    # Step 3: Interpolate along dim2 (y)
    return _interp1d_spline_scalar(F.x2, F.buf2, xq2)
end

# dispatch for mixed Real types
function (F::Interp3NonuniformCubic)(xq1::Real, xq2::Real, xq3::Real)
    T = eltype(F.V)
    return F(T(xq1), T(xq2), T(xq3))
end

# dispatch for tuple input
(F::Interp3NonuniformCubic)(xq::NTuple{3,Real}) = F(xq[1], xq[2], xq[3])
