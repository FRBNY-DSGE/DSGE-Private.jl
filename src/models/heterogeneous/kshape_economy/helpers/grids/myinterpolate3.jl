@inline function locate(x::Real, grid::AbstractVector{T}) where {T<:Real}
    # returns i in 1:(n-1) with grid[i] <= x < grid[i+1]
    n = length(grid)
    i = searchsortedlast(grid, x)
    return clamp(i, 1, n-1)
end


using Interpolations

"""
    myinterpolate3(xg, yg, zg, V, x, y, z; method=:cubic)

Cubic spline interpolation on a 3D tensor grid.

# Inputs
- `xg, yg, zg`: 1D **strictly increasing** grid vectors.
- `V`: 3D array with `size(V) == (length(xg), length(yg), length(zg))`.
- `x, y, z`: 1D query vectors (can be different lengths).

# Output
Returns `Vq` with size `(length(x), length(y), length(z))`, corresponding to
evaluation on the tensor-product query grid `(x ⊗ y ⊗ z)`.

# Notes
- `method=:cubic` uses separable cubic spline interpolation (matches MATLAB's interpn 'spline')
- `method=:linear` uses trilinear interpolation
- Queries are clamped to the grid boundaries
"""
function myinterpolate3(xg, yg, zg, V, x, y, z; method::Symbol = :cubic)
    # --- basic checks ---
    @assert ndims(V) == 3 "V must be a 3D array."
    @assert size(V) == (length(xg), length(yg), length(zg)) "V size mismatch."
    @assert issorted(xg) && all(diff(xg) .> 0) "xg must be strictly increasing."
    @assert issorted(yg) && all(diff(yg) .> 0) "yg must be strictly increasing."
    @assert issorted(zg) && all(diff(zg) .> 0) "zg must be strictly increasing."

    # Ensure 1D vectors
    xg = vec(xg); yg = vec(yg); zg = vec(zg)
    x = vec(x); y = vec(y); z = vec(z)

    # --- clamp queries to open interval (avoid last knot) ---
    tinyx = 1e-12 * (abs(last(xg) - first(xg)) + 1)
    tinyy = 1e-12 * (abs(last(yg) - first(yg)) + 1)
    tinyz = 1e-12 * (abs(last(zg) - first(zg)) + 1)

    xcl = clamp.(x, first(xg) + tinyx, last(xg) - tinyx)
    ycl = clamp.(y, first(yg) + tinyy, last(yg) - tinyy)
    zcl = clamp.(z, first(zg) + tinyz, last(zg) - tinyz)

    nx, ny, nz = length(xcl), length(ycl), length(zcl)
    nxg, nyg, nzg = length(xg), length(yg), length(zg)

    if method == :linear
        # Use Gridded linear interpolation
        itp = interpolate((xg, yg, zg), V, Gridded(Linear()))
        Vq = Array{eltype(V)}(undef, nx, ny, nz)
        @inbounds for k in 1:nz
            for j in 1:ny
                for i in 1:nx
                    Vq[i, j, k] = itp(xcl[i], ycl[j], zcl[k])
                end
            end
        end
        return Vq

    elseif method == :cubic
        # Separable cubic spline interpolation (like MATLAB's interpn 'spline')
        # Step 1: Interpolate along x-dimension for all (yg, zg) points
        V1 = Array{Float64}(undef, nx, nyg, nzg)
        for k in 1:nzg
            for j in 1:nyg
                V1[:, j, k] = cubic_interp_1d(xg, V[:, j, k], xcl)
            end
        end

        # Step 2: Interpolate along y-dimension
        V2 = Array{Float64}(undef, nx, ny, nzg)
        for k in 1:nzg
            for i in 1:nx
                V2[i, :, k] = cubic_interp_1d(yg, V1[i, :, k], ycl)
            end
        end

        # Step 3: Interpolate along z-dimension
        Vq = Array{Float64}(undef, nx, ny, nz)
        for j in 1:ny
            for i in 1:nx
                Vq[i, j, :] = cubic_interp_1d(zg, V2[i, j, :], zcl)
            end
        end
        return Vq

    else
        throw(ArgumentError("method must be :linear or :cubic, got $method"))
    end
end

"""
    cubic_interp_1d(xg, v, xq)

1D cubic spline interpolation with not-a-knot boundary conditions.
Matches MATLAB's spline() function behavior.
"""
function cubic_interp_1d(xg, v, xq)
    n = length(xg)
    xg = vec(xg)
    v = vec(v)
    xq = vec(xq)

    # Compute spline coefficients using not-a-knot conditions
    h = diff(xg)

    # Build tridiagonal system for second derivatives
    # Using not-a-knot: third derivative continuous at x[2] and x[n-1]

    # Build the matrix for not-a-knot spline (size n x n)
    A = zeros(n, n)
    rhs = zeros(n)

    # Not-a-knot condition at left: S'''(x1) = S'''(x2)
    # h2*d1 - (h1+h2)*d2 + h1*d3 = 0
    A[1, 1] = h[2]
    A[1, 2] = -(h[1] + h[2])
    A[1, 3] = h[1]
    rhs[1] = 0.0

    # Interior points: continuity of first derivative
    # h[i-1]*d[i-1] + 2*(h[i-1]+h[i])*d[i] + h[i]*d[i+1] = 6*(delta[i] - delta[i-1])
    delta = diff(v) ./ h
    for i in 2:n-1
        A[i, i-1] = h[i-1]
        A[i, i] = 2.0 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        rhs[i] = 6.0 * (delta[i] - delta[i-1])
    end

    # Not-a-knot condition at right: S'''(x_{n-1}) = S'''(x_n)
    A[n, n-2] = h[n-1]
    A[n, n-1] = -(h[n-2] + h[n-1])
    A[n, n] = h[n-2]
    rhs[n] = 0.0

    # Solve for second derivatives
    d = A \ rhs

    # Evaluate spline at query points
    nq = length(xq)
    vq = zeros(nq)

    for iq in 1:nq
        xi = xq[iq]

        # Find interval
        idx = searchsortedlast(xg, xi)
        idx = clamp(idx, 1, n-1)

        # Local coordinates
        hi = h[idx]

        # Using second derivative form:
        # S(x) = d[i]*(xg[i+1]-x)^3/(6*hi) + d[i+1]*(x-xg[i])^3/(6*hi)
        #      + (v[i]/hi - d[i]*hi/6)*(xg[i+1]-x) + (v[i+1]/hi - d[i+1]*hi/6)*(x-xg[i])

        dx1 = xg[idx+1] - xi  # xg[i+1] - x
        dx0 = xi - xg[idx]    # x - xg[i]

        vq[iq] = (d[idx] * dx1^3 + d[idx+1] * dx0^3) / (6.0 * hi) +
                 (v[idx] / hi - d[idx] * hi / 6.0) * dx1 +
                 (v[idx+1] / hi - d[idx+1] * hi / 6.0) * dx0
    end

    return vq
end
   
"""
myinterpolate3(xgrd1, xgrd2, xgrd3, ygrd, xeval1, xeval2, xeval3)

Trilineary project `ygrd` on (`xgrd1`,`xgrd2`,`xgrd3`) and use it to
interpolate value at (`xeval1`,`xeval2`,`xeval3`).

# Example
```jldoctest
julia> xgrd = [1.0,6.0];
julia> f((x,y,z)) = x+y+z;
julia> ygrd = f.(collect(Iterators.product(xgrid,xgrid,xgrid));
julia> xeval = [3.0,5.0];
julia> mylinearinterpolate3(xgrd,xgrd,xgrd,ygrd,xeval,xeval,xeval)
2x2x2 Array{Float64,3}:
[:,:,1] =
 9.0 11.0
11.0 13.0
[:,:,2] =
11.0 13.0
13.0 15.0
```

Code taken from Bayer Borne and Luetticke BASE for HANK toolbox

"""
function myinterpolate3(xgrd1::AbstractVector, xgrd2::AbstractVector,
    xgrd3::AbstractVector, ygrd::AbstractArray,
    xeval1::AbstractVector, xeval2::AbstractVector, xeval3::AbstractVector;
    method::Symbol = :cubic)

    if method == :linear
        return mylinearinterpolate3(xgrd1, xgrd2, xgrd3, ygrd, xeval1, xeval2, xeval3)
    elseif method == :akima
        return myAkimaInterp3(xgrd1, xgrd2, xgrd3, ygrd, xeval1, xeval2, xeval3)
    elseif method == :cubic
        # Use separable cubic spline interpolation (matches MATLAB's interpn 'spline')
        # Inline implementation to ensure correct dispatch
        xg = vec(xgrd1); yg = vec(xgrd2); zg = vec(xgrd3)
        V = ygrd
        x = vec(xeval1); y = vec(xeval2); z = vec(xeval3)

        # Clamp queries to open interval (avoid last knot)
        tinyx = 1e-12 * (abs(last(xg) - first(xg)) + 1)
        tinyy = 1e-12 * (abs(last(yg) - first(yg)) + 1)
        tinyz = 1e-12 * (abs(last(zg) - first(zg)) + 1)

        xcl = clamp.(x, first(xg) + tinyx, last(xg) - tinyx)
        ycl = clamp.(y, first(yg) + tinyy, last(yg) - tinyy)
        zcl = clamp.(z, first(zg) + tinyz, last(zg) - tinyz)

        nx, ny, nz = length(xcl), length(ycl), length(zcl)
        nxg, nyg, nzg = length(xg), length(yg), length(zg)

        # Step 1: Interpolate along x-dimension for all (yg, zg) points
        V1 = Array{Float64}(undef, nx, nyg, nzg)
        for k in 1:nzg
            for j in 1:nyg
                V1[:, j, k] = cubic_interp_1d(xg, V[:, j, k], xcl)
            end
        end

        # Step 2: Interpolate along y-dimension
        V2 = Array{Float64}(undef, nx, ny, nzg)
        for k in 1:nzg
            for i in 1:nx
                V2[i, :, k] = cubic_interp_1d(yg, V1[i, :, k], ycl)
            end
        end

        # Step 3: Interpolate along z-dimension
        Vq = Array{Float64}(undef, nx, ny, nz)
        for j in 1:ny
            for i in 1:nx
                Vq[i, j, :] = cubic_interp_1d(zg, V2[i, j, :], zcl)
            end
        end
        return Vq
    else
        throw(ArgumentError("method must be :linear, :akima, or :cubic, got $method"))
    end
end

"""
    _cubic_spline_interp3(xg, yg, zg, V, x, y, z)

Internal: Separable cubic spline interpolation on 3D tensor grid.
Matches MATLAB's interpn with 'spline' method.
"""
function _cubic_spline_interp3(xg, yg, zg, V, x, y, z)
    # Ensure 1D vectors
    xg = vec(xg); yg = vec(yg); zg = vec(zg)
    x = vec(x); y = vec(y); z = vec(z)

    # Clamp queries to open interval (avoid last knot)
    tinyx = 1e-12 * (abs(last(xg) - first(xg)) + 1)
    tinyy = 1e-12 * (abs(last(yg) - first(yg)) + 1)
    tinyz = 1e-12 * (abs(last(zg) - first(zg)) + 1)

    xcl = clamp.(x, first(xg) + tinyx, last(xg) - tinyx)
    ycl = clamp.(y, first(yg) + tinyy, last(yg) - tinyy)
    zcl = clamp.(z, first(zg) + tinyz, last(zg) - tinyz)

    nx, ny, nz = length(xcl), length(ycl), length(zcl)
    nxg, nyg, nzg = length(xg), length(yg), length(zg)

    # Step 1: Interpolate along x-dimension for all (yg, zg) points
    V1 = Array{Float64}(undef, nx, nyg, nzg)
    for k in 1:nzg
        for j in 1:nyg
            V1[:, j, k] = cubic_interp_1d(xg, V[:, j, k], xcl)
        end
    end

    # Step 2: Interpolate along y-dimension
    V2 = Array{Float64}(undef, nx, ny, nzg)
    for k in 1:nzg
        for i in 1:nx
            V2[i, :, k] = cubic_interp_1d(yg, V1[i, :, k], ycl)
        end
    end

    # Step 3: Interpolate along z-dimension
    Vq = Array{Float64}(undef, nx, ny, nz)
    for j in 1:ny
        for i in 1:nx
            Vq[i, j, :] = cubic_interp_1d(zg, V2[i, j, :], zcl)
        end
    end
    return Vq
end
#                                 poly_coeff_Akima3(xgrd1, xgrd2, xgrd3, F) ,
#                                 xeval1, xeval2, xeval3)
#-------------------------------------------------------------------------------
## Linear interpolations ##
#-------------------------------------------------------------------------------
# of one function
function mylinearinterpolate(xgrd::AbstractVector,#Union{AbstractArray{Float64,1},AbstractArray{DualNumbers.Dual{Float64},1}},
    ygrd::AbstractVector,#Union{AbstractArray{Float64,1},AbstractArray{DualNumbers.Dual{Float64},1}},
    xeval::AbstractVector)#Union{AbstractArray{Float64,1},AbstractArray{DualNumbers.Dual{Float64},1}})
    ind = zeros(Int,length(xeval))
    n_xgrd = length(xgrd)
    yeval= Array{eltype(ygrd),1}(undef,length(xeval))

    n_xgrd = length(xgrd)
    @views for i in eachindex(xeval)
        xi = xeval[i]
        if xi > xgrd[n_xgrd-1]
            iL = n_xgrd - 1
        elseif xi < xgrd[2]
            iL = 1
        else
            iL      = locate(xi, xgrd)
        end
        iR = iL+1
        xL = xgrd[iL]
        wR = (xi - xL) / (xgrd[iR] - xL)
        wL = 1.0-wR
        yeval[i] = wL * ygrd[iL] + wR * ygrd[iR]
    end

    return yeval
end


# of two functions on the same grids
function mylinearinterpolate_mult2(xgrd::AbstractVector,#Union{AbstractArray{Float64,1},AbstractArray{DualNumbers.Dual{Float64},1}},
    ygrd1::AbstractVector,#Union{AbstractArray{Float64,1},AbstractArray{DualNumbers.Dual{Float64},1}},
    ygrd2::AbstractVector,#Union{AbstractArray{Float64,1},AbstractArray{DualNumbers.Dual{Float64},1}},
    xeval::AbstractVector)#Union{AbstractArray{Float64,1},AbstractArray{DualNumbers.Dual{Float64},1}})
    yeval1= Array{eltype(ygrd1),1}(undef,length(xeval))
    yeval2= Array{eltype(ygrd1),1}(undef,length(xeval))

    n_xgrd = length(xgrd)
    @views @inbounds begin
        for i in eachindex(xeval)
        xi = xeval[i]
        if xi > xgrd[n_xgrd-1]
            iL = n_xgrd - 1
        elseif xi < xgrd[2]
            iL = 1
        else
            iL      = locate(xi, xgrd)
        end
        iR = iL+1
        xL = xgrd[iL]
        wR = (xeval[i] - xL) / (xgrd[iR] - xL)
        #wL = 1.0-wR
        y1L = ygrd1[iL]
        y2L = ygrd2[iL]
        yeval1[i] = y1L + wR * (ygrd1[iR] - y1L)
        yeval2[i] = y2L + wR * (ygrd2[iR] - y2L)
    end
end
    return yeval1, yeval2
end

# of three functions on the same grids
function mylinearinterpolate_mult3(xgrd::AbstractVector,#Union{AbstractArray{Float64,1},AbstractArray{DualNumbers.Dual{Float64},1}},
    ygrd1::AbstractVector,#Union{AbstractArray{Float64,1},AbstractArray{DualNumbers.Dual{Float64},1}},
    ygrd2::AbstractVector,#Union{AbstractArray{Float64,1},AbstractArray{DualNumbers.Dual{Float64},1}},
    ygrd3::AbstractVector,#Union{AbstractArray{Float64,1},AbstractArray{DualNumbers.Dual{Float64},1}},
    xeval::AbstractVector)#Union{AbstractArray{Float64,1},AbstractArray{DualNumbers.Dual{Float64},1}})

    yeval1= Array{eltype(ygrd1),1}(undef,length(xeval))
    yeval2= Array{eltype(ygrd2),1}(undef,length(xeval))
    yeval3= Array{eltype(ygrd3),1}(undef,length(xeval))

    n_xgrd = length(xgrd)
    for i in eachindex(xeval)
        xi = xeval[i]
        if xi > xgrd[n_xgrd-1]
            iL = n_xgrd - 1
        elseif xi < xgrd[2]
            iL = 1
        else
            iL      = locate(xi, xgrd)
        end
        iR = iL+1
        xL = xgrd[iL]
        wR = (xi - xL) / (xgrd[iR] - xL)
        y1L = ygrd1[iL]
        y2L = ygrd2[iL]
        y3L = ygrd3[iL]
        yeval1[i] = y1L + wR * (ygrd1[iR] - y1L)
        yeval2[i] = y2L + wR * (ygrd2[iR] - y2L)
        yeval3[i] = y3L + wR * (ygrd3[iR] - y3L)
    end
    return yeval1, yeval2, yeval3
end

#---------------- Billinear interpolation -------------------------------------
function mylinearinterpolate2(xgrd1::AbstractVector, xgrd2::AbstractVector, ygrd::AbstractArray,
    xeval1::AbstractVector, xeval2::AbstractVector)

    yeval   = zeros(eltype(xeval1),length(xeval1),length(xeval2))
    n_xgrd1 = length(xgrd1)
    n_xgrd2 = length(xgrd2)
    weight1 = Array{eltype(xeval1),1}(undef,length(xeval1))
    weight2 = Array{eltype(xeval2),1}(undef,length(xeval2))
    ind1    = zeros(Int,length(xeval1))
    ind2    = zeros(Int,length(xeval2))
    @views for i in eachindex(xeval1)
        xi = xeval1[i]
        if xi > xgrd1[n_xgrd1-1]
            iL = n_xgrd1 - 1
        elseif xi < xgrd1[2]
            iL = 1
        else
            iL      = locate(xi, xgrd1)
        end
        ind1[i]      = iL
        weight1[i]   = (xi - xgrd1[iL]) / (xgrd1[iL+1] - xgrd1[iL])
    end

    @views for i in eachindex(xeval2)
        xi = xeval2[i]
        if xi > xgrd2[n_xgrd2-1]
            iL = n_xgrd2 - 1
        elseif xi < xgrd2[2]
            iL = 1
        else
            iL      = locate(xi, xgrd2)
        end
        ind2[i]      = iL
        weight2[i]   = (xi - xgrd2[iL]) / (xgrd2[iL+1] - xgrd2[iL])
    end

    for j in eachindex(xeval2)
        w2R = weight2[j]
        w2L = 1.0-w2R
        for i in eachindex(xeval1)
            w1R = weight1[i]
            w1L = 1.0-w1R
            aux = w2L*(w1L*ygrd[ind1[i],ind2[j]] + w1R*ygrd[ind1[i]+1,ind2[j]]) +
                  w2R*(w1L*ygrd[ind1[i],ind2[j]+1] + w1R*ygrd[ind1[i]+1,ind2[j]+1])
            yeval[i,j] = aux[1]
        end
    end

    return yeval
end

function mylinearinterpolate3(xgrd1::AbstractVector, xgrd2::AbstractVector,
    xgrd3::AbstractVector, ygrd::AbstractArray,
    xeval1::AbstractVector, xeval2::AbstractVector, xeval3::AbstractVector)
    ## Define variables ##
    yeval   = Array{eltype(xeval1),3}(undef,(length(xeval1),length(xeval2),length(xeval3)))
    n_xgrd1 = length(xgrd1)
    n_xgrd2 = length(xgrd2)
    n_xgrd3 = length(xgrd3)
    weight1r = Array{eltype(xeval1),1}(undef,length(xeval1))
    weight2r = Array{eltype(xeval2),1}(undef,length(xeval2))
    weight3r = Array{eltype(xeval3),1}(undef,length(xeval3))
    ind1    = Array{Int,1}(undef,length(xeval1))
    ind2    = Array{Int,1}(undef,length(xeval2))
    ind3    = Array{Int,1}(undef,length(xeval3))
    ## find left element of xeval1 on xgrd1
    for i in eachindex(xeval1)
        xi = xeval1[i]
        if xi >= xgrd1[n_xgrd1-1]
            iL = n_xgrd1 - 1
        elseif xi < xgrd1[2]
            iL = 1
        else
            iL      = locate(xi, xgrd1)
        end
        ind1[i]      = iL
        xL          = xgrd1[iL]
        weight1r[i]  = (xi - xL) / (xgrd1[iL+1] - xL)
    end

    ## find left element of xeval2 on xgrd2
    for i in eachindex(xeval2)
        xi = xeval2[i]
        if xi >= xgrd2[n_xgrd2-1]
            iL = n_xgrd2 - 1
        elseif xi < xgrd2[2]
            iL = 1
        else
            iL      = locate(xi, xgrd2)
        end
        ind2[i]      = iL
        xL           = xgrd2[iL]
        weight2r[i]  = (xi - xL) / (xgrd2[iL+1] - xL)
    end

    ## find left element of xeval3 on xgrd3
    for i in eachindex(xeval3)
        xi = xeval3[i]
        if xi >= xgrd3[n_xgrd3-1]
            iL = n_xgrd3 - 1
        elseif xi < xgrd3[2]
            iL = 1
        else
            iL      = locate(xi, xgrd3)
        end
        ind3[i]      = iL
        xL           = xgrd3[iL]
        weight3r[i]  = (xi - xL) / (xgrd3[iL+1] - xL)
    end
@inbounds  begin
    for l in eachindex(xeval3)
        i3  = ind3[l]
        w3r = weight3r[l]
        w3l = 1.0 - w3r
        for j in eachindex(xeval2)
            i2 =ind2[j]
            w2r = weight2r[j]
            w2l = 1.0 - w2r
            for k in eachindex(xeval1)
                i1  = ind1[k]
                w1r = weight1r[k]
                w1l = 1.0 - w1r
               yeval[k,j,l] =  w1l*(w2l*(w3l*ygrd[i1,i2,i3]     + w3r*ygrd[i1,i2,i3+1])    +
                                      w2r*(w3l*ygrd[i1,i2+1,i3]   + w3r*ygrd[i1,i2+1,i3+1])) +
                                w1r*(w2l*(w3l*ygrd[i1+1,i2,i3]   + w3r*ygrd[i1+1,i2,i3+1])  +
                                      w2r*(w3l*ygrd[i1+1,i2+1,i3] + w3r*ygrd[i1+1,i2+1,i3+1]))
            end
        end
    end
end
    return yeval
end




#-------------------------------------------------------------------------
# 1-D Akima interpolation
#-------------------------------------------------------------------------

function poly_coeff_Akima1(xgrd1::Vector, F::AbstractVector)
    # determine polynomial coefficients for local polynomials using Akima's fomrula
    T       = eltype(xgrd1[1]+F[1]) # Makes sure that this works with forwardDiff, because type of ForwardDiffDual is broken (not AbstractFloat type)

    n = length(xgrd1)
        coeffs = Array{T}(undef,4,n-1)
        t = Array{T}(undef,n,1)
        m = Array{T}(undef,n+3,1)

        dx  = diff(xgrd1)
        dx2 = dx.^2
        dx3 = dx.*dx2
        # First order divided differences x-dim
        m[3:n+1] = diff(F)./dx
        # Extrapolation
        m[n+2]   = 2.0 .* m[n+1] - m[n]
        m[n+3]   = 2.0 .* m[n+2] - m[n+1]
        m[2]     = 2.0 .* m[3] - m[4]
        m[1]     = 2.0 .* m[2] - m[3]
        w = abs.(diff(m;dims=1)).+10*eps()
        t = (w[3:n+2].*m[2:n+1] + w[1:n].*m[3:n+2])./(w[3:n+2] + w[1:n])

        D = Matrix{T}(I,4,4)
        @inbounds for i = 1:n-1
            D[3:4,:] .=[
                1.0 dx[i] dx2[i] dx3[i];
                0.0 1.0 2*dx[i] 3*dx2[i]
                ]
            coeffs[:,i] = D\ [F[i]; t[i]; F[i+1]; t[i+1]]
        end
        return coeffs
    end

    function eval_Akima1(xgrd1::AbstractVector,  coeffs::AbstractArray, xeval::AbstractVector)
        # Evaluate local polynomials on a grid of evaluation points
        T       = eltype(xgrd1[1]+xeval[1]+coeffs[1])
        n_xgrd = length(xgrd1)
        yeval = Array{T,1}(undef,length(xeval))
        @views for i in eachindex(xeval)
            xi = xeval[i]
            if xi > xgrd1[n_xgrd-1]
                iL   = n_xgrd - 1
            elseif xi < xgrd1[2]
                iL   = 1
            else
                iL   = locate(xi, xgrd1)
            end
            xd        = xi - xgrd1[iL]
            xd2       = xd * xd
            xd3       = xd2 * xd
            yeval[i] = coeffs[1,iL] + coeffs[2,iL]*xd + coeffs[3,iL]*xd2 + coeffs[4,iL]*xd3
        end
        return yeval
    end
    #-------------------------------------------------------------------------
    # 2-D Akima interpolation
    #-------------------------------------------------------------------------
    function poly_coeff_Akima2(xgrd1::Vector, xgrd2::Vector, F::AbstractMatrix)
        T       = eltype(xgrd1[1]+xgrd2[1]+F[1]) # Makes sure that this works with forwardDiff, because type of ForwardDiffDual is broken (not AbstractFloat type)

        # Determine polynomial coefficients
        nx      = length(xgrd1)
        ny      = length(xgrd2)
        coeffs  = Array{T}(undef,16, (nx-1),(ny-1))

        mx      = Array{T}(undef,nx+3,ny)
        my      = Array{T}(undef,nx,ny+3)
        mxy     = zeros(T,nx+3,ny+3)

        tx      = Array{T}(undef,nx,ny)
        ty      = Array{T}(undef,nx,ny)
        txy     = Array{T}(undef,nx,ny)

        dx      = diff(xgrd1)
        dx2     = dx.^2
        dx3     = dx.*dx2

        dy      = diff(xgrd2)
        dy2     = dy.^2
        dy3     = dy.*dy2

        # First order divided differences x-dim
        mx[3:nx+1,:] = diff(F;dims=1)./dx
        # Extrapolation
        mx[nx+2,:]  = 2.0 .* mx[nx+1,:] .- mx[nx,:]
        mx[nx+3,:]  = 2.0 .* mx[nx+2,:] .- mx[nx+1,:]
        mx[2,:]     = 2.0 .* mx[3,:]    .- mx[4,:]
        mx[1,:]     = 2.0 .* mx[2,:]    .- mx[3,:]

        # First order divided differences y-dim
        my[:,3:ny+1] = diff(F;dims=2)./dy'
        # Extrapolation
        my[:,ny+2]  = 2.0 .* my[:,ny+1] .- my[:,ny]
        my[:,ny+3]  = 2.0 .* my[:,ny+2] .- my[:,ny+1]
        my[:,2]     = 2.0 .* my[:,3]    .- my[:,4]
        my[:,1]     = 2.0 .* my[:,2]    .- my[:,3]

        # Second order divided differences (croos derivative)
        # Assume cross derivative is zero outside grid
        mxy[3:nx+1,:] = diff(my;dims=1)./dx
        mxy[:,3:ny+1] = diff(mx;dims=2)./dy'


        #calculate weights
        wx = abs.(diff(mx;dims=1)).+10*eps()
        wy = abs.(diff(my;dims=2)).+10*eps()
        # calculate tangential values and cross derivatives
        tx = (wx[3:nx+2,:].*mx[2:nx+1,:] + wx[1:nx,:].*mx[3:nx+2,:])./(wx[3:nx+2,:] + wx[1:nx,:])
        ty = (wy[:,3:ny+2].*my[:,2:ny+1] + wy[:,1:ny].*my[:,3:ny+2])./(wy[:,3:ny+2] + wy[:,1:ny])

        txy = (wx[3:nx+2,:].*(wy[:,3:ny+2].*mxy[2:nx+1,2:ny+1] .+ wy[:,1:ny].*mxy[2:nx+1,3:ny+2]).+
               wx[1:nx,:]  .*(wy[:,3:ny+2].*mxy[3:nx+2,2:ny+1] .+ wy[:,1:ny].*mxy[3:nx+2,3:ny+2])) ./
             ((wy[:,3:ny+2] .+ wy[:,1:ny]).*(wx[3:nx+2,:] .+ wx[1:nx,:]))

        Dx = Matrix{T}(I,4,4)
        Dy = Matrix{T}(I,4,4)
        for i = 1:nx-1
            Dx[3,:] = [1.0 dx[i] dx2[i] dx3[i]]
            Dx[4,:] = [0.0 1.0 2*dx[i] 3*dx2[i]]
            for j=1:ny-1
                Dy[3,:] = [1.0 dy[j] dy2[j]  dy3[j]]
                Dy[4,:] = [0.0 1.0 2*dy[j] 3*dy2[j]]
                coeffs[:,i,j] = Dy\ [F[i,j]     tx[i,j]     F[i+1,j]      tx[i+1,j];
                                    ty[i,j]     txy[i,j]    ty[i+1,j]     txy[i+1,j];
                                    F[i,j+1]    tx[i,j+1]   F[i+1,j+1]    tx[i+1,j+1];
                                    ty[i,j+1]   txy[i,j+1]  ty[i+1,j+1]   txy[i+1,j+1]]/(Dx')
            end
        end
        return coeffs
    end

    function eval_Akima2(xgrd1::AbstractVector, xgrd2::AbstractVector, coeffs::AbstractArray,
        xeval1::AbstractVector, xeval2::AbstractVector)
        T       = eltype(xgrd1[1]+xgrd2[1]+xeval1[1]+xeval2[1]+coeffs[1])
        # Evaluate local polynomials on a mesh of evaluation points
        yeval   = zeros(T,length(xeval1),length(xeval2))
        n_xgrd1 = length(xgrd1)
        n_xgrd2 = length(xgrd2)
        xd1     = Array{T,1}(undef,length(xeval1))
        xd2     = Array{T,1}(undef,length(xeval2))
        ind1    = zeros(Int,length(xeval1))
        ind2    = zeros(Int,length(xeval2))

        @views for i in eachindex(xeval1)
            xi = xeval1[i]
            if xi > xgrd1[n_xgrd1-1]
                iL  = n_xgrd1 - 1
            elseif xi < xgrd1[2]
                iL  = 1
            else
                iL  = locate(xi, xgrd1)
            end
            xd1[i]  = xi - xgrd1[iL]
            ind1[i] = iL
        end
        @views for i in eachindex(xeval2)
            xi = xeval2[i]
            if xi > xgrd2[n_xgrd2-1]
                iL  = n_xgrd2 - 1
            elseif xi < xgrd2[2]
                iL  = 1
            else
                iL  = locate(xi, xgrd2)
            end
            ind2[i] = iL
            xd2[i]  = xi - xgrd2[iL]
        end
        # evaluate polynomial
        for j in eachindex(xeval2)
            x2 = xd2[j]
            x2_2 = x2 * x2
            x2_3 = x2_2 * x2
            for i in eachindex(xeval1)
                c     = coeffs[:,ind1[i],ind2[j]]
                x1 = xd1[i]
                x1_2 = x1 * x1
                x1_3 = x1_2 * x1
                yeval[i,j] = c[1]  + c[2]*x2  + c[3]*x2_2  + c[4]*x2_3 +
                            (c[5]  + c[6]*x2  + c[7]*x2_2  + c[8]*x2_3)*x1 +
                            (c[9]  + c[10]*x2 + c[11]*x2_2 + c[12]*x2_3)*x1_2 +
                            (c[13] + c[14]*x2 + c[15]*x2_2 + c[16]*x2_3)*x1_3
            end
        end
        return yeval
    end
    #-------------------------------------------------------------------------
    # 3-D Akima interpolation, following Dubey and Upadhyay (1989, Computer Physics Communications)
    #-------------------------------------------------------------------------

    myAkimaInterp3(xgrd1::AbstractVector, xgrd2::AbstractVector,
    xgrd3::AbstractVector, F::AbstractArray,
    xeval1::AbstractVector, xeval2::AbstractVector, xeval3::AbstractVector) =
                        eval_Akima3(xgrd1, xgrd2, xgrd3,
                                    poly_coeff_Akima3(xgrd1, xgrd2, xgrd3, F) ,
                                    xeval1, xeval2, xeval3)

    function poly_coeff_Akima3(xgrd1::Vector, xgrd2::Vector, xgrd3::Vector, F::AbstractArray)
        T       = eltype(xgrd1[1]+xgrd2[1]+xgrd3[1]+F[1]) # Makes sure that this works with forwardDiff, because type of ForwardDiffDual is broken (not AbstractFloat type)
        nx      = length(xgrd1)
        ny      = length(xgrd2)
        nz      = length(xgrd3)
        coeffs  = Array{T}(undef,64,(nx-1),(ny-1),(nz-1))

        mx      = Array{T}(undef,nx+3,ny,nz)
        my      = Array{T}(undef,nx,ny+3,nz)
        mz      = Array{T}(undef,nx,ny,nz+3)

        mxy     = zeros(T,nx+3,ny+3,nz)
        mxz     = zeros(T,nx+3,ny,nz+3)
        myz     = zeros(T,nx,ny+3,nz+3)

        mxyz     = zeros(T,nx+3,ny+3,nz+3)

        tx      = Array{T}(undef,nx,ny)
        ty      = Array{T}(undef,nx,ny)
        tz      = Array{T}(undef,nx,ny)
        txy     = Array{T}(undef,nx,ny)
        txz     = Array{T}(undef,nx,ny)
        tyz     = Array{T}(undef,nx,ny)
        txyz    = Array{T}(undef,nx,ny)

        dx      = diff(xgrd1)
        dx2     = dx.^2
        dx3     = dx.*dx2

        dy      = diff(xgrd2)
        dy2     = dy.^2
        dy3     = dy.*dy2

        dz      = diff(xgrd3)
        dz2     = dz.^2
        dz3     = dz.*dz2

        # First order divided differences x-dim
        mx[3:nx+1,:,:] = diff(F;dims=1)./dx
        # Extrapolation
        mx[nx+2,:,:]  = 2.0 .* mx[nx+1,:,:] .- mx[nx,:,:]
        mx[nx+3,:,:]  = 2.0 .* mx[nx+2,:,:] .- mx[nx+1,:,:]
        mx[2,:,:]     = 2.0 .* mx[3,:,:]    .- mx[4,:,:]
        mx[1,:,:]     = 2.0 .* mx[2,:,:]    .- mx[3,:,:]

        # First order divided differences y-dim
        my[:,3:ny+1,:] = diff(F;dims=2)./dy'
        # Extrapolation
        my[:,ny+2,:]  = 2.0 .* my[:,ny+1,:] .- my[:,ny,:]
        my[:,ny+3,:]  = 2.0 .* my[:,ny+2,:] .- my[:,ny+1,:]
        my[:,2,:]     = 2.0 .* my[:,3,:]    .- my[:,4,:]
        my[:,1,:]     = 2.0 .* my[:,2,:]    .- my[:,3,:]

        # First order divided differences z-dim
        mz[:,:,3:nz+1] = diff(F;dims=3)./reshape(dz,(1,1,nz-1))
        # Extrapolation
        mz[:,:,nz+2]  = 2.0 .* mz[:,:,nz+1] .- mz[:,:,nz]
        mz[:,:,nz+3]  = 2.0 .* mz[:,:,nz+2] .- mz[:,:,nz+1]
        mz[:,:,2]     = 2.0 .* mz[:,:,3]    .- mz[:,:,4]
        mz[:,:,1]     = 2.0 .* mz[:,:,2]    .- mz[:,:,3]

        # Second order divided differences (first cross derivative)
        # Assume cross derivative is zero outside grid
        # (innefficient many calculations, when filling the gaps)
        mxy[3:nx+1,:,:] = diff(my;dims=1)./dx
        mxy[:,3:ny+1,:] = diff(mx;dims=2)./dy'

        mxz[3:nx+1,:,:] = diff(mz;dims=1)./dx
        mxz[:,:,3:nz+1] = diff(mx;dims=3)./reshape(dz,(1,1,nz-1))

        myz[:,3:ny+1,:] = diff(mz;dims=2)./dy'
        myz[:,:,3:nz+1] = diff(my;dims=3)./reshape(dz,(1,1,nz-1))

        # Third order divided differences (first cross derivative)
        mxyz[3:nx+1,:,:] = diff(myz;dims=1)./dx
        mxyz[:,3:ny+1,:] = diff(mxz;dims=2)./dy'
        mxyz[:,:,3:nz+1] = diff(mxy;dims=3)./reshape(dz,(1,1,nz-1))

        #calculate weights
        wx = abs.(diff(mx;dims=1)).+10*eps()
        wy = abs.(diff(my;dims=2)).+10*eps()
        wz = abs.(diff(mz;dims=3)).+10*eps()
        # calculate tangential values
        tx = (wx[3:nx+2,:,:].*mx[2:nx+1,:,:] + wx[1:nx,:,:].*mx[3:nx+2,:,:])./(wx[3:nx+2,:,:] + wx[1:nx,:,:])
        ty = (wy[:,3:ny+2,:].*my[:,2:ny+1,:] + wy[:,1:ny,:].*my[:,3:ny+2,:])./(wy[:,3:ny+2,:] + wy[:,1:ny,:])
        tz = (wz[:,:,3:nz+2].*mz[:,:,2:nz+1] + wz[:,:,1:nz].*mz[:,:,3:nz+2])./(wz[:,:,3:nz+2] + wz[:,:,1:nz])

        # and second order (i.e., cross) derivatives
        txy = (wx[3:nx+2,:,:].*(wy[:,3:ny+2,:].*mxy[2:nx+1,2:ny+1,:] .+ wy[:,1:ny,:].*mxy[2:nx+1,3:ny+2,:]).+
               wx[1:nx,:,:]  .*(wy[:,3:ny+2,:].*mxy[3:nx+2,2:ny+1,:] .+ wy[:,1:ny,:].*mxy[3:nx+2,3:ny+2,:])) ./
             ((wy[:,3:ny+2,:] .+ wy[:,1:ny,:]).*(wx[3:nx+2,:,:] .+ wx[1:nx,:,:]))

        txz = (wx[3:nx+2,:,:].*(wz[:,:,3:nz+2].*mxz[2:nx+1,:,2:nz+1] .+ wz[:,:,1:nz].*mxz[2:nx+1,:,3:nz+2]).+
               wx[1:nx,:,:]  .*(wz[:,:,3:nz+2].*mxz[3:nx+2,:,2:nz+1] .+ wz[:,:,1:nz].*mxz[3:nx+2,:,3:nz+2])) ./
               ((wz[:,:,3:nz+2] .+ wz[:,:,1:nz]).*(wx[3:nx+2,:,:] .+ wx[1:nx,:,:]))

        tyz = (wz[:,:,3:nz+2].*(wy[:,3:ny+2,:].*myz[:,2:ny+1,2:nz+1] .+ wy[:,1:ny,:].*myz[:,3:ny+2,2:nz+1]).+
               wz[:,:,1:nz]  .*(wy[:,3:ny+2,:].*myz[:,2:ny+1,3:nz+2] .+ wy[:,1:ny,:].*myz[:,3:ny+2,3:nz+2])) ./
             ((wy[:,3:ny+2,:] .+ wy[:,1:ny,:]).*(wz[:,:,3:nz+2] .+ wz[:,:,1:nz]))
        # and third order (i.e., tripple cross) derivatives

        txyz = (wx[3:nx+2,:,:].*(wy[:,3:ny+2,:].*(wz[:,:,3:nz+2].*mxyz[2:nx+1,2:ny+1,2:nz+1] .+
                                                  wz[:,:,1:nz]  .*mxyz[2:nx+1,2:ny+1,3:nz+2] ) .+
                                 wy[:,1:ny,:]  .*(wz[:,:,3:nz+2].*mxyz[2:nx+1,3:ny+2,2:nz+1] .+
                                                  wz[:,:,1:nz]  .*mxyz[2:nx+1,3:ny+2,3:nz+2] )).+
                wx[1:nx,:,:]  .*(wy[:,3:ny+2,:].*(wz[:,:,3:nz+2].*mxyz[3:nx+2,2:ny+1,2:nz+1] .+
                                                  wz[:,:,1:nz]  .*mxyz[3:nx+2,2:ny+1,3:nz+2] ) .+
                                 wy[:,1:ny,:]  .*(wz[:,:,3:nz+2].*mxyz[3:nx+2,3:ny+2,2:nz+1] .+
                                                  wz[:,:,1:nz]  .*mxyz[3:nx+2,3:ny+2,3:nz+2] ))) ./
                ((wy[:,3:ny+2,:] .+ wy[:,1:ny,:]).*(wx[3:nx+2,:,:] .+ wx[1:nx,:,:]).*(wz[:,:,3:nz+2] .+ wz[:,:,1:nz]))
        # this it what costs time can we do this better? (i.e. vectorize instead of )
        Dx = Matrix{T}(I,4,4)
        Dy = Matrix{T}(I,4,4)
        Dz = Matrix{T}(I,4,4)
        DxInv=Vector{Matrix{T}}(undef,nx-1)
        DyInv=Vector{Matrix{T}}(undef,ny-1)
        DzInv=Vector{Matrix{T}}(undef,nz-1)
        @inbounds for k = 1:nz-1
            Dz[3,:] = [1.0 dz[k] dz2[k]  dz3[k]]
            Dz[4,:] = [0.0 1.0 2*dz[k] 3*dz2[k]]
            DzInv[k]= Dz \ I
        end
        @inbounds for j=1:ny-1
            Dy[3,:] = [1.0 dy[j] dy2[j]   dy3[j]]
            Dy[4,:] = [0.0 1.0  2*dy[j] 3*dy2[j]]
            DyInv[j] = Dy \ I
        end
        @inbounds for i = 1:nx-1
            Dx[3,:] = [1.0 dx[i] dx2[i] dx3[i]]
            Dx[4,:] = [0.0 1.0 2*dx[i] 3*dx2[i]]
            DxInv[i] = I / Dx'
        end
        A = Array{T,5}(undef,(8,2,nx,ny,nz-1))
        A[1,1,:,:,:] .= F[1:nx,  1:ny,  1:nz-1]
        A[2,1,:,:,:] .= tz[1:nx, 1:ny,  1:nz-1]
        A[3,1,:,:,:] .= F[1:nx,  1:ny,  2:nz]
        A[4,1,:,:,:] .= tz[1:nx, 1:ny,  2:nz]
        A[5,1,:,:,:] .= ty[1:nx, 1:ny,  1:nz-1]
        A[6,1,:,:,:] .= tyz[1:nx,1:ny,  1:nz-1]
        A[7,1,:,:,:] .= ty[1:nx, 1:ny,  2:nz]
        A[8,1,:,:,:] .= tyz[1:nx,1:ny,  2:nz]

        A[1,2,:,:,:] .= tx[1:nx,  1:ny,  1:nz-1]
        A[2,2,:,:,:] .= txz[1:nx, 1:ny,  1:nz-1]
        A[3,2,:,:,:] .= tx[1:nx,  1:ny,  2:nz]
        A[4,2,:,:,:] .= txz[1:nx, 1:ny,  2:nz]
        A[5,2,:,:,:] .= txy[1:nx, 1:ny,  1:nz-1]
        A[6,2,:,:,:] .= txyz[1:nx,1:ny,  1:nz-1]
        A[7,2,:,:,:] .= txy[1:nx, 1:ny,  2:nz]
        A[8,2,:,:,:] .= txyz[1:nx,1:ny,  2:nz]

         for k = 1:nz-1
             for j=1:ny-1
                DzDyInv =(kron(DyInv[j],DzInv[k]))
                B = [ A[:,:,:,j,k];
                      A[:,:,:,j+1,k] ]
                B = reshape(DzDyInv*reshape(B,16,2*nx),(16,2,nx))
                 @views for i = 1:nx-1
                    coeffs[:,i,j,k] = ([B[:,:,i] B[:,:,i+1]] * DxInv[i])[:]
                    # AX = [
                    # F[i,  j,  k]      tx[i,j,k]       F[i+1,j,k]       tx[i+1,j,k];
                    # tz[i, j,  k]      txz[i,j,k]      tz[i+1,j,k]      txz[i+1,j,k];
                    # F[i,  j,  k+1]    tx[i,j,k+1]     F[i+1,j,k+1]     tx[i+1,j,k+1];
                    # tz[i, j,  k+1]    txz[i,j,k+1]    tz[i+1,j,k+1]    txz[i+1,j,k+1];
                    # ty[i, j,  k]      txy[i,j,k]      ty[i+1,j,k]      txy[i+1,j,k];
                    # tyz[i,j,  k]      txyz[i,j,k]     tyz[i+1,j,k]     txyz[i+1,j,k];
                    # ty[i, j,  k+1]    txy[i,j,k+1]    ty[i+1,j,k+1]    txy[i+1,j,k+1];
                    # tyz[i,j,  k+1]    txyz[i,j,k+1]   tyz[i+1,j,k+1]   txyz[i+1,j,k+1];
                    # F[i,  j+1,k]      tx[i,j+1,k]     F[i+1,j+1,k]     tx[i+1,j+1,k];
                    # tz[i, j+1,k]      txz[i,j+1,k]    tz[i+1,j+1,k]    txz[i+1,j+1,k];
                    # F[i,  j+1,k+1]    tx[i,j+1,k+1]   F[i+1,j+1,k+1]   tx[i+1,j+1,k+1];
                    # tz[i, j+1,k+1]    txz[i,j+1,k+1]  tz[i+1,j+1,k+1]  txz[i+1,j+1,k+1];
                    # ty[i, j+1,k]      txy[i,j+1,k]    ty[i+1,j+1,k]    txy[i+1,j+1,k];
                    # tyz[i,j+1,k]      txyz[i,j,k]     tyz[i+1,j,k]     txyz[i+1,j,k];
                    # ty[i, j+1,k+1]    txy[i,j+1,k+1]  ty[i+1,j+1,k+1]  txy[i+1,j+1,k+1];
                    # tyz[i,j+1,k+1]    txyz[i,j+1,k+1] tyz[i+1,j+1,k+1] txyz[i+1,j+1,k+1]
                    # ]
                end
            end
        end

        return coeffs
    end

    function eval_Akima3(xgrd1::AbstractVector, xgrd2::AbstractVector, xgrd3::AbstractVector, coeffs::AbstractArray,
        xeval1::AbstractVector, xeval2::AbstractVector, xeval3::AbstractVector)
        # Evaluate local polynomials on a mesh of evaluation points
        T       = eltype(xgrd1[1]+xgrd2[1]+xgrd3[1]+xeval1[1]+xeval2[1]+xeval3[1]+coeffs[1])
        ## Define variables ##
        yeval   = zeros(T,(length(xeval1),length(xeval2),length(xeval3)))
        n_xgrd1 = length(xgrd1)
        n_xgrd2 = length(xgrd2)
        n_xgrd3 = length(xgrd3)
        pp1     = Array{T,2}(undef,(length(xeval1),4))
        pp2     = Array{T,2}(undef,(length(xeval2),4))
        pp3     = Array{T,2}(undef,(length(xeval3),4))
        ind1    = Array{Int,1}(undef,length(xeval1))
        ind2    = Array{Int,1}(undef,length(xeval2))
        ind3    = Array{Int,1}(undef,length(xeval3))

        @views for i in eachindex(xeval1)
            xi = xeval1[i]
            if xi > xgrd1[n_xgrd1-1]
                iL  = n_xgrd1 - 1
            elseif xi < xgrd1[2]
                iL  = 1
            else
                iL  = locate(xi, xgrd1)
            end
            xd1  = xi - xgrd1[iL]
            ind1[i] = iL
            pp1[i,1] = one(T)
            pp1[i,2] = xd1
            pp1[i,3] = xd1 * xd1
            pp1[i,4] = pp1[i,3] * xd1
        end
        @views for i in eachindex(xeval2)
            xi = xeval2[i]
            if xi > xgrd2[n_xgrd2-1]
                iL  = n_xgrd2 - 1
            elseif xi < xgrd2[2]
                iL  = 1
            else
                iL  = locate(xi, xgrd2)
            end
            ind2[i] = iL
            xd2  = xi - xgrd2[iL]
            pp2[i,1] = one(T)
            pp2[i,2] = xd2
            pp2[i,3] = xd2 * xd2
            pp2[i,4] = pp2[i,3] * xd2
        end
        @views for i in eachindex(xeval3)
            xi = xeval3[i]
            if xi > xgrd3[n_xgrd3-1]
                iL  = n_xgrd3 - 1
            elseif xi < xgrd3[2]
                iL  = 1
            else
                iL  = locate(xi, xgrd3)
            end
            ind3[i] = iL
            xd3  = xi - xgrd3[iL]
            pp3[i,1] = one(T)
            pp3[i,2] = xd3
            pp3[i,3] = xd3 * xd3
            pp3[i,4] = pp3[i,3] * xd3
        end
        # evaluate polynomial

        @inbounds @views for k in eachindex(xeval3)
            slice1 = reshape(pp3[k,:]'*reshape(coeffs[:,:,:,ind3[k]],(4,16*(n_xgrd1-1)*(n_xgrd2-1))),16,(n_xgrd1-1),(n_xgrd2-1))
            @inbounds @views for j in eachindex(xeval2)
                slice2= reshape(pp2[j,:]'*reshape(slice1[:,:,ind2[j]],(4,4*(n_xgrd1-1))),(4,n_xgrd1-1))
                @inbounds @views for i in eachindex(xeval1)
                    yeval[i,j,k] = pp1[i,:]'*slice2[:,ind1[i]]
                end
            end
        end
        return yeval
    end
