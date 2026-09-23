

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

