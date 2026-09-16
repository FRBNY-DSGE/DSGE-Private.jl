using Interpolations

"""
    GriddedInterpolant

A callable interpolant object similar to MATLAB's griddedInterpolant.

# Fields
- `itp`: The underlying Interpolations.jl interpolant object
- `ndims`: Number of dimensions
"""
struct GriddedInterpolant
    itp::Any  # Interpolations.jl interpolant
    ndims::Int
end

"""
    griddedInterpolant(grids, values, method="linear", extrapolation_method="linear")

Create a gridded interpolant similar to MATLAB's griddedInterpolant.

# Arguments
- `grids`: Grid vectors. Can be:
  - A single vector (1D case)
  - A tuple/vector of vectors (multi-D case), e.g., `(x, y, z)` or `[x, y, z]`
  - A tuple of meshgrids (multi-D case), e.g., `(X, Y, Z)` from ndgrid
- `values`: Array of values to interpolate. Must match the grid dimensions.
- `method`: Interpolation method. Options:
  - `"linear"` (default): Linear interpolation
  - `"spline"`: Cubic spline interpolation
  - `"nearest"`: Nearest neighbor interpolation
- `extrapolation_method`: Extrapolation method (optional, defaults to `method`)

# Returns
- `GriddedInterpolant`: A callable object that can be evaluated at query points

# Examples
```julia
# 1D case
x = [1.0, 2.0, 3.0, 4.0]
v = [10.0, 20.0, 30.0, 40.0]
F = griddedInterpolant(x, v)
result = F(2.5)  # Interpolate at x=2.5

# 3D case with cell array syntax
x = [1.0, 2.0, 3.0]
y = [4.0, 5.0]
z = [6.0, 7.0, 8.0, 9.0]
v = rand(3, 2, 4)  # Values array
F = griddedInterpolant((x, y, z), v, "spline")
result = F((1.5, 4.5, 7.0))  # Interpolate at query point
```
"""
function griddedInterpolant(grids, values, method::String="linear", extrapolation_method::String="linear")
    # Determine number of dimensions
    ndims_val = ndims(values)
    
    # Convert method string to Interpolations.jl type
    interp_type = _method_to_interp_type(method)
    extrap_type = _method_to_extrap_type(extrapolation_method)
    
    # Handle different input formats
    if isa(grids, AbstractVector) && ndims_val == 1
        # 1D case: single vector
        itp = interpolate((grids,), values, Gridded(interp_type))
        itp_ext = extrapolate(itp, extrap_type)
        return GriddedInterpolant(itp_ext, 1)
    elseif isa(grids, Tuple) || isa(grids, AbstractVector)
        # Multi-D case: tuple or vector of grid vectors or meshgrids
        grid_tuple = isa(grids, AbstractVector) ? Tuple(grids) : grids
        n_grids = length(grid_tuple)
        
        if n_grids != ndims_val
            error("Number of grid vectors ($n_grids) must match number of dimensions ($ndims_val)")
        end
        
        # Extract grid vectors from inputs (handles both vectors and meshgrids)
        grid_vectors_list = Vector{Any}(undef, n_grids)
        for i in 1:n_grids
            grid_i = grid_tuple[i]
            if ndims(grid_i) == 1
                # Already a vector
                grid_vectors_list[i] = vec(grid_i)
            elseif ndims(grid_i) == ndims_val && size(grid_i) == size(values)
                # It's a meshgrid - extract unique values along dimension i
                # For ndgrid, meshgrid i varies along dimension i
                # Extract by taking first slice along all other dimensions
                idx = ntuple(j -> j == i ? Colon() : 1, ndims_val)
                grid_vectors_list[i] = vec(grid_i[idx...])
            else
                error("Grid $i has incompatible dimensions. Expected vector or meshgrid matching values array.")
            end
        end
        
        # Convert to tuple for Interpolations.jl
        grid_vectors = Tuple(grid_vectors_list)
        
        # Create interpolant
        itp = interpolate(grid_vectors, values, Gridded(interp_type))
        itp_ext = extrapolate(itp, extrap_type)
        return GriddedInterpolant(itp_ext, ndims_val)
    else
        error("Unsupported grid format. Expected vector, tuple, or vector of vectors/meshgrids.")
    end
end

# Convenience constructor for 1D case
function griddedInterpolant(grid::AbstractVector, values::AbstractVector, method::String="linear")
    return griddedInterpolant(grid, values, method, method)
end

# Convenience constructor for multi-D case with default method
function griddedInterpolant(grids::Union{Tuple, AbstractVector}, values::AbstractArray, method::String="linear")
    return griddedInterpolant(grids, values, method, method)
end

# Handle MATLAB-style separate arguments: griddedInterpolant(x, y, z, values, method)
function griddedInterpolant(grid1, grid2, grid3, values::AbstractArray, method::String="linear")
    return griddedInterpolant((grid1, grid2, grid3), values, method, method)
end

function griddedInterpolant(grid1, grid2, grid3, values::AbstractArray, method::String, extrapolation_method::String)
    return griddedInterpolant((grid1, grid2, grid3), values, method, extrapolation_method)
end

# Make GriddedInterpolant callable
function (F::GriddedInterpolant)(query_points)
    if F.ndims == 1
        # 1D case: query_points is a scalar or vector
        if isa(query_points, Number)
            return F.itp(query_points)
        else
            return F.itp.(query_points)
        end
    else
        # Multi-D case: query_points is a tuple/vector of coordinates
        if isa(query_points, Tuple) || isa(query_points, AbstractVector)
            coords = isa(query_points, AbstractVector) ? Tuple(query_points) : query_points
            if length(coords) == F.ndims
                # Check if all coordinates are vectors (vectorized evaluation)
                if all(isa(c, AbstractVector) for c in coords)
                    # Vectorized evaluation: F({x, y, z}) where x, y, z are vectors
                    # Evaluate at each (x[i], y[i], z[i])
                    n_points = length(coords[1])
                    if !all(length(c) == n_points for c in coords)
                        error("All coordinate vectors must have the same length")
                    end
                    # Use broadcasting for efficient vectorized evaluation
                    return F.itp.(coords...)
                else
                    # Single point query: F({x, y, z}) where x, y, z are scalars
                    return F.itp(coords...)
                end
            else
                error("Number of query coordinates ($(length(coords))) must match number of dimensions ($(F.ndims))")
            end
        else
            error("Query points must be a tuple or vector of coordinates for multi-D interpolation")
        end
    end
end

# Handle vectorized evaluation for multi-D case with separate arguments
# MATLAB: F(x, y, z) where x, y, z are vectors
function (F::GriddedInterpolant)(x_query::AbstractVector, y_query::AbstractVector, z_query::AbstractVector)
    if F.ndims == 3
        return F.itp.(x_query, y_query, z_query)
    else
        error("Vectorized evaluation with 3 arguments only supported for 3D case")
    end
end

# Handle vectorized evaluation for 2D case
function (F::GriddedInterpolant)(x_query::AbstractVector, y_query::AbstractVector)
    if F.ndims == 2
        return F.itp.(x_query, y_query)
    else
        error("Vectorized evaluation with 2 arguments only supported for 2D case")
    end
end

# Helper function to convert method string to Interpolations.jl type
function _method_to_interp_type(method::String)
    if method == "linear"
        return Linear()
    elseif method == "spline"
        return BSpline(Cubic(Line(OnGrid())))
    elseif method == "nearest"
        return Constant()
    else
        error("Unsupported interpolation method: $method. Supported: 'linear', 'spline', 'nearest'")
    end
end

# Helper function to convert extrapolation method string to Interpolations.jl type
function _method_to_extrap_type(method::String)
    if method == "linear"
        return Line()
    elseif method == "spline"
        return Line()  # Use linear extrapolation for spline (MATLAB default)
    elseif method == "nearest"
        return Flat()
    else
        return Line()  # Default to linear extrapolation
    end
end
