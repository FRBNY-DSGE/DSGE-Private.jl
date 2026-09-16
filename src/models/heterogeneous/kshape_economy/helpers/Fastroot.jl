"""
    Fastroot(xgrid, fx)

Fast linear interpolation root finding.

This function finds the roots (zeros) of a function `fx` defined on grid `xgrid`
using linear interpolation. It assumes the function is piecewise linear and
monotonically increasing, allowing for fast root finding with a single Newton step.

# Arguments
- `xgrid`: Vector of grid points (column vector in MATLAB, vector in Julia)
- `fx`: Function values. Can be:
  - Vector of same length as `xgrid` (single function)
  - Matrix of size `(length(xgrid), n)` where each column is a function (MATLAB behavior)

# Returns
- `roots`: Root locations for each function
  - If `fx` is a vector: returns a scalar
  - If `fx` is a matrix: returns a vector of length `n` (one root per column)

# Notes
- Handles corner cases where the function does not cross zero:
  - If `fx[1] > 0`: returns `xgrid[1]` (minimum boundary)
  - If `fx[end] < 0`: returns `xgrid[end]` (maximum boundary)
- Uses linear interpolation (one Newton step) since function is piecewise linear

# Example
```julia
xgrid = [0.0, 1.0, 2.0, 3.0]
fx = [-1.0, -0.5, 0.5, 1.0]  # crosses zero between x=1 and x=2
root = Fastroot(xgrid, fx)  # approximately 1.5
```
"""
function Fastroot(xgrid, fx)
    # Vectorize the grid (ensure column vector in MATLAB terms, but Julia uses 1D arrays)
    xgrid = vec(xgrid)
    n_grid = length(xgrid)

    # Reshape fx to (n_grid, n_functions) where each column is a function
    fx = reshape(fx, n_grid, :)
    n_functions = size(fx, 2)

    # Compute differences
    dxgrid = diff(xgrid)  # dxgrid[i] = xgrid[i+1] - xgrid[i]
    dfx = diff(fx, dims=1)  # dfx[i, :] = fx[i+1, :] - fx[i, :]

    # Initialize idx (will hold the index of the grid point just before the root)
    idx = ones(Int, n_functions)

    # Identify corner solutions where function does not cross zero
    # idx_min: functions where fx[1, :] > 0 (corner solution at minimum)
    idx_min = fx[1, :] .> 0
    # idx_max: functions where fx[end, :] < 0 (corner solution at maximum)
    idx_max = fx[end, :] .< 0
    # index: functions where there is an interior solution (crosses zero)
    index = findall(.!idx_min .& .!idx_max)

    # For functions with interior solutions, find where sign changes from negative to positive
    # Vectorized: compute sign changes for all interior columns at once
    if !isempty(index)
        fx_sub = @view fx[:, index]
        sign_fx_sub = sign.(fx_sub)
        sign_diff_sub = diff(sign_fx_sub, dims=1)

        @inbounds for (li, col_idx) in enumerate(index)
            # Find index of maximum value in sign_diff for this column
            max_val = sign_diff_sub[1, li]
            max_idx = 1
            for row in 2:(n_grid - 1)
                v = sign_diff_sub[row, li]
                if v > max_val
                    max_val = v
                    max_idx = row
                end
            end

            if max_val > 0
                idx[col_idx] = max_idx
            else
                # No clear sign change found, use fallback
                found = false
                for row in 1:n_grid
                    if sign_fx_sub[row, li] >= 0 && row > 1
                        idx[col_idx] = row - 1
                        found = true
                        break
                    end
                end
                if !found
                    idx[col_idx] = 1
                end
            end
        end
    end

    # Initialize roots array
    roots = zeros(n_functions)

    # For functions with interior solutions, compute root using linear interpolation
    @inbounds if !isempty(index)
        for col_idx in index
            i = idx[col_idx]

            # Get function value and grid point at index i
            fxx = fx[i, col_idx]
            xl = xgrid[i]
            dx = dxgrid[i]
            dfxx = dfx[i, col_idx]

            # Linear interpolation: x_root = xl - fxx * dx / dfxx
            # This is one Newton step on a linear function
            if abs(dfxx) > 1e-14  # Avoid division by zero
                roots[col_idx] = xl - fxx * dx / dfxx
            else
                roots[col_idx] = xl  # Fallback if derivative is zero
            end
        end
    end

    # Handle corner solutions
    @inbounds roots[idx_min] .= xgrid[1]   # Boundary solution at minimum
    @inbounds roots[idx_max] .= xgrid[end] # Boundary solution at maximum

    # Return appropriate shape
    if n_functions == 1
        return roots[1]  # Return scalar for single function
    else
        return roots  # Return vector for multiple functions
    end
end


