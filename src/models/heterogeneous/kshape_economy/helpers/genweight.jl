"""
    genweight(x, xgrid)

Generate weights and indexes used for linear interpolation.

# Arguments
- `x`: Points at which function is to be interpolated (can be scalar or array)
- `xgrid`: Grid points at which function is measured (must be sorted)

# Returns
- `weight`: Interpolation weight for the higher gridpoint
- `index`: Index of the lower gridpoint for interpolation

# Notes
- No extrapolation allowed - values are clamped to valid interpolation range
- For x <= xgrid[1], uses index 1
- For x >= xgrid[end], uses index length(xgrid)-1
"""
function genweight(x::AbstractArray{T}, xgrid::AbstractVector{S}) where {T<:Real, S<:Real}
    # Initialize output arrays with same shape as x
    index = similar(x, Int)
    weight = similar(x, promote_type(T, S))

    xgrid_first = xgrid[1]
    xgrid_last = xgrid[end]
    n_grid_m1 = length(xgrid) - 1

    # Find indices using searchsortedlast (equivalent to histc in MATLAB)
    @inbounds for i in eachindex(x)
        xi = x[i]
        # searchsortedlast returns the index of the last value <= x[i]
        idx = searchsortedlast(xgrid, xi)

        # Handle edge cases
        if xi <= xgrid_first
            index[i] = 1
        elseif xi >= xgrid_last
            index[i] = n_grid_m1
        else
            index[i] = idx
        end

        # Calculate interpolation weight (weight of higher gridpoint)
        ii = index[i]
        weight[i] = (xi - xgrid[ii]) / (xgrid[ii+1] - xgrid[ii])

        # No extrapolation - clamp weights to valid range
        if weight[i] <= 0
            weight[i] = 1.0e-16
        elseif weight[i] >= 1
            weight[i] = 1.0 - 1.0e-16
        end
    end

    return weight, index
end

# Scalar version - convert to array, process, then extract scalar
function genweight(x::Real, xgrid::AbstractVector{<:Real})
    weight, index = genweight([x], xgrid)
    return weight[1], index[1]
end

