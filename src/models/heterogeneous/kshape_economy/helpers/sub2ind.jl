"""
    sub2ind(siz, i1, i2, ..., iN)

Convert subscripts to linear indices.

# Arguments
- `siz`: A tuple or vector specifying the size of the array (e.g., `[m, n, p]` for a 3D array)
- `i1, i2, ..., iN`: Subscript arrays, one for each dimension

# Returns
- Linear indices corresponding to the given subscripts

# Examples
```julia
# For a 3D array of size [2, 3, 4]
ind = sub2ind([2, 3, 4], 1, 1, 1)  # Returns 1
ind = sub2ind([2, 3, 4], 2, 1, 1)  # Returns 2
ind = sub2ind([2, 3, 4], 1, 2, 1)  # Returns 3
ind = sub2ind([2, 3, 4], 2, 3, 4)  # Returns 24
```

# Notes
This function replicates MATLAB's `sub2ind` behavior. For an array A of size `siz`,
if `ind = sub2ind(siz, i1, i2, ..., iN)`, then `A[ind[k]] = A[i1[k], i2[k], ..., iN[k]]` for all k.
"""
function sub2ind(siz, i1, i2, i3...)
    # Convert siz to a vector if needed (accepts Tuple or AbstractArray)
    siz_vec = collect(siz)
    ndims = length(siz_vec)
    
    # Collect all subscript arguments
    subscripts = [i1, i2, i3...]
    nsubs = length(subscripts)
    
    # Ensure we have the right number of dimensions
    if nsubs > ndims
        # Pad siz with ones for extra dimensions
        siz_vec = vcat(siz_vec, ones(Int, nsubs - ndims))
        ndims = nsubs
    elseif nsubs < ndims
        # Collapse trailing dimensions
        siz_vec = vcat(siz_vec[1:nsubs-1], prod(siz_vec[nsubs:end]))
        ndims = nsubs
    end
    
    # Check that all subscripts have compatible sizes
    sizes = [size(s) for s in subscripts]
    if length(unique(sizes)) > 1
        error("All subscript arrays must have the same size")
    end
    
    # Check if inputs are scalars or arrays
    is_scalar = all(isa(s, Number) for s in subscripts)
    input_shape = is_scalar ? () : size(subscripts[1])
    
    # Convert all subscripts to arrays for computation
    sub_arrays = [vec(s) for s in subscripts]
    
    # Validate indices are within bounds
    for (dim, sub) in enumerate(sub_arrays)
        if !isempty(sub)
            if any(sub .< 1) || any(sub .> siz_vec[dim])
                error("Index out of range in dimension $dim")
            end
        end
    end
    
    # Compute linear indices using MATLAB's formula:
    # For array size [m, n, p], index (i, j, k) = i + (j-1)*m + (k-1)*m*n
    # General formula: i1 + (i2-1)*s1 + (i3-1)*s1*s2 + ...
    ndx = sub_arrays[1]
    
    if nsubs > 1
        k = siz_vec[1]
        ndx = ndx .+ (sub_arrays[2] .- 1) .* k
        
        for dim in 3:nsubs
            k = k * siz_vec[dim-1]
            ndx = ndx .+ (sub_arrays[dim] .- 1) .* k
        end
    end
    
    # Return with same shape as input subscripts
    if is_scalar
        return ndx[1]  # Return scalar if input was scalar
    else
        return reshape(ndx, input_shape)  # Preserve input shape
    end
end

