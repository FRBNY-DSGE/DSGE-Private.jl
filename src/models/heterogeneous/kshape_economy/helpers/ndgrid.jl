"""
    ndgrid(vs...)

Generate N-D grids from N input vectors, similar to MATLAB's ndgrid function.

# Arguments
- `vs...`: Variable number of input vectors

# Returns
- Tuple of arrays, one for each input vector, where each array represents a grid
  that varies along its corresponding dimension.

# Example
```julia
x = [1, 2, 3]
y = [4, 5]
z = [6, 7, 8, 9]
X, Y, Z = ndgrid(x, y, z)
# X varies along dimension 1 (size: 3×2×4)
# Y varies along dimension 2 (size: 3×2×4)
# Z varies along dimension 3 (size: 3×2×4)
```
"""
function ndgrid(vs...)
    n = length(vs)
    dims = map(length, vs)
    
    grids = []
    for i in 1:n
        # Create shape where dimension i has the vector length, others are 1
        shape = ntuple(j -> j == i ? dims[j] : 1, n)
        # Reshape the vector to this shape
        grid = reshape(vs[i], shape...)
        # Broadcast to full size by multiplying with ones
        grid = grid .+ zeros(dims...)
        push!(grids, grid)
    end
    
    return Tuple(grids)
end

