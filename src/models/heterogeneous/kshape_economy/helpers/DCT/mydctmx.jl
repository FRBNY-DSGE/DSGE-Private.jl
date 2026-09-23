"""
    mydctmx(n::Integer) -> Matrix{Float64}

Construct the `n × n` Discrete Cosine Transform (DCT-II) matrix.

This is the Julia equivalent of a MATLAB-style DCT matrix construction.
The resulting matrix `DC` satisfies:

- First row: constant normalization `1 / √n`
- Remaining rows: cosine basis functions with normalization `√(2/n)`

Mathematically, the entries are:

- `DC[1, j+1] = 1 / √n`
- `DC[i+1, j+1] = √(2/n) * cos(π * (j + 0.5) * i / n)`
  for `i = 1,…,n-1`, `j = 0,…,n-1`

# Arguments
- `n::Integer`: Dimension of the DCT matrix.

# Returns
- `DC::Matrix{Float64}`: An `n × n` DCT-II transform matrix.

# Notes
- Indices are shifted by +1 relative to MATLAB because Julia is 1-based.
- This matrix is orthonormal: `DC * DC' ≈ I`.

# Examples
```julia
DC = mydctmx(4)
x  = rand(4)
y  = DC * x        # DCT coefficients
x̂  = DC' * y      # inverse transform
"""
function mydctmx(n::Integer)
    DC::Array{Float64, 2} = zeros(n, n)
    for j in 0:n-1
        DC[1, j+1] = 1 / sqrt(n)
        for i in 1:n-1
            DC[i+1, j+1] = sqrt(2 / n) * cos(pi * (j + 0.5) * i / n)
        end
    end

    return DC
end
