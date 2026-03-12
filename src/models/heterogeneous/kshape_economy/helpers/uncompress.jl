using LinearAlgebra

"""
    uncompress(compressionIndexes, XC, DC, IDC)

Expand a compressed (sparse) copula coefficient vector into its full-grid
representation by applying inverse DCT (Discrete Cosine Transform) operations
along each dimension.

This function reconstructs the full 3D copula deviation field from a sparse
set of coefficients stored at selected multi-indices. The reconstruction
uses basis-expansion operators (DCT/inverse-DCT matrices) for each dimension.

# Arguments
- `compressionIndexes::AbstractVector{<:Integer}`:
    Linear indices (1-based) into the full 3D array indicating where the
    sparse coefficients are placed. These indices use column-major (Fortran)
    ordering consistent with MATLAB's linear indexing.

- `XC::AbstractVector{T}`:
    Vector of compressed/sparse coefficients. Length must equal
    `length(compressionIndexes)`.

- `DC::Vector{<:AbstractMatrix}`:
    Cell array (Vector of matrices) containing 3 DCT basis matrices:
    - `DC[1]`: DCT matrix for dimension 1 (bonds/b), size `nm × nm`
    - `DC[2]`: DCT matrix for dimension 2 (assets/a), size `nk × nk`
    - `DC[3]`: DCT matrix for dimension 3 (productivity/se), size `ny × ny`

- `IDC::Vector{<:AbstractMatrix}`:
    Cell array (Vector of matrices) containing inverse DCT matrices:
    - `IDC[1]`: Inverse DCT matrix for dimension 1, size `nm × nm`
    (Only `IDC[1]` is used in the standard reconstruction algorithm)

# Returns
- `theta_vec::Vector{T}`:
    Full uncompressed coefficient vector of length `nm * nk * ny`,
    representing the copula deviation field on the full grid.

# Algorithm
The reconstruction proceeds in three steps:

1. **Sparse placement**: Create a 3D array `θ[nm, nk, ny]` initialized to zero,
   then place the sparse coefficients at their designated positions:
   `θ[compressionIndexes] = XC`

2. **Transform along dimensions 1 and 2**: For each slice along dimension 3 (y):
   `θ[:,:,y] = IDC[1] * θ[:,:,y] * DC[2]`
   This applies inverse-DCT along dimension 1 and DCT along dimension 2.

3. **Transform along dimension 3**: For each index along dimension 1 (m):
   `θ[m,:,:] = θ[m,:,:] * DC[3]`
   This applies DCT along dimension 3.

4. **Vectorize**: Return `vec(θ)` in column-major order.

# Mathematical Description
Let `θ̂ ∈ ℝ^k` be the sparse coefficients at indices `I`. The full field is:

    θ_full = T₃(T₂(T₁(sparse_expand(θ̂, I))))

where `Tᵢ` denotes the appropriate transform along dimension `i`.

# Example
```julia
# Typical usage in copula reconstruction
DC = [DC_b, DC_a, DC_se]      # 3 DCT matrices
IDC = [IDC_b, IDC_a, IDC_se]  # 3 inverse DCT matrices
theta_full = uncompress(grid["compressionIndexesCOP"], cop_coefs, DC, IDC)
COP_Dev = reshape(theta_full, (nb_copula, na_copula, nse_copula))
```

# Notes
- The output dimension is `nm * nk * ny` where these are determined by
  `size(DC[1], 1)`, `size(DC[2], 1)`, and `size(DC[3], 1)` respectively.
- Compression indices must be valid linear indices into an `nm × nk × ny` array.
- This implementation matches the MATLAB function `uncompress.m`.
"""
function uncompress(compressionIndexes::Vector{Float64}, 
        XC::Vector{<:AbstractMatrix}, 
        DC::Vector{<:AbstractMatrix}, 
        IDC::Vector{<:AbstractMatrix})
    # Extract dimensions from DCT matrices
    nm = size(DC[1], 1)  # Dimension 1 (bonds/b)
    nk = size(DC[2], 1)  # Dimension 2 (assets/a)
    ny = size(DC[3], 1)  # Dimension 3 (productivity/se)

    # Ensure inputs are vectors
    idxs = isa(compressionIndexes, AbstractVector) ? compressionIndexes : vec(compressionIndexes)
    xc = isa(XC, AbstractVector) ? XC : vec(XC)

    # Convert indices to integers if needed
    idxs_int = isa(eltype(idxs), Integer) ? idxs : Int.(idxs)

    # Initialize 3D array for coefficients
    theta1 = zeros(eltype(xc), nm, nk, ny)

    # Place the sparse coefficients at their positions (linear indexing)
    theta1[idxs_int] .= xc

    # Pre-allocate slab buffer for Step 1
    slab_buf = Matrix{eltype(xc)}(undef, nm, nk)

    # Step 1: For each y-slice, apply inverse-DCT along m and DCT along k
    # slab = IDC[1] * slab * DC[2]
    idc1 = IDC[1]
    dc2 = DC[2]
    for yy in 1:ny
        slab = @view theta1[:, :, yy]
        mul!(slab_buf, idc1, slab)
        mul!(slab, slab_buf, dc2)
    end

    # Pre-allocate slab buffer for Step 2
    slab_buf2 = Matrix{eltype(xc)}(undef, nk, ny)

    # Step 2: For each m-index, apply DCT along y
    # slab = slab * DC[3]
    dc3 = DC[3]
    for mm in 1:nm
        slab = @view theta1[mm, :, :]
        copyto!(slab_buf2, slab)
        mul!(slab, slab_buf2, dc3)
    end

    # Return as column vector
    return vec(theta1)
end
