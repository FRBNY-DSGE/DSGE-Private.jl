"""
    compress(theta_full, idxs)

Compress a full-length vector by selecting entries at `idxs`.

- `theta_full` is the uncompressed vector (length Nfull).
- `idxs` are 1-based indices into `theta_full`.

Returns a vector of compressed coefficients.
"""
function compress(theta_full, idxs)
    idxs_vec = vec(idxs)

    @assert all(1 .≤ idxs_vec .≤ length(theta_full)) "idxs out of bounds"
    return theta_full[idxs_vec]
end


"""
    compress(compressionIndexes, XU, DC, IDC)

Compress a 3D array to selected DCT coefficients.

This function transforms the input array from physical space to spectral (DCT)
space, then extracts the coefficients at selected positions. This is the inverse
of `uncompress`.

# Arguments
- `compressionIndexes::AbstractVector{<:Integer}`:
    Linear indices (1-based) into the transformed 3D array indicating which
    DCT coefficients to keep.

- `XU::AbstractArray{<:Real}`:
    3D array of size (nm, nk, ny) in physical space.

- `DC::Vector{<:AbstractMatrix}`:
    Cell array (Vector of matrices) containing 3 DCT basis matrices:
    - `DC[1]`: DCT matrix for dimension 1 (bonds/b), size `nm × nm`
    - `DC[2]`: DCT matrix for dimension 2 (assets/a), size `nk × nk`
    - `DC[3]`: DCT matrix for dimension 3 (productivity/se), size `ny × ny`

- `IDC::Vector{<:AbstractMatrix}`:
    Cell array (Vector of matrices) containing inverse DCT matrices:
    - `IDC[3]`: Inverse DCT matrix for dimension 3, size `ny × ny`
    (Only `IDC[3]` is used in the compress algorithm)

# Returns
- `theta::Vector{Float64}`:
    Vector of selected DCT coefficients.

# Algorithm
The compression proceeds in two steps:

1. **Transform along dimensions 2 and 3**: For each slice along dimension 1 (m):
   `XU2[m,:,:] = DC[2] * XU[m,:,:] * IDC[3]`
   This applies DCT along dimension 2 and inverse-DCT along dimension 3.

2. **Transform along dimension 1**: For each slice along dimension 3 (y):
   `XU2[:,:,y] = DC[1] * XU2[:,:,y]`
   This applies DCT along dimension 1.

3. **Extract**: Return `XU2[compressionIndexes]`.

# Notes
- This function is the inverse of `uncompress`.
- The transforms applied are: DCT along m, DCT along k, inverse-DCT along y.
- This matches the MATLAB function `compress.m`.
"""
function compress(compressionIndexes::AbstractVector{<:Integer},
                  XU::AbstractArray{<:Real},
                  DC, IDC)

    nm, nk, ny = size(XU)
    XU2 = zeros(eltype(XU), nm, nk, ny)

    # Pre-fetch DCT matrices
    dc1 = DC[1]   # nm × nm
    dc2 = DC[2]   # nk × nk
    idc3 = IDC[3] # ny × ny

    # Pre-allocate slab buffer for Step 1
    slab_buf = Matrix{eltype(XU)}(undef, nk, ny)

    # Step 1: For each m, DCT along k and inverse-DCT along y
    # slab = DC{2} * slab * IDC{3}
    for m in 1:nm
        slab = @view XU[m, :, :]  # nk × ny
        mul!(slab_buf, dc2, slab)
        XU2_view = @view XU2[m, :, :]
        mul!(XU2_view, slab_buf, idc3)
    end

    # Pre-allocate slice buffer for Step 2
    slice_buf = Matrix{eltype(XU)}(undef, nm, nk)

    # Step 2: For each y, DCT along m
    # slice = DC{1} * slice
    for y in 1:ny
        slice = @view XU2[:, :, y]  # nm × nk
        copyto!(slice_buf, slice)
        mul!(slice, dc1, slice_buf)
    end

    # Keep selected entries
    idxs_vec = vec(compressionIndexes)
    return vec(XU2)[idxs_vec]
end


function compress(theta_full::AbstractMatrix, idxs)
    return theta_full[idxs, :]
end

