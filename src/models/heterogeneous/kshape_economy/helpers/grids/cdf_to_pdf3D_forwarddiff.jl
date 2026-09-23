"""
    cdf_to_pdf3D_forwarddiff(C) -> P

Triple forward difference with zero padding on the 0-faces (cell masses).

Given a 3D CDF array `C`, this returns `P` where each entry is the probability
mass in the corresponding 3D cell. This matches the MATLAB implementation:

    S = zeros(size(C)+[1 1 1], 'like', C);
    S(2:end,2:end,2:end) = C;
    P = diff(diff(diff(S,1,1),1,2),1,3);

# Arguments
- `C::AbstractArray{T,3}`: 3D CDF values.

# Returns
- `P::Array{T,3}`: 3D cell masses with `size(P) == size(C)`.
"""
function cdf_to_pdf3D_forwarddiff(C::AbstractArray{T,3}) where {T}
    n1, n2, n3 = size(C)
    P = Array{T,3}(undef, n1, n2, n3)

    @inbounds for k in 1:n3
        km1 = k - 1
        for j in 1:n2
            jm1 = j - 1
            for i in 1:n1
                im1 = i - 1
                # Forward difference in all 3 dims with zero-padded boundaries
                # P[i,j,k] = C[i,j,k] - C[i-1,j,k] - C[i,j-1,k] - C[i,j,k-1]
                #           + C[i-1,j-1,k] + C[i-1,j,k-1] + C[i,j-1,k-1]
                #           - C[i-1,j-1,k-1]
                # where C[0,...] = 0
                val = C[i, j, k]
                if im1 >= 1; val -= C[im1, j, k]; end
                if jm1 >= 1; val -= C[i, jm1, k]; end
                if km1 >= 1; val -= C[i, j, km1]; end
                if im1 >= 1 && jm1 >= 1; val += C[im1, jm1, k]; end
                if im1 >= 1 && km1 >= 1; val += C[im1, j, km1]; end
                if jm1 >= 1 && km1 >= 1; val += C[i, jm1, km1]; end
                if im1 >= 1 && jm1 >= 1 && km1 >= 1; val -= C[im1, jm1, km1]; end
                P[i, j, k] = val
            end
        end
    end

    return P
end


