#---------------------------------------------------------
# Discrete Cosine Transform
#---------------------------------------------------------
@inline mydctmx(n::Int) = mydctmx(Float64, n)
function mydctmx(::Float64, n::Int)
    DC = zeros(n, n)
    for j = 0:(n - 1)
        DC[1, j + 1] = (1. / sqrt(n))::Float64
        for i = 1:(n - 1)
            DC[i + 1, j + 1] = sqrt(2. / n) * cos(π * (j + 1. / 2.) * i / n)
        end
    end
    return DC
end

function mydctmx(T::DataType, n::Int)
    DC = zeros(T, n, n)
    for j = 0:(n - 1)
        DC[1, j + 1] = convert(T, (1. / sqrt(n)))
        for i = 1:(n - 1)
            DC[i + 1, j + 1] = convert(T, sqrt(2. / n) * cos(π * ((j + 1. / 2.) * i / n)))
        end
    end
    return DC
end

function _orig_mydctmx(n::Int) # original port from HANKEstim
    DC::Array{Float64,2} =zeros(n,n);
    for j=0:n-1
        DC[1,j+1]=float(1/sqrt(n));
        for i=1:n-1
            DC[i+1,j+1]=(pi*float((j+1/2)*i/n));
            DC[i+1,j+1] = sqrt(2/n).*cos.(DC[i+1,j+1])
        end
    end
    return DC
end

function produceCompMat(DC::AbstractArray{S1}, compressionIndexes::AbstractArray{S2},
                        dims::Int) where {S1 <: AbstractArray{<: Real}, S2 <: AbstractArray{Int}}
    IDD = DC[1]'
    for j=2:dims
        IDD = kron(DC[j]',IDD)
    end
    UnCompMat = IDD[:,compressionIndexes]
    CompMat   = inv(UnCompMat'*UnCompMat)*UnCompMat'
    return CompMat, UnCompMat
end

# Either create generated function or use a more intelligent way of determining index sizes and indexing
# TODO: GENERALIZE
function uncompress(compressionIndexes::AbstractArray{S1}, XC::AbstractArray{S2},
                    DC::AbstractArray{S3}, IDC::AbstractArray{S4}, n_par # n_par => Named
                    ) where {S1 <: AbstractArray{<: Real}, S2 <: Real, S3 <: AbstractArray{<: Real}, S4 <: AbstractArray{<: Real}}
    # POTENTIAL FOR SPEEDUP BY SPLITTING INTO DUAL AND REAL PART AND USE BLAS
    θ1 = zeros(eltype(XC), n_par.nm, n_par.nk, n_par.ny)
    for j  = 1:length(XC)
        θ1[compressionIndexes[j]] = copy(XC[j])
    end
    @views for yy = 1:n_par.ny
        θ1[:,:,yy] = IDC[1]*θ1[:,:,yy]*DC[2]
    end
    @views for mm = 1:n_par.nm
        θ1[mm,:,:] = θ1[mm,:,:]*DC[3]
    end
    θ = vec(θ1)
    return θ
end

# TODO: GENERALIZE
function compress(compressionIndexes::AbstractArray{S1}, XU::AbstractArray{S2},
                  DC::AbstractArray{S3}, IDC::AbstractArray{S4},
                  n_par) where {S1 <: AbstractArray{<: Real}, S2 <: Real, S3 <: AbstractArray{<: Real}, S4 <: AbstractArray{<: Real}}

    θ   = zeros(eltype(XU), length(compressionIndexes))
    XU2 = zeros(eltype(XU), size(XU))

    # preallocate mm and kk (subs from compressionIndexes)
    mm  = Vector{Int}(undef, length(compressionIndexes))
    kk  = Vector{Int}(undef, length(compressionIndexes))
    zz  = Vector{Int}(undef, length(compressionIndexes))
    # Use the fact that not all elements of the grid are used in the compression index
    for j = 1:length(compressionIndexes) # index to subs
        zz[j] = div(compressionIndexes[j], n_par.nm * n_par.nk) + 1
        kk[j] = div(compressionIndexes[j] - (zz[j] - 1) * n_par.nm * n_par.nk, n_par.nk) + 1
        mm[j] = compressionIndexes[j] - (zz[j] - 1) * n_par.nm * n_par.nk - (kk[j] - 1) * n_par.nk
    end
    # unique(mm) unique(kk) identify the grid elements retained in the
    # discrete cosine transformation

    # Eliminate unused rows/columns from the transformation matrix
    KK   = unique(kk)
    MM   = unique(mm)
    dc1  = DC[1][MM, :]
    dc2  = DC[2][KK, :]

    # Perform the DCT
    # TODO: add @simd?
    @inbounds @views begin
        for mmm in MM
           XU2[mmm, KK, :] .= (dc2 * XU[mmm, :, :]) * IDC[3]
        end
        for yy in unique(zz)
           XU2[MM, KK, yy] .= (dc1 * XU2[:, KK, yy]) # * idc2
        end

        for j = 1:length(compressionIndexes)
            θ[j] = XU2[compressionIndexes[j]]
        end
    end
    return θ
end
