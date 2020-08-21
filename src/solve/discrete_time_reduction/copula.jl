"""
```
function copula(μ::AbstractArray{S, N}, quadrature_weights::AbstractArray{S, N} = Array{S, N}(undef, zeros(Int, N)...);
                method::InterpolationType = Gridded(Linear())) where {S <: Real, N <: Int}
```

creates an Interpolations object that is the copula for the discretized probability density function `μ`.
If `c = copula(μ, ...)` where `μ` is a `Matrix`, then `c(x, y)` maps the marginal `x` and marginal `y` cumulative distributions
to the cumulative distribution implied by `μ`.
"""
function copula(μ::AbstractArray{S, N}, quadrature_weights::AbstractArray{S, N} = Array{S, N}(undef, zeros(Int, N)...);
                method::InterpolationType = Gridded(Linear())) where {S <: Real, N <: Int}

    # Get overall CDF and CDFs of marginals
    Fμ         = cdf_quadrature(μ, quadrature_weights)
    marginals  = marginal_cdf_quadrature(μ, quadrature_weights)

    # Construct interpolation object via Gridded(Linear())
    return interpolate(marginals, marginals, method)
end

function _bayer_luetticke_mylinearinterpolate3(xgrd1::AbstractVector, xgrd2::AbstractVector,
    xgrd3::AbstractVector, ygrd::AbstractArray,
    xeval1::AbstractVector, xeval2::AbstractVector, xeval3::AbstractVector)

end

function marginal_cdf_quadrature(μ::AbstractArray{S, N}, quadrature_weights::AbstractArray{S, N} =
                                 Array{S, N}(undef, zeros(Int, N)...)) where {S <: Real, N <: Int}
    marg_cdfs = Tuple(Vector{S}(undef, n + 1) for n in size(μ))
    all_dims  = collect(1:N)

    if isempty(quadrature_weights)
        μ = μ .* quadrature_weights # Re-weight distribution to incorporate quadrature
    end

    for (i, marg_cdf) in enumerate(marg_cdfs)
        marg_cdf[1]     = 0. # set first index to zero
        i_dims          = setdiff(all_dims, i)
        if quadrature_weights
        marg_cdf[2:end] = cumsum(dropdims(sum(μ, dims = i_dims), dims = i_dims)) # sum to marginals, flatten to vector, cumsum for CDF
    end

    return marg_cdfs
end

function cdf_quadrature(μ::AbstractArray{S, N}, quadrature_weights::AbstractArray{S, N} =
                        Array{S, N}(undef, zeros(Int, N)...)) where {S <: Real, N <: Int}
    # Initialize CDF matrix as zeros. Slightly inefficient, but avoid complication of setting boundaries of CDF to zero
    Fμ_sz = size(μ) .+ 1
    Fμ    = zeros(S, size(μ) .+ 1) # CDF of μ
    inds  = CartesianIndices([2:i for i in 1:Fμ_sz])

    # Perform integration for CDF
    A     = @view Fμ[inds]                                 # portion of CDF to actually fill in
    A    .= isempty(quadrature_weights) ? μ : μ .* weights # multiply by quadrature weights so that cumsum works if needed
    for i in 1:N
        cumsum!(A, A, dims = i) # same as cumsum(cumsum(A, dims = 1), dims = 2) if μ is 2D
    end

    return Fμ
end


# The following function is for testing purposes.
# It has been ported from Bayer and Luetticke (2021) "Solving discrete time heterogeneous agent models with
# aggregate risk and many idiosyncratic states by perturbation", Quantitative Economics.
function _bayer_luetticke_mylinearinterpolate3(xgrd1::AbstractVector, xgrd2::AbstractVector,
    xgrd3::AbstractVector, ygrd::AbstractArray,
    xeval1::AbstractVector, xeval2::AbstractVector, xeval3::AbstractVector)
    ## Define variables ##
    yeval   = Array{eltype(xeval1),3}(undef,(length(xeval1),length(xeval2),length(xeval3)))
    n_xgrd1 = length(xgrd1)
    n_xgrd2 = length(xgrd2)
    n_xgrd3 = length(xgrd3)
    weight1r = Array{eltype(xeval1),1}(undef,length(xeval1))
    weight2r = Array{eltype(xeval2),1}(undef,length(xeval2))
    weight3r = Array{eltype(xeval3),1}(undef,length(xeval3))
    ind1    = Array{Int,1}(undef,length(xeval1))
    ind2    = Array{Int,1}(undef,length(xeval2))
    ind3    = Array{Int,1}(undef,length(xeval3))
    ## find left element of xeval1 on xgrd1
    for i in eachindex(xeval1)
        xi = xeval1[i]
        if xi .>= xgrd1[n_xgrd1-1]
            iL = n_xgrd1 - 1
        elseif xi .< xgrd1[2]
            iL = 1
        else
            iL      = locate(xi, xgrd1)
        end
        ind1[i]      = copy(iL)
        xL          = xgrd1[iL]
        weight1r[i]  = copy((xi .- xL)./ (xgrd1[iL.+1] .- xL))
    end

    ## find left element of xeval2 on xgrd2
    for i in eachindex(xeval2)
        xi = xeval2[i]
        if xi .>= xgrd2[n_xgrd2-1]
            iL = n_xgrd2 - 1
        elseif xi .< xgrd2[2]
            iL = 1
        else
            iL      = locate(xi, xgrd2)
        end
        ind2[i]      = copy(iL)
        xL           = xgrd2[iL]
        weight2r[i]  = copy((xi .- xL)./ (xgrd2[iL.+1] .- xL))
    end

    ## find left element of xeval3 on xgrd3
    for i in eachindex(xeval3)
        xi = xeval3[i]
        if xi .>= xgrd3[n_xgrd3-1]
            iL = n_xgrd3 - 1
        elseif xi .< xgrd3[2]
            iL = 1
        else
            iL      = locate(xi, xgrd3)
        end
        ind3[i]      = copy(iL)
        xL           = xgrd3[iL]
        weight3r[i]  = copy((xi .- xL)./ (xgrd3[iL.+1] .- xL))
    end
@inbounds  begin
    for l in eachindex(xeval3)
        i3  = ind3[l]
        w3r = weight3r[l]
        w3l = 1.0 - w3r
        for j in eachindex(xeval2)
            i2 =ind2[j]
            w2r = weight2r[j]
            w2l = 1.0 - w2r
            for k in eachindex(xeval1)
                i1  = ind1[k]
                w1r = weight1r[k]
                w1l = 1.0 - w1r
               yeval[k,j,l] =  w1l.*(w2l.*(w3l.*ygrd[i1,i2,i3]     + w3r.*ygrd[i1,i2,i3+1])    +
                                      w2r.*(w3l.*ygrd[i1,i2+1,i3]   + w3r.*ygrd[i1,i2+1,i3+1])) +
                                w1r.*(w2l.*(w3l.*ygrd[i1+1,i2,i3]   + w3r.*ygrd[i1+1,i2,i3+1])  +
                                      w2r.*(w3l.*ygrd[i1+1,i2+1,i3] + w3r.*ygrd[i1+1,i2+1,i3+1]))
            end
        end
    end
end
    return yeval
end
