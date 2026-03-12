function sub2ind(siz, v1, v2, varargin...)
    """
    SUB2IND Linear index from multiple subscripts.
    
    sub2ind is used to determine the equivalent single index
    corresponding to a given set of subscript values.
    
    ind = sub2ind(siz, i, j) returns the linear index equivalent to the
    row and column subscripts in the arrays i and j for a matrix of
    size siz.
    
    ind = sub2ind(siz, i1, i2, ..., iN) returns the linear index
    equivalent to the N subscripts in the arrays i1, i2, ..., iN for an
    array of size siz.
    
    i1, i2, ..., iN must have the same size, and ind will have the same size
    as i1, i2, ..., iN. For an array A, if ind = sub2ind(size(A), i1, ..., iN),
    then A[ind[k]] = A[i1[k], ..., iN[k]] for all k.
    """
    
    siz = Float64.(siz)
    lensiz = length(siz)
    
    if !all(isreal, siz) || ndims(siz) > 1 || lensiz < 2
        error("Invalid size vector")
    end
    
    numOfIndInput = 1 + 1 + length(varargin)  # v1 + v2 + varargin
    
    if numOfIndInput > 0
        # Adjust siz if needed
        if lensiz < numOfIndInput
            # Adjust for trailing singleton dimensions
            siz = vcat(siz, ones(numOfIndInput - lensiz))
        elseif lensiz > numOfIndInput
            # Adjust for linear indexing on last element
            siz = vcat(siz[1:numOfIndInput-1], prod(siz[numOfIndInput:end]))
        end
        
        # Check index arguments
        set_size = false
        s = nothing
        
        for i in 1:numOfIndInput
            if i == 1
                v = v1
            elseif i == 2
                v = v2
            else
                ind = i - 2
                v = varargin[ind]
            end
            
            if !set_size
                if !isscalar(v)
                    s = size(v)
                    set_size = true
                end
            else
                if !isscalar(v) && s != size(v)
                    # Verify sizes of subscripts
                    error("Subscript vectors must be the same size")
                end
            end
            
            if !all(isreal, v)
                # Verify subscripts are real
                error("Subscripts must be real")
            end
            
            siz_i = siz[i]
            if !isempty(v) && (any(isnan, v) || minimum(v) < 1 || maximum(v) > siz_i)
                # Verify subscripts are within range
                error("Index out of range")
            end
        end
    end
    
    # Compute linear indices
    if numOfIndInput <= 1
        ndx = Float64.(v1)
    else
        k = siz[1]
        ndx = Float64.(v1) .+ (Float64.(v2) .- 1) .* k
        k = k * siz[2]
        
        for i in 3:numOfIndInput
            ind = i - 2
            ndx = ndx .+ (Float64.(varargin[ind]) .- 1) .* k
            k = k * siz[i]
        end
    end
    
    return ndx
end