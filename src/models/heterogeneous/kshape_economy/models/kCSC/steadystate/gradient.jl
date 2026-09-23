function gradient(v::AbstractVector)
    n = length(v)
    g = similar(v)

    g[1] = v[2] - v[1]
    g[end] = v[end] - v[end-1]
    for i in 2:n-1
        g[i] = (v[i+1] - v[i-1]) / 2
    end
    return g
end

    
