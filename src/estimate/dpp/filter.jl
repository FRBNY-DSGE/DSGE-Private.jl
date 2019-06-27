"""
```
Filter model weights for dynamic prediction pool.
```
"""
function filter(mvec::Vector{AbstractModel}, loglh::Matrix{T},
                data::Matrix{T}; params::Dict{Symbol,T} = Dict{Symbol,T}())
                where {T<:AbstractFloat}

    ######################################################################
    # Step 1: Set up
    ######################################################################
    if isempty(params)
        # Default to static pool
        params[:μ] = 0
        params[:σ] = 1
        params[:ρ] = 1
    end
    F_ϵ = Distributions.Normal(params[:μ], params[:σ])
    F_u = Distributions.Normal(0, 0) # no measurement error
    s_init = cdf(Distributions.Normal(0,1), rand(F_ϵ, 1))

    ######################################################################
    # Step 2: Iterate
    ######################################################################
    T = size(data, 2)


    for t = 1:T
        sum loglik, loglhconditional, ~ = tempered_particle_filter(data, x -> transition(x, params),
                                 y -> measurement(y, dropdims(loglh[:,t]; dims = 2)),
                                 F_ϵ, F_u, s_init)
    end


end
