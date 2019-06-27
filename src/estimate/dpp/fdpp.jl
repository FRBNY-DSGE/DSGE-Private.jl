# Current pseudocode
# a function dpp takes dictionary of ParticleCloud objects (see smc.jl and particle.jl) with the particular model as the key, data as well
# settings: estimate models (take as a vector) or a symbol of :all,
# generate forecasts,
# already generated predictive densities/forecasts (from ParticleCloud)
# transition equation
# measurement equation, takes predictive densities from each model object, the loglh;
# feed tpf these things

"""
```
To be done
```

"""
function dpp(mvec::Vector{AbstractModel},
             mparticles::Vector{ParticleCloud} = Vector{ParticleCloud}(),
             data::Matrix{T}; params::Dict{Symbol,T} = Dict{Symbol,T}(),
             verbose::Symbol = :low, estimate::Bool = false, run_test::Bool = false,
             method::Symbol = :SMC, estimate_kwargs...) where {T<:AbstractFloat}

if estimate == true || isempty(mparticles)
    # Estimate every model in the vector of models
    # NOTE TO SELF: this method only works for SMC
    mparticles = Vector{ParticleCloud}(size(mvec))
    for (i,m) in enumerate(mvec)
        mparticles[i] = estimate(m, data; method = method, estimate_kwargs...)
    end
end

# Step 1: Initialize state and extract data for estimation
if isempty(params)
    # Default to static pool
    params[:μ] = 0
    params[:σ] = 1
    params[:ρ] = 1
    x0 = rand(Normal(), 1)
    lambda0 = cdf(Normal(), x0)
else
    x0 = rand(Normal(params[:μ], params[:σ]), 1)
end

# Get log-likelihoods of each model
mloglh = zeros(length(mvec), size(data, 2)) # row=which model, col=period
for i = 1:size(data, 2)
    for (j,pc) in enumerate(mparticles)
        mloglh[j,i] = pc.particles[i].loglh
    end
end

# Step 2: Estimate with SMC


filter(mvec, mloglh, data)



end
