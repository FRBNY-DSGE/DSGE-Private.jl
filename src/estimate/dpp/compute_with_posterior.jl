# This script temporarily holds functions related to any posterior computations
# done after estimating a PoolModel. For example,
# sample_λ constructs pdfs of λ by sampling from the
# the joint density of (λ,θ) in the sense that,
# given the posterior of θ, which is not conditional on θ,
# we are then sampling from the density of p(λ|θ), hence
# the resulting pair (λ,θ) will be a sample from the joint density.
# If we have lots of samples of θ, this will be a good
# approximation of the joint density.

"""
```
sample_λ(m, θs, T = -1; parallel = true) where T<:AbstractFloat
```

Computes and samples from the conditional density p(λ_t|θ, I_t, P) for
particle in `θs`, which represents the posterior distribution.

### Inputs

- `m::PoolModel{T}`: `PoolModel` object
- `θs::Matrix{T}`: matrix of particles representing posterior distribution of θ
- `T::Int64`: final period for tempered particle filter

### Keyword Argument

- `parallel::Bool`: use parallel computing to compute and sample λ

### Outputs

- `λ_sample::Vector{Float64}`: sample of λs; together with (θ,λ) represents a joint density

```
"""
function sample_λ(m::PoolModel{T}, θs::Matrix{T}, T::Int64 = -1;
                  parallel::Bool = true) where T<:AbstractFloat
    # Check size and orientation of θs is correct: assume parameter_num x particle_num
    if length(m.parameters) != size(θs,1)
        error("number of parameters in PoolModel do not match number of parameters in matrix of posterior draws of θ")
    end

    # Initialize necessary objects
    λ_sample = parallel ? SharedVector{Float64}(undef, size(θs,2)) : Vector{Float64}(undef, size(θs,2))
    if parallel
        θs_share = SharedArray(θs)
    end
    if T == -1
        T = get_periods(m) # No period provided, so default to entire length of loglhs in PoolModel
    end

    # Sample from p(λ|θ, I_t^P, P) for each θ in posterior
    if parallel
        @distributed for i in 1:size(θs_share,2)
            m.update!(θs_share[:,i])
            tuning = deepcopy(get_setting(m, :tuning)) # avoid changing settings of m
            tuning[:get_t_particle_dist] = true
            tuning[:allout] = false
            ~, λ_particles, λ_weights = DSGE.filter(m, zeros(1,T); tuning = tuning)
            λ_sample[i] = DSGE.sample(λ_particles, DSGE.Weights(λ_weights))
        end
        λ_sample = Array(λ_sample)
    else
        for i in 1:size(θs,2)
            m.update!(θs[:,i])
            tuning = deepcopy(get_setting(m, :tuning)) # avoid changing settings of m
            tuning[:get_t_particle_dist] = true
            tuning[:allout] = false
            ~, λ_particles, λ_weights = DSGE.filter(m, zeros(1,T); tuning = tuning)
            λ_sample[i] = DSGE.sample(λ_particles, DSGE.Weights(λ_weights))
        end
    end

    return λ_sample
end

"""
```
compute_Eλ(m, λmat, weights = []; current_period = true, parallel = true) where T<:AbstractFloat
```

Computes and samples from the conditional density p(λ_t|θ, I_t, P) for
particle in `θs`, which represents the posterior distribution.

### Inputs

- `m::PoolModel{T}`: `PoolModel` object
- `λmat::Vector{T}`: vector of particles of λ samples from (θ,λ) joint distribution
- `weights::Vector{T}`: weights of λ particles, defaults to equal weights

### Keyword Argument

- `current_period::Bool`: compute Eλ for current period t
- `parallel::Bool`: use parallel computing to compute and sample λ
- `get_dpp_pred_dens::Bool`: compute predictive densities according to dynamic prediction pools

### Outputs

- `λhat_tplush::Float64`: E[λ_{t+h|t} | I_t^P, P]
- `λhat_t::Float64`: E[λ_{t|t} | I_t^P, P]

```
"""
function compute_Eλ(m::PoolModel{T}, λmat::Vector{T}, weights::Vector{T} = Vector{Float64}(undef,0);
                    current_period::Bool = true, parallel::Bool = false) where T<:AbstractFloat

    # Set up
    if isempty(weights)
        weights = ones(length(λmat)) # assume equal weights
    end
    λ_mat = parallel ? SharedArray(λmat) : copy(λmat) # so we don't alter this matrix in place
    n_particles = length(λ_mat)
    h = get_forecast_horizon(m)
    λhat_t = if current_period mean(λ_mat .* weights) end # compute expected lambda in current period t

    # Push forward states and compute mean
    if parallel
        λhat_tplush = @sync @distributed (+) for i in 1:n_particles
            ϵ = rand(get_F_ϵ(pm), h)
            for j in 1:h
                λ_mat[i] = get_Φ(pm)([λ_mat[i]; 1 - λ_mat[i]], [ϵ[j]])
            end
        end
        λhat_tplush /= n_particles
    else
        for i in 1:n_particles
            ϵ = rand(get_F_ϵ(pm), h) # h is small, so this is not expensive to loop through
            for j in 1:h
                λ_mat[i] = get_Φ(pm)([λ_mat[i]; 1 - λ_mat[i]], [ϵ[j]])
            end
        end
        λhat_tplush = mean(λ_mat .* weights)
    end

    if current_period
        return λhat_tplush, λhat_t
    else
        return λhat_tplush
    end
end
