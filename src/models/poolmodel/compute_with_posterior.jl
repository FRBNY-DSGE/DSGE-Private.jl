using SharedArrays
"""
```
update_λ!(λmat, m, pred_dens, θs, T = -1; parallel = true) where S<:AbstractFloat
```

Computes and samples from the conditional density p(λ_t|θ, I_t, P) for
particle in `θs`, which represents the posterior distribution.

### Inputs

- `λmat`: matrix to populate with samples of λ from posterior distribution of θ, assumed to
    have dimensions n_particles × n_periods
- `m::PoolModel{S}`: `PoolModel` object
- `pred_dens`: n_model × n_periods matrix of predictive densities for each underlying model
- `θs::Matrix{S}`: matrix of particles representing posterior distribution of θ
- `T::Int64`: final period for tempered particle filter

where `S<:AbstractFloat`.

### Keyword Argument

- `parallel::Bool`: use parallel computing to compute and sample draws of λ

### Outputs

- `λ_sample::Vector{Float64}`: sample of draws of λs; together with (θ,λ) represents a joint density

```
"""
function sample_λ(m::PoolModel{S}, pred_dens::Matrix{S},
                   θs::Matrix{S}, T::Int64 = -1;
                   parallel::Bool = false) where S<:AbstractFloat
    # Check size and orientation of θs is correct: assume particle_num x parameter_num
    if length(m.parameters) != size(θs,2)
        error("number of parameters in PoolModel do not match number of parameters in matrix of posterior draws of θ")
    end

    # Initialize necessary objects
    λ_sample = parallel ? SharedVector{Float64}(size(θs,1)) : Vector{Float64}(undef, size(θs,1))
    if parallel
        θs_share = SharedArray(θs)
    end
    if T <= 0
        T = size(pred_dens,2)
    end

    # Sample from p(λ|θ, I_t^P, P) for each θ in posterior
    if parallel
        @distributed for i in 1:size(θs_share,1)
            update!(m, vec(θs_share[i,:]))
            tuning = deepcopy(get_setting(m, :tuning)) # avoid changing settings of m
            tuning[:get_t_particle_dist] = true
            tuning[:allout] = false
            ~, λ_particles, λ_weights = DSGE.filter(m, pred_dens[:,T]; tuning = tuning)
            λ_sample[i] = DSGE.sample(λ_particles[1], DSGE.Weights(λ_weights[:,T]))
        end
        λ_sample = Array(λ_sample)
        # λmat[:,T] = Array(λ_sample)
    else
        for i in 1:size(θs,1)
            update!(m, vec(θs[i,:]))
            tuning = deepcopy(get_setting(m, :tuning)) # avoid changing settings of m
            tuning[:get_t_particle_dist] = true
            tuning[:allout] = false
            ~, λ_particles, λ_weights = DSGE.filter(m, pred_dens[:,T]); tuning = tuning)
            # λ_sample[i] = DSGE.sample(λ_particles, DSGE.Weights(λ_weights[:,T]))
            λmat[i,T] = DSGE.sample(λ_particles[1], DSGE.Weights(λ_weights[:,T]))
        end
    end

    return λ_sample
end

"""
```
compute_Eλ(m, h, λvec, weights = []; current_period = true, parallel = true) where T<:AbstractFloat
```

Computes and samples from the conditional density p(λ_t|θ, I_t, P) for
particle in `θs`, which represents the posterior distribution.

### Inputs

- `m::PoolModel{T}`: `PoolModel` object
- `h::Int64`: forecast horizon
- `λvec::Vector{T}`: vector of particles of λ samples from (θ,λ) joint distribution
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
function compute_Eλ(m::PoolModel{T}, h::Int64, λvec ::Vector{T}, weights::Vector{T} = Vector{Float64}(undef,0);
                    current_period::Bool = true, parallel::Bool = false) where T<:AbstractFloat

    # Set up
    if isempty(weights)
        weights = ones(length(λvec)) # assume equal weights
    end
    λ_vec = parallel ? SharedArray(λvec) : copy(λvec) # so we don't alter this vector in place
    n_particles = length(λ_vec)
    λhat_t = if current_period mean(λ_vec .* weights) end # compute expected lambda in current period t

    # Push forward states and compute mean
    Φ, F_ϵ, ~ = solve(pm)
    if parallel
        @sync @distributed for i in 1:n_particles
            ϵ = rand(F_ϵ, h)
            for j in 1:h
                λ_vec[i] = Φ([λ_vec[i]; 1 - λ_vec[i]], [ϵ[j]])[1]
            end
        end
    else
        for i in 1:n_particles
            ϵ = rand(F_ϵ, h) # h is small, so this is not expensive to loop through
            for j in 1:h
                λ_vec[i] = Φ([λ_vec[i]; 1 - λ_vec[i]], [ϵ[j]])[1]
            end
        end
    end
    λhat_tplush = mean(λ_vec .* weights)

    if current_period
        return λhat_tplush, λhat_t
    else
        return λhat_tplush
    end
end
