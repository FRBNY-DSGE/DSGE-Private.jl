"""
```
mutation(m, data, p, d, blocks_free, blocks_all, ϕ_n, ϕ_n1; c = 1., α = 1., old_data)
```

Execute one proposed move of the Metropolis-Hastings algorithm for a given parameter

### Arguments:
- `m::AbstractModel`: Model of type AbstractModel being estimated.
- `data::Matrix{Float64}`: Matrix of data
- `p::Particle`: Initial particle value
- `d::Distribution`: A distribution with μ = the weighted mean, and Σ = the weighted variance/covariance matrix
- `blocks_free::Vector{Vector{Int64}}`: A vector of index blocks, where the indices in each block corresponds to the ordering of free parameters only (e.g. all the indices will be ∈ 1:n_free_parameters)
- `blocks_all::Vector{Vector{Int64}}`: A vector of index blocks, where the indices in each block corresponds to the ordering of all parameters (e.g. all the indices will be in ∈ 1:n_para, free and fixed)
- `ϕ_n::Float64`: The current tempering factor
- `ϕ_n1::Float64`: The previous tempering factor

### Keyword Arguments:
- `c::Float64`: The scaling parameter for the proposed covariance matrix
- `α::Float64`: The mixing proportion
- `old_data::Matrix{Float64}`: The matrix of old data to be used in calculating the old_loglh, old_logpost in time tempering

### Outputs:

- `p::Particle`: An updated particle containing updated parameter values, log-likelihood, posterior, and acceptance indicator.

"""
function mutation(m::AbstractModel, data::Matrix{Float64}, p::Particle, d::Distribution,
                  blocks_free::Vector{Vector{Int64}}, blocks_all::Vector{Vector{Int64}},
                  ϕ_n::Float64, ϕ_n1::Float64; c::Float64 = 1., α::Float64 = 1.,
                  old_data::T = T(undef, size(data, 1), 0),
                  use_chand_recursion::Bool = false,
                  verbose::Symbol = :low) where {T <: AbstractMatrix}

    n_steps     = get_setting(m, :n_mh_steps_smc)
    n_free_para = length([!θ.fixed for θ in m.parameters])
    step_prob   = rand() # Draw initial step probability

    para = p.value
    like = p.loglh
    post_init = p.logpost
    like_prev = p.old_loglh # The likelihood evaluated at the old data (for time tempering)

    # Previous posterior needs to be updated (due to tempering)
    post = post_init #+ (ϕ_n - ϕ_n1) * like
    accept = 0.0

    for step in 1:n_steps
        for (block_f, block_a) in zip(blocks_free, blocks_all)

            # Index out the parameters corresponding to a given random block
            # And also create a distribution centered at the weighted mean
            # and with Σ corresponding to the same random block
            para_subset = para[block_a]
            d_subset    = MvNormal(d.μ[block_f], d.Σ.mat[block_f, block_f])

            para_draw, para_new_density, para_old_density = mvnormal_mixture_draw(para_subset,
                                                                                  d_subset; c = c, α = α)
            para_new = deepcopy(para)
            para_new[block_a] = para_draw

            lik0 = like
            pr0  = post

            q0 = α * exp(logpdf(MvNormal(para_draw,   c^2 * d_subset.Σ.mat), para_subset))
            q1 = α * exp(logpdf(MvNormal(para_subset, c^2 * d_subset.Σ.mat), para_draw))

            ind_pdf = 1.0

            for i=1:length(block_a)
                sigi  = sqrt(d_subset.Σ.mat[i,i])
                zstat = (para_subset[i] - para_draw[i]) / sigi
                ind_pdf = ind_pdf / (sigi * sqrt(2.0 * π)) * exp(-0.5 * zstat ^ 2)
            end

            q0 = q0 + (1. - α) / 2. * ind_pdf
            q1 = q1 + (1. - α) / 2. * ind_pdf

            q0 = q0 + (1. - α) / 2. * exp(logpdf(MvNormal(d_subset.μ, c^2 * d_subset.Σ.mat),
                                                 para_subset))
            q1 = q1 + (1. - α) / 2. * exp(logpdf(MvNormal(d_subset.μ, c^2 * d_subset.Σ.mat),
                                                 para_draw))
            q0 = log(q0)
            q1 = log(q1)

            like_new      = -Inf
            prior_new     = -Inf
            like_old_data = -Inf

            try
                update!(m, para_new)
                prior_new = prior(m)
                from_steady_state = false
                try
                    #if the settings there, use the settings
                    from_steady_state = get_setting(m, :from_steady_state)
                catch
                    nothing #if the settings not there, keep it as false
                end
                if from_steady_state
                    ψ_penalty_new = get_setting(m, :ψ_penalty)
                    ψ_likelihood_new = get_setting(m, :ψ_likelihood)
                    ψ_penalty_old = get_setting(m, :ψ_penalty_old)
                    ψ_likelihood_old = get_setting(m, :ψ_likelihood_old)
                    target_vars = get_setting(m, :target_vars)
                    target_σt   = get_setting(m, :target_σt)
                    targets     = get_setting(m, :targets)

                    system = try
                        compute_system(m, verbose = verbose)
                    catch err
                        if catch_errors && (isa(err, GensysError) || isa(err, KleinError))
                            return -Inf
                        else
                            rethrow(err)
                        end
                    end
                    try
                        lik = chand_recursion(data, system[:TTT], system[:RRR], system[:CCC],
                                              system[:QQ], system[:ZZ], system[:DD], system[:EE],
                                              allout= true, Nt0 = n_presample_periods(m))[1]
                    catch err
                        if catchₑrrors && isa(err, DomainError)
                            @warn "Log of incremental likelihood is negative; returning -Inf"
                            return -Inf
                        else
                            rethrow(err)
                        end
                    end
                        like_new = ψ_penalty_new * penalty_function(m, target_vars, targets, target_σt) + ψ_likelihood_new * lik
                else
                    like_new = likelihood(m, data; sampler = true,
                                          use_chand_recursion = use_chand_recursion,
                                          verbose = verbose)
                end
                if like_new == -Inf
                    prior_new = like_old_data = -Inf
                end

                if from_steady_state
                    like_old_data = ψ_penalty_old * penalty_function(m, target_vars, targets, target_σt) + ψ_likelihood_old * lik # chand_recursion(data, system[:TTT], system[:RRR], system[:CCC], system[:QQ], system[:ZZ], system[:DD], system[:DD], system[:EE], allout= true, Nt0 = n_presample_periods(m))[1]
                else
                    like_old_data = isempty(old_data) ? 0. : likelihood(m, old_data; sampler = true,
                                                                        use_chand_recursion = use_chand_recursion,
                                                                        verbose = verbose)
                end
            catch err
                if isa(err, ParamBoundsError)
                    prior_new = like_new = like_old_data = -Inf
                elseif isa(err, PosDefException) || isa(err, SingularException)
                    prior_new = -Inf
                    like_new = -Inf
                    like_old_data = -Inf
                elseif isa(err, LinearAlgebra.LAPACKException)
                    prior_new = -Inf
                    like_new = -Inf
                    like_old_data = -Inf
                else
                    rethrow(err)
                end
            end

            if (q0 == Inf && q1 == Inf)
                q0 = 0.0
            end

            η = exp(ϕ_n * (like_new - lik0) + prior_new - pr0 + q0 - q1)

            if step_prob < η
                para      = para_new
                like      = like_new
                post      = prior_new
                like_prev = like_old_data
                accept   += length(block_a)
            end
            # Draw again for next step
            step_prob = rand()
        end
    end
    update_mutation!(p, para, like, post, like_prev, accept / n_free_para)
    return p
end
