# This script temporarily holds functions for replicating the
# Dynamic Prediction Pools paper before being later refactored
# once the functionality of the code is made more apparent
# e.g. we might add these functions to the forecast folder

function plot_posterior_λ_evolution(m::PoolModel{T}, λmat::Matrix{T},
                                    weights::Matrix{T} = Matrix{Float64}(undef,0,0);
                                    bins::Symbol = :freedman_diaconis) where T<:AbstractFloat
    # Check λmat and weights are same size
    if size(λmat) != size(weights)
        error("λmat must be the same size as weights")
    end

    # Set weights to one if empty
    if isempty(weights)
        weights = ones(size(λmat))
    end
    T = size(λmat,2)
    stack_time_mat = kron(Vector(1:T), ones(size(λmat,2))) # == vec(ones(size(λmat,2)) .* Vector(1:T)')
    stack_λmat     = vec(λmat)


    # Plot in 3D
    return histogram2d(stack_time_mat, stack_λmat; weights = weights)
end

function plot_posterior_hyperparameter(m::PoolModel, pc::ParticleCloud)
    weights = DSGE.get_weights(pc)
    θ_particles = DSGE.get_vals(pc)
    is_estim = falses(size(θ_particles,1))
    for (i,param) in enumerate(values(m.parameters))
        is_estim[i] = !param.fixed
    end
    rows = Vector(1:size(θ_particles,1))[is_estim]
    plots = Dict{Symbol,Any}()
    for (row,param) in zip(rows, values(m.parameters))
        plots[param.key] = histogram(θ_particles[row,:]; weights = weights[row,:])
        plot!(param.prior.value)
    end
    return plots
end

####################
# Helper functions
####################

@inline function freedman_diaconis(data::Vector{T}) where T<:AbstractFloat = 2 * iqr(data) * length(data)^(-1/3)

@inline function scott(data::Vector{T}) where T<:AbstractFloat = 3.5 * Statistics.std(data) * length(data)^(-1/3)
