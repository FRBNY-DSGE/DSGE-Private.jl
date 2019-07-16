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

    # Compute histogram at each t
    T = size(λmat,2)
    hv = Vector{Histogram}(undef,T) # vector of histograms, i.e. hvec
    edgesL = Vector{Vector{Float64}}(undef,T) # vector holding edge vectors in each period, may vary in length, hence matrix
    for t = 1:T
        nbins = freedman_diaconis(λmat[:,t])
        hv[t] = fit(Histogram, λmat[:,t], weights[:,t]; nbins = nbins)

    end

    # Plot

end

function plot_posterior_hyperparameter(m::PoolModel)
    if m[:ρ].fixed
    end
    if m[:σ].fixed
    end
    if m[:μ].fixed
    end
end

####################
# Helper functions
####################

@inline function freedman_diaconis(data::Vector{T}) where T<:AbstractFloat = 2 * iqr(data) * length(data)^(-1/3)

@inline function scott(data::Vector{T}) where T<:AbstractFloat = 3.5 * Statistics.std(data) * length(data)^(-1/3)
