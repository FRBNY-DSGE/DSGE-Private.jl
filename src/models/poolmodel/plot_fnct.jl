using StatsBase

# This script temporarily holds functions for replicating the
# Dynamic Prediction Pools paper before being later refactored
# once the functionality of the code is made more apparent
# e.g. we might add these functions to the forecast folder

function heatmap_posterior_λ_evolution(m::PoolModel{S}, datevec::Vector{Date}, λmat::Matrix{S},
                                    λweight::Matrix{S} = Matrix{Float64}(undef,0,0);
                                    bins::Symbol = :freedman_diaconis) where S<:AbstractFloat
    # Set weights to one if empty
    if isempty(λweight)
        λweight = ones(size(λmat))
    end

    # Check λmat and weights are same size
    if size(λmat) != size(λweight)
        error("λmat must be the same size as weights")
    end

    # Create inputs to histogram2d
    T = size(λmat,2)
    stack_timemat = Matrix{Date}(undef,size(λmat))
    for i in 1:T
        stack_timemat[:,i] .= datevec[i]
    end
    stack_timemat = vec(stack_timemat)
    stack_λmat    = vec(λmat)

    # Plot in 3D
    return histogram2d(stack_timemat, stack_λmat, weights = StatsBase.Weights(vec(λweight)))
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

function surface_posterior_λ_evolution(datevec::Vector{Date}, λmat::Matrix{S},
                                    λweight::Matrix{S} = Matrix{Float64}(undef,0,0);
                                    bins::Symbol = :freedman_diaconis) where S<:AbstractFloat
    # Set weights to one if empty
    if isempty(λweight)
        λweight = ones(size(λmat))
    end

    # Check λmat and weights are same size
    if size(λmat) != size(λweight)
        error("λmat must be the same size as weights")
    end

    # Create x, y, and z inputs to surf
    λ_edges = Dict{Int64,Vector{Float64}}() # x, location along [0,1] axis
    dates = Dict{Int64,Vector{Date}}() # y, vector of repeated values for the date
    λ_plotwts = Dict{Int64,Vector{Float64}}() # z, height of histogram
    t_nbins = Dict{Int64,Int64}()
    T = length(datevec)
    λ_modes = Vector{Float64}(undef,T)
    n_elem = 0 # number of elements
    for t in 1:T
        # t_nbins[t] = ceil(1 / freedman_diaconis(vec(λmat[:,t])))
        # bin_width = 1 / t_nbins[t]
        tmp = fit(Histogram, λmat[:,t], StatsBase.Weights(λweight[:,t]); nbins = 20)
        t_nbins[t] = length(tmp.weights)
        n_elem += t_nbins[t]
        tmp_edges = collect(tmp.edges[1])
        λ_edges[t] = diff(tmp_edges) + tmp_edges[1:end-1]
        dates[t] = repeat([datevec[t]], t_nbins[t])
        λ_plotwts[t] = tmp.weights ./ sum(tmp.weights)
        ind = argmax(λ_plotwts[t])
        λ_modes[t] = λ_edges[t][ind]
    end

    # Insert inputs into vectors for plotting
    X = Vector{Float64}(undef, n_elem)
    Z = Vector{Float64}(undef, n_elem)
    i = 0
    for t in 1:T
        X[i+1:i+t_nbins[t]] = λ_edges[t]
        Z[i+1:i+t_nbins[t]] = λ_plotwts[t]
        i += t_nbins[t]
    end
    Y = date_to_floats(datevec, length(λ_edges[1]))

    # Plot in 3D
    return surface(X, Y, Z, size = (800,600), camera = (50,60)), λ_edges, dates, λ_plotwts, λ_modes
end

####################
# Helper functions
####################

@inline function freedman_diaconis(data::Vector{S}) where S<:AbstractFloat
    return 2 * iqr(data) / length(data)^(1. / 3)
end

@inline function scott(data::Vector{S}) where S<:AbstractFloat
    return 3.5 * Statistics.std(data) * length(data)^(-1/3)
end

# function date_to_floats(datevec::Vector{Date}, reps::Int64)
#     date_floats = zeros(reps, length(datevec))
#     tofloats = Dict{String,Float64}("03" => .25, "06" => .5, "09" => .75, "12" => 0.)
#     for j = 1:length(datevec)
#         tmp = string(datevec[j])
#         s = tmp[6:7]
#         num = parse(Float64,tmp[1:4]) + tofloats[s]
#         date_floats[:,j] .= num
#     end
#     return date_floats
# end
