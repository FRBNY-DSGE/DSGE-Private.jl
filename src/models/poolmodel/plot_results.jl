fn = dirname(@__FILE__)

using OrderedCollections, Plots, FileIO
# This script produces a PDF of plots replicating the main figures of
# the DPP paper, with side by side plots with old and corrected predictive densities.

savefile_vec = ["analyze_realtime_estimation_preddens=wrong805orig904.jld2",
                "analyze_realtime_estimation_preddens=right805orig904.jld2"]
#                 "analyze_realtime_estimation_preddens=right805current904.jld2"]

# Create vector of plots, properly ordered
plotvec = Matrix{Plots.Plot}(undef,length(savefile_vec),4)
for (i,sf) in enumerate(savefile_vec)
    jlddata = load("$(fn)/save/output_data/poolmodel/ss0/estimate/work/$(sf)")
    plotvec[i,1] = jlddata["logscores_nodpp"]
    plotvec[i,2] = jlddata["logscores_dpp"]
#    plotvec[i,3] = jlddata["heatmap"]
    plotvec[i,3] = jlddata["posterior_rho"]
end
plotvec = vec(plotvec)
nrows = ceil(length(plotvec) / length(savefile_vec))

# Make joint plot to be saved as a PDF
# "Posterior distribution of λ over time (wrong)", "Posterior distribution of λ over time (corrected)",
titles = ["Log scores comparison: SWFF vs SWπ (wrong)", "Log scores comparison: SWFF vs SWπ (corrected)",
          "Log scores comparison over time (wrong)", "Log scores comparison over time (corrected)",
          "Prior and end-of-sample posterior for ρ ∼ U[0,1] (wrong)", "Prior and end-of-sample posterior for ρ ∼ U[0,1] (corrected)"]
for (i,p) in enumerate(plotvec)
    p.attr[:plot_title] = titles[i]
end

plotroot = figurespath(pm, "estimate")

lastplot = plot(plotvec..., layout = grid(nrows, length(savefile_vec)), size = (650, 775), titlefontsize = 10)
fn = joinpath(plotroot, "ReplicateDPP.pdf")
Plots.savefig(lastpot, fn)
println("Done with saving final plot")
