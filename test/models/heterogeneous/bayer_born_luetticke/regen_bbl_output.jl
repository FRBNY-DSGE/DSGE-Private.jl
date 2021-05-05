using Pkg, JLD2, Random
hankestim_path = joinpath(dirname(@__FILE__), "..", "..", "..", "..", "..", "HANKEstim")
refpath = joinpath(dirname(@__FILE__), "..", "..", "..", "reference")
jld2_version = "0.1.11" # change to whichever version you want to use. For legacy reasons, the FRBNY DSGE team is using v0.1.11
Pkg.activate(hankestim_path)
Pkg.pin(name = "JLD2", version = jld2_version)
push!(LOAD_PATH, joinpath(hankestim_path, "src"))
using HANKEstim, ForwardDiff

# initialize model parameters
m_par = ModelParameters()
Random.seed!(1793)
sr = compute_steadystate(m_par)
include("save_sr_regen_bbl_output.jl")
lr = linearize_full_model(sr, m_par)
JLD2.jldopen(joinpath(refpath, "bayer_born_luetticke_original_linearize_seed1793.jld2"), true, true, true, IOStream) do file
    write(file, "gx", lr.State2Control)
    write(file, "hx", lr.LOMstate)
    write(file, "A", lr.A)
    write(file, "B", lr.B)
end

# Now run a linearization of just the block for aggregate equations
gx_aggr, hx_aggr, _, _, A_aggr, B_aggr = HANKEstim.SGU_estim(sr.XSSaggr, copy(lr.A), copy(lr.B),
                                                             sr.m_par, sr.n_par, sr.indexes,
                                                             sr.indexes_aggr, sr.distrSS; estim = true)
JLD2.jldopen(joinpath(refpath, "bayer_born_luetticke_original_linearize_aggr_update_seed1793.jld2"), true, true, true, IOStream) do file
    write(file, "gx", gx_aggr)
    write(file, "hx", hx_aggr)
    write(file, "A", A_aggr)
    write(file, "B", B_aggr)
end

# Finally, compute the selection matrix and shock variance-covariance matrix
include(joinpath(hankestim_path, "src/3_Model/input_aggregate_names.jl"))
e_set = EstimationSettings(mode_start_file = joinpath(hankestim_path, "src", "7_Saves", "HANKXplus_postmean.jld2"))
observed_vars = [e_set.observed_vars_input[e_set.growth_rate_select]; e_set.observed_vars_input[.!e_set.growth_rate_select]]
nobservables = length(observed_vars)
H_sel = zeros(nobservables, sr.n_par.nstates + sr.n_par.ncontrols)
for i in eachindex(observed_vars)
    H_sel[i,   getfield(sr.indexes, (observed_vars[i]))]  = 1.0
end

SCov = zeros(sr.n_par.nstates, sr.n_par.nstates)
σ_vals = [i * .01 for i in 1:9]
for (j, i) in enumerate(e_set.shock_names)
    SCov[getfield(sr.indexes, i), getfield(sr.indexes, i)] = σ_vals[j]^2
end
JLD2.jldopen(joinpath(refpath, "bayer_born_luetticke_original_measurement.jld2"), true, true, true, IOStream) do file
    write(file, "SCov", SCov)
    write(file, "H_sel", H_sel)
end
