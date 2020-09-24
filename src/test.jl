#addprocs_frbny(32)
#ENV["juliafrbnymemory"] = "2G"
using DSGE, DelimitedFiles, JLD2, BenchmarkTools

m = GHLS()
#@load "model_MH.jld2" m
m <= Setting(:sampling_method, :SMC)
m <= Setting(:use_parallel_workers, true)
m <= Setting(:n_particles, 50000)
m <= Setting(:date_presample_start, quartertodate("1983-Q1"))
m <= Setting(:date_mainsample_start, quartertodate("1983-Q1"))
m <= Setting(:date_mainsample_end, quartertodate("2014-Q1"))

m <= Setting(:optimization_attempts, 5)
m <= Setting(:optimization_xtol, .0001)
m <= Setting(:optimization_ftol, .0001)
m <= Setting(:optimization_gtol, .0001)

# Reads in data and keeps only series we need
#data_raw = readdlm("glss_data.txt")
#data_matrix = data_raw
#data_matrix = data_raw[:, [1, 2, 3, 6, 7]]
#data_matrix = data_raw[:, [1, 2, 3, 6, 7]]
#t_data_matrix = convert(Matrix, data_matrix')
#we want the data with periods as columns
#estimate(m,t_data_matrix,verbose=:none)

ghls_vars = [:β, :π_bar, :gz, :ψ_L, :γ, :σ_L, :ϕ_p, :ϕ_w, :ϵ_p, :ϵ_w,
             :ap, :aw, :bw, :lamhp, :α, :δ, :ϕ_I, :σ_a, :ρ_R ,:γ_π ,:γ_x, :γ_g,
             :shrgy, :σ_Z, :ρ_g, :σ_g, :ρ_μ, :σ_μ, :ρ_η, :σ_η, :ρ_int,
             :σ_R, :ρ_elast, :σ_elast, :ρ_elastw, :σ_elastw, :e_y, :e_c,
             :e_i, :e_u, :e_u, :e_π, :e_R]
# Delete bw, lamhp and 2 unnecessary measurement errors
rm_inds = findall(x -> x == :bw || x == :lamhp || x == :e_u, ghls_vars)

props = readdlm("cholM.txt")
props = props[setdiff(1:end, rm_inds), setdiff(1:end, rm_inds)]

paraming = [i.key for i in m.parameters]

props = props * props'

ghls_vars = ghls_vars[setdiff(1:end, rm_inds)]

for i in 1:length(paraming)
    ind_kp = findall(x -> x == paraming[i], ghls_vars)
    tmp = props[i,:]
    props[i,:] = props[ind_kp,:]
    props[ind_kp, :] = tmp

    tmp2 = props[:,i]
    props[:,i] = props[:,ind_kp]
    props[:,ind_kp] = tmp2
end

@show "Running"

#m <= Setting(:calculate_hessian, false)
#m <= Setting(:recalculate_hessian, false)

#estimate(m,verbose=:high; proposal_covariance = props)
estimate(m, verbose=:none)
