using DSGE
using ModelConstructors
using StateSpaceRoutines
using HDF5
using Random
using CSV
using Dates
using DataFrames
using Distributed

path = dirname(@__FILE__)
#writing_output = false


#===============config======================#
calculate_posterior_mode = true
calculate_hessian = true

save_output_posterior_mode = true
save_output_hessian = true
#===============configure optimizer=================#
#optimizer_config = :xnes
optimizer_config = :cmaes
#optimizer_config = :conjugate_gradient
#optimizer_config = :csminwel
#optimizer_config = :lbfgs
#optimizer_config = :trust_region_newton
#optimizer_config = :pso
println("running $(optimizer_config)...")

#===============DSGE-VECM configuration=============#
vecm_object = true
λ = Inf
lags = 4
horizon = 16



#load model
m = DSSW()

if vecm_object == true
    vecm = DSGE.DSGEVECM(m)
    DSGE.update!(vecm,
              shocks = collect(keys(m.exogenous_shocks)),
              observables = collect(keys(m.observables)),
              λ = λ,
              lags = lags)

end
#xx = Ref{Any}()



#load data
file = "$path/../reference/vecm2007_data_log.csv"
file2 = "$path/../reference/vecm2007_cointdata_log.csv"

function construct_data()
     # Load in US dataset #3
     df = DataFrame(CSV.File(file))
     df_coint = DataFrame(CSV.File(file2))
 
     # Add dates
     dates_vec = []
     start_date = Date(1954, 9, 30)
     n_obs = size(df, 1)
     push!(dates_vec, start_date)
     for i = 2:n_obs
         push!(dates_vec, DSGE.iterate_quarters(start_date, i-1))
     end
     dates_col = DataFrame(dates = dates_vec)
 
     # Return df_to_matrix output
     df_mat = Matrix(df[1:end-1, :])
     df_coint = Matrix(df_coint)
 
     data = hcat(df_mat, df_coint)
 
     return data'
end




data = construct_data()

#data = data'

#file = "$path/../reference/optimize_out.h5"
#minimizer  = h5read(file, "minimizer")
#minimum    = h5read(file, "minimum")
#H_expected = h5read(file, "H")
 
# See src/estimate/estimate.jl
#DSGE.update!(m, x0)
end
n_iterations = 1000

# Set seed for reproducibility
Random.seed!(42)

#=
x0 = Float64[]
for p in m.parameters
    lower = p.valuebounds[1]
    upper = p.valuebounds[2]
    push!(x0, lower + rand() * (upper - lower))
end

=#
# Update x0
#DSGE.update!(m, x0)

#=
ENV["frbnyjuliamemory"] = "8G"
n_workers = 3
addprocs_frbny(n_workers)
#@everywhere using DSGE, SMC, OrderedCollections, CMAEvolutionStrategy
@everywhere begin
    push!(LOAD_PATH, pwd())   # make sure workers can see the local module
    using DSGE, SMC, OrderedCollections, CMAEvolutionStrategy
end
=#


if calculate_posterior_mode == true
    start_time_optimizer = time()

    if vecm_object == false
        out, H, callback_data = optimize!(m, data; method = optimizer_config, iterations = n_iterations, show_trace = true)

    else
        out, H, callback_data = optimize!(vecm, Matrix(data); method = optimizer_config, iterations = n_iterations, show_trace = true)
    end
seconds_optimizer = time() - start_time_optimizer
    if save_posterior_mode == true

    #TODO: output posterior mode
    
    end



else
    if vecm_object == true
        #minimized_x = #read in 
    end
    #TODO: read in posterior mode from reference/xxx.h5
end
    
if calculate_hessian == true
    start_time_hessian = time()
    
    if vecm_object == false
        hessian, _ = hessian!(m, out.minimizer, data; toggle = true, verbose = :low) 
    else
        hessian, _ = hessian!(vecm, out.minimizer, Matrix(data); toggle = true, verbose = :low)        
    end

    if save_hessian == true
    #TODO: save hessian
#=
h5open(rawpath(m, "estimate","hessian.h5"),"w") do file
                 file["hessian"] = hessian
             end
=#
    end

    seconds_hessian = time() - seconds_hessian
end

println(out)
println("optimizer time: $(seconds_optimizer)")
println("hessian time: $(seconds_hessian)")
