using DSGE
using ModelConstructors
using StateSpaceRoutines
using HDF5
using Random
using CSV
using Dates
using DataFrames
#include("../../../../../data/dsge_data_dir/dsgejl/michael/proc/includeall.jl")
path = dirname(@__FILE__)
writing_output = false
file_path = "/data/dsge_data_dir/dsgejl/michael/proc/includeall.jl"
include(file_path)

#optimizer_config = :csminwel
#optimizer_config = :lbfgs
optimizer_config = :trust_region_newton
#optimizer_config = :pso
println(optimizer_config)

#custom_settings = [Setting(:date_forecast_start, quartertodate("2015-Q4"))]
#m = AnSchorfheide(custom_settings = custom_settings, testing = true)
#m = DSSW()
m = Model1002("ss10")


#load data
#file = "$path/../reference/optimize_in.h5"
#x0   = h5read(file, "params")
#data = h5read(file, "data")'

#test file
#params_test = deepcopy(x0)
#data_test   = Matrix{Float64}(data)

m_old_vint = "190829"
m_old_fcast_date = Date(2019, 09, 30)
usual_settings!(m, m_old_vint, fcast_date = m_old_fcast_date)

old_data = df_to_matrix(m, load_data(m))

#file = "$path/../reference/vecm2007_data_log.csv"
#file2 = "$path/../reference/vecm2007_cointdata_log.csv"

#=
# Read in matlab dataset
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
 
     return data
end
=#



#data = construct_data()

#file = "$path/../reference/optimize_out.h5"
#minimizer  = h5read(file, "minimizer")
#minimum    = h5read(file, "minimum")
#H_expected = h5read(file, "H")
 
# See src/estimate/estimate.jl
#DSGE.update!(m, x0)
n_iterations = 1000


# Set seed for reproducibility
#Random.seed!(42)

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

#start timer
start_time = time()
out, H, callback_data = optimize!(m, old_data; method = optimizer_config, iterations = n_iterations, show_trace = true)
seconds = time() - start_time

println(out)
println(seconds)
