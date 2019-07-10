using DSGE, DataFrames, JLD2
using Dates, Test

path = dirname(@__FILE__)

# Set up arguments
m = AnSchorfheide(testing = true)
m <= Setting(:date_forecast_start, quartertodate("2015-Q4"))

df, system, z0, P0 = jldopen("$path/../reference/forecast_args.jld2", "r") do file
    read(file, "df"), read(file, "system"), read(file, "z0"), read(file, "P0")
end

# Read expected output
exp_kal = jldopen("$path/../reference/filter_out.jld2", "r") do file
    read(file, "exp_kal")
end
df2 = DataFrame()
df2[:date] = df[:date]
df2[:obs_cpi] = df[:obs_cpi]
df2[:obs_gdp] = df[:obs_gdp]
df2[:obs_nominalrate] = df[:obs_nominalrate]

# Without providing z0 and P0
@testset "Check Kalman filter outputs without initializing state/state-covariance" begin
    kal = DSGE.filter(m, df, system)
    for out in fieldnames(typeof(kal))
        expect = exp_kal[out]
        actual = kal[out]

        if ndims(expect) == 0
            @test expect ≈ actual
        else
            @test @test_matrix_approx_eq(expect, actual)
        end
    end
end

# Providing z0 and P0
@testset "Check Kalman filter outputs initializing state/state-covariance" begin
    kal = DSGE.filter(m, df, system, z0, P0)
    for out in fieldnames(typeof(kal))
        expect = exp_kal[out]
        actual = kal[out]

        if ndims(expect) == 0
            @test expect ≈ actual
        else
            @test @test_matrix_approx_eq(expect, actual)
        end
    end
end


############################################################
# PoolModel tests
############################################################
# PROVIDE ONLY ONE PERIOD OF DATA SO IT RUNS JUST ONE PERIOD
# Set up arguments
SWFF = SmetsWoutersFF()
m = PoolModel() # to be completed
# m <= Setting(:date_forecast_start, quartertodate("2015-Q4"))

statespace, distributions, z0 = jldopen("$path/../reference/forecast_args_pool.jld2", "r") do file
    read(file, "df"), read(file, "statespace"), read(file, "distributions"), read(file, "z0")
end
update_statespace!(m, statespace)
update_distributions!(m, distributions)

# Read expected output
exp_pool, seed_num = jldopen("$path/../reference/filter_out_pool.jld2", "r") do file
    read(file, "exp_pool"), read(file, "seed_num")
end
Random.seed(seed_num)



# Without providing z0 and P0
@testset "Check TPF filter outputs without initializing state/state-covariance" begin
    tpf_sum, tpf_cond, tpf_time = DSGE.filter(m)
    @test tpf_sum ≈ exp_pool["tfp_sum"]
    @test tpf_cond ≈ exp_pool["tfp_cond"]
    @test tpf_time ≈ exp_pool["tfp_time"]
end

# Providing z0 and P0
Random.seed!(seed_num)
@assert draw_prior(m) == z0 # enforce we have the right seed number
Random.seed!(seed_num)
@testset "Check TPF filter outputs initializing state/state-covariance" begin
    tpf_sum, tpf_cond, tpf_time = DSGE.filter(m, z0)
    @test tpf_sum ≈ exp_pool["tfp_sum"]
    @test tpf_cond ≈ exp_pool["tfp_cond"]
    @test tpf_time ≈ exp_pool["tfp_time"]
end



nothing
