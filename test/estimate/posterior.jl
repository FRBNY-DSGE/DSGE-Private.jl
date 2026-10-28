using BenchmarkTools

path = dirname(@__FILE__)
@testset "Check prior overloading" begin
    custom_settings = [Setting(:date_forecast_start, quartertodate("2015-Q4"))]
    m = AnSchorfheide(custom_settings = custom_settings, testing = true)
    priordensity = exp(DSGE.prior(m))
    dsgevar = DSGE.DSGEVAR(m, collect(keys(m.exogenous_shocks)))
    @test prior(m) == prior(m.parameters)
    @test 0 <= priordensity <= 1 # ensure prior density is a density
    @test prior(dsgevar) == prior(dsgevar.dsge) == prior(m.parameters)
end

@testset "Check DSGE likelihood and posterior calculations" begin
    custom_settings = [Setting(:date_forecast_start, quartertodate("2015-Q4"))]
    m = AnSchorfheide(custom_settings = custom_settings, testing = true)

    file = "$path/../reference/posterior.jld2"
    data = Matrix{Float64}(load(file, "data")')
    lh_expected = load(file, "lh_expected")
    post_expected = load(file, "post_expected")

    lh = likelihood(m, data)
    @test lh_expected ≈ lh

    post = DSGE.posterior(m, data)
    @test post_expected ≈ post

    x = map(α->α.value, m.parameters)
    post_at_start = DSGE.posterior!(m, x, data)
    @test post_expected ≈ post_at_start

    # Ensure if we are not evaluating at start vector, then we do not get the reference
    # posterior
    global y = x .+ 0.01
    post_not_at_start = DSGE.posterior!(m, y, data)
    ϵ = 1.0
    @test abs(post_at_start - post_not_at_start) > ϵ

    # Check keywords
    m.parameters[2].value = 3.
    @test !isinf(likelihood(m, data; sampler = false))
    @test isinf(likelihood(m, data; sampler = true))
end

@testset "Reuse one state-space system for time-tempered likelihoods" begin
    m = AnSchorfheide(testing = true)
    data = Matrix{Float64}(load(joinpath(@__DIR__, "../reference/posterior.jld2"), "data")')
    old_data = data[:, 1:div(size(data, 2), 2)]

    for use_chand_recursion in (false, true)
        expected_new = likelihood(m, data; use_chand_recursion = use_chand_recursion)
        expected_old = likelihood(m, old_data; use_chand_recursion = use_chand_recursion)
        shared_system = Ref{Any}(nothing)
        actual_new = likelihood(m, data; use_chand_recursion = use_chand_recursion,
                                system_cache = shared_system)
        @test !isnothing(shared_system[])
        cached = shared_system[]
        actual_old = likelihood(m, old_data; use_chand_recursion = use_chand_recursion,
                                system_cache = shared_system)
        @test shared_system[] === cached
        @test actual_new ≈ expected_new
        @test actual_old ≈ expected_old
    end

    m <= Setting(:auto_reject, true)
    shared_system = Ref{Any}(nothing)
    @test likelihood(m, data; system_cache = shared_system) == -Inf
    @test isnothing(shared_system[])
end

@testset "Likelihood constructs only observed independent regimes" begin
    m = AnSchorfheide(testing = true)
    m <= Setting(:regime_switching, true)
    m <= Setting(:n_regimes, 4)
    first_date = date_presample_start(m)
    m <= Setting(:date_zlb_start, first_date)
    m <= Setting(:regime_dates, Dict(1 => first_date,
                                     2 => DSGE.iterate_quarters(first_date, 6),
                                     3 => DSGE.iterate_quarters(first_date, 48),
                                     4 => DSGE.iterate_quarters(first_date, 60)))
    setup_regime_switching_inds!(m)
    data = Matrix{Float64}(load(joinpath(@__DIR__, "../reference/posterior.jld2"), "data")')[:, 1:24]
    full = compute_system(m)
    observed = compute_system(m; sample_end_date = DSGE.iterate_quarters(first_date, 23))

    @test length(full.transitions) == 4
    @test length(observed.transitions) == 2
    @test length(observed.measurements) == 2
    @test length(observed.pseudo_measurements) == 2
    for reg in 1:2, matrix in (:TTT, :RRR, :CCC, :QQ, :ZZ, :DD, :EE)
        @test observed[reg, matrix] == full[reg, matrix]
    end
    first_only = compute_system(m; sample_end_date = DSGE.iterate_quarters(first_date, 5))
    @test length(first_only.transitions) == 1
    for matrix in (:TTT, :RRR, :CCC, :QQ, :ZZ, :DD, :EE)
        @test first_only[1, matrix] == full[1, matrix]
    end
    @test length(compute_system(m; sample_end_date = DSGE.iterate_quarters(first_date, 65)).transitions) == 4

    cached = Ref{Any}(nothing)
    actual = likelihood(m, data; system_cache = cached)
    expected = likelihood(m, data; system_cache = Ref{Any}(full))
    @test length(cached[].transitions) == 2
    @test actual ≈ expected

    shorter_first = Ref{Any}(nothing)
    short_likelihood = likelihood(m, data[:, 1:6]; system_cache = shorter_first)
    @test short_likelihood ≈ likelihood(m, data[:, 1:6]; system_cache = Ref{Any}(full))
    @test length(shorter_first[].transitions) == 1
    @test likelihood(m, data; system_cache = shorter_first) ≈ actual
    @test length(shorter_first[].transitions) == 2
end

@testset "Likelihood skips the pseudo-measurement equations" begin
    data = Matrix{Float64}(load(joinpath(@__DIR__, "../reference/posterior.jld2"), "data")')
    for regime_switching in (false, true)
        m = AnSchorfheide(testing = true)
        if regime_switching
            m <= Setting(:regime_switching, true)
            m <= Setting(:n_regimes, 4)
            first_date = date_presample_start(m)
            m <= Setting(:date_zlb_start, first_date)
            m <= Setting(:regime_dates, Dict(1 => first_date,
                                             2 => DSGE.iterate_quarters(first_date, 6),
                                             3 => DSGE.iterate_quarters(first_date, 48),
                                             4 => DSGE.iterate_quarters(first_date, 60)))
            setup_regime_switching_inds!(m)
        end
        sample_end_date = DSGE.iterate_quarters(date_presample_start(m), size(data, 2) - 1)
        full = compute_system(m; sample_end_date = sample_end_date)
        no_pseudo = compute_system(m; sample_end_date = sample_end_date, compute_pseudo = false)
        systems = regime_switching ? [(System(full, reg), System(no_pseudo, reg)) for reg in 1:length(full.transitions)] :
            [(full, no_pseudo)]
        for (sys_full, sys_no_pseudo) in systems
            for matrix in (:TTT, :RRR, :CCC, :QQ, :ZZ, :DD, :EE)
                @test sys_no_pseudo[matrix] == sys_full[matrix]
            end
            @test size(sys_full[:ZZ_pseudo], 1) == n_pseudo_observables(m)
            @test size(sys_no_pseudo[:ZZ_pseudo], 1) == 0
        end

        # Bitwise equal to the likelihood from the full system
        lh = likelihood(m, data)
        @test lh == sum(DSGE.filter_likelihood(m, data, full; include_presample = false))

        # A likelihood call doesn't change the systems built afterwards from the same model
        after = compute_system(m)
        sys_after = regime_switching ? System(after, 1) : after
        @test size(sys_after[:ZZ_pseudo], 1) == n_pseudo_observables(m)
        @test sys_after[:ZZ_pseudo] == systems[1][1][:ZZ_pseudo]
        @test sys_after[:DD_pseudo] == systems[1][1][:DD_pseudo]
    end
end

@testset "Check DSGEVAR likelihood and posterior calculations" begin
    dsge = Model1002("ss10")
    obs_i = [dsge.observables[:obs_nominalrate], dsge.observables[:obs_gdp],
             dsge.observables[:obs_gdpdeflator]]
    m = DSGE.DSGEVAR(dsge, collect(keys(dsge.exogenous_shocks)))
    DSGE.update!(m; observables = [:obs_nominalrate, :obs_gdp, :π_t], lags = 4, λ = 0.5)
    dsge_data =
        df_to_matrix(dsge,
                     CSV.read(joinpath(path, "../reference/test_dsgevar_likelihood_dsge_data.csv"), DataFrame))[obs_i, :]
    lh = likelihood(m, dsge_data)
    @test lh ≈ DSGE.dsgevar_likelihood(m, dsge_data) ≈
        load(joinpath(path, "../reference/test_dsgevar_likelihood_dsge.jld2"), "llh")

    post = DSGE.posterior(m, dsge_data)
    @test post ≈ lh + prior(m)

    x = map(α -> α.value, m.dsge.parameters)
    post_at_start = DSGE.posterior!(m, x, dsge_data)
    @test post ≈ post_at_start

    # Ensure if we are not evaluating at start vector, then we do not get the reference
    # posterior
    global y = x .+ 0.01
    post_not_at_start = DSGE.posterior!(m, y, dsge_data)
    ϵ = 1.0
    @test abs(post_at_start - post_not_at_start) > ϵ

    # Check keywords
    m.dsge.parameters[1].value = 3.
    @test !isinf(likelihood(m, dsge_data; sampler = false))
    @test isinf(likelihood(m, dsge_data; sampler = true))
end

################
# Benchmarking #
################
# Flip to true to run; off by default.
run_benchmarks = false

if run_benchmarks
    # DSGE (AnSchorfheide): prior / likelihood / posterior / posterior!.
    custom_settings = [Setting(:date_forecast_start, quartertodate("2015-Q4"))]
    m_dsge = AnSchorfheide(custom_settings = custom_settings, testing = true)
    data_dsge = Matrix{Float64}(load("$path/../reference/posterior.jld2", "data")')
    x_dsge = map(α -> α.value, m_dsge.parameters)

    b_prior = @benchmark prior($m_dsge)
    b_lh    = @benchmark likelihood($m_dsge, $data_dsge)
    b_post  = @benchmark DSGE.posterior($m_dsge, $data_dsge)
    b_postx = @benchmark DSGE.posterior!($m_dsge, $x_dsge, $data_dsge)

    # DSGE-VAR (Model1002 ss10): likelihood.
    dsge_var = Model1002("ss10")
    obs_i = [dsge_var.observables[:obs_nominalrate], dsge_var.observables[:obs_gdp],
             dsge_var.observables[:obs_gdpdeflator]]
    m_var = DSGE.DSGEVAR(dsge_var, collect(keys(dsge_var.exogenous_shocks)))
    DSGE.update!(m_var; observables = [:obs_nominalrate, :obs_gdp, :π_t], lags = 4, λ = 0.5)
    data_var = df_to_matrix(dsge_var,
                            CSV.read(joinpath(path, "../reference/test_dsgevar_likelihood_dsge_data.csv"),
                                     DataFrame))[obs_i, :]
    b_var_lh = @benchmark likelihood($m_var, $data_var)

    println("\n===== estimate/posterior benchmark results =====")
    for (name, b) in [("prior (dsge)       ", b_prior),
                      ("likelihood (dsge)  ", b_lh),
                      ("posterior (dsge)   ", b_post),
                      ("posterior! (dsge)  ", b_postx),
                      ("likelihood (dsgevar)", b_var_lh)]
        println(name, "  time:   ", BenchmarkTools.prettytime(median(b).time),
                "   memory: ", BenchmarkTools.prettymemory(median(b).memory))
    end
end
