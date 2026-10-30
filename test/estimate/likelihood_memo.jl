using DSGE, ModelConstructors, HDF5, Random, Test, Dates, LinearAlgebra

# :preprocessed_transitions memoizes regime transition matrices within one solve. Its key
# does not include parameter values. A model that carries the setting must therefore give the
# same posterior, Hessian and MH chain as a model without it: likelihood clears the memo
# before it solves at new parameters. Callers such as csminwel, hessian! and
# metropolis_hastings do not clear it themselves.

path = dirname(@__FILE__)
data = h5read(joinpath(path, "..", "reference", "sim_data_regswitch_anschorfheide.h5"), "data")

function memo_rs_as(saveroot::String)
    m = AnSchorfheide()
    m <= Setting(:saveroot, saveroot)
    m <= Setting(:data_vintage, "210101")
    m <= Setting(:cond_vintage, "210101")
    m <= Setting(:sampling_method, :MH)
    m <= Setting(:n_mh_simulations, 30)
    m <= Setting(:n_mh_blocks, 1)
    m <= Setting(:n_mh_burn, 0)
    m <= Setting(:n_mh_thin, 1)
    m <= Setting(:mh_adaptive_accept, false)
    m <= Setting(:use_parallel_workers, false)
    m <= Setting(:regime_switching, true, true, "rs2", "")
    m <= Setting(:n_regimes, 3)
    m <= Setting(:regime_dates, Dict(1 => date_presample_start(m),
                                     2 => DSGE.iterate_quarters(Date(1960, 3, 31), 50),
                                     3 => DSGE.iterate_quarters(Date(1960, 3, 31), 100)))
    setup_regime_switching_inds!(m)
    m <= Setting(:model2para_regimes, Dict{Int, Int}(1 => 1, 2 => 1, 3 => 2))
    for i in 1:length(m.parameters), k in 1:2
        ModelConstructors.set_regime_val!(m.parameters[i], k, m.parameters[i].value)
    end
    DSGE.setup_param_regimes!(m, repeat([1 1 2], length(m.parameters)))
    return m
end

@testset "Likelihood clears :preprocessed_transitions at new parameters" begin
    root  = mktempdir()
    m0    = memo_rs_as(root)
    modeθ = h5read(joinpath(path, "..", "reference", "regime_switching_anschorfheide_paramsmode_output.h5"), "params")
    H     = h5read(joinpath(path, "..", "reference", "hessian_rs2=true_vint=210101.h5"), "hessian")

    # Parameter vectors near the mode
    Random.seed!(20261029)
    θs = Vector{Float64}[]
    while length(θs) < 4
        θ = modeθ .+ 0.05 .* (vec(rand(m0.parameters, 1; regime_switching = true)) .- modeθ)
        mt = deepcopy(m0)
        isfinite(posterior!(mt, θ, data; catch_errors = true)) && push!(θs, θ)
    end

    # Fresh: the setting is absent, so the memo is never read
    fresh = map(θs) do θ
        mt = deepcopy(m0)
        posterior!(mt, θ, data; catch_errors = true)
    end
    @test length(unique(fresh)) == length(fresh)

    # One model that carries the setting, evaluated in sequence and never reset by the caller
    mm = deepcopy(m0)
    mm <= Setting(:preprocessed_transitions, Dict())
    seq = [posterior!(mm, θ, data; catch_errors = true) for θ in θs]
    @test seq == fresh
    @test !isempty(get_setting(mm, :preprocessed_transitions))   # the memo is used within a solve

    # Hessian (first 3 free parameters)
    ma = deepcopy(m0); ma <= Setting(:n_hessian_test_params, 3)
    mp = deepcopy(ma); mp <= Setting(:preprocessed_transitions, Dict())
    Ha, _ = hessian!(ma, copy(modeθ), data; check_neg_diag = false)
    Hp, _ = hessian!(mp, copy(modeθ), data; check_neg_diag = false)
    @test Hp == Ha
    @test any(!iszero, Ha)

    # Metropolis-Hastings: the same chain with and without the setting
    F = svd(H)
    S_inv = zeros(size(H))
    for i in findall(x -> x > 1e-6, F.S)
        S_inv[i, i] = 1 / F.S[i]
    end
    function chain(mt, tag)
        mt <= Setting(:saveroot, joinpath(root, tag))
        propdist = DegenerateMvNormal(copy(modeθ), F.V * S_inv * F.U'; stdev = false)
        fixed_dirs = findall(iszero, diag(propdist.Σ))
        propdist.σ[fixed_dirs, :] .= 0.0
        propdist.σ[:, fixed_dirs] .= 0.0
        Random.seed!(mt.rng, 11); Random.seed!(11)
        DSGE.metropolis_hastings(propdist, mt, data, 0.0005, 0.0005;
                                 regime_switching = true, verbose = :none)
        h5read(rawpath(mt, "estimate", "mhsave.h5"), "mhparams")
    end
    ca = chain(deepcopy(m0), "absent")
    mp = deepcopy(m0); mp <= Setting(:preprocessed_transitions, Dict())
    cp = chain(mp, "present")
    @test cp == ca
    @test size(unique(ca, dims = 1), 1) > 1   # the chain moved

    rm(root; recursive = true, force = true)
end

@testset "optimize!'s solvability check solves at the proposal, not from a stale memo" begin
    root  = mktempdir()
    m0    = memo_rs_as(root)
    modeθ = h5read(joinpath(path, "..", "reference", "regime_switching_anschorfheide_paramsmode_output.h5"), "params")
    Random.seed!(20261030)
    θ2 = modeθ
    while true
        θ2 = modeθ .+ 0.3 .* (vec(rand(m0.parameters, 1; regime_switching = true)) .- modeθ)
        isfinite(posterior!(deepcopy(m0), θ2, data; catch_errors = true)) && break
    end
    mf = deepcopy(m0)                    # no memo
    DSGE.update!(mf, θ2)
    fresh = compute_system(mf)

    # A model whose memo was filled at the mode, as after a posterior evaluation there
    filled() = (mt = deepcopy(m0); mt <= Setting(:preprocessed_transitions, Dict());
                DSGE.update!(mt, modeθ); compute_system(mt); mt)
    # Without clearing, the solve at θ2 reuses the mode's transitions
    ms = filled(); DSGE.update!(ms, θ2)
    stale = compute_system(ms)
    @test any(stale[i, :TTT] != fresh[i, :TTT] for i in 1:DSGE.n_regimes(fresh))
    # solve_at! (used by optimize!'s proposal check) clears the memo first
    sys = DSGE.solve_at!(filled(), θ2)
    @test all(sys[i, :TTT] == fresh[i, :TTT] for i in 1:DSGE.n_regimes(fresh))
    @test all(sys[i, :RRR] == fresh[i, :RRR] for i in 1:DSGE.n_regimes(fresh))
    @test all(sys[i, :CCC] == fresh[i, :CCC] for i in 1:DSGE.n_regimes(fresh))

    rm(root; recursive = true, force = true)
end

@testset "optimize!'s simulated-annealing proposals go through solve_at!" begin
    # The stale-memo bug cannot be seen from optimize!'s output (the objective clears the memo
    # itself; only the solvability check of each proposal did not), so check the wiring: both
    # neighbor functions solve via solve_at!, and optimize! never calls compute_system directly
    src  = read(joinpath(pkgdir(DSGE), "src", "estimate", "optimize.jl"), String)
    body = src[1:first(findfirst("\nfunction solve_at!", src))]
    @test count("solve_at!(m, x_proposal_all)", body) == 2
    @test !occursin("compute_system(", body)
end

@testset "solve_at! for a DSGE-VAR clears the memo of its DSGE" begin
    # optimize!'s DSGE-VAR neighbor calls solve_at!; DSGE-VARs have no `tvis` keyword and keep
    # the memo in their DSGE
    function as_var(θ)
        dsge = AnSchorfheide()
        DSGE.update!(dsge, θ)
        mv = DSGE.DSGEVAR(dsge, collect(keys(dsge.exogenous_shocks)))
        DSGE.update!(mv; observables = [:obs_nominalrate, :obs_gdp, :π_t], lags = 4, λ = 0.5)
        return mv
    end
    θ0 = map(x -> x.value, DSGE.get_parameters(AnSchorfheide()))
    θ1 = copy(θ0)
    θ1[findfirst(p -> p.key == :τ, DSGE.get_parameters(AnSchorfheide()))] *= 1.5
    m = as_var(θ0)
    DSGE.get_dsge(m) <= Setting(:preprocessed_transitions, Dict{Any, Any}(:stale => true))
    β, Σ = DSGE.solve_at!(m, θ1)
    @test !haskey(get_setting(DSGE.get_dsge(m), :preprocessed_transitions), :stale)
    @test map(x -> x.value, DSGE.get_parameters(m)) == θ1
    β1, Σ1 = compute_system(as_var(θ1))
    @test β == β1 && Σ == Σ1
end
