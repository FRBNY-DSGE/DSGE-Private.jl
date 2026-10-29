using DSGE, ModelConstructors, HDF5, Random, JLD2, FileIO, SMC, Test, Dates

# Bridge with a distinct old_model that carries :preprocessed_transitions.
# The memo is keyed without parameter values, so smc2 must clear it on the model it
# evaluates for every old-data likelihood. Here old_model is the model of a previous
# estimation (as in a chain of bridges), so it is left with a populated memo. Every
# old-data log-likelihood in the bridged cloud must equal a fresh evaluation.

path = dirname(@__FILE__)
data = h5read(joinpath(path, "..", "..", "reference", "sim_data_regswitch_anschorfheide.h5"), "data")
# The old sample must reach date_zlb_start (the regime-switching AS likelihood needs it)
old_data = data[:, 1:(size(data, 2) - 1)]

function oldmodel_memo_rs_as(vint::String)
    m = AnSchorfheide()
    m <= Setting(:saveroot, normpath(joinpath(path, "save", "old_model_memo")))
    m <= Setting(:sampling_method, :SMC)
    m <= Setting(:data_vintage, vint)
    m <= Setting(:n_Φ, 500)
    m <= Setting(:λ, 2.0)
    m <= Setting(:n_particles, 100)
    m <= Setting(:n_smc_blocks, 1)
    m <= Setting(:use_parallel_workers, false)
    m <= Setting(:step_size_smc, 0.5)
    m <= Setting(:n_mh_steps_smc, 1)
    m <= Setting(:resampler_smc, :multinomial)
    m <= Setting(:target_accept, 0.25)
    m <= Setting(:mixture_proportion, 0.9)
    m <= Setting(:resampling_threshold, 0.5)
    m <= Setting(:use_fixed_schedule, false)
    m <= Setting(:adaptive_tempering_target_smc, 0.95)
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

@testset "Bridge with a distinct old_model carrying :preprocessed_transitions" begin
    m_old = oldmodel_memo_rs_as("000000")
    DSGE.smc2(m_old, old_data; regime_switching = true, run_csminwel = false,
              verbose = :none, smc_seed = 11)
    old_cloud = load(rawpath(m_old, "estimate", "smc_cloud.jld2"), "cloud")
    # The previous estimation leaves its memo on m_old
    @test haskey(DSGE.get_settings(m_old), :preprocessed_transitions)
    @test !isempty(get_setting(m_old, :preprocessed_transitions))

    m_new = oldmodel_memo_rs_as("210101")
    m_new <= Setting(:previous_data_vintage, "000000")
    @test m_new !== m_old
    DSGE.smc2(m_new, data; regime_switching = true, run_csminwel = false, verbose = :none,
              old_data = old_data, old_cloud = old_cloud, old_model = m_old, smc_seed = 12)
    cloud = load(rawpath(m_new, "estimate", "smc_cloud.jld2"), "cloud")

    # Fresh old-data likelihoods: memo cleared before each evaluation
    P = SMC.get_vals(cloud)
    pv = deepcopy(m_new.parameters)
    m_fresh = deepcopy(m_old)
    fresh_old_loglh = map(1:size(P, 2)) do i
        ModelConstructors.update!(pv, P[:, i])
        DSGE.update!(m_fresh, pv; regime_switching = true)
        m_fresh <= Setting(:preprocessed_transitions, Dict())
        likelihood(m_fresh, old_data; sampler = false, catch_errors = true, verbose = :none)
    end
    @test all(isapprox.(SMC.get_old_loglh(cloud), fresh_old_loglh; rtol = 1e-10))

    rm(joinpath(path, "save", "old_model_memo"); recursive = true, force = true)
end
