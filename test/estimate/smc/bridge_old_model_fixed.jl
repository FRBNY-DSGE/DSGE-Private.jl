using DSGE, ModelConstructors, HDF5, Random, JLD2, FileIO, SMC, Test, Distributions, Statistics

# A bridge whose old model fixes a parameter that the new model estimates (as ss207 fixes the
# regime-2 shock sds that ss104 estimates). The old-data likelihood must be evaluated at the old
# model's fixed value, not at the particle's draw: update! never overwrites a fixed parameter, and
# the check after update! must not reject that. It must still catch a misaligned update.

path = dirname(@__FILE__)
bf_data     = h5read(joinpath(path, "reference/smc.h5"), "data")
bf_old_data = bf_data[:, 1:(size(bf_data, 2) - 30)]
bf_dir      = mktempdir()
const BF_FIXED_ρ_R = 0.3847

function bf_model(name; fix_ρ_R::Bool = false)
    m = AnSchorfheide()
    fix_ρ_R && (m <= parameter(:ρ_R, BF_FIXED_ρ_R, fixed = true))
    m <= Setting(:saveroot, joinpath(bf_dir, name))
    m <= Setting(:n_particles, 100)
    m <= Setting(:n_Φ, 20)
    m <= Setting(:λ, 2.0)
    m <= Setting(:n_smc_blocks, 1)
    m <= Setting(:use_parallel_workers, false)
    m <= Setting(:step_size_smc, 0.5)
    m <= Setting(:n_mh_steps_smc, 2)
    m <= Setting(:resampler_smc, :systematic)
    m <= Setting(:target_accept, 0.25)
    m <= Setting(:mixture_proportion, 0.9)
    m <= Setting(:use_fixed_schedule, true)
    m <= Setting(:resampling_threshold, 0.5)
    m <= Setting(:smc_iteration, 0)
    m <= Setting(:use_chand_recursion, true)
    m <= Setting(:tempered_update_prior_weight, 0.0)
    return m
end
bf_loglh(m, θ, data) = (DSGE.update!(m, θ);
    likelihood(m, data; sampler = false, catch_errors = true, verbose = :none, use_chand_recursion = true))

@testset "Bridge from an old model that fixes a parameter the new model frees" begin
    m_old = bf_model("old"; fix_ρ_R = true)
    k = m_old.keys[:ρ_R]
    @test m_old[:ρ_R].fixed && !bf_model("x")[:ρ_R].fixed
    smc2(m_old, bf_old_data; verbose = :none, run_csminwel = false, smc_seed = 31)
    old_cloud = load(rawpath(m_old, "estimate", "smc_cloud.jld2"), "cloud")
    @test all(old_cloud.particles[:, k] .== BF_FIXED_ρ_R)

    # θ̃ = ρ_R: draw it from its prior under the new model (BRIDGE-NOTES-REVIEW B3). The old
    # likelihood does not depend on it, so the old cloud's loglh stays valid; its log prior
    # becomes the new model's.
    m_new = bf_model("new")
    n = size(old_cloud.particles, 1)
    old_cloud.particles[:, k] = rand(Xoshiro(5), get(m_new[:ρ_R].prior), n)
    pv = deepcopy(m_new.parameters)
    SMC.update_logprior!(old_cloud, [(ModelConstructors.update!(pv, old_cloud.particles[i, 1:length(pv)]);
                                      ModelConstructors.prior(pv)) for i in 1:n])

    # Before the fix: AssertionError at the first proposal (old_model[:ρ_R].value != particle)
    smc2(m_new, bf_data; verbose = :none, run_csminwel = false, smc_seed = 32,
         old_data = bf_old_data, old_cloud = old_cloud, old_model = m_old)
    cloud = load(rawpath(m_new, "estimate", "smc_cloud.jld2"), "cloud")
    P = SMC.get_vals(cloud)
    @test std(P[k, :]) > 0                                   # ρ_R is estimated in the bridge
    @test m_old[:ρ_R].value == BF_FIXED_ρ_R                  # and never written into the old model

    # The old-data loglh of every final particle is the old model's, at the fixed ρ_R
    m_chk   = bf_model("chk"; fix_ρ_R = true)
    at_fix  = [bf_loglh(m_chk, P[:, i], bf_old_data) for i in 1:size(P, 2)]
    @test m_chk[:ρ_R].value == BF_FIXED_ρ_R
    @test isequal(SMC.get_old_loglh(cloud), at_fix)
    # ... and not the old data's loglh at the particle's own ρ_R
    m_free  = bf_model("free")
    at_draw = [bf_loglh(m_free, P[:, i], bf_old_data) for i in 1:size(P, 2)]
    @test all(at_draw .!= at_fix)
    # The new-data loglh is at the particle's ρ_R
    @test isequal(SMC.get_loglh(cloud), [bf_loglh(m_free, P[:, i], bf_data) for i in 1:size(P, 2)])
end

@testset "check_old_model_update still catches a misaligned update" begin
    θ = [p.value for p in bf_model("v").parameters]
    i_g, i_z, i_R = (bf_model("v").keys[s] for s in (:ρ_g, :ρ_z, :ρ_R))
    θ[i_R] = 0.7                                             # ≠ the old model's fixed 0.3847
    para2 = deepcopy(bf_model("v").parameters)
    ModelConstructors.update!(para2, θ)

    m_old = bf_model("a"; fix_ρ_R = true)
    DSGE.update!(m_old, para2)
    @test DSGE.check_old_model_update(m_old, para2) === nothing          # fixed ρ_R skipped
    @test m_old[:ρ_R].value == BF_FIXED_ρ_R

    # Positional mismatch between two free parameters
    swapped = copy(para2); swapped[i_g], swapped[i_z] = para2[i_z], para2[i_g]
    m_old = bf_model("b"; fix_ρ_R = true)
    DSGE.update!(m_old, swapped)
    @test_throws AssertionError DSGE.check_old_model_update(m_old, para2)
    # ... and between a free parameter and the fixed one
    swapped = copy(para2); swapped[i_g], swapped[i_R] = para2[i_R], para2[i_g]
    m_old = bf_model("c"; fix_ρ_R = true)
    DSGE.update!(m_old, swapped)
    @test_throws AssertionError DSGE.check_old_model_update(m_old, para2)
    # Without fixed parameters the check is the old one
    m_old = bf_model("d")
    DSGE.update!(m_old, para2)
    @test DSGE.check_old_model_update(m_old, para2) === nothing
    @test para2[i_g].value != para2[i_z].value && para2[i_g].value != para2[i_R].value
    m_old = bf_model("e")
    DSGE.update!(m_old, swapped)
    @test_throws AssertionError DSGE.check_old_model_update(m_old, para2)

    # Regimes: a regime fixed in the old model is skipped, a free one is checked
    function with_regimes!(ps, v2)
        for p in ps
            if p.key == :ρ_g
                ModelConstructors.set_regime_val!(p, 1, p.value)
                ModelConstructors.set_regime_val!(p, 2, v2)
            end
        end
        return ps
    end
    para_rs = with_regimes!(deepcopy(para2), 0.6)
    m_old = bf_model("f")
    with_regimes!(m_old.parameters, 0.2)
    ModelConstructors.set_regime_fixed!(m_old[:ρ_g], 1, false)
    ModelConstructors.set_regime_fixed!(m_old[:ρ_g], 2, true)
    DSGE.update!(m_old, para_rs; regime_switching = true)
    @test m_old[:ρ_g].regimes[:value][2] == 0.2                          # not overwritten
    @test m_old[:ρ_g].regimes[:value][1] == para_rs[i_g].value
    @test DSGE.check_old_model_update(m_old, para_rs) === nothing
    m_old = bf_model("g")
    with_regimes!(m_old.parameters, 0.2)
    DSGE.update!(m_old, para_rs; regime_switching = true)
    @test DSGE.check_old_model_update(m_old, para_rs) === nothing        # regime 2 free: written
    m_old[:ρ_g].regimes[:value][2] = 0.3                                 # as if misaligned
    @test_throws AssertionError DSGE.check_old_model_update(m_old, para_rs)
end

rm(bf_dir; recursive = true, force = true)
