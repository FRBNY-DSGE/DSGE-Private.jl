# RTWALK114 FINAL STEP (13 Nov 2026; RTWALK114-FULLCOMPARE.md): the full-sample data of the last walk step
# 2026Q1 (vintage 260710) -> 2026Q2 (vintage 261013 = the ss114 direct estimation's data).
# The pin pinned/rt_ss114_2026Q2_vint=261013_full.jld2 is the ss114 frozen matrix (work/ss114/frozen/data_ss114.jld2,
# 26 x 268, 1959Q3-2026Q2), copied by scripts/pin_full.jl after these checks (NaN-aware isequal):
#   - the ss114 frozen file's sha256 equals work/ss114/frozen/data_ss114.sha256;
#   - pin data === the ss114 frozen matrix, and === ss104v3's frozen 28 x 268 matrix minus obs_pgap / obs_ygap
#     (by observable name; = bridgeB.jl SAMPLE_END=full);
#   - observables in the walk model's order.
# full_data_pinned(m) re-checks all of this on every load (master and workers). Included after setup.jl + rt_data.jl.
const S114_FROZEN = "/data/dsge_data_dir/estimation-AI/work/ss114/frozen/data_ss114.jld2"
const S114_FROZEN_SHA = "/data/dsge_data_dir/estimation-AI/work/ss114/frozen/data_ss114.sha256"
const V3_FROZEN = "/data/dsge_data_dir/estimation-AI/work/ss104v3/frozen/data_ss104v3.jld2"
const FULL_T = Date(2026, 6, 30); const FULL_V = "261013"
const FULL_PIN = joinpath(W, "pinned", "rt_ss114_2026Q2_vint=261013_full.jld2")
const FULL_PIN_SHA = joinpath(W, "pinned", "rt_ss114_2026Q2_full.sha256")

function full_frozen_checked(m)
    sh = bytes2hex(open(sha256, S114_FROZEN))
    @assert sh == strip(read(S114_FROZEN_SHA, String)) "ss114 frozen data changed: $sh"
    d114 = load(S114_FROZEN, "data"); d104 = load(V3_FROZEN, "data")
    @assert size(d114) == (26, 268) && size(d104) == (28, 268)
    @assert n_observables(m) == 26 && size(d114, 2) == length(DSGE.quarter_range(date_presample_start(m), FULL_T))
    # ss104 observable order (rtwalk104-style ss104 build; the same order as ss104v3's, bridgeB.jl FULL)
    m104o = build_walk_model(Date(2026, 9, 30), "ss104"; avg = false, antmu = 0)
    o114 = sort(collect(keys(m.observables)), by = k -> m.observables[k])
    @assert isequal(d104[[m104o.observables[k] for k in o114], :], d114) "ss114 frozen data != ss104v3 frozen data minus the gap rows"
    return d114, sh, o114
end

function full_data_pinned(m)
    @assert isfile(FULL_PIN) "no $FULL_PIN: run scripts/pin_full.jl first"
    psh = bytes2hex(open(sha256, FULL_PIN))
    @assert psh == split(readline(FULL_PIN_SHA))[1] "full pin sha256 $psh differs from $FULL_PIN_SHA"
    p = load(FULL_PIN)
    @assert p["sha_ss114_frozen"] == bytes2hex(open(sha256, S114_FROZEN)) "ss114 frozen data changed since the pin"
    @assert isequal(p["data"], load(S114_FROZEN, "data")) "full pin != ss114 frozen matrix"
    @assert p["observables"] == [string(k) for k in sort(collect(keys(m.observables)), by = k -> m.observables[k])]
    return p["data"], FULL_V
end
