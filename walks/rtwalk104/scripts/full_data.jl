# RTWALK104 FINAL STEP (13 Nov 2026; RTWALK104.md "13 Nov 2026: final step 2026Q1 -> 2026Q2"): the full-sample data
# of the last walk step 2026Q1 (vintage 260710) -> 2026Q2 (vintage 261013 = the ss104v3 / ss104v3mh5 direct
# estimations' data). Mirrors work/rtwalk114/scripts/full_data.jl (same agent pattern, ss104 instead of ss114).
# The pin pinned/rt_ss104_2026Q2_vint=261013_full.jld2 is the ss104v3 frozen matrix
# (work/ss104v3/frozen/data_ss104v3.jld2, 28 x 268, 1959Q3-2026Q2), copied by scripts/pin_full.jl after these checks:
#   - the ss104v3 frozen file's sha256 equals FULL_SRC_SHA (sha256sum of 13 Nov; = data_ss104v2_ref.jld2's);
#   - pin data === the ss104v3 frozen matrix === the ss104v2 reference matrix (NaN-aware isequal);
#   - 28 observables in the walk model's order, T = 268 = presample start .. 2026Q2.
# full_data_pinned(m) re-checks all of this on every load (master and workers). Included after setup.jl + rt_data.jl.
const V3_FROZEN     = "/data/dsge_data_dir/estimation-AI/work/ss104v3/frozen/data_ss104v3.jld2"
const V2_FROZEN_REF = "/data/dsge_data_dir/estimation-AI/work/ss104v3/frozen/data_ss104v2_ref.jld2"
const FULL_SRC_SHA  = "f9671b7a451615a295d0c9c914ebe17b1a778ccc1a20613f530cf41f005a4b4c"   # both files, sha256sum 13 Nov 2026
const FULL_T = Date(2026, 6, 30); const FULL_V = "261013"
const FULL_PIN = joinpath(W, "pinned", "rt_ss104_2026Q2_vint=261013_full.jld2")
const FULL_PIN_SHA = joinpath(W, "pinned", "rt_ss104_2026Q2_full.sha256")
walk_obs_order(m) = sort(collect(keys(m.observables)), by = k -> m.observables[k])

function full_frozen_checked(m)
    sh = bytes2hex(open(sha256, V3_FROZEN))
    @assert sh == FULL_SRC_SHA "ss104v3 frozen data changed: sha256 $sh"
    d = load(V3_FROZEN, "data")
    @assert isequal(d, load(V2_FROZEN_REF, "data")) "ss104v3 frozen != ss104v2 reference matrix"
    @assert size(d) == (28, 268) "ss104v3 frozen data are $(size(d)), expected 28 x 268"
    @assert n_observables(m) == 28 && size(d, 2) == length(DSGE.quarter_range(date_presample_start(m), FULL_T))
    return d, sh, walk_obs_order(m)
end

function full_data_pinned(m)
    @assert isfile(FULL_PIN) "no $FULL_PIN: run scripts/pin_full.jl first"
    psh = bytes2hex(open(sha256, FULL_PIN))
    @assert psh == split(readline(FULL_PIN_SHA))[1] "full pin sha256 $psh differs from $FULL_PIN_SHA"
    p = load(FULL_PIN)
    @assert p["sha_ss104v3_frozen"] == bytes2hex(open(sha256, V3_FROZEN)) == FULL_SRC_SHA "ss104v3 frozen data changed since the pin"
    @assert isequal(p["data"], load(V3_FROZEN, "data")) "full pin != ss104v3 frozen matrix"
    @assert p["observables"] == string.(walk_obs_order(m))
    return p["data"], FULL_V
end
