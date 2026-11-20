# RTWALK114: = work/rtwalk104/scripts/rt_data.jl for ss114 (no obs_pgap / obs_ygap rows; pins rt_ss114_*; walk from 2020Q3; each new pin is
# checked against rtwalk104's ss104 pin of the same sample end with those two rows dropped, bitwise, see rt_data_pinned).
# Original header: the real-time dataset for ss104 at any sample end t (2019Q4 .. 2026Q1), "rev" OIS convention.
#
# Vintage: the Blue Chip origin of sample end t is the quarter after t; its vintage is the 10th of the first month
# of the quarter after the origin (realtime_scripts/src/util.jl:19-43, BLUECHIP_FORECAST_MONTH = 1), e.g.
# 2019Q4 -> origin 2020Q1 -> 200410 (as work/rtvint104), 2020Q1 -> 200710, ..., 2026Q1 -> 260710.
#
# Macro / financial observables: MP's cached real-time frame realtime_spec=m1002_hp=true_re=ois_vint=<v>.csv, read
# only (the frame load_realtime_data(try_disk = true) reads; no FRED here). 241010 has only the re=bluechip frame; its
# non-rate columns are the same construction (checked for 200410 in RTVINT104 PART A, and checked again here for
# every vintage that has both).
#
# "rev" convention (Marco, 4 Nov; = RTVINT104 RT_OIS=rev, = the ss104 estimation's measurement model): the
# rate-expectation rows obs_nominalrate1..6 (OIS) and obs_exp_nominalrate1..6, and obs_pgap / obs_ygap, are taken from
# the revised 261013 frame (Elena's ss104 data) truncated at t, with its observation pattern: OIS h observed
# 2008Q4 .. 2015Q1-(h-1) and 2020Q1-2020Q3, NaN 2020Q4-2021Q4; expected FFR from 2022Q1; pgap/ygap only 2020Q2
# (-0.125 / -12). obs_nominalrate is real-time but NaN over 2020Q4-2021Q4 (the temporary-ZLB data treatment). Those
# rate rows are market data that are not revised; the revised frame is simply the source with the right pattern.
# MP's realtime caches carry no OIS (ois_170615.csv ends 2017Q2) and no SPD expected-FFR rows at all, and MP's
# Realtime2026 tempzlb treatment NaNs the OIS from 2020Q4 to the end (BRIDGE-CHECK-FW finding 1), so they cannot be
# used for these rows. At t = 2019Q4 this reproduces work/rtvint104's rev pin bitwise (checked in preflight PART A).
#
# Defines: RT_DIR, sample_end_vintage(t), rt_cache_path(v), build_rt_matrix(m, t), rt_data_pinned(m, t), REV_DF
using SHA

const RT_DIR  = "/data/dsge_data_dir/dsgejl/realtime/input_data/data"
const RT_PIN_DIR = joinpath(W, "pinned")
const RTVINT104_REVPIN = "/data/dsge_data_dir/estimation-AI/work/rtvint104/pinned/rt_ss104_2019Q4_vint=200410_revois.jld2"
const WALK_FIRST = Date(2020, 3, 31)       # RTWALK114: first origin. Main chain: from the reweighted ss104 2020Q3 cloud; validation bridges (steps_bridge/): from the mapped ss104 2020Q1 cloud
const STEP_BASE  = Date(2019, 12, 31)      # step numbers as rtwalk104 (step 4 = 2020Q3 -> 2020Q4; chain step 1)
const RTW104_PIN_DIR = "/data/dsge_data_dir/estimation-AI/work/rtwalk104/pinned"   # read-only
const WALK_LAST  = Date(2026, 3, 31)      # last sample end with a real-time frame (vintage 260710)
const ZLB_NAN    = (Date(2020, 12, 31), Date(2021, 12, 31))
const REV_ROWS   = vcat([Symbol("obs_nominalrate$h") for h in 1:6], [Symbol("obs_exp_nominalrate$h") for h in 1:6],
                        Symbol[])    # RTWALK114: obs_pgap / obs_ygap are not ss114 observables

function sample_end_vintage(t::Date)
    o = nextq(t)                                   # forecast origin
    y, q = Dates.year(o), Dates.quarterofyear(o)
    rq = q % 4 + 1; ry = rq == 1 ? y + 1 : y        # = rtvint104 rt_vintage
    return Dates.format(Date(ry, 3 * (rq - 1) + 1, 10), "yymmdd")
end
@assert sample_end_vintage(Date(2019, 12, 31)) == "200410" && sample_end_vintage(Date(2026, 3, 31)) == "260710"

function rt_cache_path(v::AbstractString)
    f = joinpath(RT_DIR, "realtime_spec=m1002_hp=true_re=ois_vint=$v.csv")
    isfile(f) && return f
    f = joinpath(RT_DIR, "realtime_spec=m1002_hp=true_re=bluechip_vint=$v.csv")
    isfile(f) && return f
    error("no real-time frame for vintage $v")
end

fdate(x) = x isa Date ? x : Date(string(x))
function read_rt_csv(path)
    df = CSV.read(path, DataFrame)
    df[!, :date] = fdate.(df[!, :date])
    for c in propertynames(df)
        c == :date && continue
        df[!, c] = Float64[ismissing(x) ? NaN : Float64(x) for x in df[!, c]]   # = clean_df!
    end
    sort!(df, :date)
    return df
end

const REV_DF = read_rt_csv(FROZEN_DF)

"""Real-time ss104 data matrix (n_obs x T) for sample end t, rev convention (see the header)."""
function build_rt_matrix(m, t::Date)
    v = sample_end_vintage(t); path = rt_cache_path(v)
    df = read_rt_csv(path)
    @assert df.date[end] == t "frame $path ends $(df.date[end]), expected $t"
    @assert DSGE.quarter_range(date_presample_start(m), t) ⊆ df.date
    mat = data_matrix(m, df, t)
    rev = data_matrix(m, REV_DF, t)
    @assert size(mat) == size(rev)
    dates = DSGE.quarter_range(date_presample_start(m), t)
    for k in REV_ROWS
        r = m.observables[k]; mat[r, :] = rev[r, :]
    end
    r = m.observables[:obs_nominalrate]
    for (j, d) in enumerate(dates)
        ZLB_NAN[1] <= d <= ZLB_NAN[2] && (mat[r, j] = NaN)
    end
    # the observation pattern of every row equals the revised data's, except where the revised frame itself lacks the
    # series (none expected); report, don't fail (preflight prints the coverage table)
    return mat, v, path
end

"""Build the matrix for sample end t, pin it once (pinned/rt_ss114_<t>_vint=<v>_revois.jld2), and on later calls require
the rebuild to equal the pin and the source files to be unchanged."""
function rt_data_pinned(m, t::Date)
    mat, v, path = build_rt_matrix(m, t)
    sha_cache = bytes2hex(open(sha256, path)); sha_rev = bytes2hex(open(sha256, FROZEN_DF))
    mkpath(RT_PIN_DIR)
    pin = joinpath(RT_PIN_DIR, "rt_ss114_$(qlabel(t))_vint=$(v)_revois.jld2")
    if isfile(pin)
        p = load(pin)
        @assert p["sha_cache"] == sha_cache && p["sha_rev"] == sha_rev "real-time source files changed since the pin $pin"
        @assert isequal(p["data"], mat) "rebuilt real-time data differ from the pin $pin"
        return p["data"], v
    end
    myid() == 1 || error("pin $pin missing: build it on the master first")
    # RTWALK114: the ss114 matrix must equal rtwalk104's ss104 pin of the same sample end without obs_pgap / obs_ygap,
    # row by row (by observable name), bitwise; and the two sources must be the files that pin was built from
    obs114 = [string(k) for k in sort(collect(keys(m.observables)), by = k -> m.observables[k])]
    p104f = joinpath(RTW104_PIN_DIR, "rt_ss104_$(qlabel(t))_vint=$(v)_revois.jld2")
    p104 = load(p104f)
    @assert p104["sha_cache"] == sha_cache && p104["sha_rev"] == sha_rev "sources differ from those of $p104f"
    obs104 = Vector{String}(p104["observables"])
    @assert setdiff(obs104, obs114) == ["obs_pgap", "obs_ygap"] && isempty(setdiff(obs114, obs104))
    @assert isequal(p104["data"][[findfirst(==(o), obs104) for o in obs114], :], mat) "ss114 matrix at $(qlabel(t)) != rtwalk104 pin minus the gap rows"
    tmp = pin * ".tmp$(getpid())"
    jldsave(tmp; data = mat, vint = v, sample_end = qlabel(t), cache = path, sha_cache = sha_cache, rev = FROZEN_DF,
            sha_rev = sha_rev, ois_rows = "rev", observables = obs114,
            rtwalk104_pin = p104f, sha_rtwalk104_pin = bytes2hex(open(sha256, p104f)),
            built_at = string(now()))
    try                                    # RTWALK114 (verifier I1): concurrent builders of the same pin
        mv(tmp, pin)
    catch e
        isfile(pin) || rethrow()
        rm(tmp; force = true)
        @assert isequal(load(pin, "data"), mat) "concurrent pin $pin differs"
    end
    println("  real-time data: built and pinned ", pin)
    return mat, v
end
