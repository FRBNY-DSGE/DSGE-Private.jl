# RTWALK104 pre-flight (no SMC). Run with workers (`julia -p NW`, run.sh NW=...).
#   PART=A  data and model checks, for every sample end t = 2019Q4 .. 2026Q1:
#           A1 real-time matrix built + pinned (rt_data.jl); at 2019Q4 it must equal work/rtvint104's rev pin bitwise
#           A2 coverage vs the revised data truncated at t; which series are not real-time (equal across vintages);
#              re=ois vs re=bluechip frames
#           A3 in-sample regime count k(t), and the FREE SET: for each of the 38 parameter-regimes the seed run fixed,
#              does it enter the likelihood on the data to t? (bitwise perturbation test at 2 base points, prior draws)
#              -> pinned/freeset.jld2 (FIXED_AT[label] = members that still do not enter)
#           A4 "model built for origin t": build_ss104(forecast start = t + 1q) vs the 2026Q3 build: regimes, parameter
#              layout and likelihood on the data to t
#           A5 likelihood time per sample end
#   PART=B  incremental log weights Δ = log p(Y_{t+1}|θ) - log p(Y_t|θ) for every step, on NSUB particles of
#           (proxy A) the 2019Q4 real-time seed cloud with all 38 slots drawn from their priors, and
#           (proxy B) the full-sample ss104v2 cloud. Mean, sd, ESS at φ = 1, static stage-count simulation, timings.
#           Output out/preflightB.jld2 and out/preflightB_steps.csv.
PART = get(ENV, "PART", "A")
using Distributed
@everywhere include(joinpath("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk104/scripts", "setup.jl"))
@everywhere include(joinpath("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk104/scripts", "rt_data.jl"))
using Printf, Statistics, Random, StatsBase
println("host ", gethostname(), "  PART ", PART, "  nworkers ", nworkers(), "  ", now())
const TS = collect(DSGE.quarter_range(WALK_FIRST, WALK_LAST))
const LABELS = qlabel.(TS)
const OBS = [string(k) for k in sort(collect(keys(m.observables)), by = k -> m.observables[k])]
const SEED0 = get(ENV, "SEED_CLOUD", "/data/dsge_data_dir/estimation-AI/work/rtvint104/saveroot/output_data/m1002/ss104/estimate/raw/" *
    "smc_cloud_bridge=true_mhsteps=3_nparts=15000_phi=0.95_priorwt=0.0_rtvint104revois_sampleend=2019Q4_vint=200410.jld2")
const V2CLOUD = "/data/dsge_data_dir/estimation-AI/work/ss104full/saveroot/output_data/m1002/ss104/estimate/raw/" *
    "smc_cloud_fcastdate=2026Q3_mhsteps=3_nparts=15000_phi=0.95_ss104v2_vint=261013.jld2"

# all real-time matrices are built and pinned on the master; workers read the pins
DATA = Dict{String, Matrix{Float64}}(); VINT = Dict{String, String}()
for t in TS
    d, v = rt_data_pinned(m, t); DATA[qlabel(t)] = d; VINT[qlabel(t)] = v
end
@everywhere workers() begin
    const WDATA = Dict(qlabel(t) => rt_data_pinned(m, t)[1] for t in DSGE.quarter_range(WALK_FIRST, WALK_LAST))
    configure_freeset!(m, Set{Tuple{Symbol, Int}}())     # all 101 free: every candidate can be perturbed
end
configure_freeset!(m, Set{Tuple{Symbol, Int}}())
WDATA = DATA

if PART == "A"
    # ------------------------------------------------------------------ A1 / A2: data
    println("\n=== A1  origin -> vintage -> sample end ===")
    println("sample end  origin  vintage  frame                                   T")
    for t in TS
        println(rpad(qlabel(t), 12), rpad(qlabel(nextq(t)), 8), rpad(VINT[qlabel(t)], 9), rpad(basename(rt_cache_path(VINT[qlabel(t)])), 40), size(DATA[qlabel(t)], 2))
    end
    p = load(RTVINT104_REVPIN)
    println("2019Q4 matrix == work/rtvint104 rev pin, bitwise: ", isequal(p["data"], DATA["2019Q4"]))
    @assert isequal(p["data"], DATA["2019Q4"])

    println("\n=== A2  coverage vs the revised data truncated at t (cells: RT-only / REV-only; revised cells) ===")
    for t in TS
        rt = DATA[qlabel(t)]; rv = data_matrix(m, REV_DF, t)
        bad = String[]; nrev = 0
        for (r, nm) in enumerate(OBS)
            a = count(k -> !isnan(rt[r, k]) && isnan(rv[r, k]), axes(rt, 2)); b = count(k -> isnan(rt[r, k]) && !isnan(rv[r, k]), axes(rt, 2))
            (a > 0 || b > 0) && push!(bad, "$nm +$a/-$b")
            nrev += count(k -> !isnan(rt[r, k]) && !isnan(rv[r, k]) && rt[r, k] != rv[r, k], axes(rt, 2))
        end
        println(rpad(qlabel(t), 8), " revised cells ", lpad(nrev, 5), "  pattern differences: ", isempty(bad) ? "none" : join(bad, ", "))
    end
    # series that are not real time: equal (on the common quarters) to the last vintage
    dlast = DATA[LABELS[end]]
    println("\nper series: number of vintages whose values on the common quarters differ from the last vintage (", VINT[LABELS[end]], "), and max |Δ| over vintages")
    for (r, nm) in enumerate(OBS)
        nd = 0; mx = 0.0
        for l in LABELS[1:end-1]
            d = DATA[l]; T = size(d, 2)
            both = [k for k in 1:T if !isnan(d[r, k]) && !isnan(dlast[r, k])]
            isempty(both) && continue
            dd = maximum(abs.(d[r, both] .- dlast[r, both]))
            dd > 0 && (nd += 1); mx = max(mx, dd)
        end
        println("  ", rpad(nm, 24), lpad(nd, 3), " of ", length(LABELS) - 1, @sprintf("   max|Δ| %.3g", mx))
    end
    println("\nre=ois vs re=bluechip frames (non-rate columns identical?):")
    for t in TS
        v = VINT[qlabel(t)]
        fo = joinpath(RT_DIR, "realtime_spec=m1002_hp=true_re=ois_vint=$v.csv"); fb = joinpath(RT_DIR, "realtime_spec=m1002_hp=true_re=bluechip_vint=$v.csv")
        (isfile(fo) && isfile(fb)) || (println("  $v: only ", isfile(fo) ? "ois" : "bluechip"); continue)
        a = read_rt_csv(fo); b = read_rt_csv(fb)
        cs = [c for c in intersect(propertynames(a), propertynames(b)) if c != :date && !occursin("exp_nominal", string(c))]
        dif = [c for c in cs if !isequal(a[!, c], b[!, c])]
        println("  $v: ", isempty(dif) ? "identical" : "DIFFER in $dif")
    end

    # ------------------------------------------------------------------ A3: regimes and the free set
    println("\n=== A3  in-sample regimes and the free set ===")
    rd = get_setting(m, :regime_dates)
    println("n_regimes ", get_setting(m, :n_regimes), "; regime start dates: ", join(["$r:$(qlabel(rd[r]))" for r in sort(collect(keys(rd)))], " "))
    m2p = get_setting(m, :model2para_regime)
    println("model regimes -> parameter regime, for the 38 candidates:")
    for key in unique(first.(CAND38))
        haskey(m2p, key) || (println("  ", rpad(string(key), 16), " (no model2para map: regime 1 only)"); continue)
        mp = m2p[key]
        inv = Dict{Int, Vector{Int}}()
        for (mr, pr) in mp; push!(get!(inv, pr, Int[]), mr); end
        println("  ", rpad(string(key), 16), join(["par reg $pr <- model regs $(minimum(v))..$(maximum(v)) (first $(qlabel(rd[minimum(v)])))" for (pr, v) in sort(collect(inv))], ";  "))
    end
    kins(t) = DSGE.insample_regime_count(m, t; tvis = haskey(DSGE.get_settings(m), :tvis_information_set), compute_pseudo = false)
    sc = load(SEED0, "cloud"); P = sc.particles; N = size(P, 2)
    θb = P[argmax(P[:, N-4] .+ P[:, N-3]), 1:N-5]
    θa = copy(VALS0)
    bases = [θa, θb]
    Random.seed!(20261104)
    draws = [ModelConstructors.rand_regime_switching(m.parameters) for _ in 1:2]
    ModelConstructors.toggle_regime!(m.parameters, 1)
    tasks = [(l, j, b, bases[b], draws[b]) for l in LABELS for j in eachindex(CAND38) for b in 1:2]   # data travel with the task
    @everywhere workers() begin
        function enters(l, j, b, θ, dr)
            s = slot_index(CAND38[j]...)
            θp = copy(θ); θp[s] = dr[s]
            L0 = LIK(m, θ, WDATA[l]); L1 = LIK(m, θp, WDATA[l])
            return (l, j, b, L0, L1, !(L0 === L1))
        end
    end
    t0 = time()
    PCACHE = joinpath(W, "out", "preflightA_perturb.jld2")      # cache of the 1976-likelihood-pair test
    res = if isfile(PCACHE) && load(PCACHE, "keys") == [(x[1], x[2], x[3]) for x in tasks] && load(PCACHE, "bases") == bases && load(PCACHE, "draws") == draws
        println("perturbation test: cached ", PCACHE); load(PCACHE, "res")
    else
        r = pmap(x -> enters(x...), tasks; batch_size = 4)
        jldsave(PCACHE; res = r, keys = [(x[1], x[2], x[3]) for x in tasks], bases = bases, draws = draws); r
    end
    println(@sprintf("perturbation test: %d tasks in %.1f s on %d workers", length(tasks), time() - t0, nworkers()))
    ENT = Dict((r[1], r[2]) => false for r in res)
    # "enters" = the perturbation moves loglh by more than rounding (EST-2019Q4 A3: ρ_ait_rm moves the 2019Q4 loglh by
    # ≤ 1e-8 through the initial-state covariance although its shock s.d. is 0), or makes it non-finite
    # ENTER_TOL comes from setup.jl (read from the pin: 1e-6 for pinned/freeset.jld2; the 1e-4 alternative is built by repin_freeset.jl)
    enters_material(r) = !(isfinite(r[4]) && isfinite(r[5]) && abs(r[5] - r[4]) <= ENTER_TOL)
    for r in res; enters_material(r) && (ENT[(r[1], r[2])] = true); end
    noise = [abs(r[5] - r[4]) for r in res if r[6] && !enters_material(r)]
    println(@sprintf("bitwise-different but below %.0e (rounding): %d of %d perturbations, max |ΔL| %.3g; slots: %s", ENTER_TOL,
                     length(noise), length(res), isempty(noise) ? 0.0 : maximum(noise),
                     join(unique(["$(CAND38[r[2]][1])[$(CAND38[r[2]][2])]@$(r[1])" for r in res if r[6] && !enters_material(r)])[1:min(end, 12)], " ")))
    FIXED_AT = Dict{String, Vector{Tuple{Symbol, Int}}}()
    println("\nsample end  k  newly entering (vs previous sample end)                       still fixed")
    prevset = Set{Tuple{Symbol, Int}}()
    for (i, l) in enumerate(LABELS)
        entering = Set(CAND38[j] for j in eachindex(CAND38) if ENT[(l, j)])
        FIXED_AT[l] = sort([c for c in CAND38 if !(c in entering)])
        new = setdiff(entering, prevset); gone = setdiff(prevset, entering)
        println(rpad(l, 12), lpad(kins(qdate(l)), 2), "  ", isempty(new) ? "-" : join(["$(k)[$r]" for (k, r) in sort(collect(new))], " "),
                isempty(gone) ? "" : "   !!! NO LONGER ENTERING: $(collect(gone))", "   | fixed ", length(FIXED_AT[l]), ", free ", 101 - length(FIXED_AT[l]))
        global prevset = entering
    end
    # base-point loglh (finite?) per sample end
    for l in LABELS
        r0 = [r for r in res if r[1] == l && r[2] == 1]
        println("  loglh at defaults / seed modal particle, ", l, ": ", join([@sprintf("%.4f", r[4]) for r in sort(r0, by = x -> x[3])], " / "))
    end
    @assert FIXED_AT["2019Q4"] == CAND38 "at 2019Q4 some of the 38 enter the likelihood: $(setdiff(CAND38, FIXED_AT["2019Q4"]))"
    for i in 2:length(LABELS)
        @assert Set(FIXED_AT[LABELS[i]]) ⊆ Set(FIXED_AT[LABELS[i-1]]) "free set not monotone at $(LABELS[i])"
    end
    if isfile(FREESET_PIN)
        @assert load(FREESET_PIN, "fixed_at") == FIXED_AT "freeset pin differs from this run"
        println("free set == pin ", FREESET_PIN)
    else
        jldsave(FREESET_PIN; fixed_at = FIXED_AT, labels = LABELS, cand38 = CAND38, built_at = string(now()),
                note = "bitwise perturbation test (prior draw of the slot) at the defaults and the 2019Q4 RT seed modal particle")
        println("free set pinned: ", FREESET_PIN)
    end

    # ------------------------------------------------------------------ A4: model built for the origin
    println("\n=== A4  model built with forecast start = origin (t + 1q) vs the 2026Q3 build ===")
    @everywhere function origin_check(l, θ)
        t = qdate(l)
        mo = build_ss104(nextq(t); avg = false, antmu = 0)
        mo <= Setting(:likelihood_insample_regimes_only, true)
        same_layout = n_parameters_regime_switching(mo) == n_parameters_regime_switching(m) &&
                      [p.key for p in mo.parameters] == [p.key for p in m.parameters]
        rd_o = get_setting(mo, :regime_dates); rd_m = get_setting(m, :regime_dates)
        same_dates_insample = all(r -> haskey(rd_o, r) && rd_o[r] == rd_m[r], [r for r in keys(rd_m) if rd_m[r] <= t])
        # structure by (key, regime): number of regimes, fixed flag, and fixed value; then the likelihood with the free
        # slots of the origin build set BY NAME from θ (its fixed slots keep the origin build's own values)
        ModelConstructors.toggle_regime!(mo.parameters, 1)
        so = let s = Symbol[p.key for p in mo.parameters]
            for p in mo.parameters
                if !isempty(p.regimes) && haskey(p.regimes, :value); for i in 2:length(p.regimes[:value]); push!(s, Symbol(p.key, "_reg$i")); end; end
            end; s
        end
        vo = copy(ModelConstructors.get_values(mo.parameters))
        free_o = Set(ModelConstructors.get_free_para_inds(mo.parameters; regime_switching = true))
        free_m = Set(ModelConstructors.get_free_para_inds(m.parameters; regime_switching = true))
        diffs = String[]
        for (i, s) in enumerate(so)
            j = findfirst(==(s), SLOTSYMS)
            if j === nothing
                push!(diffs, "$s only in origin build")
            else
                (i in free_o) != (j in free_m) && push!(diffs, "$s $(i in free_o ? "free" : "fixed") in origin build, $(j in free_m ? "free" : "fixed") in 2026Q3")
                !(i in free_o) && !(j in free_m) && vo[i] != VALS0[j] && push!(diffs, "$s fixed at $(vo[i]) vs $(VALS0[j])")
                (i in free_o) && (vo[i] = θ[j])
            end
        end
        for s in setdiff(SLOTSYMS, so); push!(diffs, "$s only in 2026Q3 build"); end
        Lm = LIK(m, θ, WDATA[l])
        Lo = try LIK(mo, vo, WDATA[l]) catch e; push!(diffs, "LIK error: " * first(sprint(showerror, e), 120)); NaN end
        return (l, get_setting(mo, :n_regimes), same_layout, same_dates_insample, Lm, Lo, length(free_o), diffs)
    end
    t0 = time()
    oc = pmap(origin_check, LABELS, fill(θb, length(LABELS)))
    println(@sprintf("built %d origin models in %.1f s", length(oc), time() - t0))
    println("sample end  n_regimes  same layout  same in-sample dates  free  loglh 2026Q3 build   loglh origin build   diff")
    for r in oc
        println(rpad(r[1], 12), lpad(r[2], 9), lpad(string(r[3]), 13), lpad(string(r[4]), 22), lpad(r[7], 6), @sprintf("%20.8f %20.8f %10.3g", r[5], r[6], r[6] - r[5]))
        isempty(r[8]) || println("      structure differences (", length(r[8]), "): ", join(r[8][1:min(end, 10)], "; "), length(r[8]) > 10 ? " ..." : "")
    end

    # ------------------------------------------------------------------ A5: likelihood times
    println("\n=== A5  likelihood time per sample end (seed modal particle, warm, min of 3, master) ===")
    for l in LABELS
        LIK(m, θb, DATA[l])
        tt = minimum([@elapsed LIK(m, θb, DATA[l]) for _ in 1:3])
        println(rpad(l, 10), @sprintf("%.4f s  k = %d", tt, kins(qdate(l))))
    end
else
    # ------------------------------------------------------------------ PART B
    NSUB = parse(Int, get(ENV, "NSUB", "300"))
    function subsample(path; redraw38 = false, seed = 1)
        c = load(path, "cloud"); P = c.particles; N = size(P, 2)
        @assert N - 5 == length(VALS0) "$path: $(N-5) slots, model $(length(VALS0))"
        w = P[:, end] ./ sum(P[:, end])
        idx = SMC.resample(w; n_parts = NSUB, method = :systematic, rng = Xoshiro(seed))
        Θ = P[idx, 1:N-5]
        if redraw38
            for i in 1:NSUB
                Random.seed!(seed * 1_000_003 + i)
                d = ModelConstructors.rand_regime_switching(m.parameters)
                for (k, r) in CAND38; s = slot_index(k, r); Θ[i, s] = d[s]; end
            end
            ModelConstructors.toggle_regime!(m.parameters, 1)
        end
        return Θ
    end
    ΘA = subsample(SEED0; redraw38 = true, seed = 20261104)
    ΘB = subsample(V2CLOUD; seed = 20261105)
    @everywhere function lik_all(θ)
        out = Float64[]; tm = Float64[]
        for t in DSGE.quarter_range(WALK_FIRST, WALK_LAST)
            s = time(); push!(out, LIK(m, θ, WDATA[qlabel(t)])); push!(tm, time() - s)
        end
        return out, tm
    end
    t0 = time()
    resA = pmap(lik_all, [ΘA[i, :] for i in 1:NSUB]); tA = time() - t0
    resB = pmap(lik_all, [ΘB[i, :] for i in 1:NSUB]); tB = time() - t0 - tA
    println(@sprintf("likelihoods: proxy A %.1f s, proxy B %.1f s on %d workers (%d particles x %d sample ends each)", tA, tB, nworkers(), NSUB, length(TS)))
    LA = reduce(hcat, first.(resA))'; LB = reduce(hcat, first.(resB))'
    TA = reduce(hcat, last.(resA))'; TB = reduce(hcat, last.(resB))'
    tlik = vec(median(vcat(TA, TB), dims = 1))
    ess(w) = sum(w)^2 / sum(w .^ 2)
    function simulate_schedule(Δ; target = 0.95, thr = 0.5, rng = MersenneTwister(1))   # = rtvint104 preflight.jl (equal start weights)
        n = length(Δ); w = ones(n); d = copy(Δ); d[.!isfinite.(d)] .= -1e300
        φ = 0.0; essprev = ess(w); resampled = true; stages = 0; nres = 0
        while φ < 1.0 && stages < 5000
            stages += 1
            bar = resampled ? target * n : target * essprev
            f(x) = (lw = log.(w) .+ (x - φ) .* d; e = exp.(lw .- maximum(lw)); ess(e) - bar)
            φn = if f(1.0) >= 0
                1.0
            else
                lo, hi = φ, 1.0
                for _ in 1:200; mid = (lo + hi) / 2; f(mid) >= 0 ? (lo = mid) : (hi = mid); end
                lo
            end
            lw = log.(w) .+ (φn - φ) .* d; e = exp.(lw .- maximum(lw)); w = e ./ sum(e) .* n
            essprev = ess(w); φ = φn; resampled = false
            if essprev < thr * n
                cw = cumsum(w) ./ n; u = (rand(rng) .+ (0:n-1)) ./ n
                idx = [min(searchsortedfirst(cw, x), n) for x in u]
                d = d[idx]; w = ones(n); resampled = true; nres += 1; essprev = n
            end
        end
        return stages, nres
    end
    rows = []
    println("\nstep  old -> new      | proxy A (2019Q4 RT seed, 38 slots from prior)               | proxy B (full-sample ss104v2 cloud)                       | lik time old/new (s)")
    println("                     |  mean Δ   sd Δ   n-Inf  ESS/N(φ=1)  stages(static) resamp | mean Δ   sd Δ   n-Inf  ESS/N(φ=1)  stages(static) resamp |")
    for i in 1:length(TS)-1
        r = Dict{Symbol, Any}(:step => i, :old => LABELS[i], :new => LABELS[i+1], :vint_old => VINT[LABELS[i]], :vint_new => VINT[LABELS[i+1]],
                              :t_old => tlik[i], :t_new => tlik[i+1])
        for (lab, L) in (("A", LA), ("B", LB))
            Δ = L[:, i+1] .- L[:, i]; fin = isfinite.(Δ)
            μ = mean(Δ[fin]); s = std(Δ[fin])
            lw = copy(Δ); lw[.!fin] .= -Inf; e = exp.(lw .- maximum(lw[fin]))
            st, nr = simulate_schedule(Δ)
            r[Symbol("mean_$lab")] = μ; r[Symbol("sd_$lab")] = s; r[Symbol("ninf_$lab")] = count(!, fin)
            r[Symbol("essN_$lab")] = ess(e) / NSUB; r[Symbol("stages_$lab")] = st; r[Symbol("resamples_$lab")] = nr
        end
        push!(rows, r)
        println(@sprintf("%3d  %s -> %s | %8.2f %6.2f %5d %10.4f %10d %8d | %8.2f %6.2f %5d %10.4f %10d %8d | %.3f / %.3f",
                         i, r[:old], r[:new], r[:mean_A], r[:sd_A], r[:ninf_A], r[:essN_A], r[:stages_A], r[:resamples_A],
                         r[:mean_B], r[:sd_B], r[:ninf_B], r[:essN_B], r[:stages_B], r[:resamples_B], r[:t_old], r[:t_new]))
    end
    df = DataFrame([(; (k => v for (k, v) in sort(collect(r), by = first))...) for r in rows])
    CSV.write(joinpath(W, "out", "preflightB_steps.csv"), df)
    jldsave(joinpath(W, "out", "preflightB.jld2"); LA = LA, LB = LB, TA = TA, TB = TB, labels = LABELS)
end
println("done ", now())
