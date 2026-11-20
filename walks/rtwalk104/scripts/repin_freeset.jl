# RTWALK104: build the ENTER_TOL = 1e-4 free-set pin (RTWALK104-REVIEW §1, note (d)) — an ALTERNATIVE, see setup.jl — from the cached perturbation
# test of preflight.jl PART=A (out/preflightA_perturb.jld2; no likelihood is evaluated here). qsub only, 1 slot:
#   R=/data/dsge_data_dir/estimation-AI/work/rtwalk104
#   qsub -b y -q all.q -wd $R -j y -N rtw_repin -o $R/logs/repin_freeset.log -l h_vmem=8G,h_rt=00:20:00 /bin/bash $R/scripts/run.sh repin_freeset.jl
# 1. recomputes FIXED_AT with the preflight rule at the OLD tolerance (1e-6) and asserts it equals the current pin
#    (so the classification below is the same code path as the pin);
# 2. recomputes it at ENTER_TOL (default 1e-4), lists every perturbation with 1e-6 < |ΔL| <= ENTER_TOL and every
#    change of FIXED_AT per sample end; asserts FIXED_AT[2019Q4] == CAND38 and monotonicity;
# 3. if anything changed (WRITE=1): writes pinned/freeset_tol<TOL>.jld2; pinned/freeset.jld2 (the 1e-6 production pin) is
#    never overwritten. [First run, job 343092, wrote over freeset.jld2 and kept a copy; the files were then swapped back
#    by hand: freeset.jld2 = the original 1e-6 pin (46007 bytes, unchanged), freeset_tol1e-4.jld2 = the 1e-4 pin.]
using JLD2, Printf, Dates
const W = "/data/dsge_data_dir/estimation-AI/work/rtwalk104"
const PIN = joinpath(W, "pinned", "freeset.jld2")         # production pin, ENTER_TOL = 1e-6 (read only here)
const TOL = parse(Float64, get(ENV, "ENTER_TOL", "1e-4")); const NEWPIN = joinpath(W, "pinned", "freeset_tol" * get(ENV, "ENTER_TOL", "1e-4") * ".jld2")
src = PIN
pin = load(src); CAND38 = pin["cand38"]; LABELS = pin["labels"]; OLD = pin["fixed_at"]
c = load(joinpath(W, "out", "preflightA_perturb.jld2")); res = c["res"]
println("perturbation cache: ", length(res), " results; labels ", LABELS[1], "..", LABELS[end], "; source pin ", src)
# = preflight.jl enters_material, with the tolerance as an argument
enters_material(r, tol) = !(isfinite(r[4]) && isfinite(r[5]) && abs(r[5] - r[4]) <= tol)
function fixed_at(tol)
    ENT = Dict((r[1], r[2]) => false for r in res)
    for r in res; enters_material(r, tol) && (ENT[(r[1], r[2])] = true); end
    Dict(l => sort([CAND38[j] for j in eachindex(CAND38) if !ENT[(l, j)]]) for l in LABELS)
end
F6 = fixed_at(1e-6)
@assert F6 == OLD "the cache does not reproduce the 1e-6 pin"
println("1e-6 classification from the cache == pin: true")
println("\nperturbations with 1e-6 < |ΔL| <= $(TOL) (reclassified from 'enters' to 'does not enter' at this (sample end, slot, base)):")
for r in res
    d = abs(r[5] - r[4])
    isfinite(d) && 1e-6 < d <= TOL && println(@sprintf("  %s  %s[%d]  base %d  |ΔL| %.3g", r[1], CAND38[r[2]]..., r[3], d))
end
println("\nper slot that changes: max |ΔL| over the 2 bases at each sample end (Inf = non-finite)")
FN = fixed_at(TOL)
chg = sort(unique(vcat([collect(symdiff(Set(FN[l]), Set(F6[l]))) for l in LABELS]...)))
for s in chg
    j = findfirst(==(s), CAND38)
    println("  ", s[1], "[", s[2], "]: ", join([@sprintf("%s %.2g", l, maximum(r -> (isfinite(r[4]) && isfinite(r[5])) ? abs(r[5] - r[4]) : Inf,
                                                        [r for r in res if r[1] == l && r[2] == j])) for l in LABELS], ", "))
end
@assert FN["2019Q4"] == CAND38
for i in 2:length(LABELS); @assert Set(FN[LABELS[i]]) ⊆ Set(FN[LABELS[i-1]]) "not monotone at $(LABELS[i])"; end
println("\nsample end   free(1e-6) free($(TOL))   change")
for l in LABELS
    a = setdiff(FN[l], F6[l]); b = setdiff(F6[l], FN[l])
    println(rpad(l, 12), lpad(101 - length(F6[l]), 8), lpad(101 - length(FN[l]), 12), "   ",
            isempty(a) && isempty(b) ? "-" : join(vcat(["now FIXED $(k)[$r]" for (k, r) in a], ["now FREE $(k)[$r]" for (k, r) in b]), ", "))
end
if FN == F6
    println("\nfree sets UNCHANGED at ENTER_TOL = $TOL")
elseif get(ENV, "WRITE", "0") == "1"

    tmp = NEWPIN * ".tmp"
    jldsave(tmp; fixed_at = FN, labels = LABELS, cand38 = CAND38, enter_tol = TOL, built_at = string(now()),
            note = "re-pinned by repin_freeset.jl from out/preflightA_perturb.jld2 with ENTER_TOL = $TOL (RTWALK104-REVIEW §1 d); production pin freeset.jld2 = ENTER_TOL 1e-6")
    mv(tmp, NEWPIN; force = true)
    println("\nfree sets CHANGED: written to $NEWPIN ($PIN unchanged)")
else
    println("\nfree sets CHANGED; WRITE=1 not set, pin not written")
end
