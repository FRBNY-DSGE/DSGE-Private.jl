# RTWALK114: pinned/freeset.jld2 = rtwalk104's pinned/freeset.jld2 (ENTER_TOL 1e-6) with the 8 AIT slots removed from
# every FIXED_AT[t]. ss114 fixes those slots for good; every other walk candidate keeps the ss104 schedule. The pin is
# validated against ss114 by tests/test_walk114.jl (perturbation test at every sample end: no slot in FIXED_AT[t]
# enters the ss114 likelihood on the data to t).
# usage: run.sh pin_freeset.jl   (1 slot; refuses to overwrite an existing pin with different content)
using JLD2, SHA, Dates
const SRC = "/data/dsge_data_dir/estimation-AI/work/rtwalk104/pinned/freeset.jld2"
const DST = "/data/dsge_data_dir/estimation-AI/work/rtwalk114/pinned/freeset.jld2"
const AIT = Set([(:ρ_ait_rm, 1), (:σ_ait_rm, 2), [(Symbol("σ_ait_r_m$i"), 2) for i in 1:6]...])
src = load(SRC)
println("source ", SRC, " sha256 ", bytes2hex(open(sha256, SRC)), "; keys ", collect(keys(src)))
fa104 = src["fixed_at"]; tol = haskey(src, "enter_tol") ? src["enter_tol"] : 1e-6
@assert tol == 1e-6
fa = Dict{String, Vector{Tuple{Symbol, Int}}}(l => sort([x for x in v if !(x in AIT)]) for (l, v) in fa104)
for l in sort(collect(keys(fa)))
    println(rpad(l, 8), " ss104 fixed ", lpad(length(fa104[l]), 2), "  ss114 fixed ", lpad(length(fa[l]), 2),
            "  (AIT removed: ", join(sort(["$(k)[$r]" for (k, r) in fa104[l] if (k, r) in AIT]), " "), ")")
end
labs = sort(collect(keys(fa)))
for i in 2:length(labs); @assert Set(fa[labs[i]]) ⊆ Set(fa[labs[i-1]]) "not monotone at $(labs[i])"; end
if isfile(DST)
    @assert load(DST, "fixed_at") == fa "existing $DST differs"
    println("existing pin equal: ", DST)
else
    jldsave(DST; fixed_at = fa, enter_tol = tol, source = SRC, sha_source = bytes2hex(open(sha256, SRC)),
            removed = sort(collect(AIT)), built_at = string(now()))
    println("written ", DST, " sha256 ", bytes2hex(open(sha256, DST)))
end
