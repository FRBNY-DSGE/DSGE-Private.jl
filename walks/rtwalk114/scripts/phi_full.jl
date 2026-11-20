# RTWALK114 FINAL STEP: print φ, ESS of the newest stage files of the 2026Q2 step (read only; 1 slot SGE)
using SMC, JLD2
D = "/data/dsge_data_dir/estimation-AI/work/rtwalk114/saveroot/output_data/m1002/ss114/estimate/raw"
fs = filter(f -> occursin("sampleend=2026Q2_vint=261013_stage=", f) && endswith(f, ".jld2"), readdir(D))
for f in sort(fs, by = f -> parse(Int, match(r"stage=(\d+)", f)[1]))
    try
        c = load(joinpath(D, f), "cloud")
        println(f[end-15:end], "  phi ", c.tempering_schedule[end], "  ESS ", round(c.ESS[end], digits = 0), "  n_stages ", length(c.tempering_schedule))
    catch e
        println(f, " does not load: ", e)
    end
end
