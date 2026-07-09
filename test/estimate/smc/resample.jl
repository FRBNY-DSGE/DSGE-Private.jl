writing_output = false
if VERSION < v"1.5"
    ver = "111"
elseif VERSION < v"1.6"
    ver = "150"
elseif VERSION < v"1.7"
    ver = "160"
else
    # Julia 1.7 switched default RNG from MersenneTwister to per-Task Xoshiro256++
    ver = "1126"
end

Random.seed!(42)  # use_parallel_workers=false; @everywhere advances calling-task RNG non-deterministically

# Anchor fixture path to the test file's location so this works from any CWD
# (the resample fixtures live in test/reference/, two levels up from here).
resample_ref = joinpath(dirname(@__FILE__), "..", "..", "reference",
                        "resample_version=" * ver * ".jld2")

weights = rand(400)
weights = weights ./ sum(weights)

test_sys_resample    = SMC.resample(weights, method = :systematic)
test_multi_resample  = SMC.resample(weights, method = :multinomial)
test_poly_resample   = SMC.resample(weights, method = :polyalgo)

if writing_output
    jldopen(resample_ref,
            true, true, true, IOStream) do file
        write(file, "sys", test_sys_resample)
        write(file, "multi", test_multi_resample)
        write(file, "poly", test_poly_resample)
    end
end

saved_sys_resample   = load(resample_ref, "sys")
saved_multi_resample = load(resample_ref, "multi")
saved_poly_resample  = load(resample_ref, "poly")

####################################################################

@testset "Resampling methods" begin
    @test test_sys_resample   == saved_sys_resample
    @test test_multi_resample == saved_multi_resample
    @test test_poly_resample  == saved_poly_resample
end
