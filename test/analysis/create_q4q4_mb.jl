using DSGE, Dates, BenchmarkTools
path = dirname(@__FILE__)

write_test_output = false

# Load existing MeansBands reference (product = :forecast, 60 periods from 2015-Q4)
mb_full = load("$path/../reference/MeansBands.jld2", "mb")

# create_q4q4_mb requires a 4q product — make a copy with the right product
mb_4q = deepcopy(mb_full)
mb_4q.metadata[:product] = :forecast4q

mb_q4q4 = create_q4q4_mb(mb_4q)

if write_test_output
    JLD2.jldopen("$path/../reference/create_q4q4_mb_out.jld2", "w") do f
        f["mb_q4q4"] = mb_q4q4
    end
end

saved_mb_q4q4 = load("$path/../reference/create_q4q4_mb_out.jld2", "mb_q4q4")

@testset "create_q4q4_mb" begin
    # Error on non-4q product
    @test_throws ErrorException create_q4q4_mb(mb_full)

    # Product name updated correctly (:forecast4q -> :forecastq4q4)
    @test mb_q4q4.metadata[:product] == :forecastq4q4

    # All dates in metadata are Q4
    @test all(Dates.quarterofyear(d) == 4 for d in keys(mb_q4q4.metadata[:date_inds]))

    # All rows in means are Q4
    @test all(Dates.quarterofyear(d) == 4 for d in mb_q4q4.means[!, :date])

    # Fewer rows than original (Q4 only is a subset)
    @test size(mb_q4q4.means, 1) < size(mb_4q.means, 1)

    # Bands also filtered to Q4
    for var in keys(mb_q4q4.bands)
        @test all(Dates.quarterofyear(d) == 4 for d in mb_q4q4.bands[var][!, :date])
    end

    # Matches saved reference
    @test mb_q4q4.means == saved_mb_q4q4.means
    @test mb_q4q4.metadata[:product] == saved_mb_q4q4.metadata[:product]

    # All valid product types pass through without error
    for prod in [:hist4q, :histforecast4q, :bddforecast4q, :bddhistforecast4q]
        mb_test = deepcopy(mb_full)
        mb_test.metadata[:product] = prod
        result = create_q4q4_mb(mb_test)
        @test all(Dates.quarterofyear(d) == 4 for d in result.means[!, :date])
    end
end

display(@benchmark create_q4q4_mb($mb_4q))
