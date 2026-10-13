using DSGE, Test, ModelConstructors, DataFrames, Dates, BenchmarkTools

N = 6
m = AnSchorfheide()
m <= Setting(:n_mon_anticipated_shocks, N)
m <= Setting(:date_presample_start, Date(2019, 12, 31))

# Quarterly (end-of-quarter) dates spanning the pgap/ygap init and ZLB windows
# the function keys off of (2020-06-30 init; 2020-12-31..2021-12-31 ZLB).
dates = [Date(2020, 3, 31), Date(2020, 6, 30), Date(2020, 9, 30), Date(2020, 12, 31),
         Date(2021, 3, 31), Date(2021, 6, 30), Date(2021, 9, 30), Date(2021, 12, 31)]

function make_df()
    df = DataFrame(:date => copy(dates))
    df[!, :obs_nominalrate] = fill(1.0, length(dates))
    # The source's expected-rate masking (manual_data_adjustments.jl:53-56) maps
    # submatrix columns back with `k[2] + minimum(exp_inds) - 1`, which only works
    # if the obs_exp_nominalrate columns are CONTIGUOUS. Real data groups them, so
    # add all obs_nominalrate{i} first, then all obs_exp_nominalrate{i}.
    for i in 1:N
        df[!, "obs_nominalrate$i"] = fill(1.0, length(dates))
    end
    for i in 1:N
        df[!, "obs_exp_nominalrate$i"] = fill(0.5, length(dates))    # > 0.033
    end
    allowmissing!(df, ["obs_exp_nominalrate$i" for i in 1:N])
    return df
end

# fcast_date < 2022-04-01 avoids the (broken) mon_ant_ait_shocks branch; cond_type
# = :none and add_22Q1_ffr = false (default) keep us on the main path.
out = DSGE.post_covid_data_mods(m, make_df(), :none, DataFrame(); fcast_date = Date(2021, 1, 1))

@testset "post_covid_data_mods preserves rows when all are in-sample" begin
    @test nrow(out) == length(dates)
    @test out[!, :date] == dates
end

@testset "post_covid_data_mods adds and seeds obs_pgap / obs_ygap" begin
    @test "obs_pgap" in names(out)
    @test "obs_ygap" in names(out)
    # Seeded at the init date (2020-06-30, row 2); NaN elsewhere
    @test out[2, :obs_pgap] == -0.125
    @test out[2, :obs_ygap] == -12.0
    @test isnan(out[1, :obs_pgap])
    @test isnan(out[1, :obs_ygap])
end

@testset "post_covid_data_mods NaNs the ZLB nominal-rate window" begin
    # ZLB runs 2020-12-31..2021-12-31 (rows 4:8)
    @test all(isnan, out[4:8, :obs_nominalrate])
    @test all(out[1:3, :obs_nominalrate] .== 1.0)          # pre-ZLB untouched
    for i in 1:N
        @test all(isnan, out[4:8, "obs_nominalrate$i"])
    end
end

@testset "post_covid_data_mods NaNs the final anticipated-rate obs" begin
    # Last row's anticipated nominal-rate obs are zeroed out (cond_type = :none)
    for i in 1:N
        @test isnan(out[end, "obs_nominalrate$i"])
    end
end

@testset "post_covid_data_mods masks the expected-rate history" begin
    # keep_Q1_spd path sets rows 1:end-3 of each obs_exp_nominalrate column missing
    for i in 1:N
        @test all(ismissing, out[1:end-3, "obs_exp_nominalrate$i"])
    end
end

@testset "post_covid_data_mods keeps the expected-rate path from :exp_ffr_data_start_date" begin
    m3 = AnSchorfheide()
    m3 <= Setting(:n_mon_anticipated_shocks, N)
    m3 <= Setting(:date_presample_start, Date(2019, 12, 31))
    m3 <= Setting(:exp_ffr_data_start_date, Date(2021, 3, 31))
    out3 = DSGE.post_covid_data_mods(m3, make_df(), :none, DataFrame(); fcast_date = Date(2021, 1, 1))
    for i in 1:N
        @test all(ismissing, out3[out3[!, :date] .< Date(2021, 3, 31), "obs_exp_nominalrate$i"])
        @test all(out3[out3[!, :date] .>= Date(2021, 3, 31), "obs_exp_nominalrate$i"] .== 0.5)
    end
end

@testset "post_covid_data_mods drops out-of-sample (pre-presample) rows" begin
    m2 = AnSchorfheide()
    m2 <= Setting(:n_mon_anticipated_shocks, N)
    # Drop only the first row (2020-03-31); keep 2020-06-30, which the source
    # requires as the pgap/ygap init date (see the BUG testset below).
    m2 <= Setting(:date_presample_start, Date(2020, 6, 30))
    out2 = DSGE.post_covid_data_mods(m2, make_df(), :none, DataFrame(); fcast_date = Date(2021, 1, 1))
    @test nrow(out2) == length(dates) - 1
    @test minimum(out2[!, :date]) == Date(2020, 6, 30)
end

@testset "ss104_estimation returns the dataframe unchanged" begin
    df = make_df()
    @test DSGE.ss104_estimation(df) === df
end

#################################################
# Regression tests for fixed bugs (bugs doc §1.11) #
#################################################

@testset "§1.11: the 2022Q2 market-expected FFR block runs, and only for 2022Q2" begin
    # Used to throw UndefVarError (`mon_ant_ait_shocks` was defined nowhere).
    out_q2 = DSGE.post_covid_data_mods(m, make_df(), :none, DataFrame(); fcast_date = Date(2022, 6, 1))
    @test out_q2[end, "obs_exp_nominalrate1"] == 0.19939
    @test out_q2[end, "obs_exp_nominalrate6"] == 0.783
    # A later unconditional forecast must keep its own data, not the May 2022 values.
    out_later = DSGE.post_covid_data_mods(m, make_df(), :none, DataFrame(); fcast_date = Date(2026, 9, 30))
    for i in 1:N
        @test out_later[end, "obs_exp_nominalrate$i"] == 0.5
    end
end

@testset "§1.11: data without the pgap/ygap init date does not throw" begin
    mbug = AnSchorfheide()
    mbug <= Setting(:n_mon_anticipated_shocks, N)
    mbug <= Setting(:date_presample_start, Date(2020, 9, 30))   # drops 2020-06-30
    out_bug = DSGE.post_covid_data_mods(mbug, make_df(), :none, DataFrame(); fcast_date = Date(2021, 1, 1))
    @test all(isnan, out_bug[!, :obs_pgap])
    @test all(isnan, out_bug[!, :obs_ygap])
end

@testset "§1.11: exported name matches the function" begin
    @test isdefined(DSGE, :post_covid_data_mods)
    @test :post_covid_data_mods in names(DSGE)
    @test !(Symbol("post_covid_data_mods!") in names(DSGE))
end

@testset "§1.11: the caller's expected_ffr vector is not mutated" begin
    exp_ffr = [0, 1, 2, 3, 4, 5, 6]
    DSGE.post_covid_data_mods(m, make_df(), :none, DataFrame(); fcast_date = Date(2021, 1, 1),
                              expected_ffr = exp_ffr)
    @test exp_ffr == [0, 1, 2, 3, 4, 5, 6]
end

@testset "§1.11: expected-rate masking works when the columns are not contiguous" begin
    df = make_df()
    df[!, "obs_exp_nominalrate3"] = allowmissing(fill(0.01, nrow(df)))   # <= 0.033, must be masked
    front = ["date", "obs_exp_nominalrate1", "obs_nominalrate"]   # exp1 now sits apart from exp2..exp6
    df = select(df, front, Not(front))
    out_nc = DSGE.post_covid_data_mods(m, df, :none, DataFrame(); fcast_date = Date(2021, 1, 1))
    @test all(ismissing, out_nc[!, "obs_exp_nominalrate3"])
    @test all(out_nc[end-2:end, "obs_exp_nominalrate1"] .== 0.5)
    @test !any(ismissing, out_nc[!, :obs_nominalrate][1:3])               # a neighboring column is untouched
end

@testset "§1.11: add_22Q1_ffr leaves unfilled horizons missing instead of 0" begin
    cm = DataFrame(:Maturity => ["8/15/2022", "11/15/2022", "8/15/2023", "8/15/2024"],
                   :rate     => [1.0, 2.0, 3.0, 4.0])
    out_cm = DSGE.post_covid_data_mods(m, make_df(), :full, DataFrame(); cm_ffr = cm, add_22Q1_ffr = true,
                                       fcast_date = Date(2021, 1, 1))
    @test out_cm[end, "obs_exp_nominalrate2"] == 0.25
    @test out_cm[end, "obs_exp_nominalrate3"] == 0.5
    @test out_cm[end, "obs_exp_nominalrate6"] == 0.75
    for i in (1, 4, 5)
        @test ismissing(out_cm[end, "obs_exp_nominalrate$i"])
    end
end

@testset "§1.11: an expected_ffr that includes 0 (current quarter) is accepted" begin
    # Before the §1.11 fix the alias-and-deleteat! dropped the 0 from the caller's vector; once the caller's
    # vector was left alone, both expected-rate blocks indexed position 0 (BoundsError)
    exp0 = [0, 1, 2, 3, 4, 5, 6]
    cm = DataFrame(:Maturity => ["8/15/2022", "11/15/2022", "8/15/2023", "8/15/2024"],
                   :rate     => [1.0, 2.0, 3.0, 4.0])
    out_cm0 = DSGE.post_covid_data_mods(m, make_df(), :full, DataFrame(); cm_ffr = cm, add_22Q1_ffr = true,
                                        fcast_date = Date(2021, 1, 1), expected_ffr = exp0)
    @test out_cm0[end, "obs_exp_nominalrate2"] == 0.25
    out_q2_0 = DSGE.post_covid_data_mods(m, make_df(), :none, DataFrame(); fcast_date = Date(2022, 6, 1),
                                         expected_ffr = exp0)
    @test out_q2_0[end, "obs_exp_nominalrate1"] == 0.19939
    @test exp0 == [0, 1, 2, 3, 4, 5, 6]
end

################
# Benchmarking #
################
run_benchmarks = false

if run_benchmarks
    b_mods = @benchmark DSGE.post_covid_data_mods($m, make_df(), :none, DataFrame();
                                                  fcast_date = Date(2021, 1, 1))
    b_ss104 = @benchmark DSGE.ss104_estimation(make_df())

    println("\n===== manual_data_adjustments benchmark results =====")
    for (name, b) in [("post_covid_data_mods", b_mods),
                      ("ss104_estimation",     b_ss104)]
        println(rpad(name, 24), " time: ", rpad(BenchmarkTools.prettytime(median(b).time), 12),
                "memory: ", BenchmarkTools.prettymemory(median(b).memory))
    end
end
