using DSGE, Test, ModelConstructors, Dates

# Drop the legacy packet/plot workflow: its 191010 population data and
# generated forecast fixtures are absent from the repository. Keep the
# independent packet-label checks, without requiring a FRED credential.
@testset "Packet forecast month labels" begin
    m = Model1002("ss10")
    m <= Setting(:data_vintage, "191010")
    for (forecast_start, expected) in ((Date(2019, 3, 31), "Oct"),
                                       (Date(2019, 12, 31), "Dec"),
                                       (Date(2018, 12, 31), "Oct"),
                                       (Date(2020, 3, 31), "Mar"))
        m <= Setting(:date_forecast_start, forecast_start)
        @test DSGE.month_label(m) == expected
    end
end
