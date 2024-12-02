@inline eye(n::Integer) = Matrix{Float64}(I,n,n)

function measurement(m::OnionModel{T},
                     TTT::Matrix{T},
                     RRR::Matrix{T},
                     CCC::Vector{T}) where {T<: AbstractFloat}

    endo     = m.endogenous_states
    endo_new = m.endogenous_states_augmented
    exo      = m.exogenous_shocks
    obs      = m.observables

    _n_observables      = n_observables(m)
    _n_states           = n_states_augmented(m)
    _n_shocks_exogenous = n_shocks_exogenous(m)
    _n = get_setting(m, :n_sectors)

    ZZ = zeros(_n_observables, _n_states)
    DD = zeros(_n_observables)
    EE = zeros(_n_observables, _n_observables)
    QQ = zeros(_n_shocks_exogenous, _n_shocks_exogenous)

    ## Demeaned Consumption Growth
    ZZ[obs[:consumption_growth], endo[:c_t]]  = 1.0
    ZZ[obs[:consumption_growth], endo_new[:c_t1]] = -1.0

    ## Demeaned Real Wage Growth
    ZZ[obs[:real_wage_growth], endo[:w_t]] = 1.
    ZZ[obs[:real_wage_growth], endo_new[:w_t1]] = - 1.



    ## Demeaned CPI Inflation
    ZZ[obs[:cpi_inflation], endo[:πc_t]] = 1.


    ## Demeaned Sector "i" CPI
    inflation_sector_names = get_setting(m, :sector_names)
    for i in 1:get_setting(m, :n_sectors)
        ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
        if get_setting(m, :sectoral_nan)
            ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = NaN
        end
    end

    ## Demeaned FFR
    ZZ[obs[:NominalFFR], endo[:r_t]] = 1.

    ## Populating QQ Matrix
    # Until this point, all iid shocks are off -- QQs are initialized as the 0 matrix
    # All trend shocks are off
    # All sector obs are off (other than experiment 4 where all the sector obs are on).
    # Tau is off
    # pistar is off

    #All aggregate shocks are populated AFTER the experiment settings block


    QQ[exo[:τ_sh], exo[:τ_sh]] = 0.0 #This is just here to be explicit. Was 1.0 to replciate Kanzig IRFs


    if get_setting(m, :marco_test_num) == 0
        println("running base case")

    elseif get_setting(m, :marco_test_num) == 1
        #One sector iid shocks, no observations
        QQ[exo[Symbol("μ_iid_1_sh")], exo[Symbol("μ_iid_1_sh")]] =  get_setting(m,:std_overrides)[1] * m[:σ_μ]^2

    elseif get_setting(m, :marco_test_num) == 2
        #Pi star shocks on and one sector observed
        QQ[exo[:πstar_sh], exo[:πstar_sh]] = m[:σ_πstar]^2
        ZZ[obs[Symbol("Inflation, $(inflation_sector_names[8])")], endo[Symbol("π_8")]] = 1.0

    elseif get_setting(m, :marco_test_num) == 3
        #One sector iid shocks and corresponding sector obs. No pi star is implicit
        QQ[exo[Symbol("μ_iid_8_sh")], exo[Symbol("μ_iid_8_sh")]] =  get_setting(m,:std_overrides)[8] * m[:σ_μ]^2

        ZZ[obs[Symbol("Inflation, $(inflation_sector_names[8])")], endo[Symbol("π_8")]] = 1.0

    elseif get_setting(m, :marco_test_num) == 4
        #No pi star, all sectors iid and all sectors observations
        QQ[exo[:πstar_sh], exo[:πstar_sh]] = 0.0
        QQ[exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")], exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")]] = diagm(get_setting(m,:std_overrides)) * m[:σ_μ]^2
        for i in 1:get_setting(m, :n_sectors)
            ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
        end


    elseif get_setting(m, :marco_test_num) == 5 #Works, matches 4

        #In exp 5, we make trend shocks as iid by setting ρ_μ_trend = 0.0 -- should exactly replicate 4, but with different color bars
        #Rationale: When we ran persistant μ shocks, we got crazy things. Natural experiment is to set ρ_μ_trend = 0 and variance of trend shocks to that of iid shocks.

        QQ[exo[:πstar_sh], exo[:πstar_sh]] = 0.0
        QQ[exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")], exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")]] = diagm(get_setting(m,:std_overrides)) * m[:σ_μ]^2
        #m[:ρ_μ_trend].value = 0.0 set upon initializing the model for this setting.


        QQ[exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")], exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")]] .= 0.0   #diagm(get_setting(m,:std_overrides)) * m[:σ_μ]^2
        for i in 1:get_setting(m, :n_sectors)
            ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
        end


    elseif get_setting(m, :marco_test_num) == 6
        #Same as 5, but ρ_μ_trend markup is set to DSGE value, not 0.999 (0.88ish)
        #m[:ρ_μ_trend] = m[:ρ_μ]

        QQ[exo[:πstar_sh], exo[:πstar_sh]] = 0.0
        QQ[exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")], exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")]] = diagm(get_setting(m,:std_overrides)) * m[:σ_μ]^2

        QQ[exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")], exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")]] .= 0.0   #diagm(get_setting(m,:std_overrides)) * m[:σ_μ]^2
        for i in 1:get_setting(m, :n_sectors)
            ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
        end


    elseif get_setting(m, :marco_test_num) == 7
        #Getting closer to the initial model. We have iid and trend markup shocks both on, ρ_μ_trend back to 0.999 for trend.

        #m[ρ_μ_trend] = 0.999 , set where model is initialized

        QQ[exo[:πstar_sh], exo[:πstar_sh]] = 0.0
        QQ[exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")], exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")]] = diagm(get_setting(m,:std_overrides)) * m[:σ_μ]^2

        #We now set the variance of markup trend shocks to the variance of π⋆ shocks.
        QQ[exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")], exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")]] =  eye(_n) * m[:σ_πstar]^2


        for i in 1:get_setting(m, :n_sectors)
            ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
        end



         elseif get_setting(m, :marco_test_num) == 8
        #Same as 6, but ρ_μ_trend = 0.7 set upon initializing the model. No iid markup shocks or π⋆.

        #m[:ρ_μ_trend].value = 0.7 set upon initializing the model for this setting.

        QQ[exo[:πstar_sh], exo[:πstar_sh]] = 0.0
        QQ[exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")], exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")]] = diagm(get_setting(m,:std_overrides)) * m[:σ_μ]^2

        QQ[exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")], exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")]] .= 0.0   #diagm(get_setting(m,:std_overrides)) * m[:σ_μ]^2
        for i in 1:get_setting(m, :n_sectors)
            ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
        end


    elseif get_setting(m, :marco_test_num) == 9
        #Same as 7, but we divide variance of μ_trend by 100. Still ρ_μ_trend = 0.999

        #m[ρ_μ_trend] = 0.999 , set where model is initialized

        QQ[exo[:πstar_sh], exo[:πstar_sh]] = 0.0
        QQ[exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")], exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")]] = diagm(get_setting(m,:std_overrides)) * m[:σ_μ]^2

        #Set the variance of markup trend shocks to the variance of π⋆ shocks/100
        QQ[exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")], exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")]] =  (eye(_n) * m[:σ_πstar]^2)/100


        for i in 1:get_setting(m, :n_sectors)
            ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
        end

elseif get_setting(m, :marco_test_num) == 10
#Back to variance of trend = variance of iid
#Set ρ_μ_trend back to 0.0 in the model. Ideally we get back to some markup shocks that add back to what we get in experiment 4

        QQ[exo[:πstar_sh], exo[:πstar_sh]] = 0.0
        QQ[exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")], exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")]] = diagm(get_setting(m,:std_overrides)) * m[:σ_μ]^2

        #We now set the variance of markup trend shocks to the variance of π⋆ shocks.
        QQ[exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")], exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")]] =  diagm(get_setting(m,:std_overrides)) * m[:σ_μ]^2

        #m[ρ_μ_trend] = 0.0 , set where model is initialized
        for i in 1:get_setting(m, :n_sectors)
            ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
        end


elseif get_setting(m, :marco_test_num) == 11
#Same as 6, but ρ_μ_trend = 0.7 (no iid, variance of trend shocks is variance of π⋆ / 100)

#m[:ρ_μ_trend].value = 0.7 set upon initializing the model for this setting.

QQ[exo[:πstar_sh], exo[:πstar_sh]] = m[:σ_πstar]^2
QQ[exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")], exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")]] = (diagm(get_setting(m,:std_overrides)) * m[:σ_μ]^2)/100

        QQ[exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")], exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")]] .= 0.0   #diagm(get_setting(m,:std_overrides)) * m[:σ_μ]^2
        for i in 1:get_setting(m, :n_sectors)
            ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
        end
    end

## Populating aggregate shocks ####

    #QQ[exo[:πstar_sh], exo[:πstar_sh]] = m[:σ_πstar]^2 * 0.0
    QQ[exo[:mp_sh], exo[:mp_sh]] = m[:σ_r_m]^2
    QQ[exo[:a_sh], exo[:a_sh]] = m[:σ_a_t]^2
    QQ[exo[:b_sh], exo[:b_sh]] = m[:σ_b_t]^2
    QQ[exo[:μw_sh], exo[:μw_sh]] = m[:σ_μw]^2


    ## Demeaned 10Y Inflation Expectations
    # TTT10, CCC10 = k_periods_ahead_expected_sums(TTT, CCC, 40)
    # ZZ[obs[:inflation_expectations_10year], :] = view(TTT10, endo[:li], :)

    return Measurement(ZZ, DD, QQ, EE)
end
