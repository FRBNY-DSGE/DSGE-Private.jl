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


    ######################
    #First, define all aggregate observables. These do not change across models
    #Remember that all ZZ and QQ matrices are initialized as 0's. If no entry is explicitly added for an observable, it remains 0.
    #######################

    ## Demeaned Consumption Growth
    ZZ[obs[:consumption_growth], endo[:c_t]]  = 1.0
    ZZ[obs[:consumption_growth], endo_new[:c_t1]] = -1.0

    ## Demeaned Real Wage Growth
    ZZ[obs[:real_wage_growth], endo[:w_t]] = 1.
    ZZ[obs[:real_wage_growth], endo_new[:w_t1]] = - 1.

    ## Demeaned FFR
    ZZ[obs[:NominalFFR], endo[:r_t]] = 1.

    #Demeaned CPI Inflation, using Keshav's Gamma (final consumption shares)
    ZZ[obs[:cpi_inflation], endo[:πKc_t]] = 1.


    ## Demeaned CPI Inflation -- will leave here commented out. For much of the briefing results, we used the above K-CPI
    #ZZ[obs[:cpi_inflation], endo[:πc_t]] = 1.




    ##### Demeaned sector i CPI. Left here to show that it is not ignored -- we only add in certain specifications.
    #=
    ## Demeaned Sector "i" CPI
    inflation_sector_names = get_setting(m, :sector_names)
    for i in 1:get_setting(m, :n_sectors)
    ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
    if get_setting(m, :sectoral_nan)
    ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = NaN
    end
    end
    =#






    ######################
    #Populating QQ Matrix
    # QQs are initialized as the 0 matrix
    # All trend shocks are off
    # All sector obs are off (other than experiment 4 where all the sector obs are on).
    # Tau is off for all -- We leave that below explicitly for tractibility.
    ######################

    QQ[exo[:τ_sh], exo[:τ_sh]] = 0.0 #This is just here to be explicit. Was 1.0 to replciate Kanzig IRFs

    ## Populating aggregate shocks ####

    #π* shocks: Aggregate shock, but off in most of the cases. Leaving here just for tractibility
    #QQ[exo[:πstar_sh], exo[:πstar_sh]] = m[:σ_πstar]^2

    #Monetary policy
    QQ[exo[:mp_sh], exo[:mp_sh]] = m[:σ_r_m]^2
    #TFP
    QQ[exo[:a_sh], exo[:a_sh]] = m[:σ_a_t]^2
    #Discount rate
    QQ[exo[:b_sh], exo[:b_sh]] = m[:σ_b_t]^2
    #Wage markup
    QQ[exo[:μw_sh], exo[:μw_sh]] = m[:σ_μw]^2


    if get_setting(m, :marco_test_num) == 0
        println("running base case")
        #We have the 4 main aggregate observables and 4 main aggregate shocks. Nothing sectoral to give a baseline.


    elseif get_setting(m, :marco_test_num) == 1
        #Now, we include one sector's iid shocks, without observing that sector

        QQ[exo[Symbol("μ_iid_1_sh")], exo[Symbol("μ_iid_1_sh")]] =  get_setting(m,:std_overrides)[1] * m[:σ_μ]^2

    elseif get_setting(m, :marco_test_num) == 2
        #Pi star shocks on, no sector shocks. One sector is observed

        QQ[exo[:πstar_sh], exo[:πstar_sh]] = m[:σ_πstar]^2

        #CPI Sector 8 is Utilities
        ZZ[obs[Symbol("Inflation, $(inflation_sector_names[8])")], endo[Symbol("π_8")]] = 1.0

    elseif get_setting(m, :marco_test_num) == 3

        #One sector iid shocks and corresponding sector obs. We return to no pi star. Henceforth no πstar is implicit unless it is explicitly included.

        QQ[exo[Symbol("μ_iid_8_sh")], exo[Symbol("μ_iid_8_sh")]] =  get_setting(m,:std_overrides)[8] * m[:σ_μ]^2
        ZZ[obs[Symbol("Inflation, $(inflation_sector_names[8])")], endo[Symbol("π_8")]] = 1.0

    elseif get_setting(m, :marco_test_num) == 4
        #Include all sectoral iid shocks and all sectors observations

        QQ[exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")], exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")]] = diagm(get_setting(m,:std_overrides)) * m[:σ_μ]^2
        for i in 1:get_setting(m, :n_sectors)
            ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
        end


    elseif get_setting(m, :marco_test_num) == 5 #Works, matches 4
        #Make the sectoral trend shocks iid by setting ρ_μ_trend = 0.0 and setting variance of these shocks to the variance of iid sectoral shocks.

        #In exp 5, we make trend shocks as iid by setting ρ_μ_trend = 0.0 -- should exactly replicate 4, but with different color bars
        #Rationale: When we ran persistant μ shocks, we got crazy things. Natural experiment is to set ρ_μ_trend = 0 and variance of trend shocks to that of iid shocks.


        QQ[exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")], exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")]] = diagm(get_setting(m,:std_overrides)) * m[:σ_μ]^2
        #m[:ρ_μ_trend].value = 0.0 set upon initializing the model for this setting.

        #The line below is implicit, but leaving for interpretability
        #QQ[exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")], exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")]] .= 0.0

        for i in 1:get_setting(m, :n_sectors)
            ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
        end


    elseif get_setting(m, :marco_test_num) == 6
        #Same as 5, but ρ_μ_trend markup is set to DSGE value, not 0.999 (0.88ish)
        #m[:ρ_μ_trend] = m[:ρ_μ]


        QQ[exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")], exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")]] = diagm(get_setting(m,:std_overrides)) * m[:σ_μ]^2



        #= BP -- WHY IS THIS HERE?? DON'T THINK WE EVER RAN WITH THIS. INITIAL FIRST EXP 6 WAS WITHOUT =#
        #Populate common and categorical shocks
        #QQ[exo[:μ_com_sh], exo[:μ_com_sh]] = m[:σ_μ]^2

        for i in 1:get_setting(m, :n_sectors)
            ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
        end



    elseif get_setting(m, :marco_test_num) == 66
        #Same as 6, but we do not observe or track sectoral CPI. Instead, we track CPI categories (energy, core goods, core services)
        #m[:ρ_μ_trend] = 0.95 (was 0.8827)
        #No π⋆

        #Populate common and categorical shocks
        QQ[exo[:μ_com_sh], exo[:μ_com_sh]] = 0.0  #m[:σ_μ]^2
        QQ[exo[:μ_com_goods_sh], exo[:μ_com_goods_sh]] = m[:σ_μ]^2
        QQ[exo[:μ_com_services_sh], exo[:μ_com_services_sh]] = m[:σ_μ]^2
        QQ[exo[:μ_com_energy_sh], exo[:μ_com_energy_sh]] = m[:σ_μ]^2


        #Kill prodcutivity:
        QQ[exo[:a_sh], exo[:a_sh]] = 0.0


        #No observables in individual sectors, but in categorical sectors
        ZZ[obs[:cpi_inflation], endo[:πKc_t]] = 0.0

        #Demeaned Core Services:
        core_services_sum = sum(m[:Kgam].value[get_setting(m, :core_service_sectors)])
        for i in get_setting(m, :core_service_sectors)
            ZZ[obs[:cpi_core_services], endo[Symbol("π_$i")]] = m[:Kgam].value[i] / core_services_sum
        end


        #Demeaned Core Goods:
        core_goods_sum = sum(m[:Kgam].value[get_setting(m, :core_goods_sectors)])
        for i in get_setting(m, :core_goods_sectors)
            ZZ[obs[:cpi_core_goods], endo[Symbol("π_$i")]] = m[:Kgam].value[i] / core_goods_sum
        end


        #Demeaned Energy CPI
        energy_sum = sum(m[:Kgam].value[get_setting(m, :energy_sectors)])
        for i in get_setting(m, :energy_sectors)
            ZZ[obs[:cpi_energy], endo[Symbol("π_$i")]] = m[:Kgam].value[i] / energy_sum
        end



        elseif get_setting(m, :marco_test_num) == 68
        #Same as 5, but ρ_μ_trend markup is set to DSGE value, not 0.999 (0.88ish)
        #No π⋆
        #MAIN MODEL FOR THE BRIEFING!!!

        #No common shock and categorical shocks
        QQ[exo[:μ_com_sh], exo[:μ_com_sh]] = 0.0  #Explicit here for clarity
        QQ[exo[:μ_com_goods_sh], exo[:μ_com_goods_sh]] = m[:σ_μ]^2
        QQ[exo[:μ_com_services_sh], exo[:μ_com_services_sh]] = m[:σ_μ]^2
        QQ[exo[:μ_com_energy_sh], exo[:μ_com_energy_sh]] = m[:σ_μ]^2


#Bring productivity shocks back (Done at the top already)

# Include tfp measurement
ZZ[obs[:obs_tfp], endo[:a_t]] = 1.


#No observables in individual sectors, but in categorical sectors
ZZ[obs[:cpi_inflation], endo[:πKc_t]] = 0.0

#Demeaned Core Services:
core_services_sum = sum(m[:Kgam].value[get_setting(m, :core_service_sectors)])
for i in get_setting(m, :core_service_sectors)
    ZZ[obs[:cpi_core_services], endo[Symbol("π_$i")]] = m[:Kgam].value[i] / core_services_sum
end


#Demeaned Core Goods:
core_goods_sum = sum(m[:Kgam].value[get_setting(m, :core_goods_sectors)])
for i in get_setting(m, :core_goods_sectors)
    ZZ[obs[:cpi_core_goods], endo[Symbol("π_$i")]] = m[:Kgam].value[i] / core_goods_sum
end


#Demeaned Energy CPI
energy_sum = sum(m[:Kgam].value[get_setting(m, :energy_sectors)])
for i in get_setting(m, :energy_sectors)
    ZZ[obs[:cpi_energy], endo[Symbol("π_$i")]] = m[:Kgam].value[i] / energy_sum
end




elseif get_setting(m, :marco_test_num) == 69
#Same as 5, but ρ_μ_trend markup is set to DSGE value, not 0.999 (0.88ish)
#m[:ρ_μ_trend] =  0.8827
#No π⋆

#Populate common and categorical shocks
QQ[exo[:μ_com_sh], exo[:μ_com_sh]] =  m[:σ_μ]^2
QQ[exo[:μ_com_goods_sh], exo[:μ_com_goods_sh]] = m[:σ_μ]^2
QQ[exo[:μ_com_services_sh], exo[:μ_com_services_sh]] = m[:σ_μ]^2
QQ[exo[:μ_com_energy_sh], exo[:μ_com_energy_sh]] = m[:σ_μ]^2

#Now, add back observable CPI
ZZ[obs[:cpi_inflation], endo[:πKc_t]] = 1.

#Kill prodcutivity:
QQ[exo[:a_sh], exo[:a_sh]] = 0.0


#No observables in individual sectors, but in categorical sectors

#Demeaned Core Services:
core_services_sum = sum(m[:Kgam].value[get_setting(m, :core_service_sectors)])
for i in get_setting(m, :core_service_sectors)
    ZZ[obs[:cpi_core_services], endo[Symbol("π_$i")]] = m[:Kgam].value[i] / core_services_sum
end


#Demeaned Core Goods:
core_goods_sum = sum(m[:Kgam].value[get_setting(m, :core_goods_sectors)])
for i in get_setting(m, :core_goods_sectors)
    ZZ[obs[:cpi_core_goods], endo[Symbol("π_$i")]] = m[:Kgam].value[i] / core_goods_sum
end


#Demeaned Energy CPI
energy_sum = sum(m[:Kgam].value[get_setting(m, :energy_sectors)])
for i in get_setting(m, :energy_sectors)
    ZZ[obs[:cpi_energy], endo[Symbol("π_$i")]] = m[:Kgam].value[i] / energy_sum
end



elseif get_setting(m, :marco_test_num) == 70
#Same as 5, but ρ_μ_trend markup is set to DSGE value, not 0.999 (0.88ish)
#m[:ρ_μ_trend] = 0.95 (was 0.8827)
#π⋆ ON, long run inflation expectations are on (minus 2.3, divided by 4)
#Not well behaved, not used.



#Populate common and categorical shocks
QQ[exo[:πstar_sh], exo[:πstar_sh]] = m[:σ_πstar]^2
QQ[exo[:μ_com_sh], exo[:μ_com_sh]] = 0.0  #m[:σ_μ]^2
QQ[exo[:μ_com_goods_sh], exo[:μ_com_goods_sh]] = m[:σ_μ]^2
QQ[exo[:μ_com_services_sh], exo[:μ_com_services_sh]] = m[:σ_μ]^2
QQ[exo[:μ_com_energy_sh], exo[:μ_com_energy_sh]] = m[:σ_μ]^2


#Bring productivity shocks back

# Include tfp measurement
ZZ[obs[:obs_tfp], endo[:a_t]] = 1.


#No observables in individual sectors, but in categorical sectors
ZZ[obs[:cpi_inflation], endo[:πKc_t]] = 1.
#Demeaned Core Services:
core_services_sum = sum(m[:Kgam].value[get_setting(m, :core_service_sectors)])
for i in get_setting(m, :core_service_sectors)
    ZZ[obs[:cpi_core_services], endo[Symbol("π_$i")]] = m[:Kgam].value[i] / core_services_sum
end


#Demeaned Core Goods:
core_goods_sum = sum(m[:Kgam].value[get_setting(m, :core_goods_sectors)])
for i in get_setting(m, :core_goods_sectors)
    ZZ[obs[:cpi_core_goods], endo[Symbol("π_$i")]] = m[:Kgam].value[i] / core_goods_sum
end


#Demeaned Energy CPI
energy_sum = sum(m[:Kgam].value[get_setting(m, :energy_sectors)])
for i in get_setting(m, :energy_sectors)
    ZZ[obs[:cpi_energy], endo[Symbol("π_$i")]] = m[:Kgam].value[i] / energy_sum
end


#Include long run inflation expectations: Need to calculate 40 quarter ahead inflation
TTTs = Matrix{T}[]
CCCs = Matrix{T}[]
memo = nothing
permanent_t = 1
TTT10 = (I - TTT) \ (TTT - TTT^40)
#Confirm all of my Cs are 0

CCC10 = CCC

TTT10        = TTT10 ./ 40.
CCC10        = CCC10 ./ 40.

ZZ[obs[:obs_longinflation], :] = view(TTT10, endo[:πKc_t], :)
#Leave out the constant:
#DD[obs[:obs_longinflation]]    = 100*(m[:π_star]-1) + CCC10[endo[:π_t]]




elseif get_setting(m, :marco_test_num) == 71
#Same as 68, but with no I/O matrices -- this is done in the model set up.

#The goal of this specification is to run a counterfactual exercise, done in /data/dsge_data_dir/proc/dsge/briefings/202412/counterfactual_noIO.jl
#We initialize a model, under spec 68, and smooth to find the intial state and history of shocks.
#Then, we find the system matrices for model 71 (no IO) and run a counterfactual forecast of this new system from the initial state and smoothed shock series found from smoothing the original system.
#Plotting functions are naive but included in the spec.

#Populate common and categorical shocks
#QQ[exo[:πstar_sh], exo[:πstar_sh]] = m[:σ_πstar]^2
QQ[exo[:μ_com_sh], exo[:μ_com_sh]] = 0.0  #m[:σ_μ]^2
QQ[exo[:μ_com_goods_sh], exo[:μ_com_goods_sh]] = m[:σ_μ]^2
QQ[exo[:μ_com_services_sh], exo[:μ_com_services_sh]] = m[:σ_μ]^2
QQ[exo[:μ_com_energy_sh], exo[:μ_com_energy_sh]] = m[:σ_μ]^2


#Bring productivity shocks back (Done at the top)

# Include tfp measurement
ZZ[obs[:obs_tfp], endo[:a_t]] = 1.



#No observables in individual sectors, but in categorical sectors
ZZ[obs[:cpi_inflation], endo[:πKc_t]] = 1.
#Demeaned Core Services:
core_services_sum = sum(m[:Kgam].value[get_setting(m, :core_service_sectors)])
for i in get_setting(m, :core_service_sectors)
    ZZ[obs[:cpi_core_services], endo[Symbol("π_$i")]] = m[:Kgam].value[i] / core_services_sum
end


#Demeaned Core Goods:
core_goods_sum = sum(m[:Kgam].value[get_setting(m, :core_goods_sectors)])
for i in get_setting(m, :core_goods_sectors)
    ZZ[obs[:cpi_core_goods], endo[Symbol("π_$i")]] = m[:Kgam].value[i] / core_goods_sum
end


#Demeaned Energy CPI
energy_sum = sum(m[:Kgam].value[get_setting(m, :energy_sectors)])
for i in get_setting(m, :energy_sectors)
    ZZ[obs[:cpi_energy], endo[Symbol("π_$i")]] = m[:Kgam].value[i] / energy_sum
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

        for i in 1:get_setting(m, :n_sectors)
            ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
        end


    elseif get_setting(m, :marco_test_num) == 9
        #Same as 7, but we divide variance of μ_trend by 100. Still ρ_μ_trend = 0.999

        #m[ρ_μ_trend] = 0.999 , set where model is initialized

        QQ[exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")], exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")]] = diagm(get_setting(m,:std_overrides)) * m[:σ_μ]^2

        #Set the variance of markup trend shocks to the variance of π⋆ shocks/100
        QQ[exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")], exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")]] =  (eye(_n) * m[:σ_πstar]^2)/100


        for i in 1:get_setting(m, :n_sectors)
            ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
        end

elseif get_setting(m, :marco_test_num) == 10
#Back to variance of trend = variance of iid
#Set ρ_μ_trend back to 0.0 in the model. Ideally we get back to some markup shocks that add back to what we get in experiment 4

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

        for i in 1:get_setting(m, :n_sectors)
            ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
        end
    end


    return Measurement(ZZ, DD, QQ, EE)
end
