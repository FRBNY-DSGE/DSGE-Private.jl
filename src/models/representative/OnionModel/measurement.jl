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
        if i ∉ get_setting(m, :nan_sects)
            ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
            #QQ[exo[Symbol("μ_trend_$(i)_sh")], exo[Symbol("μ_trend_$(i)_sh")] = (m[:σ_πstar]/100)^2
        else
            ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = NaN
            #QQ[exo[Symbol("μ_trend_$(i)_sh")], exo[Symbol("μ_trend_$(i)_sh")] = 0.0
        end

    end

    ## Demeaned FFR
    ZZ[obs[:NominalFFR], endo[:r_t]] = 1.

    QQ[exo[:τ_sh], exo[:τ_sh]] = 1.0 #1.0 works to replicate the Kanzig oil IRFs




    QQ[exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")], exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")]] = eye(_n) * (m[:σ_πstar]/100)^2


    if get_setting(m, :override_stds)
        #std_devs = [x^2 for x in get_setting(m, :std_overrides)
        QQ[exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")], exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")]] = diagm(get_setting(m,:std_overrides))
        #=for i in 1:get_setting(m, :n_sectors)
            if i ∉ get_setting(m, :nan_sects)
                QQ[exo[Symbol("μ_iid_$(_n)_sh")], exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")]] = diagm(get_setting(m,:std_overrides))
            else
                QQ[exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")], exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")]] = diagm(get_setting(m,:std_overrides)) * 0.0
            end

        end=#

    else
        #QQ[exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")], exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")]] = eye(n)
        QQ[exo[Symbol("μ_1_sh")]:exo[Symbol("μ_$(_n)_sh")], exo[Symbol("μ_1_sh")]:exo[Symbol("μ_$(_n)_sh")]] = eye(n)
    end

    QQ[exo[:πstar_sh], exo[:πstar_sh]] = m[:σ_πstar]^2 * 0.0
    QQ[exo[:mp_sh], exo[:mp_sh]] = m[:σ_r_m]^2
    QQ[exo[:a_sh], exo[:a_sh]] = m[:σ_a_t]^2
    QQ[exo[:b_sh], exo[:b_sh]] = m[:σ_b_t]^2
    QQ[exo[:μw_sh], exo[:μw_sh]] = m[:σ_μw]^2


    ## Demeaned 10Y Inflation Expectations
    # TTT10, CCC10 = k_periods_ahead_expected_sums(TTT, CCC, 40)
    # ZZ[obs[:inflation_expectations_10year], :] = view(TTT10, endo[:li], :)

    return Measurement(ZZ, DD, QQ, EE)
end
