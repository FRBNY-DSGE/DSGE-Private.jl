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
    ZZ[obs[:consumption_growth], endo[:c]]       = 1.0
    ZZ[obs[:consumption_growth], endo_new[:c_1]] = -1.0

    ## Demeaned Real Wage Growth
    ZZ[obs[:real_wage_growth], :] = TTT[endo[:lw], :]
    ZZ[obs[:real_wage_growth], endo[:lw]]   = ZZ[obs[:real_wage_growth], endo[:lw]] - 1.0

    ## Demeaned CPI Inflation
    ZZ[obs[:cpi_inflation], endo[:πc]] = 1.0

    ## Demeaned Sector "i" CPI
    inflation_sector_names = get_setting(m, :sector_names)
    for i in 1:get_setting(m, :n_sectors)
        ZZ[obs[Symbol("Inflation, $(inflation_sector_names[i])")], endo[Symbol("π_$i")]] = 1.0
    end

    ## Demeaned FFR
    ZZ[obs[:NominalFFR], :] = TTT[endo[:li], :]


    QQ[exo[:τ_sh], exo[:τ_sh]] = 1.0 #This should be m[:σ_τ]^2
    QQ[exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")], exo[Symbol("μ_trend_1_sh")]:exo[Symbol("μ_trend_$(_n)_sh")]] = eye(_n)
    QQ[exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")], exo[Symbol("μ_iid_1_sh")]:exo[Symbol("μ_iid_$(_n)_sh")]] = eye(_n)
    ## Demeaned 10Y Inflation Expectations
    # TTT10, CCC10 = k_periods_ahead_expected_sums(TTT, CCC, 40)
    # ZZ[obs[:inflation_expectations_10year], :] = view(TTT10, endo[:li], :)

    return Measurement(ZZ, DD, QQ, EE)
end
