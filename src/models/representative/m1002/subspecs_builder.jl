"""
'''
setup_flexait_tempzlb!(m::AbstractDSGEModel, cond_type::Symbol, start_zlb_date::Date, end_zlb_date::Date,
                                θ::NamedTuple; pgap_ygap_init_date::Date = Date(2020, 6, 30),
                                set_regime_vals_fnct::Function = baseline_covid_set_regime_vals,
                                altpolicy::Bool = false, skip_altpolicy_state_init::Bool = false,
                                include_zlb::Bool = true, uncertain_altpolicy::Bool = true,
                                tvcred_dates::Union{Tuple{Date, Date}, Nothing} = nothing,
                                start_tvcred_level::Union{Number, Nothing} = nothing,
                                end_tvcred_level::Union{Number, Nothing} = nothing,
                                chosen_altpolicy::AltPolicy = flexible_ait(),
                                adjusted_historical_expectations::Bool = true,
                                spd_expect::Bool = true,
                                tworule_zlb::Bool = true, endo_zlb::Bool = false
'''

This is a helper function used in subspecs.jl to make the relevant model setting changes needed to setup flexible ait and temporary zero lower bound policies.

"""
function setup_flexait_tempzlb!(m::AbstractDSGEModel, cond_type::Symbol, start_zlb_date::Date, end_zlb_date::Date,
                                θ::NamedTuple; pgap_ygap_init_date::Date = Date(2020, 6, 30),
                                set_regime_vals_fnct::Function = baseline_covid_set_regime_vals,
                                altpolicy::Bool = false, skip_altpolicy_state_init::Bool = false,
                                include_zlb::Bool = true, uncertain_altpolicy::Bool = true,
                                tvcred_dates::Union{Tuple{Date, Date}, Nothing} = nothing,
                                start_tvcred_level::Union{Number, Nothing} = nothing,
                                end_tvcred_level::Union{Number, Nothing} = nothing,
                                chosen_altpolicy::AltPolicy = flexible_ait(),
                                adjusted_historical_expectations::Bool = true,
                                spd_expect::Bool = true,
                                tworule_zlb::Bool = true, endo_zlb::Bool = false)

    ## Set up imperfectly credible Flexible AIT rule
    m <= Setting(:flexible_ait_φ_π, θ[:φ_π])
    m <= Setting(:flexible_ait_φ_y, θ[:φ_y])
    m <= Setting(:ait_Thalf, θ[:Thalf])
    m <= Setting(:gdp_Thalf, θ[:Thalf])
    m <= Setting(:pgap_value, θ[:pgap])
    m <= Setting(:pgap_type, :flexible_ait)
    m <= Setting(:ygap_value, θ[:ygap])
    m <= Setting(:ygap_type, :flexible_ait)
    m <= Setting(:flexible_ait_ρ_smooth, θ[:ρ_smooth])
    if haskey(θ, :historical_policy)
        m <= Setting(:alternative_policies, AltPolicy[θ[:historical_policy]])
    elseif !haskey(DSGE.get_settings(m), :alternative_policies)
        error("The DSGE model have not the required setting :alternative_policies")
    end

    if altpolicy
        @warn "altpolicy kwarg is deprecated and does nothing"
    end
    m <= Setting(:skip_altpolicy_state_init, skip_altpolicy_state_init)


    if include_zlb
        ## Set up temporary ZLB
        m <= Setting(:gensys2, true)

        m <= Setting(:zlb_rule_value, 0.1)

        # Set up credibility for ZLB and alternative policy, if applicable
        m <= Setting(:uncertain_temporary_altpolicy, uncertain_altpolicy)
        m <= Setting(:uncertain_altpolicy, uncertain_altpolicy)
        m <= Setting(:temporary_altpolicy_names, [:zlb_rule]) # using zlb_rule instead of zero_rate
        m <= Setting(:zlb_rule_remove_mon_anticipated_shocks, true) # remove anticipated MP shocks upon entering ZLB

        # Find the first regime in which gensys2 should apply
        gensys2_first_regime = findfirst([get_setting(m, :regime_dates)[i] .== start_zlb_date for i in 1:get_setting(m, :n_regimes)])
        if isnothing(gensys2_first_regime)
            # Temporary ZLB doesn't occur in the current number of regimes
            @warn "No regime matches the start date of the ZLB period. Assuming that the start date begins in a regime after the last regime. If this assumption is wrong, please re-write the regime dates or update this function."
            gensys2_first_regime = get_setting(m, :n_regimes) + 1
            get_setting(m, :regime_dates)[gensys2_first_regime] = start_zlb_date
        end

        n_zlb_reg = DSGE.subtract_quarters(end_zlb_date, start_zlb_date) + 1
        if end_zlb_date < get_setting(m, :regime_dates)[get_setting(m, :n_regimes)]
            set_regime_vals_fnct(m, get_setting(m, :n_regimes))
        else
            # technically gensys2_first_regime + (n_zlb_reg - 1) + 1 b/c
            # gensys2_first_regime is the first regime for the temporary ZLB, so need to subtract 1
            # from n_zlb_reg, but need to also add 1 for the lift-off regime
            set_regime_vals_fnct(m, gensys2_first_regime + n_zlb_reg)
        end

        # Set up regime_eqcond_info
        m <= Setting(:replace_eqcond, true)
        reg_dates = deepcopy(get_setting(m, :regime_dates))
        regime_eqcond_info = Dict{Int, DSGE.EqcondEntry}()
        weights = !isnothing(end_tvcred_level) ? [end_tvcred_level, 1. - end_tvcred_level] : [θ[:cred], 1. - θ[:cred]]
        for (regind, date) in zip(gensys2_first_regime:(n_zlb_reg - 1 + gensys2_first_regime), # See comments starting at line 57
                                  quarter_range(reg_dates[gensys2_first_regime],
                                                iterate_quarters(reg_dates[gensys2_first_regime], n_zlb_reg - 1)))
            reg_dates[regind] = date
            regime_eqcond_info[regind] = DSGE.EqcondEntry(DSGE.zlb_rule(), weights)
        end
        reg_dates[n_zlb_reg + gensys2_first_regime] = iterate_quarters(reg_dates[gensys2_first_regime], n_zlb_reg)
        regime_eqcond_info[n_zlb_reg + gensys2_first_regime] = DSGE.EqcondEntry(chosen_altpolicy, weights)
        m <= Setting(:regime_dates,               reg_dates)
        m <= Setting(:regime_eqcond_info,         regime_eqcond_info)
        m <= Setting(:temporary_altpolicy_length, n_zlb_reg)

        # Set up regime indices
        setup_regime_switching_inds!(m; cond_type = cond_type)

        # Set up TVIS information set
        m <= Setting(:tvis_information_set, vcat([i:i for i in 1:(gensys2_first_regime - 1)],
                                                 [i:get_setting(m, :n_regimes) for i in
                                                  gensys2_first_regime:get_setting(m, :n_regimes)]))

        if !isnothing(tvcred_dates)
            if isnothing(end_tvcred_level) || isnothing(start_tvcred_level) || end_tvcred_level > 1. || end_tvcred_level < 0. ||
                start_tvcred_level > 1. || start_tvcred_level < 0.
                error("The kwargs start_tvcred_level and end_tvcred_level need to be numbers between 0 and 1")
            end
            n_tvcred_reg = DSGE.subtract_quarters(tvcred_dates[2], tvcred_dates[1]) + 1 # number of regimes for time-varying credibility
            reg_start = DSGE.subtract_quarters(tvcred_dates[1], start_zlb_date) + gensys2_first_regime # first regime for time-varying credibility
            for (regind, date) in zip(reg_start:(reg_start + n_tvcred_reg),
                                      DSGE.quarter_range(reg_dates[reg_start], DSGE.iterate_quarters(reg_dates[reg_start], n_tvcred_reg - 1)))
                reg_dates[regind] = date
                if date > end_zlb_date
                    regime_eqcond_info[regind] = DSGE.EqcondEntry(chosen_altpolicy, [end_tvcred_level, 1. - end_tvcred_level])
                end
            end
            m <= Setting(:regime_dates,             reg_dates)
            m <= Setting(:regime_eqcond_info, regime_eqcond_info)
            setup_regime_switching_inds!(m; cond_type = cond_type)
            set_regime_vals_fnct(m, get_setting(m, :n_regimes))
            m <= Setting(:tvis_information_set, vcat([i:i for i in 1:(gensys2_first_regime - 1)],
                                                     [i:get_setting(m, :n_regimes) for i in
                                                      gensys2_first_regime:get_setting(m, :n_regimes)]))
            credvec = collect(range(start_tvcred_level, stop = end_tvcred_level, length = n_tvcred_reg))
            for (i, k) in enumerate(sort!(collect(keys(regime_eqcond_info))))
                if tvcred_dates[1] <= reg_dates[k] <= tvcred_dates[2]
                    get_setting(m, :regime_eqcond_info)[k].weights = [credvec[i], 1. - credvec[i]]
                end
            end
        end

        if tworule_zlb
            tworule_eqcond_info = deepcopy(get_setting(m, :regime_eqcond_info))
            for (reg, eq_entry) in tworule_eqcond_info
                if reg >= 12
                    tworule_eqcond_info[reg] = EqcondEntry(default_policy(), [1.])
                end
                tworule_eqcond_info[reg].weights = [1.]
            end
            old_rule_start_reg = 12
            tworule_iden_eqcond = Dict{Int, Int}()
            for i in (old_rule_start_reg + 1):get_setting(m, :n_regimes)
                tworule_iden_eqcond[i] = old_rule_start_reg
            end

            tworule = MultiPeriodAltPolicy(:two_rule, get_setting(m, :n_regimes), tworule_eqcond_info, gensys2 = true,
                                           temporary_altpolicy_names = [:zlb_rule],                               temporary_altpolicy_length = 6,
                                           infoset = copy(get_setting(m, :tvis_information_set)))
            delete!(DSGE.get_settings(m), :alternative_policies)
            m <= Setting(:alternative_policies, DSGE.AbstractAltPolicy[tworule])
        end

        if adjusted_historical_expectations
            setup_historical_expectations!(m, start_zlb_date, end_zlb_date, spd_expect = spd_expect)
        end

        if endo_zlb
            m <= Setting(:max_temporary_altpolicy_length, max_zlb)
            m <= Setting(:historical_temporary_altpolicy_length, DSGE.subtract_quarters(fcast_date, start_zlb_date))
            min_zlb = DSGE.subtract_quarters(end_zlb_date, get_setting(m, :regime_dates)[get_setting(m, :reg_forecast_start)])
            m <= Setting(:min_temporary_altpolicy_length, max(min_zlb, 0)) # NOTE -- number of altpol regimes *starting from forecast*
            # Create two-rule as alternatie policy
            tvcred_n_qtrs = DSGE.subtract_quarters(end_tvcred_date, start_tvcred_date)
            m <= Setting(:cred_vary_until, tvcred_n_qtrs + 4)
        end
    end
end


"""
'''
zlb_and_taylor!(m::AbstractDSGEModel, cond_type::Symbol,  start_zlb_date::Date, end_zlb_date::Date;
                         set_regime_vals_fnct::Function = baseline_covid_set_regime_vals,
                         include_zlb::Bool = true,
                         tvcred_dates::Union{Tuple{Date, Date}, Nothing} = nothing)
'''

This is a helper function that can be used in subspecs.jl to modify a model for implementing zlb and taylor policies.

"""

function zlb_and_taylor!(m::AbstractDSGEModel, cond_type::Symbol,  start_zlb_date::Date, end_zlb_date::Date;
                         set_regime_vals_fnct::Function = baseline_covid_set_regime_vals,
                         include_zlb::Bool = true,
                         tvcred_dates::Union{Tuple{Date, Date}, Nothing} = nothing)

    m <= Setting(:skip_altpolicy_state_init, true)
    # Nominal rates data during temporary ZLB
    tempzlb_data = DSGE.quarter_range(start_zlb_date, date_conditional_end(m))
    if include_zlb
        ## Set up temporary ZLB
        m <= Setting(:gensys2, true)
        # Set up credibility for ZLB and alternative policy, if applicable
        m <= Setting(:uncertain_temp_alt, false)
        m <= Setting(:temporary_altpolicy_names, [:zlb_rule])
        m <= Setting(:zlb_rule_remove_mon_anticipated_shocks, true)

        # Find the first regime in which gensys2 should apply
        gensys2_first_regime = findfirst([get_setting(m, :regime_dates)[i] .== start_zlb_date for i in 1:get_setting(m, :n_regimes)])
        if isnothing(gensys2_first_regime)
            # Temporary ZLB doesn't occur in the current number of regimes
            @warn "No regime matches the start date of the ZLB period. Assuming that the start date begins in a regime after the last regime. If this assumption is wrong, please re-write the regime dates or update this function."
            gensys2_first_regime = get_setting(m, :n_regimes) + 1
            get_setting(m, :regime_dates)[gensys2_first_regime] = start_zlb_date
        end

        n_zlb_reg = DSGE.subtract_quarters(end_zlb_date, start_zlb_date) + 1
        if end_zlb_date < get_setting(m, :regime_dates)[get_setting(m, :n_regimes)]
            set_regime_vals_fnct(m, get_setting(m, :n_regimes))
        else
            # technically gensys2_first_regime + (n_zlb_reg - 1) + 1 b/c
            # gensys2_first_regime is the first regime for the temporary ZLB, so need to subtract 1
            # from n_zlb_reg, but need to also add 1 for the lift-off regime
            set_regime_vals_fnct(m, gensys2_first_regime + n_zlb_reg)
        end

        # Set up regime_eqcond_info
        m <= Setting(:replace_eqcond, true)
        reg_dates = deepcopy(get_setting(m, :regime_dates))
        regime_eqcond_info = Dict{Int, DSGE.EqcondEntry}()
        weights = [1.0]
        for (regind, date) in zip(gensys2_first_regime:(n_zlb_reg - 1 + gensys2_first_regime), # See comments starting at line 57
                                  quarter_range(reg_dates[gensys2_first_regime],
                                                iterate_quarters(reg_dates[gensys2_first_regime], n_zlb_reg - 1)))
            reg_dates[regind] = date
            regime_eqcond_info[regind] = DSGE.EqcondEntry(DSGE.zlb_rule(), [1.0])
        end
        reg_dates[n_zlb_reg + gensys2_first_regime] = iterate_quarters(reg_dates[gensys2_first_regime], n_zlb_reg)
        regime_eqcond_info[n_zlb_reg + gensys2_first_regime] = DSGE.EqcondEntry(DSGE.taylor_rule(), [1.0])
        m <= Setting(:regime_dates,               reg_dates)
        m <= Setting(:regime_eqcond_info,         regime_eqcond_info)
        m <= Setting(:temporary_altpolicy_length, n_zlb_reg)

        # Set up regime indices
        setup_regime_switching_inds!(m; cond_type = cond_type)

        # Set up TVIS information set
        m <= Setting(:tvis_information_set, vcat([i:i for i in 1:(gensys2_first_regime - 1)],
                                                 [i:get_setting(m, :n_regimes) for i in
                                                  gensys2_first_regime:get_setting(m, :n_regimes)]))

        if !isnothing(tvcred_dates)
            n_tvcred_reg = DSGE.subtract_quarters(tvcred_dates[2], tvcred_dates[1]) + 1 # number of regimes for time-varying credibility
            reg_start = DSGE.subtract_quarters(tvcred_dates[1], start_zlb_date) + gensys2_first_regime # first regime for time-varying credibility
            for (regind, date) in zip(reg_start:(reg_start + n_tvcred_reg),
                                      DSGE.quarter_range(reg_dates[reg_start], DSGE.iterate_quarters(reg_dates[reg_start], n_tvcred_reg - 1)))
                reg_dates[regind] = date
                if date > end_zlb_date
                    regime_eqcond_info[regind] = DSGE.EqcondEntry(DSGE.taylor_rule(), [1.0])
                end
            end
            m <= Setting(:regime_dates,             reg_dates)
            m <= Setting(:regime_eqcond_info, regime_eqcond_info)
            setup_regime_switching_inds!(m; cond_type = cond_type)
            set_regime_vals_fnct(m, get_setting(m, :n_regimes))
            m <= Setting(:tvis_information_set, vcat([i:i for i in 1:(gensys2_first_regime - 1)],
                                                     [i:get_setting(m, :n_regimes) for i in
                                                      gensys2_first_regime:get_setting(m, :n_regimes)]))
            # credvec = collect(range(start_tvcred_level, stop = end_tvcred_level, length = n_tvcred_reg))
            for (i, k) in enumerate(sort!(collect(keys(regime_eqcond_info))))
                if tvcred_dates[1] <= reg_dates[k] <= tvcred_dates[2]
                    get_setting(m, :regime_eqcond_info)[k].weights = [1.0]
                end
            end
        end
    end
end


"""
'''
setup_historical_expectations!(m::AbstractDSGEModel, start_zlb_date::Date, end_zlb_date::Date; spd_expect::Bool = true)
'''

Changing alternative policies and expectation weights at specific model regimes.
"""
function setup_historical_expectations!(m::AbstractDSGEModel, start_zlb_date::Date, end_zlb_date::Date; spd_expect::Bool = true)
    reg_forecast_start = get_setting(m, :reg_forecast_start)
    ## True Policy - ZLB until 2021Q4 and then AIT (or Taylor if no_altpol_2022)
    for (i, reg) in get_setting(m, :regime_eqcond_info)
        pre_wts = get_setting(m, :regime_eqcond_info)[i].weights
        if i > 9 || (spd_expect && i >= 9 && end_zlb_date == Date(2022,3,31))
            if i > DSGE.subtract_quarters(end_zlb_date, Date(2020,12,31)) + 5
                get_setting(m, :regime_eqcond_info)[i].alternative_policy = DSGE.flexible_ait()
            end
            get_setting(m, :regime_eqcond_info)[i].weights = if spd_expect && end_zlb_date == Date(2021,12,31)
                [pre_wts[1], pre_wts[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            elseif spd_expect
                [pre_wts[1], pre_wts[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else
                [pre_wts[1], pre_wts[2], 0.0, 0.0]
            end
        elseif !spd_expect
            get_setting(m, :regime_eqcond_info)[i].weights = [0.0, 0.0, pre_wts[1], pre_wts[2]]
        elseif spd_expect && end_zlb_date == Date(2021,12,31)
            if i == 5
                get_setting(m, :regime_eqcond_info)[i].weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pre_wts[1], pre_wts[2]]
            elseif i <= 7
                get_setting(m, :regime_eqcond_info)[i].weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pre_wts[1], pre_wts[2], 0.0, 0.0]
            elseif i == 8
                get_setting(m, :regime_eqcond_info)[i].weights = [0.0, 0.0, 0.0, 0.0, pre_wts[1], pre_wts[2], 0.0, 0.0, 0.0, 0.0]
            else
                get_setting(m, :regime_eqcond_info)[i].weights = [0.0, 0.0, pre_wts[1], pre_wts[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            end
        else
            if i == 5
                get_setting(m, :regime_eqcond_info)[i].weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pre_wts[1], pre_wts[2]]
            elseif (i <= 8 && !spd_expect) || (spd_expect && i <= 7)
                get_setting(m, :regime_eqcond_info)[i].weights = [0.0, 0.0, 0.0, 0.0, pre_wts[1], pre_wts[2], 0.0, 0.0]
            else
                get_setting(m, :regime_eqcond_info)[i].weights = [0.0, 0.0, pre_wts[1], pre_wts[2], 0.0, 0.0, 0.0, 0.0]
            end
        end
    end

    # Alt 1: ZLB until 2021Q4 and then Taylor
    for (i, reg) in get_setting(m, :alternative_policies)[1].regime_eqcond_info
        if i > DSGE.subtract_quarters(end_zlb_date, Date(2020,12,31)) + 5
            get_setting(m, :alternative_policies)[1].regime_eqcond_info[i].alternative_policy = DSGE.default_policy()
        else
            get_setting(m, :regime_eqcond_info)[i].alternative_policy = DSGE.zlb_rule()
            get_setting(m, :alternative_policies)[1].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
        end
    end

    # Alt 2-3: ZLB until 2022Q3 and then AIT (or 3: Taylor)
    m <= Setting(:alternative_policies, repeat(get_setting(m, :alternative_policies), (spd_expect && end_zlb_date == Date(2021,12,31) ? 9 : 7)))
    get_setting(m, :alternative_policies)[1] = deepcopy(get_setting(m, :alternative_policies)[1])
    get_setting(m, :alternative_policies)[2] = deepcopy(get_setting(m, :alternative_policies)[2])
    get_setting(m, :alternative_policies)[3] = deepcopy(get_setting(m, :alternative_policies)[3])
    if spd_expect
        get_setting(m, :alternative_policies)[4] = deepcopy(get_setting(m, :alternative_policies)[4])
        get_setting(m, :alternative_policies)[5] = deepcopy(get_setting(m, :alternative_policies)[5])
        get_setting(m, :alternative_policies)[6] = deepcopy(get_setting(m, :alternative_policies)[6])
        get_setting(m, :alternative_policies)[7] = deepcopy(get_setting(m, :alternative_policies)[7])
    end
    if spd_expect && end_zlb_date == Date(2021,12,31)
        get_setting(m, :alternative_policies)[8] = deepcopy(get_setting(m, :alternative_policies)[8])
        get_setting(m, :alternative_policies)[9] = deepcopy(get_setting(m, :alternative_policies)[9])
    end
    get_setting(m, :alternative_policies)[2].key = :longzlb_ait
    get_setting(m, :alternative_policies)[3].key = :longzlb_taylor
    if spd_expect
        get_setting(m, :alternative_policies)[4].key = :longzlb_ait_2021Q1_Q3
        get_setting(m, :alternative_policies)[5].key = :longzlb_taylor_2021Q1_Q3
        get_setting(m, :alternative_policies)[6].key = :longzlb_ait_2020Q4
        get_setting(m, :alternative_policies)[7].key = :longzlb_taylor_2020Q4
    end
    if spd_expect && end_zlb_date == Date(2021,12,31)
        get_setting(m, :alternative_policies)[8].key = :incorrect
        get_setting(m, :alternative_policies)[9].key = :useless
    end
    for (i, reg) in get_setting(m, :alternative_policies)[1].regime_eqcond_info
        if spd_expect && end_zlb_date == Date(2021,12,31)
            if i <= 10
                get_setting(m, :alternative_policies)[2].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
                get_setting(m, :alternative_policies)[3].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
            else
                get_setting(m, :alternative_policies)[2].regime_eqcond_info[i].alternative_policy = DSGE.flexible_ait()
                get_setting(m, :alternative_policies)[3].regime_eqcond_info[i].alternative_policy = DSGE.default_policy()
            end
        elseif spd_expect
            if i <= 14
                get_setting(m, :alternative_policies)[2].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
                get_setting(m, :alternative_policies)[3].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
            else
                get_setting(m, :alternative_policies)[2].regime_eqcond_info[i].alternative_policy = DSGE.flexible_ait()
                get_setting(m, :alternative_policies)[3].regime_eqcond_info[i].alternative_policy = DSGE.default_policy()
            end
        else
            if i <= 11
                get_setting(m, :alternative_policies)[2].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
                get_setting(m, :alternative_policies)[3].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
            else
                get_setting(m, :alternative_policies)[2].regime_eqcond_info[i].alternative_policy = DSGE.flexible_ait()
                get_setting(m, :alternative_policies)[3].regime_eqcond_info[i].alternative_policy = DSGE.default_policy()
            end
        end

        if spd_expect && end_zlb_date == Date(2021,12,31)
            if i <= 14
                get_setting(m, :alternative_policies)[4].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
                get_setting(m, :alternative_policies)[5].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
                get_setting(m, :alternative_policies)[6].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
                get_setting(m, :alternative_policies)[7].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
                get_setting(m, :alternative_policies)[8].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
                get_setting(m, :alternative_policies)[9].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
            elseif i == 15
                get_setting(m, :alternative_policies)[4].regime_eqcond_info[i].alternative_policy = DSGE.flexible_ait()
                get_setting(m, :alternative_policies)[5].regime_eqcond_info[i].alternative_policy = DSGE.default_policy()
                get_setting(m, :alternative_policies)[6].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
                get_setting(m, :alternative_policies)[7].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
                get_setting(m, :alternative_policies)[8].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
                get_setting(m, :alternative_policies)[9].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
            elseif i <= 18
                get_setting(m, :alternative_policies)[4].regime_eqcond_info[i].alternative_policy = DSGE.flexible_ait()
                get_setting(m, :alternative_policies)[5].regime_eqcond_info[i].alternative_policy = DSGE.default_policy()
                get_setting(m, :alternative_policies)[6].regime_eqcond_info[i].alternative_policy = DSGE.flexible_ait()
                get_setting(m, :alternative_policies)[7].regime_eqcond_info[i].alternative_policy = DSGE.default_policy()
                get_setting(m, :alternative_policies)[8].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
                get_setting(m, :alternative_policies)[9].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
            else
                get_setting(m, :alternative_policies)[4].regime_eqcond_info[i].alternative_policy = DSGE.flexible_ait()
                get_setting(m, :alternative_policies)[5].regime_eqcond_info[i].alternative_policy = DSGE.default_policy()
                get_setting(m, :alternative_policies)[6].regime_eqcond_info[i].alternative_policy = DSGE.flexible_ait()
                get_setting(m, :alternative_policies)[7].regime_eqcond_info[i].alternative_policy = DSGE.default_policy()
                get_setting(m, :alternative_policies)[8].regime_eqcond_info[i].alternative_policy = DSGE.flexible_ait()
                get_setting(m, :alternative_policies)[9].regime_eqcond_info[i].alternative_policy = DSGE.default_policy()
            end
        elseif spd_expect && i <= 15 && (!spd_expect || (spd_expect && end_zlb_date == Date(2022,3,31)))
            get_setting(m, :alternative_policies)[4].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
            get_setting(m, :alternative_policies)[5].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
            get_setting(m, :alternative_policies)[6].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
            get_setting(m, :alternative_policies)[7].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
        elseif (i <= 18 && !spd_expect) || (spd_expect && end_zlb_date == Date(2022,3,31) && i <= 18)
            get_setting(m, :alternative_policies)[4].regime_eqcond_info[i].alternative_policy = DSGE.flexible_ait()
            get_setting(m, :alternative_policies)[5].regime_eqcond_info[i].alternative_policy = DSGE.default_policy()
            get_setting(m, :alternative_policies)[6].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
            get_setting(m, :alternative_policies)[7].regime_eqcond_info[i].alternative_policy = DSGE.zlb_rule()
        elseif !spd_expect || (spd_expect && end_zlb_date == Date(2022,3,31))
            get_setting(m, :alternative_policies)[4].regime_eqcond_info[i].alternative_policy = DSGE.flexible_ait()
            get_setting(m, :alternative_policies)[5].regime_eqcond_info[i].alternative_policy = DSGE.default_policy()
            get_setting(m, :alternative_policies)[6].regime_eqcond_info[i].alternative_policy = DSGE.flexible_ait()
            get_setting(m, :alternative_policies)[7].regime_eqcond_info[i].alternative_policy = DSGE.default_policy()
        end
    end

    if spd_expect && end_zlb_date == Date(2021,12,31)
        get_setting(m, :alternative_policies)[2].temporary_altpolicy_length = 6
        get_setting(m, :alternative_policies)[3].temporary_altpolicy_length = 6
        get_setting(m, :alternative_policies)[4].temporary_altpolicy_length = 10
        get_setting(m, :alternative_policies)[5].temporary_altpolicy_length = 10
        get_setting(m, :alternative_policies)[6].temporary_altpolicy_length = 11
        get_setting(m, :alternative_policies)[7].temporary_altpolicy_length = 11
        get_setting(m, :alternative_policies)[8].temporary_altpolicy_length = 14
        get_setting(m, :alternative_policies)[9].temporary_altpolicy_length = 14
    elseif spd_expect
        get_setting(m, :alternative_policies)[2].temporary_altpolicy_length = 10
        get_setting(m, :alternative_policies)[3].temporary_altpolicy_length = 10
        get_setting(m, :alternative_policies)[4].temporary_altpolicy_length = 11
        get_setting(m, :alternative_policies)[5].temporary_altpolicy_length = 11
        get_setting(m, :alternative_policies)[6].temporary_altpolicy_length = 14
        get_setting(m, :alternative_policies)[7].temporary_altpolicy_length = 14
    else
        get_setting(m, :alternative_policies)[2].temporary_altpolicy_length = 7
        get_setting(m, :alternative_policies)[3].temporary_altpolicy_length = 7
        if new_expect
            get_setting(m, :alternative_policies)[4].temporary_altpolicy_length = 11
            get_setting(m, :alternative_policies)[5].temporary_altpolicy_length = 11
            get_setting(m, :alternative_policies)[6].temporary_altpolicy_length = 14
            get_setting(m, :alternative_policies)[7].temporary_altpolicy_length = 14
        end
    end
    m <= Setting(:temporary_altpolicy_length, DSGE.subtract_quarters(end_zlb_date, start_zlb_date) + 1)
    get_setting(m, :alternative_policies)[1].temporary_altpolicy_length = DSGE.subtract_quarters(end_zlb_date, start_zlb_date) + 1

    m <= Setting(:uncertain_altpolicy, true)
    m <= Setting(:uncertain_temporary_altpolicy, true)
end


"""
'''
model2para_covid_set_regime_vals(m::AbstractDSGEModel, n::Int, new_model2para_reg::Int = 1; start_regime::Int = 6)
'''

Helper function when using temporary alternative policies with these scenarios and when there is a model2para_regimes dictionary. `start_regime` specifies the first regime for which we may or may not need to add extra regimes for parameters `new_model2para_reg` specifies what parameter regime to which extra model regimes are mapped. It is assumed that all regime-switching parameters are in model2para_regime, which is assumed to avoid looping over unnecessary parameters.
"""

# helper function when using temporary alternative policies with these scenarios
# and when there is a model2para_regimes dictionary.
# `start_regime` specifies the first regime for which we may or may not need to add extra regimes for parameters
# `new_model2para_reg` specifies what parameter regime to which extra model regimes are mapped.
# It is assumed that all regime-switching parameters are in model2para_regime, which
# is assumed to avoid looping over unnecessary parameters.
function model2para_covid_set_regime_vals(m::AbstractDSGEModel, n::Int, new_model2para_reg::Int = 1; start_regime::Int = 6)

    start_reg = max(start_regime, 6)

    if n >= start_reg
        m2p = get_setting(m, :model2para_regime)
        for (k, v) in m2p
            horizon_m2p_reg = v[maximum(keys(v))]
            for i in start_regime:n
                v[i] = horizon_m2p_reg # model regime i maps to same para regime as last known mapping
            end
        end
    end

    m
end


function add_sigma_mkup_iid!(m::AbstractDSGEModel)
    get_setting(m, :model2para_regime)[:σ_λ_f_iid] = Dict(1 => 1, 2 => 2, 3 => 2, 4 => 2, 5 => 3)
    for i in 6:get_setting(m, :n_regimes)
        get_setting(m, :model2para_regime)[:σ_λ_f_iid][i] = 1
    end

    # Set values (priors are set already unless regime-switching is desired in 2020:Q4)
    set_regime_val!(m[:σ_λ_f_iid], 1, 0.)
    set_regime_val!(m[:σ_λ_f_iid], 2, 5.0)
    set_regime_val!(m[:σ_λ_f_iid], 3, 0.05)

    # Fix shocks to 0 in para regime 1
    set_regime_fixed!(m[:σ_λ_f_iid], 1, true)
    set_regime_fixed!(m[:σ_λ_f_iid], 2, false)
    set_regime_fixed!(m[:σ_λ_f_iid], 3, false)

    # Regime-switching priors for regime 3
    for i in 1:2
        set_regime_prior!(m[:σ_λ_f_iid], i, m[:σ_λ_f_iid].prior)
    end
    set_regime_prior!(m[:σ_λ_f_iid], 3, RootInverseGamma(10.0, 0.0501))
end


function add_meas_pi!(m::AbstractDSGEModel)
    get_setting(m, :model2para_regime)[:ρ_meas_π] = Dict(1 => 1, 2 => 2, 3 => 2, 4 => 2, 5 => 2, 6 => 2, 7 => 2, 8 => 2, 9 => 2, 10 => 2)
    get_setting(m, :model2para_regime)[:σ_meas_π] = Dict(1 => 1, 2 => 2, 3 => 2, 4 => 2, 5 => 2, 6 => 2, 7 => 2, 8 => 2, 9 => 2, 10 => 1)
    for i in 10:get_setting(m, :n_regimes)
        get_setting(m, :model2para_regime)[:ρ_meas_π][i] = 2
        get_setting(m, :model2para_regime)[:σ_meas_π][i] = 1
    end

    # Set regime value bounds
    set_regime_valuebounds!(m[:ρ_meas_π], 1, (0.0, 5.0))
    set_regime_valuebounds!(m[:σ_meas_π], 1, (0.0, 5.0))
    set_regime_valuebounds!(m[:ρ_meas_π], 2, (1.0e-8, 5.0))
    m[:ρ_meas_π].valuebounds = (1.0e-8, 5.0)
    set_regime_valuebounds!(m[:σ_meas_π], 2, (1.0e-8, 5.0))

    # Set values (priors are set already unless regime-switching is desired in 2020:Q4)
    set_regime_val!(m[:ρ_meas_π], 1, 0.)
    set_regime_val!(m[:ρ_meas_π], 2, 0.2320)
    m[:ρ_meas_π].value = 0.2320
    set_regime_val!(m[:σ_meas_π], 1, 0.)
    set_regime_val!(m[:σ_meas_π], 2, 0.0999)

    # Fix shocks to 0 in para regime 1
    m[:ρ_meas_π].fixed = false#true
    m[:σ_meas_π].fixed = true
    set_regime_fixed!(m[:ρ_meas_π], 1, true)
    set_regime_fixed!(m[:ρ_meas_π], 2, false)
    set_regime_fixed!(m[:σ_meas_π], 1, true)
    set_regime_fixed!(m[:σ_meas_π], 2, false)

    set_regime_prior!(m[:σ_meas_π], 1, m[:σ_meas_π].prior)
    set_regime_prior!(m[:σ_meas_π], 2, m[:σ_meas_π].prior)
    # set_regime_prior!(m[:ρ_meas_π], 1, m[:ρ_meas_π].prior)
    # set_regime_prior!(m[:ρ_meas_π], 2, m[:ρ_meas_π].prior)
end

function add_zero_meas_pi!(m::AbstractDSGEModel)
    # Set measurement errors from ss87 to 0
    get_setting(m, :model2para_regime)[:ρ_meas_π] = Dict(1 => 1, 2 => 2, 3 => 2, 4 => 2, 5 => 2, 6 => 2, 7 => 2, 8 => 2, 9 => 2, 10 => 2)
    get_setting(m, :model2para_regime)[:σ_meas_π] = Dict(1 => 1, 2 => 2, 3 => 2, 4 => 2, 5 => 2, 6 => 2, 7 => 2, 8 => 2, 9 => 2, 10 => 1)
    for i in 10:get_setting(m, :n_regimes)
        get_setting(m, :model2para_regime)[:ρ_meas_π][i] = 2
        get_setting(m, :model2para_regime)[:σ_meas_π][i] = 1
    end

    # Set values (priors are set already unless regime-switching is desired in 2020:Q4)
    set_regime_val!(m[:ρ_meas_π], 1, 0.)
    set_regime_val!(m[:ρ_meas_π], 2, 0.0)
    m[:ρ_meas_π].value = 0.0
    set_regime_val!(m[:σ_meas_π], 1, 0.)
    set_regime_val!(m[:σ_meas_π], 2, 0.0)

    # Fix shocks to 0 in para regime 1
    m[:ρ_meas_π].fixed = false#true
    m[:σ_meas_π].fixed = true
    set_regime_fixed!(m[:ρ_meas_π], 1, true)
    set_regime_fixed!(m[:ρ_meas_π], 2, true)
    set_regime_fixed!(m[:σ_meas_π], 1, true)
    set_regime_fixed!(m[:σ_meas_π], 2, true)
end

function remove_persist_mkup!(m::AbstractDSGEModel)
    # get_setting(m, :model2para_regime)[:ρ_λ_f] = Dict(1 => 1, 2 => 2, 3 => 2, 4 => 2, 5 => 2)
    get_setting(m, :model2para_regime)[:σ_λ_f] = Dict(1 => 1, 2 => 2, 3 => 2, 4 => 2, 5 => 2)
    for i in 6:get_setting(m, :n_regimes)
        # get_setting(m, :model2para_regime)[:ρ_λ_f][i] = 1
        get_setting(m, :model2para_regime)[:σ_λ_f][i] = 1
    end

    # Set values (priors are set already unless regime-switching is desired in 2020:Q4)
    # set_regime_val!(m[:ρ_meas_π], 1, m[:ρ_λ_f].value)
    # set_regime_val!(m[:ρ_meas_π], 2, 0.0)
    set_regime_val!(m[:σ_λ_f], 1, m[:σ_λ_f].value)
    set_regime_val!(m[:σ_λ_f], 2, 0.0)

    # Fix shocks to 0 in para regime 2
    # set_regime_fixed!(m[:ρ_meas_π], 1, false)
    # set_regime_fixed!(m[:ρ_meas_π], 2, true)
    m[:σ_λ_f].fixed = false
    set_regime_fixed!(m[:σ_λ_f], 1, false)
    set_regime_fixed!(m[:σ_λ_f], 2, true)
end

function rm_iid_pce_meas_err!(m::AbstractDSGEModel)

    #get_setting(m, :model2para_regime)[:ρ_gdpdef] = Dict(1 => 1, 2 => 2, 3 => 2, 4 => 2, 5 => 2, 6 => 2, 7 => 2)
    #get_setting(m, :model2para_regime)[:σ_gdpdef] = Dict(1 => 1, 2 => 2, 3 => 2, 4 => 2, 5 => 2, 6 => 2, 7 => 2)
    get_setting(m, :model2para_regime)[:ρ_corepce] = Dict(1 => 1) ## Don't need to change ρ_corepce
    get_setting(m, :model2para_regime)[:σ_corepce] = Dict(1 => 1, 2 => 2, 3 => 2, 4 => 2, 5 => 2, 6 => 2, 7 => 2, 8 => 2, 9 => 2, 10 => 1)
    for i in 2:9
        get_setting(m, :model2para_regime)[:ρ_corepce][i] = 1
    end
    for i in 10:get_setting(m, :n_regimes)
        #get_setting(m, :model2para_regime)[:ρ_gdpdef][i] = 1
        #get_setting(m, :model2para_regime)[:σ_gdpdef][i] = 1
        get_setting(m, :model2para_regime)[:ρ_corepce][i] = 1
        get_setting(m, :model2para_regime)[:σ_corepce][i] = 1
    end

    # Change valuebounds in regime 2 and keep those in regime 1
    #set_regime_valuebounds!(m[:ρ_gdpdef], 1, m[:ρ_gdpdef].valuebounds)
    #set_regime_valuebounds!(m[:σ_gdpdef], 1, m[:σ_gdpdef].valuebounds)
    set_regime_valuebounds!(m[:ρ_corepce], 1, m[:ρ_corepce].valuebounds)
    set_regime_valuebounds!(m[:σ_corepce], 1, m[:σ_corepce].valuebounds)

    #set_regime_valuebounds!(m[:ρ_gdpdef], 2, (0.0, m[:ρ_gdpdef].valuebounds[2]))
    #set_regime_valuebounds!(m[:σ_gdpdef], 2, (0.0, m[:σ_gdpdef].valuebounds[2]))
    set_regime_valuebounds!(m[:ρ_corepce], 2, (0.0, m[:ρ_corepce].valuebounds[2]))
    set_regime_valuebounds!(m[:σ_corepce], 2, (0.0, m[:σ_corepce].valuebounds[2]))

    # Set values (priors are set already)
    #set_regime_val!(m[:ρ_gdpdef], 1, 0.5379)
    #set_regime_val!(m[:ρ_gdpdef], 2, 0.0)
    #set_regime_val!(m[:σ_gdpdef], 1, 0.1575)
    #set_regime_val!(m[:σ_gdpdef], 2, 0.0)

    set_regime_val!(m[:ρ_corepce], 1, 0.2320)
    set_regime_val!(m[:ρ_corepce], 2, 0.0)
    set_regime_val!(m[:σ_corepce], 1, 0.0999)
    set_regime_val!(m[:σ_corepce], 2, 0.0)

    # Fix shocks to 0 in para regime 1
    #set_regime_fixed!(m[:ρ_gdpdef], 1, false)
    #set_regime_fixed!(m[:ρ_gdpdef], 2, true)
    #set_regime_fixed!(m[:σ_gdpdef], 1, false)
    #set_regime_fixed!(m[:σ_gdpdef], 2, true)

    m[:ρ_corepce].fixed = false
    m[:σ_corepce].fixed = false

    set_regime_fixed!(m[:ρ_corepce], 1, false)
    set_regime_fixed!(m[:ρ_corepce], 2, true)
    set_regime_fixed!(m[:σ_corepce], 1, false)
    set_regime_fixed!(m[:σ_corepce], 2, true)

    # Change valuebounds in regime 2 and keep those in regime 1
    #set_regime_valuebounds!(m[:ρ_gdpdef], 1, m[:ρ_gdpdef].valuebounds)
    #set_regime_valuebounds!(m[:σ_gdpdef], 1, m[:σ_gdpdef].valuebounds)
    set_regime_valuebounds!(m[:ρ_corepce], 1, m[:ρ_corepce].valuebounds)
    set_regime_valuebounds!(m[:σ_corepce], 1, m[:σ_corepce].valuebounds)

    #set_regime_valuebounds!(m[:ρ_gdpdef], 2, (0.0, m[:ρ_gdpdef].valuebounds[2]))
    #set_regime_valuebounds!(m[:σ_gdpdef], 2, (0.0, m[:σ_gdpdef].valuebounds[2]))
    set_regime_valuebounds!(m[:ρ_corepce], 2, (0.0, m[:ρ_corepce].valuebounds[2]))
    set_regime_valuebounds!(m[:σ_corepce], 2, (0.0, m[:σ_corepce].valuebounds[2]))

    # Set values
    ## Should I set rho_gdpdef to 0 or not?
    #set_regime_val!(m[:ρ_gdpdef], 1, 0.5379)
    #set_regime_val!(m[:ρ_gdpdef], 2, 0.0)

    set_regime_val!(m[:ρ_corepce], 1, 0.2320)
    set_regime_val!(m[:ρ_corepce], 2, 0.0)
    set_regime_val!(m[:σ_corepce], 1, 0.0999)
    set_regime_val!(m[:σ_corepce], 2, 0.0)
end

function expected_nominal_rates!(m::AbstractDSGEModel; irf_reg::Integer = Dict(map(reverse, collect(get_setting(m, :regime_dates))))[date_forecast_start(m)])
    if mon_anticipated_ait_shocks(m) != false
        for i in mon_anticipated_ait_shocks(m) ## AIT expected FFR
            symb_i = Symbol("σ_ait_r_m$(i)")
            get_setting(m, :model2para_regime)[symb_i] = Dict(1 => 1)
            for j in 1:irf_reg
                if j < 10
                    get_setting(m, :model2para_regime)[symb_i][j] = 1
                else
                    get_setting(m, :model2para_regime)[symb_i][j] = 2
                end
            end
            set_regime_valuebounds!(m[symb_i], 1, m[symb_i].valuebounds)
            set_regime_valuebounds!(m[symb_i], 2, m[symb_i].valuebounds)
            m[symb_i].fixed = false

            set_regime_val!(m[symb_i], 1, 0.0)
            set_regime_val!(m[symb_i], 2, m[symb_i].value)

            set_regime_fixed!(m[symb_i], 1, true)
            set_regime_fixed!(m[symb_i], 2, false)
        end
    end

    for i in 1:n_mon_anticipated_shocks_padding(m) ## Taylor Rule expected FFR
        symb_i = Symbol("σ_r_m$(i)")
        if symb_i in [m.parameters[j].key for j in 1:length(m.parameters)] && !m[symb_i].fixed
            get_setting(m, :model2para_regime)[symb_i] = Dict(1 => 1)
            for j in 1:irf_reg
                if j < 10
                    get_setting(m, :model2para_regime)[symb_i][j] = 1
                else
                    get_setting(m, :model2para_regime)[symb_i][j] = 2
                end
            end
            set_regime_valuebounds!(m[symb_i], 1, m[symb_i].valuebounds)
            set_regime_valuebounds!(m[symb_i], 2, m[symb_i].valuebounds)
            m[symb_i].fixed = false

            set_regime_val!(m[symb_i], 2, 0.0)
            set_regime_val!(m[symb_i], 1, m[symb_i].value)

            set_regime_fixed!(m[symb_i], 2, true)
            set_regime_fixed!(m[symb_i], 1, false)
        end
    end

    # Contemporaneous AIT shocks
    if haskey(m.settings, :add_ait_rm) && get_setting(m, :add_ait_rm)
        get_setting(m, :model2para_regime)[:σ_ait_rm] = Dict(1 => 1)
        for i in 1:9
            get_setting(m, :model2para_regime)[:σ_ait_rm][i] = 1
        end
        for i in 10:irf_reg
            get_setting(m, :model2para_regime)[:σ_ait_rm][i] = 2
        end
        set_regime_valuebounds!(m[:σ_ait_rm], 1, m[:σ_ait_rm].valuebounds)
        set_regime_valuebounds!(m[:σ_ait_rm], 2, m[:σ_ait_rm].valuebounds)
        m[:σ_ait_rm].fixed = false

        set_regime_val!(m[:σ_ait_rm], 1, 0.0)
        set_regime_val!(m[:σ_ait_rm], 2, m[:σ_ait_rm].value)

        set_regime_fixed!(m[:σ_ait_rm], 1, true)
        set_regime_fixed!(m[:σ_ait_rm], 2, false)
    end

    # Contemporaneous Taylor shock
    get_setting(m, :model2para_regime)[:σ_r_m] = Dict(1 => 1)
    for i in 1:9
        get_setting(m, :model2para_regime)[:σ_r_m][i] = 1
    end
    for i in 10:irf_reg
        get_setting(m, :model2para_regime)[:σ_r_m][i] = 2
    end
    set_regime_valuebounds!(m[:σ_r_m], 1, m[:σ_r_m].valuebounds)
    set_regime_valuebounds!(m[:σ_r_m], 2, m[:σ_r_m].valuebounds)
    m[:σ_r_m].fixed = false

    set_regime_val!(m[:σ_r_m], 2, 0.0)
    set_regime_val!(m[:σ_r_m], 1, m[:σ_r_m].value)

    set_regime_fixed!(m[:σ_r_m], 2, true)
    set_regime_fixed!(m[:σ_r_m], 1, false)

    # iid measurement error on expected AIT shock
    for i in expected_ffr(m)
        symb_i = Symbol("σ_exp_rm$(i)")
        get_setting(m, :model2para_regime)[symb_i] = Dict(1 => 1)
        for j in 1:irf_reg
            if j < 10
                get_setting(m, :model2para_regime)[symb_i][j] = 1
            else
                get_setting(m, :model2para_regime)[symb_i][j] = 2
            end
        end
        set_regime_valuebounds!(m[symb_i], 1, m[symb_i].valuebounds)
        set_regime_valuebounds!(m[symb_i], 2, m[symb_i].valuebounds)
        m[symb_i].fixed = false

        set_regime_val!(m[symb_i], 1, 0.0)
        set_regime_val!(m[symb_i], 2, m[symb_i].value)

        set_regime_fixed!(m[symb_i], 1, true)
        set_regime_fixed!(m[symb_i], 2, false)
   end
end
