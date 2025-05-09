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


"""
```
init_settings!(m::Model1002)
```

Initializes the model's settings as per sub specification. These settings are intrinsic to building the rest of the model (as opposed to other settings which may just affect the forecasting or estimation process once the model has been created) and are likely to contain dependencies in `subspecs.jl` or `subspecs_builder.jl`.
"""
function init_subspec_settings!(m::Model1002)
    ss_num = parse(Int, SubString(subspec(m),3,subspec_ind))

    if get_setting(m, :cond_id) in collect(1:5)
        m <= Setting(:cond_full_names, [:obs_gdp, :obs_corepce, :obs_spread, :obs_nominalrate, :obs_longrate],
                     "Observables used in conditional forecasts")
    elseif get_setting(m, :cond_id) == 6
        m <= Setting(:cond_full_names, [:obs_gdp, :obs_corepce, :obs_spread, :obs_nominalrate, :obs_longrate,
                                        :obs_gdpdeflator], "Observables used in conditional forecasts")
    end

    if subspec(m) in ["ss16", "ss17"]
        m <= Setting(:laborshare_base_period, DSGE.quartertodate("1964-Q1"), "Base year for labor share series to provide an initial condition")
    end


    # COVID-19 settings
    if ss_num >= 59 && ss_num <= 104
        m <= Setting(:antshocks, Dict{Symbol, Int}(:biidc => 1, :φ => 1, :ziid => 1))
        m <= Setting(:ant_eq_mapping, Dict{Symbol, Symbol}(:biidc => :biidc, :φ => :φ, :ziid => :ziid))
        m <= Setting(:ant_eq_E_mapping, Dict{Symbol, Symbol}(:φ => :Eφ))
        m <= Setting(:proportional_antshocks, Symbol[:biidc, :φ, :ziid])

        m <= Setting(:add_cprod_params, true)
        m <= Setting(:add_cprod_group, true)
        m <= Setting(:add_cprod_states, true)
        m <= Setting(:add_std_shock_group, true)
    end
    if ss_num >= 62 && ss_num <= 104
        m <= Setting(:add_pseudo_gdp, true),
        m <= Setting(:add_pseudo_corepce, true)
        m <= Setting(:add_anticipated_obs_gdp, true)
        m <= Setting(:n_anticipated_obs_gdp, 1)
        m <= Setting(:filename_anticipated_obs_gdp, "MEDIANANTGDP")
        m <= Setting(:contemporaneous_and_proportional_antshocks, Symbol[:biidc])
        m <= Setting(:meas_err_anticipated_obs_gdp, 1.)
        m <= Setting(:add_iid_anticipated_obs_gdp_meas_err, false)
        m <= Setting(:add_initialize_pgap_ygap_pseudoobs, true)
    end

    if ss_num >= 87 && ss_num <= 104
        m <= Setting(:add_meas_pi_states, true)
        m <= Setting(:add_meas_pi_params, true)
        m <= Setting(:add_meas_pi_shock_group, true)
        m <= Setting(:add_meas_pi_error_group, true)
    end

    if ss_num >= 100 && ss_num <= 103
        m <= Setting(:add_pol_smooth_group, true)
    end

    if subspec(m) in ["ss30", "ss59", "ss60", "ss61"]
        m <= Setting(:flexible_ait_policy_change, false,
                     "Indicator for whether 2020-Q3 switch in monetary policy rule to AIT is on")
        m <= Setting(:add_pgap, true)
        m <= Setting(:add_ygap, true)
    elseif ss_num >= 62 && ss_num <= 104
        m <= Setting(:flexible_ait_policy_change, false,
                     "Indicator for whether 2020-Q3 switch in monetary policy rule to AIT is on")
        m <= Setting(:add_pgap, true)
        m <= Setting(:add_ygap, true)
        m <= Setting(:skip_altpolicy_state_init, true)
    else
        m <= Setting(:flexible_ait_policy_change, false,
                     "Indicator for whether 2020-Q3 switch in monetary policy rule to AIT is on")
        m <= Setting(:add_pgap, false)
        m <= Setting(:add_ygap, false)
    end


    if subspec(m) in ["ss104"]
        m <= Setting(:mon_anticipated_ait_shocks, [1, 2, 3, 4, 5, 6])
        m <= Setting(:expected_ffr, [1, 2, 3, 4, 5, 6])
        m <= Setting(:all_ffr_qs, [1, 2, 3, 4, 5, 6])
        m <= Setting(:forecast_horizons, 40)
        m <= Setting(:add_iid_cond_obs_gdp_meas_err, true)
        m <= Setting(:add_iid_cond_obs_corepce_meas_err, true)
        m <= Setting(:add_ait_rm, true)
        m <= Setting(:add_taylor_rm, true)
        m <= Setting(:standard_shocks_mode_adjust, 1, false, "modeadj", "")
        m <= Setting(:standard_shocks_spread_adjust, 2, false, "spreadadj", "")
        m <= Setting(:add_altpolicy_pgap, true)
        m <= Setting(:add_altpolicy_ygap, true)
    end
end


"""
"""
function init_subspec_params!(m::AbstractDSGEModel)

    if haskey(m.settings, :add_cprod_params) ? get_setting(m, :add_cprod_params) : false # parse(Int, SubString(subspec(m),3,subspec_ind)) >= 59
        m <= parameter(:ρ_ziid, 0., (0., 0.999), (0., 0.999), ModelConstructors.Untransformed(), BetaAlt(0.5, 0.2), fixed=true,
                       description="ρ_ziid: AR(1) coefficient in the iid component of the technology process.",
                       tex_label="\\rho_{z, iid}")
        m <= parameter(:σ_ziid, 0., (0., 1e2), (0., 1e2), ModelConstructors.Exponential(),
                       RootInverseGamma(2. * (5.)^2 ./ 5., sqrt((5.)^2 + .1)), fixed=false,
                       description="σ_ziid: The standard deviation of the process describing the iid component of productivity.",
                       tex_label="\\sigma_{z, iid}")
        m <= parameter(:ρ_biidc, 0., (0., 0.999), (0., 0.999), ModelConstructors.Untransformed(), BetaAlt(0.5, 0.2), fixed=true,
                       description="ρ_biidc: AR(1) coefficient in the iid component of the preference process.",
                       tex_label="\\rho_{b, iid, c}")
        m <= parameter(:σ_biidc, 0., (0., 1e2), (0., 1e2), ModelConstructors.Exponential(),
                       RootInverseGamma(2. * (4.)^2 ./ 4., sqrt((4.)^2  + .1)), fixed=false, # If σ_φ ∼ RootInverseGamma(ν, τ), then σ_φ² ∼ InverseGamma(ν/2, ντ²/2), with mode M given by ν (τ² - M²) = 2 * M²
                       description="σ_biidc: The standard deviation of the process describing the iid component of preferences.",
                       tex_label="\\sigma_{b, iid, c}")
        m <= parameter(:ρ_φ, 0., (0., 0.999), (0., 0.999), ModelConstructors.Untransformed(), BetaAlt(0.5, 0.2), fixed=true,
                       description="ρ_φ: AR(1) coefficient in the labor supply preference process.",
                       tex_label="\\rho_{\\varphi}")
        m <= parameter(:σ_φ, 0., (0., 1e3), (0., 1e3), ModelConstructors.Exponential(),
                       RootInverseGamma(2 * (400.0)^2 ./ (400. * 10), sqrt(1. + (400.0)^2)), fixed=false,
                       description="σ_φ: The standard deviation of the process describing the labor supply preference.",
                       tex_label="\\sigma_{\\varphi}") # If σ_φ ∼ RootInverseGamma(ν, τ), then σ_φ² ∼ InverseGamma(ν/2, ντ²/2)
    end

    if haskey(m.settings, :add_meas_pi_params) ? get_setting(m, :add_meas_pi_params) : false # parse(Int, SubString(subspec(m),3,subspec_ind)) >= 87
        m <= parameter(:ρ_meas_π, 0.2320, (0.0, 0.999), (0.0, 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
                       tex_label="\\rho_{meas_\\pi}")
        m <= parameter(:σ_meas_π, 0.0999, (0.0, 5.),(0.0, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false,
                       tex_label="\\sigma_{meas_\\pi}")
    end

    if subspec(m) in ["ss62"] # TODO: add new specs
        m <= parameter(:damp_standard_shocks, 1., (0., 1e3), (0., 1e3), fixed=false, # will be fixed later in subspec!(m)
                       description="damp_standard_shocks: Damping factor for standard business cycle shocks during COVID-19.",
                       tex_label="damp standard shocks")
        m <= parameter(:amplify_σ_r_m, 1., (0., 1e3), (0., 1e3), fixed=false, # will be fixed later in subspec!(m)
                       description="amplify_σ_r_m: Amplification factor for monetary policy shock during COVID-19.",
                       tex_label="amplify \\sigma_{r^m}")
        m <= parameter(:amplify_inflation_me, 1., (0., 1e3), (0., 1e3), fixed=false, # will be fixed later in subspec!(m)
                       description="amplify_inflation_me: Amplification factor for inflation measurement error during COVID-19.",
                       tex_label="amplify inflation measurement error")
    end

    if subspec(m) in ["ss67", "ss68", "ss71", "ss72", "ss75", "ss76", "ss80", "ss82", "ss83"]
        m <= parameter(:ρ_g_covid, 0.9863, (0., 0.999), (0., 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
                       description="ρ_g: AR(1) coefficient in the government spending process.",
                       tex_label="\\rho_g")

        m <= parameter(:ρ_μ_covid, 0.8735, (0., 0.999), (0., 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
                       description="ρ_μ: AR(1) coefficient in capital adjustment cost process.",
                       tex_label="\\rho_{\\mu}")

        m <= parameter(:ρ_λ_f_covid, 0.8827, (0., 0.999), (0., 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
                       description="ρ_λ_f: AR(1) coefficient in the price mark-up shock process.",
                       tex_label="\\rho_{\\lambda_f}")

        m <= parameter(:ρ_σ_w_covid, 0.9898, (0., 0.99999), (0., 0.99999), ModelConstructors.SquareRoot(), BetaAlt(0.75, 0.15), fixed=false,
                       description="ρ_σ_w: The standard deviation of entrepreneurs' capital productivity follows an exogenous process with mean ρ_σ_w. Innovations to the process are called _spread shocks_.",
                       tex_label="\\rho_{\\sigma_\\omega}")

        m <= parameter(:ρ_lr_covid, 0.6936, (0., 0.999), (0., 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
                       tex_label="\\rho_{10y}")

        m <= parameter(:ρ_tfp_covid, 0.1953, (0., 0.999), (0., 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
                       tex_label="\\rho_{tfp}")

        m <= parameter(:ρ_gdp_covid, 0., (-0.999, 0.999), (-0.999, 0.999), ModelConstructors.SquareRoot(), Normal(0.0, 0.2), fixed=false,
                       tex_label="\\rho_{gdp}")

        m <= parameter(:ρ_gdi_covid, 0., (-0.999, 0.999), (-0.999, 0.999), ModelConstructors.SquareRoot(), Normal(0.0, 0.2), fixed=false,
                       tex_label="\\rho_{gdi}")

        m <= parameter(:ρ_gdpvar_covid, 0., (-0.999, 0.999), (-0.999, 0.999), ModelConstructors.SquareRoot(), Normal(0.0, 0.4), fixed=false,
                       tex_label="\\varrho_{gdp}")

        m <= parameter(:σ_g_covid, 2.5230, (0., 5.), (1e-8, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false,
                       description="σ_g: The standard deviation of the government spending process.",
                       tex_label="\\sigma_{g}")

        m <= parameter(:σ_μ_covid, 0.4559, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false,
                       description="σ_μ: The standard deviation of the exogenous marginal efficiency of investment shock process.",
                       tex_label="\\sigma_{\\mu}")

        m <= parameter(:σ_λ_f_covid, 0.1314, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false,
                       description="σ_λ_f: The mean of the process that generates the price elasticity of the composite good. Specifically, the elasticity is (1+λ_{f,t})/(λ_{f_t}).",
                       tex_label="\\sigma_{\\lambda_f}")

        m <= parameter(:σ_σ_ω_covid, 0.0428, (1e-7,100.), (0., 0.), ModelConstructors.Exponential(), RootInverseGamma(4, 0.05), fixed=false,
                       description="σ_σ_ω: The standard deviation of entrepreneurs' capital productivity follows an exogenous process with standard deviation σ_σ_ω.",
                       tex_label="\\sigma_{\\sigma_\\omega}")

        m <= parameter(:σ_lr_covid, 0.1766, (1e-8,10.), (1e-8, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.75), fixed=false,
                       tex_label="\\sigma_{10y}")

        m <= parameter(:σ_tfp_covid, 0.9391, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false,
                       tex_label="\\sigma_{tfp}")

        m <= parameter(:σ_gdp_covid, 0.1, (1e-8, 5.),(1e-8, 5.),ModelConstructors.Exponential(),RootInverseGamma(2, 0.10), fixed=false,
                       tex_label="\\sigma_{gdp}")

        m <= parameter(:σ_gdi_covid, 0.1, (1e-8, 5.),(1e-8, 5.),ModelConstructors.Exponential(),RootInverseGamma(2, 0.10), fixed=false,
                       tex_label="\\sigma_{gdi}")
    end

    if subspec(m) in ["ss69", "ss70", "ss73", "ss74", "ss77", "ss78"]
        m <= parameter(:ρ_z_p_covid, 0.8910, (0., 0.999), (0., 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
                       description="ρ_z_p: AR(1) coefficient in the process describing the permanent component of productivity.",
                       tex_label="\\rho_{z^p}")

        m <= parameter(:σ_z_p_covid, 0.1662, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false,
                       description="σ_z_p: The standard deviation of the shock to the permanent component of productivity.",
                       tex_label="\\sigma_{z^p}")

        m <= parameter(:ρ_g_covid, 0.9863, (0., 0.999), (0., 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
                       description="ρ_g: AR(1) coefficient in the government spending process.",
                       tex_label="\\rho_g")

        m <= parameter(:ρ_μ_covid, 0.8735, (0., 0.999), (0., 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
                       description="ρ_μ: AR(1) coefficient in capital adjustment cost process.",
                       tex_label="\\rho_{\\mu}")

        m <= parameter(:ρ_λ_f_covid, 0.8827, (0., 0.999), (0., 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
                       description="ρ_λ_f: AR(1) coefficient in the price mark-up shock process.",
                       tex_label="\\rho_{\\lambda_f}")

        m <= parameter(:ρ_σ_w_covid, 0.9898, (0., 0.99999), (0., 0.99999), ModelConstructors.SquareRoot(), BetaAlt(0.75, 0.15), fixed=false,
                       description="ρ_σ_w: The standard deviation of entrepreneurs' capital productivity follows an exogenous process with mean ρ_σ_w. Innovations to the process are called _spread shocks_.",
                       tex_label="\\rho_{\\sigma_\\omega}")

        m <= parameter(:ρ_lr_covid, 0.6936, (0., 0.999), (0., 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
                       tex_label="\\rho_{10y}")

        m <= parameter(:ρ_tfp_covid, 0.1953, (0., 0.999), (0., 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
                       tex_label="\\rho_{tfp}")

        m <= parameter(:ρ_gdp_covid, 0., (-0.999, 0.999), (-0.999, 0.999), ModelConstructors.SquareRoot(), Normal(0.0, 0.2), fixed=false,
                       tex_label="\\rho_{gdp}")

        m <= parameter(:ρ_gdi_covid, 0., (-0.999, 0.999), (-0.999, 0.999), ModelConstructors.SquareRoot(), Normal(0.0, 0.2), fixed=false,
                       tex_label="\\rho_{gdi}")

        m <= parameter(:ρ_gdpvar_covid, 0., (-0.999, 0.999), (-0.999, 0.999), ModelConstructors.SquareRoot(), Normal(0.0, 0.4), fixed=false,
                       tex_label="\\varrho_{gdp}")

        m <= parameter(:σ_g_covid, 2.5230, (0., 5.), (1e-8, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false,
                       description="σ_g: The standard deviation of the government spending process.",
                       tex_label="\\sigma_{g}")

        m <= parameter(:σ_μ_covid, 0.4559, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false,
                       description="σ_μ: The standard deviation of the exogenous marginal efficiency of investment shock process.",
                       tex_label="\\sigma_{\\mu}")

        m <= parameter(:σ_λ_f_covid, 0.1314, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false,
                       description="σ_λ_f: The mean of the process that generates the price elasticity of the composite good. Specifically, the elasticity is (1+λ_{f,t})/(λ_{f_t}).",
                       tex_label="\\sigma_{\\lambda_f}")

        m <= parameter(:σ_σ_ω_covid, 0.0428, (1e-7,100.), (0., 0.), ModelConstructors.Exponential(), RootInverseGamma(4, 0.05), fixed=false,
                       description="σ_σ_ω: The standard deviation of entrepreneurs' capital productivity follows an exogenous process with standard deviation σ_σ_ω.",
                       tex_label="\\sigma_{\\sigma_\\omega}")

        m <= parameter(:σ_lr_covid, 0.1766, (1e-8,10.), (1e-8, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.75), fixed=false,
                       tex_label="\\sigma_{10y}")

        m <= parameter(:σ_tfp_covid, 0.9391, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false,
                       tex_label="\\sigma_{tfp}")

        m <= parameter(:σ_gdp_covid, 0.1, (1e-8, 5.),(1e-8, 5.),ModelConstructors.Exponential(),RootInverseGamma(2, 0.10), fixed=false,
                       tex_label="\\sigma_{gdp}")

        m <= parameter(:σ_gdi_covid, 0.1, (1e-8, 5.),(1e-8, 5.),ModelConstructors.Exponential(),RootInverseGamma(2, 0.10), fixed=false,
                       tex_label="\\sigma_{gdi}")
    end


    if subspec(m) in ["ss88", "ss90", "ss92"]
        m <= parameter(:ρ_λ_f_iid, 0.0, (1e-5, 0.999), (1e-5, 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2),
                       fixed=true,
                       description="ρ_λ_f_iid: The persistence of the mean-reverting shock to the price markup.",
                       tex_label="\\rho_{\\lambda_f, ziid}")
        m <= parameter(:σ_λ_f_iid, 0.0, (0., 100.), (0., 100.), ModelConstructors.Exponential(), RootInverseGamma(10.0, sqrt(25.1)),
                       fixed=false,
                       description="σ_λ_f_iid: The standard deviation of the mean-reverting shock to the price markup.",
                       tex_label="\\sigma_{\\lambda_f, ziid}")
    end

    if subspec(m) in ["ss86", "ss89", "ss91"]
        m <= parameter(:σ_λ_f_iid, 0.0, (0., 100.), (0., 100.), ModelConstructors.Exponential(), RootInverseGamma(10.0, sqrt(25.1)),
                       fixed=false,
                       description="σ_λ_f_iid: The standard deviation of the mean-reverting shock to the price markup.",
                       tex_label="\\sigma_{\\lambda_f, ziid}")
    end

    if subspec(m) in ["ss94", "ss95", "ss96"]
        m <= parameter(:ρ_λ_f_iid, 0.8827, (1e-5, 0.999), (1e-5, 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2),
                       fixed=false,
                       description="ρ_λ_f_iid: The persistence of the mean-reverting shock to the price markup.",
                       tex_label="\\rho_{\\lambda_f, ziid}")
        m <= parameter(:σ_λ_f_iid, 0.0, (0., 100.), (0., 100.), ModelConstructors.Exponential(), RootInverseGamma(10.0, sqrt(25.1)),
                       fixed=false,
                       description="σ_λ_f_iid: The standard deviation of the mean-reverting shock to the price markup.",
                       tex_label="\\sigma_{\\lambda_f, ziid}")
    end

    if subspec(m) in ["ss98"]
        m <= parameter(:ρ_biidc_sh, 0.75, (0., 0.999), (0., 0.999), ModelConstructors.Untransformed(), BetaAlt(0.5, 0.2), fixed=false,
                       description="ρ_biidc_sh: AR(1) coefficient for the shock of the preference process.",
                       tex_label="\\rho_{b, iid, c, sh}")
    end

    if subspec(m) in ["ss99"]
        m <= parameter(:meas_π1, 0.0, (0.0, 5.0), (0.0, 5.0), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=true,
                       tex_label="meas_\\pi1")
    end

    if subspec(m) in ["ss100"]
        m <= parameter(:φ_π, 4.0, (1.25, 15.0), (1.25, 15.0), ModelConstructors.Untransformed(), Normal(4.0, 3.0), fixed=false,
                       description="φ_π: Weight on inflation in AIT Rule",
                       tex_label="\\varphi_{\\pi}")

        m <= parameter(:φ_y, 3.0, (1.25, 15.0), (1.25, 15.0), ModelConstructors.Untransformed(), Normal(3.0, 3.0), fixed=false,
                       description="φ_y: Weight on output gap in AIT Rule",
                       tex_label="\\varphi_y")

        m <= parameter(:ρ_smooth, 0.9, (1e-5, 0.999), (1e-5, 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.75, 0.10), fixed=false,
                       description="ρ_smooth: Degree of inertia in AIT Rule",
                       tex_label="\\rho_{smooth}")
    end

    if subspec(m) in ["ss103"]
        m <= parameter(:κ_std_bcshocks, 1.0, (0.0, 1.0), (0.0, 1.0), ModelConstructors.SquareRoot(), Uniform(0,1), fixed=false,
                       description="κ_std_bcshocks: scaling factor for standard business cycle shocks during covid",
                       tex_label="\\kappa_{bcshocks}")
        m <= parameter(:κ_covid, 1.0, (0.0, 1.0), (0.0, 1.0), Untransformed(), Uniform(0,1), fixed=false,
                       description="Fraction of regime 2 value used in regime 3 for σ_{covid}",
                       tex_label = "\\kappa_{covid}")
        m <= parameter(:κ_pce, 1.0, (0.0, 1.0), (0.0, 1.0), Untransformed(), Uniform(0,1), fixed=false,
                       description="Fraction of regime 2 value used in regime 3 for σ_{meas,π}",
                       tex_label = "\\kappa_{pce}")
    end

    if haskey(get_settings(m), :add_ait_rm) ? get_setting(m, :add_ait_rm) : false
        m <= parameter(:ρ_ait_rm, 0.2135, (-1e-5, 0.999), (-1e-5, 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
                       description="ρ_ait_rm: AR(1) coefficient in the AIT monetary policy shock process.",
                       tex_label="\\rho_{ait,r^m}")
        m <= parameter(:σ_ait_rm, 0.2380, (0.0, 5.), (0.0, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false,
                       description="σ_ait_rm: The standard deviation of the AIT monetary policy shock.",
                       tex_label="\\sigma_{ait,r^m}")

        for i in mon_anticipated_ait_shocks(m)
            m <= parameter(Symbol("σ_ait_r_m$i"), .2, (0.0, 100.), (0.0, 0.), ModelConstructors.Exponential(),
                           RootInverseGamma(4, .2), fixed=false,
                           description="σ_ait_r_m$i: Standard deviation of the $i-period-ahead anticipated AIT policy shock.",
                           tex_label=@sprintf("\\sigma_{ait,r^m%d}",i))
        end
    end


# Kappa to restrict values to fixed proportion of value in earlier regime
    if haskey(m.settings, :add_κ_covid) && get_setting(m, :add_κ_covid)
        m <= parameter(:κ_covid, 1.0, (0.0, 2.0), (0.0, 2.0), Untransformed(), Uniform(0,1), fixed=false,
                       description="Fraction of regime 2 value used in regime 3 for σ_{covid}",
                       tex_label = "\\kappa_{covid}")
    end

    if haskey(m.settings, :add_κ_pce) && get_setting(m, :add_κ_pce)
        m <= parameter(:κ_pce, 1.0, (0.0, 2.0), (0.0, 2.0), Untransformed(), Uniform(0,1), fixed=true,
                       description="Fraction of regime 2 value used in regime 3 for σ_{meas,π}",
                       tex_label = "\\kappa_{pce}")
    end

    if haskey(get_settings(m), :add_initialize_pgap_ygap_pseudoobs) ?
        get_setting(m, :add_initialize_pgap_ygap_pseudoobs) : false
        m <= parameter(:σ_pgap, 0., (0., 1e2), (0., 5.), ModelConstructors.Exponential(),
                       RootInverseGamma(2. * (20.)^2 ./ .1, sqrt((4.)^2  + .1)), fixed=true,
                       tex_label="\\sigma_{pgap}")
        m <= parameter(:σ_ygap, 0., (0., 1e2), (0., 5.), ModelConstructors.Exponential(),
                       RootInverseGamma(2. * (20.)^2 ./ .1, sqrt((4.)^2  + .1)), fixed=true,
                       tex_label="\\sigma_{ygap}")
    end

    if haskey(get_settings(m), :add_iid_cond_obs_gdp_meas_err) ?
        get_setting(m, :add_iid_cond_obs_gdp_meas_err) : false
        m <= parameter(:ρ_condgdp, 0., (-0.999, 0.999), (-0.999, 0.999),
                       ModelConstructors.SquareRoot(), Normal(0.0, 0.2), fixed=true,
                       tex_label="\\rho_{cond gdp}")
        m <= parameter(:σ_condgdp, 0.1, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                       RootInverseGamma(2, 0.10), fixed=false,
                       tex_label="\\sigma_{cond gdp}")
    end

    if haskey(get_settings(m), :add_iid_anticipated_obs_gdp_meas_err) ?
        get_setting(m, :add_iid_anticipated_obs_gdp_meas_err) : false
        m <= parameter(:ρ_gdpexp, 0., (-0.999, 0.999), (-0.999, 0.999),
                       ModelConstructors.SquareRoot(), Normal(0.0, 0.2), fixed=true,
                       tex_label="\\rho_{gdpexp}")
        m <= parameter(:σ_gdpexp, 0.1, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                       RootInverseGamma(2, 0.10), fixed=false,
                       tex_label="\\sigma_{gdpexp}")
    end

    if haskey(get_settings(m), :add_iid_cond_obs_corepce_meas_err) ?
        get_setting(m, :add_iid_cond_obs_corepce_meas_err) : false
        m <= parameter(:ρ_condcorepce, 0., (-0.999, 0.999), (-0.999, 0.999),
                       ModelConstructors.SquareRoot(), Normal(0.0, 0.2), fixed=true,
                       tex_label="\\rho_{cond corepce}")
        m <= parameter(:σ_condcorepce, 0.0999, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                       RootInverseGamma(2, 0.10), fixed=false,
                       tex_label="\\sigma_{cond corepce}")
    end

# SPD expected FFR measurement error parameters
    if !isempty(expected_ffr(m))
        for i in expected_ffr(m)
            m <= parameter(Symbol("σ_exp_rm$i"), 0.0375 + 0.00625 * i, (0.0, 5.), (0.0, 5.), ModelConstructors.Exponential(),
                           RootInverseGamma(4, .2), fixed=true,
                           description="σ_exp_rm$i: Standard deviation of the $i-period-ahead FFR measurement error.",
                           tex_label=@sprintf("\\sigma_{exp_{r^m}%d}",i))

        end
        m <= parameter(:ρ_exp_rm, 0., (-1e-5, 0.999), (-1e-5, 0.999), ModelConstructors.SquareRoot(), Normal(0.0, 0.2), fixed=true,
                       tex_label="\\rho_{exp_rm}")
    end

    for i = 1:n_mon_anticipated_shocks_padding(m)
        if i <= n_mon_anticipated_shocks(m)
            m <= parameter(Symbol("σ_r_m$i"), .2, (0.0, 100.), (0.0, 0.), ModelConstructors.Exponential(),
                           RootInverseGamma(4, .2), fixed=false,
                           description="σ_r_m$i: Standard deviation of the $i-period-ahead anticipated policy shock.",
                           tex_label=@sprintf("\\sigma_{ant%d}",i))
        else

            m <= parameter(Symbol("σ_r_m$i"), .0, (0.0, 100.), (1e-5, 0.), ModelConstructors.Exponential(), RootInverseGamma(4, .2), fixed=true,
                           description="σ_r_m$i: Standard deviation of the $i-period-ahead anticipated policy shock.",
                           tex_label=@sprintf("\\sigma_{ant%d}",i))
        end
    end

    for (sh, ant_num) in get_setting(m, :antshocks)
        ant_tex_label = DSGE.detexify(sh) == sh ? string(sh) : "\\" * string(DSGE.detexify(sh))
        for i in 1:ant_num
            m <= parameter(Symbol("σ_$(sh)$i"), 0., (0., 1e3), (0., 1e3), ModelConstructors.Exponential(), RootInverseGamma(4, .2),
                           fixed=false,
                           description="σ_$(sh)$i: Standard deviation of the $i-period-ahead anticipated policy shock.",
                           tex_label="\\sigma_{$(ant_tex_label), ant$(i)}")
        end
    end

    for key in get_setting(m, :proportional_antshocks)
        propant_tex_label = DSGE.detexify(key) == key ? string(key) : "\\" * string(DSGE.detexify(key))
        m <= parameter(Symbol(:σ_, key, :_prop), 0., (0., 1e3), (0., 0.), ModelConstructors.Exponential(),
                       RootInverseGamma(2, 1.), fixed=false,
                       description="σ_$(key)_prop: proportional of anticipated shock to contemporaneous shock to $key",
                       tex_label="\\sigma_{$(propant_tex_label)}^{prop}")
    end

end

"""
init_subspec_indices(m::AbstractDSGEModel)

There is a lot of easy work that can be done here to make this function more forward looking in case people want to add a group.
"""
function init_subspec_indices(m::AbstractDSGEModel,
                              endogenous_states::Vector{Symbol},
                              exogenous_shocks::Vector{Symbol},
                              expected_shocks::Vector{Symbol},
                              equilibrium_conditions::Vector{Symbol},
                              observables::Vector{Symbol}
                              pseudo_observables::Vector{Symbol})

    for (key, val) in get_setting(m, :antshocks)
        endogenous_states = vcat(endogenous_states, [Symbol(key, "_tl$i") for i = 1:val])
        equilibrium_conditions = vcat(equilibrium_conditions, [Symbol("eq_", key, "l$i") for i = 1:val])
        exogenous_shocks = vcat(exogenous_shocks, [Symbol(key, "_shl$i") for i = 1:val])
    end

    # SPD expected FFR measurement error
    if !isempty(expected_ffr(m))
        for i in expected_ffr(m)
            push!(endogenous_states_augmented, Symbol("e_exp_rm$i"))
            push!(exogenous_shocks, Symbol("exp_rm_sh$i"))
        end
    end


    if haskey(get_settings(m), :add_cprod_states) ? get_setting(m, :add_cprod_states) : false # parse(Int, SubString(subspec(m), 3,subspec_ind)) >= 59
        push!(endogenous_states, :ziid_t)
        push!(equilibrium_conditions, :eq_ziid)
        push!(exogenous_shocks, :ziid_sh)
        push!(endogenous_states, :biidc_t)
        push!(equilibrium_conditions, :eq_biidc)
        push!(exogenous_shocks, :biidc_sh)
        push!(endogenous_states, :φ_t)
        push!(endogenous_states, :Eφ_t)
        push!(equilibrium_conditions, :eq_φ)
        push!(equilibrium_conditions, :eq_Eφ)
        push!(exogenous_shocks, :φ_sh)
    end

    if haskey(get_settings(m), :add_meas_pi_states) ? get_setting(m, :add_meas_pi_states) : false# parse(Int, SubString(subspec(m),3,subspec_ind)) >= 87
        push!(endogenous_states_augmented, :e_meas_π_t, :e_meas_π_t1)

        push!(exogenous_shocks, :meas_π_sh)
    end


    if subspec(m) in ["ss13", "ss17", "ss20"]
        push!(endogenous_states_augmented, :Sinf_t, :πtil_t, :πtil_t1)
    end

    if subspec(m) in ["ss14", "ss15", "ss16", "ss18", "ss19"]
        push!(endogenous_states_augmented, :e_tfp_t1)
        push!(endogenous_states_augmented, :Sinf_t, :πtil_t, :πtil_t1)
    end

    # COVID counterparts for standard business cycle shocks
    if subspec(m) in ["ss67", "ss68", "ss71", "ss72", "ss75", "ss76", "ss80", "ss82", "ss83"]
        push!(endogenous_states, :g_covid_t)
        push!(equilibrium_conditions, :eq_g_covid)
        push!(exogenous_shocks, :g_covid_sh)
        push!(endogenous_states, :μ_covid_t)
        push!(equilibrium_conditions, :eq_μ_covid)
        push!(exogenous_shocks, :μ_covid_sh)
        push!(endogenous_states, :λ_f_covid_t)
        push!(equilibrium_conditions, :eq_λ_f_covid)
        push!(exogenous_shocks, :λ_f_covid_sh)
        push!(endogenous_states, :σ_ω_covid_t)
        push!(equilibrium_conditions, :eq_σ_ω_covid)
        push!(exogenous_shocks, :σ_ω_covid_sh)

        push!(endogenous_states_augmented, :e_lr_covid_t)
        push!(exogenous_shocks, :lr_covid_sh)
        push!(endogenous_states_augmented, :e_tfp_covid_t)
        push!(exogenous_shocks, :tfp_covid_sh)
        push!(endogenous_states_augmented, :e_gdp_covid_t)
        push!(exogenous_shocks, :gdp_covid_sh)
        push!(endogenous_states_augmented, :e_gdi_covid_t)
        push!(exogenous_shocks, :gdi_covid_sh)
    end

    if subspec(m) in ["ss69", "ss70", "ss73", "ss74", "ss77", "ss78"]
        push!(endogenous_states, :g_covid_t)
        push!(equilibrium_conditions, :eq_g_covid)
        push!(exogenous_shocks, :g_covid_sh)
        push!(endogenous_states, :μ_covid_t)
        push!(equilibrium_conditions, :eq_μ_covid)
        push!(exogenous_shocks, :μ_covid_sh)
        push!(endogenous_states, :λ_f_covid_t)
        push!(equilibrium_conditions, :eq_λ_f_covid)
        push!(exogenous_shocks, :λ_f_covid_sh)
        push!(endogenous_states, :σ_ω_covid_t)
        push!(equilibrium_conditions, :eq_σ_ω_covid)
        push!(exogenous_shocks, :σ_ω_covid_sh)

        push!(endogenous_states_augmented, :e_lr_covid_t)
        push!(exogenous_shocks, :lr_covid_sh)
        push!(endogenous_states_augmented, :e_tfp_covid_t)
        push!(exogenous_shocks, :tfp_covid_sh)
        push!(endogenous_states_augmented, :e_gdp_covid_t)
        push!(exogenous_shocks, :gdp_covid_sh)
        push!(endogenous_states_augmented, :e_gdi_covid_t)
        push!(exogenous_shocks, :gdi_covid_sh)

        push!(endogenous_states, :zp_covid_t)
        push!(equilibrium_conditions, :eq_zp_covid)
        push!(exogenous_shocks, :zp_covid_sh)
    end

    if subspec(m) in ["ss86", "ss89", "ss91"]
        push!(equilibrium_conditions, :eq_λ_f_persist)
        push!(endogenous_states, :λ_f_t_persist)

        push!(exogenous_shocks, :λ_f_iid_sh)
    end

    if subspec(m) in ["ss88", "ss90", "ss92", "ss94", "ss95", "ss96"]
        push!(endogenous_states, :λ_f_mr)
        push!(equilibrium_conditions, :eq_λ_f_mr)

        push!(equilibrium_conditions, :eq_λ_f_persist)
        push!(endogenous_states, :λ_f_t_persist)

        push!(exogenous_shocks, :λ_f_iid_sh)
    end

    if subspec(m) in ["ss98"]
        push!(equilibrium_conditions, :eq_biidc_sh)
        push!(endogenous_states, :biidc_sh1)
    end


    # Add various pseudo-observables
    if get_setting(m, :add_laborproductivity_measurement)
        push!(endogenous_states_augmented, :cum_z_t)
        m <= Setting(:integrated_series, [:cum_z_t])
    end
    if get_setting(m, :add_nominalgdp_level)
        integ_series = [:cum_z_t, :cum_y_t, :cum_e_gdp_t, :cum_π_t]
        push!(endogenous_states_augmented, setdiff(integ_series, endogenous_states_augmented)...)
        if haskey(get_settings(m), :integrated_series)
            integ_series = union(get_setting(m, :integrated_series), integ_series)
        end
        m <= Setting(:integrated_series, integ_series)
    end
    if get_setting(m, :add_cumulative)
        push!(endogenous_states_augmented, setdiff([:y_f_t1, :c_f_t1, :i_f_t1], endogenous_states_augmented)...)
        integ_series = [:cum_z_t, :cum_y_t, :cum_e_gdp_t, :cum_π_t,
                        :cum_y_f_t, :cum_c_t, :cum_c_f_t,
                        :cum_i_t, :cum_i_f_t]
        push!(endogenous_states_augmented, setdiff(integ_series, endogenous_states_augmented)...)
        if haskey(get_settings(m), :integrated_series)
            integ_series = union(get_setting(m, :integrated_series), integ_series)
        end
        m <= Setting(:integrated_series, integ_series)
    end
    if get_setting(m, :add_flexible_price_growth)
        push!(endogenous_states_augmented, setdiff([:y_f_t1, :c_f_t1, :i_f_t1], endogenous_states_augmented)...)
    end
    if haskey(get_settings(m), :add_pgap) ? get_setting(m, :add_pgap) : false
        push!(endogenous_states, setdiff([:pgap_t], endogenous_states)...)
        push!(equilibrium_conditions, setdiff([:eq_pgap], equilibrium_conditions)...)
    end
    if haskey(get_settings(m), :add_ygap) ? get_setting(m, :add_ygap) : false
        push!(endogenous_states, setdiff([:ygap_t], endogenous_states)...)
        push!(equilibrium_conditions, setdiff([:eq_ygap], equilibrium_conditions)...)
    end
    if haskey(get_settings(m), :add_altpolicy_pgap) ? get_setting(m, :add_altpolicy_pgap) : false
        push!(endogenous_states, setdiff([:pgap_t], endogenous_states)...)
        push!(equilibrium_conditions, setdiff([:eq_pgap], equilibrium_conditions)...)
    end
    if haskey(get_settings(m), :add_altpolicy_ygap) ? get_setting(m, :add_altpolicy_ygap) : false
        push!(endogenous_states, setdiff([:ygap_t], endogenous_states)...)
        push!(equilibrium_conditions, setdiff([:eq_ygap], equilibrium_conditions)...)
    end
    if (haskey(get_settings(m), :add_initialize_pgap_ygap_pseudoobs) ? get_setting(m, :add_initialize_pgap_ygap_pseudoobs) : false)
        push!(exogenous_shocks, setdiff([:pgap_sh, :ygap_sh], exogenous_shocks)...)
    end
    if haskey(get_settings(m), :add_rw) ? get_setting(m, :add_rw) : false
        push!(endogenous_states, setdiff([:rw_t, :Rref_t], endogenous_states)...)
        push!(equilibrium_conditions, setdiff([:eq_rw, :eq_Rref], equilibrium_conditions)...)
    end
    if haskey(get_settings(m), :add_ait_rm) ? get_setting(m, :add_ait_rm) : false
        push!(endogenous_states, setdiff([:ait_rm_t], endogenous_states)...)
        push!(equilibrium_conditions, setdiff([:eq_ait_rm], equilibrium_conditions)...)
        push!(exogenous_shocks, setdiff([:rm_ait_sh], exogenous_shocks)...)
        if !isempty(mon_anticipated_ait_shocks(m))
            for i = 1:maximum(mon_anticipated_ait_shocks(m))
                push!(endogenous_states, setdiff([Symbol("rm_ait_tl$i")], endogenous_states)...)
                push!(exogenous_shocks, setdiff([Symbol("rm_ait_shl$i")], exogenous_shocks)...)
                push!(equilibrium_conditions, setdiff([Symbol("eq_ait_rml$i")], equilibrium_conditions)...)
            end
        end
    end

    if haskey(get_settings(m), :add_iid_cond_obs_gdp_meas_err) ?
        get_setting(m, :add_iid_cond_obs_gdp_meas_err) : false
        push!(endogenous_states_augmented, :e_condgdp_t)
        push!(exogenous_shocks, :condgdp_sh)
    end

    if haskey(get_settings(m), :add_iid_anticipated_obs_gdp_meas_err) ?
        get_setting(m, :add_iid_anticipated_obs_gdp_meas_err) : false
        push!(endogenous_states_augmented, :e_gdpexp_t)
        push!(exogenous_shocks, :gdpexp_sh)
    end

    if haskey(get_settings(m), :add_iid_cond_obs_corepce_meas_err) ?
        get_setting(m, :add_iid_cond_obs_corepce_meas_err) : false
        push!(endogenous_states_augmented, :e_condcorepce_t)
        push!(exogenous_shocks, :condcorepce_sh)
    end

    return endogenous_states, exogenous_shocks, expected_shocks, equilibrium_conditions, observables, pseudo_observables
end


function init_subspec_parameter_groupings(m::Model1002, groupings::OrderedDict{String, Vector{Parameter}})

    subspec_ind = parse(Int, SubString(subspec(m),3,subspec_ind))

    if haskey(get_settings(m), :add_meas_pi_error_group) ? get_setting(m, :add_meas_pi_error_group) : false # subspec_num >= 87
        push!(groupings["Measurement Error Parameters"], :ρ_meas_π, :σ_meas_π)
    end
    if haskey(get_settings(m), :add_pol_smooth_group) ? get_setting(m, :add_pol_smooth_group) : false # subspec_num >= 100 && subspec_num != 104
        push!(groupings["Policy Parameters"], :φ_π, :φ_y, :ρ_smooth)
    end
    if haskey(get_settings(m), :add_ait_rm) && get_setting(m, :add_ait_rm)
        push!(groupings["Policy Parameters"], :σ_ait_rm, :ρ_ait_rm)
    end

    if !isempty(expected_ffr(m))
        for i in expected_ffr(m)
            push!(groupings["Measurement Error Parameters"], Symbol("σ_exp_rm$i"))
        end
        push!(groupings["Measurement Error Parameters"], Symbol("ρ_exp_rm"))
    end

    if haskey(get_settings(m), :add_cprod_group) ? get_setting(m, :add_cprod_group) : false # subspec_num >= 59
        covid = [:σ_ziid, :σ_biidc, :σ_φ]
        for (sh, ant_num) in get_setting(m, :antshocks)
            for i in 1:ant_num
                push!(covid, Symbol("σ_$(sh)$i"))
            end
        end
        for key in get_setting(m, :proportional_antshocks)
            push!(covid, Symbol(:σ_, key, :_prop))
        end
        if haskey(m.settings, :add_κ_covid) && get_setting(m, :add_κ_covid)
            push!(covid, :κ_covid)
        end
        if haskey(m.settings, :add_κ_pce) && get_setting(m, :add_κ_pce)
            push!(covid, :κ_pce)
        end
        push!(all_keys, covid)
        push!(descriptions, "COVID-19 Parameters")
    end

    # Ensure no parameters missing
    incl_params    = vcat(collect(values(groupings))...)
    excl_params_sym = vcat([:Upsilon, :ρ_μ_e, :ρ_γ, :σ_μ_e, :σ_γ, :Iendoα, :γ_gdi, :δ_gdi],
                           [Symbol("σ_r_m$i") for i=n_mon_anticipated_shocks(m)+1:n_mon_anticipated_shocks_padding(m)])
    if haskey(get_settings(m), :add_cprod_group) ? get_setting(m, :add_cprod_group) # subspec_num >= 59
        push!(excl_params_sym, :ρ_ziid, :ρ_biidc, :ρ_φ)
    end
    if haskey(get_settings(m), :add_initialize_pgap_ygap_pseudoobs) ?
        get_setting(m, :add_initialize_pgap_ygap_pseudoobs) : false
        push!(excl_params_sym, :σ_pgap, :σ_ygap)
    end
    if haskey(get_settings(m), :add_iid_cond_obs_gdp_meas_err) ?
        get_setting(m, :add_iid_cond_obs_gdp_meas_err) : false
        push!(excl_params_sym, :σ_condgdp, :ρ_condgdp)
    end
    if haskey(get_settings(m), :add_iid_cond_obs_corepce_meas_err) ?
        get_setting(m, :add_iid_cond_obs_corepce_meas_err) : false
        push!(excl_params_sym, :σ_condcorepce, :ρ_condcorepce)
    end
    excl_params = [m[θ] for θ in excl_params_sym]

    @assert isempty(setdiff(m.parameters, vcat(incl_params, excl_params)))

    return groupings

end



function init_subspec_shock_groupings(m::Model1002)
    # For parsing model subspec to Int
    subspec_ind = isletter(subspec(m)[end]) ? length(subspec(m)) - 1 : length(subspec(m))

    if haskey(get_settings(m), :add_std_shock_group) ? get_setting(m, :add_std_shock_group) : false # parse(Int, SubString(subspec(m),3,subspec_ind)) >= 59
        gov = ShockGroup("g", [:g_sh], RGB(0.70, 0.13, 0.13)) # firebrick
        bet = ShockGroup("b", [:b_sh], RGB(0.3, 0.3, 1.0))
        fin = ShockGroup("FF", [:γ_sh, :μ_e_sh, :σ_ω_sh], RGB(0.29, 0.0, 0.51)) # indigo
        # tfp = ShockGroup("tfp", [:ztil_sh], RGB(1.0, 0.55, 0.0)) # darkorange
        tfp = ShockGroup("tfp", [:ztil_sh, :zp_sh], RGB(1.0, 0.55, 0.0)) # darkorange
        pmu = ShockGroup("mkp", [:λ_f_sh, :λ_w_sh], RGB(0.60, 0.80, 0.20)) # yellowgreen
        wmu = ShockGroup("w-mkp", [:λ_w_sh], RGB(0.0, 0.5, 0.5)) # teal
        phi = ShockGroup("phi", [:φ_sh], RGB(0.5, 0.5, 0.))

        rm_vec = vcat([:rm_sh], [Symbol("rm_shl$i") for i = 1:n_mon_anticipated_shocks(m)])
        if haskey(get_settings(m), :add_ait_rm) ? get_setting(m, :add_ait_rm) : false
            append!(rm_vec, [:rm_ait_sh])
            if !isempty(mon_anticipated_ait_shocks(m))
                for i in mon_anticipated_ait_shocks(m)
                    push!(rm_vec, Symbol("ait_rm_sh$i"))
                end
            end
        end

        pol = ShockGroup("pol", rm_vec, RGB(1.0, 0.84, 0.0)) # gold
        pis = ShockGroup("pi-LR", [:π_star_sh], RGB(1.0, 0.75, 0.793)) # pink
        mei = ShockGroup("mu", [:μ_sh], :cyan)

        mea_vec = [:lr_sh, :tfp_sh, :gdpdef_sh, :corepce_sh, :gdp_sh, :gdi_sh]
        if haskey(get_settings(m), :add_meas_pi_shock_group) ? get_setting(m, :add_meas_pi_shock_group) : false # parse(Int, SubString(subspec(m),3,subspec_ind)) >= 87
            push!(mea_vec, :meas_π_sh)
        end
        if !isempty(expected_ffr(m))
            for i in expected_ffr(m)
                push!(rm_vec, Symbol("exp_rm_sh$i"))
                #push!(exogenous_shocks, Symbol("exp_rm_sh$i"))
            end
        end

        if haskey(get_settings(m), :add_iid_cond_obs_gdp_meas_err) ?
            get_setting(m, :add_iid_cond_obs_gdp_meas_err) : false
            mea = ShockGroup("me", push!(mea_vec, :condgdp_sh, :condcorepce_sh),
                             RGB(0.0, 0.8, 0.0))
        elseif haskey(get_settings(m), :add_iid_anticipated_obs_gdp_meas_err) ?
            get_setting(m, :add_iid_anticipated_obs_gdp_meas_err) : false
            mea = ShockGroup("me", push!(mea_vec, :gdpexp_sh),
                             RGB(0.0, 0.8, 0.0))
        else
            mea = ShockGroup("me", mea_vec, RGB(0.0, 0.8, 0.0))
        end

        zpe = ShockGroup("zp", [:zp_sh], RGB(0.0, 0.3, 0.0))
        det = ShockGroup("dt", [:dettrend], :gray40)
        oth = ShockGroup("other", [:dettrend, :g_sh, :π_star_sh, :μ_sh], :gray40) # :dettrend

        # COVID-19 Shocks
        betcovid = ShockGroup("biidc", haskey(m.exogenous_shocks, :biidc_shl1) ? [:biidc_sh, :biidc_shl1] : [:biidc_sh],
                              RGB(0.70, 0.13, 0.13))
        zcovid   = ShockGroup("ziid", [:ziid_sh], RGB(0., 0.5, 0.5))
        φcovid   = ShockGroup("phi", [:φ_sh], RGB(0.5, 0.5, 0.))
        ocovid   = ShockGroup("Other COVID", [:ziid_sh, :φ_sh], RGB(0., 0.5, 0.5))

        if haskey(m.exogenous_shocks, :pgap_sh)
            push!(ocovid.shocks, :pgap_sh)
        end
        if haskey(m.exogenous_shocks, :ygap_sh)
            push!(ocovid.shocks, :ygap_sh)
        end

        # Time-varying CCC, e.g. from temporary ZLB
        if haskey(get_settings(m), :gensys2) ? get_setting(m, :gensys2) : false
            st = ShockGroup("States Trend", [:StatesTrend], :darkgreen) # :dettrend
            return [betcovid, ocovid, bet, fin, tfp, pmu, pol, mea, oth, st]
        else
            return [betcovid, ocovid, bet, fin, tfp, pmu, pol, mea, oth]
        end
        # return [gov, bet, fin, tfp, pmu, wmu, pol, pis, mei, mea, zpe, det]
        # return [betcovid, zcovid, φcovid, bet, fin, tfp, pmu, pol, mea, oth]
    elseif subspec(m) != "ss12"
        gov = ShockGroup("g", [:g_sh], RGB(0.70, 0.13, 0.13)) # firebrick
        bet = ShockGroup("b", [:b_sh], RGB(0.3, 0.3, 1.0))
        fin = ShockGroup("FF", [:γ_sh, :μ_e_sh, :σ_ω_sh], RGB(0.29, 0.0, 0.51)) # indigo
        tfp = ShockGroup("z", [:ztil_sh], RGB(1.0, 0.55, 0.0)) # darkorange
        pmu = ShockGroup("p-mkp", [:λ_f_sh], RGB(0.60, 0.80, 0.20)) # yellowgreen
        wmu = ShockGroup("w-mkp", [:λ_w_sh], RGB(0.0, 0.5, 0.5)) # teal
        pol = ShockGroup("pol", vcat([:rm_sh], [Symbol("rm_shl$i") for i = 1:n_mon_anticipated_shocks(m)]),
                         RGB(1.0, 0.84, 0.0)) # gold
        pis = ShockGroup("pi-LR", [:π_star_sh], RGB(1.0, 0.75, 0.793)) # pink
        mei = ShockGroup("mu", [:μ_sh], :cyan)
        mea = ShockGroup("me", [:lr_sh, :tfp_sh, :gdpdef_sh, :corepce_sh, :gdp_sh, :gdi_sh], RGB(0.0, 0.8, 0.0))
        zpe = ShockGroup("zp", [:zp_sh], RGB(0.0, 0.3, 0.0))
        det = ShockGroup("dt", [:dettrend], :gray40)

        return [gov, bet, fin, tfp, pmu, wmu, pol, pis, mei, mea, zpe, det]
    else
        # financial, productivity, other, measurement errors
        fin = ShockGroup("FF", [:γ_sh, :μ_e_sh, :σ_ω_sh], RGB(0.29, 0.0, 0.51)) # indigo
        prod = ShockGroup("Prod", [:ztil_sh, :zp_sh], RGB(1.0, 0.55, 0.0)) # darkorange
        mea = ShockGroup("Measurement", [:lr_sh, :tfp_sh, :gdpdef_sh, :corepce_sh,
                                         :gdp_sh, :gdi_sh], RGB(0.0, 0.8, 0.0))
        other = ShockGroup("Other", [:g_sh, :b_sh, :λ_f_sh, :λ_w_sh,
                                     vcat([:rm_sh],
                                     [Symbol("rm_shl$i") for i = 1:n_mon_anticipated_shocks(m)])...,
                                     :π_star_sh, :μ_sh,
                                     :dettrend], RGB(0.70, 0.13, 0.13)) # firebrick
        return [fin, prod, mea, other]
    end
end



function init_subspec_steadystate!(m::Model1002)
    println("not implemented yet!")
end
