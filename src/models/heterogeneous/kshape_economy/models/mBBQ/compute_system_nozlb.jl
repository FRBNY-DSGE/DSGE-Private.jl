

function compute_system_nozlb!(m::mBBQ)
    
    #update_ss_v5
    DSGE.update_ss_v5!(m) 
    
    #check certain values >0

    #state_reduc_tv_copula 
    DSGE.state_reduc_tvcopula!(m)
    

    #maybe square some values?

    #compute jacob TESTING ONLY, read csv jacob instead of actually compute jacob for testing
    # F21_ad, F22_ad, F23_ad, F24_ad, F41_ad, F42_ad, F43_ad, F44_ad, F21_ad_zlb, F22_ad_zlb, F23_ad_zlb, F24_ad_zlb, F41_ad_zlb, F42_ad_zlb, F43_ad_zlb, F44_ad_zlb = DSGE.jacobian!(m)
    _jcsv = joinpath(@__DIR__, "tests", "csv")
    F21_ad = Matrix(CSV.read(joinpath(_jcsv, "F21_ad.csv"), DataFrame))
    F22_ad = Matrix(CSV.read(joinpath(_jcsv, "F22_ad.csv"), DataFrame))
    F23_ad = Matrix(CSV.read(joinpath(_jcsv, "F23_ad.csv"), DataFrame))
    F24_ad = Matrix(CSV.read(joinpath(_jcsv, "F24_ad.csv"), DataFrame))
    F41_ad = Matrix(CSV.read(joinpath(_jcsv, "F41_ad.csv"), DataFrame))
    F42_ad = Matrix(CSV.read(joinpath(_jcsv, "F42_ad.csv"), DataFrame))
    F43_ad = Matrix(CSV.read(joinpath(_jcsv, "F43_ad.csv"), DataFrame))
    F44_ad = Matrix(CSV.read(joinpath(_jcsv, "F44_ad.csv"), DataFrame))

    # F21_ad_zlb = Matrix(CSV.read(joinpath(_jcsv, "F21_ad_zlb.csv"), DataFrame))
    # F22_ad_zlb = Matrix(CSV.read(joinpath(_jcsv, "F22_ad_zlb.csv"), DataFrame))
    # F23_ad_zlb = Matrix(CSV.read(joinpath(_jcsv, "F23_ad_zlb.csv"), DataFrame))
    # F24_ad_zlb = Matrix(CSV.read(joinpath(_jcsv, "F24_ad_zlb.csv"), DataFrame))



    # SGU solver
    hx, gx, F1_aux, F2_aux, F3_aux, F4_aux, indicator_1 = DSGE.SGU_solver(m, F21_ad, F22_ad, F23_ad, F24_ad, F41_ad, F42_ad, F43_ad, F44_ad)

    if indicator_1 == 0
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, false
    end

    df = DSGE.load_data_bbq(m)

    H_aux, P_ref, Q_ref, HQ, SIGMA_full, SIGMA = DSGE.pq(m, hx, gx, F1_aux, F2_aux, F3_aux, F4_aux)

    regime_inds, y, Ts, Rs, Cs, Qs, Zs, Ds, Es = DSGE.prepare_PQ_regimes(m, df, H_aux, P_ref, Q_ref, HQ, SIGMA_full, SIGMA)


    return regime_inds, y, Ts, Rs, Cs, Qs, Zs, Ds, Es, true
end
