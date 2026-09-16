using JLD2
include(joinpath(@__DIR__, "../../../helpers/genweight.jl"))
include(joinpath(@__DIR__, "IRFs_CSC_tvcopula.jl"))
include(joinpath(@__DIR__, "update_ss_csc.jl"))
"""
    dyn_CSC_tvcopula(
        final_table,
        param, grid, SS_stats,
        mu_dist, Value, mutil_c, Va;
        pool=nothing
    ) -> Dict{String,Any}

Run the CSC (two-sector labour, financial intermediary) pipeline under a
time-varying copula specification.  Mirrors `dyn_ZLB_tvcopula_new` but
calls `F_sys_CSC_ref_tvcopula` and applies CSC-specific parameter overrides
via `parameters_agg_EST_v3!(...; model=:csc)`.
"""
function dyn_CSC_tvcopula(
    final_table,
    param, grid, SS_stats,
    mu_dist, Value, mutil_c, Va;
    pool=nothing
)

    param["copula"] = "time-varying"

    for jkjkjkjkjk in 1:1
        for lllllll in 1:1

            if jkjkjkjkjk == 1
                param["adjust"] = "G"
            else
                param["adjust"] = "LT"
            end

            # -----------------------------
            # Parameter parsing
            # -----------------------------
            xhat = final_table[1:26, :]

            parameters_agg_CSC(param, SS_stats)

            param_update = Dict{String,Any}()

            getx(i) = xhat isa AbstractVector ? xhat[i] : xhat[i]

            param_update["kappa"]      = getx(1)
            param_update["rho_w"]      = getx(2)        # mapped to rho_w_1/2 by :csc branch
            param_update["d"]          = 1 - getx(3)    # mapped to d_1/2
            param_update["eps_w"]      = 1
            param_update["phi"]        = getx(4) * 10
            param_update["phi_x"]      = 0
            param_update["phi_pi"]     = getx(5)
            param_update["phi_u"]      = getx(6)        # overridden to 0 by :csc branch
            param_update["rho_BB"]     = getx(7)
            param_update["rho_B"]      = getx(8)
            param_update["rho_D"]      = getx(9)
            param_update["rho_G"]      = getx(10)
            param_update["rho_R"]      = getx(11)
            param_update["rho_Z"]      = getx(12)
            param_update["rho_eta"]    = getx(13)
            param_update["rho_iota"]   = getx(14)
            param_update["rho_PSI_W"]  = 0
            param_update["sig_D"]      = getx(15) / 100
            param_update["sig_G"]      = getx(16) / 100
            param_update["sig_R"]      = getx(17) / 100
            param_update["sig_Z"]      = getx(18) / 100
            param_update["sig_eta"]    = getx(19) / 100
            param_update["sig_iota"]   = getx(20) / 100
            param_update["sig_w"]      = getx(21) / 100
            param_update["sig_BB"]     = getx(22) / 100
            param_update["iota"]       = getx(26)       # mapped to iota_1/2 by :csc branch
            param_update["varphi"]     = 0
            param_update["alpha_lk"]   = 0
            param_update["theta_b"]    = 0.97
            param_update["GAMMA"]      = getx(23)
            param_update["rho_B_F"]    = getx(25)
            param_update["sig_B_F"]    = getx(24) / 100
            param_update["rho_X_QE"]   = 0.95
            param_update["phi_pi_QE"]  = 0.5
            param_update["phi_u_QE"]   = 10
            param_update["gamma_B"]    = 0
            param_update["gamma_pi"]   = 0
            param_update["gamma_Y"]    = 0
            param_update["rho_MP"]     = 0
            param_update["gamma_T"]    = 0
            param_update["rho_PSI_RP"] = 0
            param_update["sig_RP"]     = 0.1 / 100
            param_update["Calvo"]      = 0.7

            update_ss_csc!(SS_stats, param, param_update, grid)
            parameters_agg_EST_v3!(param, param_update; model = :csc, SS_stats = SS_stats)

            # -----------------------------
            # State-space reduction
            # -----------------------------
            reduc = state_reduc_tvcopula!(param, grid, SS_stats, mu_dist, Value, mutil_c, Va)
            Xss                   = reduc.Xss
            Yss                   = reduc.Yss
            Gamma_state           = reduc.Gamma_state
            Gamma_control         = reduc.Gamma_control
            InvGamma              = reduc.InvGamma
            State                 = reduc.State
            State_m               = reduc.State_m
            Contr                 = reduc.Contr
            Contr_m               = reduc.Contr_m
            DC                    = reduc.DC
            IDC                   = reduc.IDC
            DCD                   = reduc.DCD
            IDCD                  = reduc.IDCD
            distrSS               = reduc.distrSS
            CDF_SS                = reduc.CDF_SS
            COP_SS                = reduc.COP_SS
            distr_b_SS            = reduc.distr_b_SS
            distr_a_SS            = reduc.distr_a_SS
            distr_se_SS           = reduc.distr_se_SS
            CDF_b_SS              = reduc.CDF_b_SS
            CDF_a_SS              = reduc.CDF_a_SS
            CDF_se_SS             = reduc.CDF_se_SS
            compressionIndexesCOP = reduc.compressionIndexesCOP
            Poly                  = reduc.Poly
            InvCheb               = reduc.InvCheb
            Gamma2                = reduc.Gamma2
            nPoly                 = reduc.nPoly
            nFullCtrl             = reduc.nFullCtrl
            nRedCtrl              = reduc.nRedCtrl
            nFullMarg             = reduc.nFullMarg
            nRedMarg              = reduc.nRedMarg
            nRedStates            = reduc.nRedStates

            StateSS   = Xss
            ControlSS = Yss
            jld2file  = "SSInfo.jld2"

            @save jld2file Xss Yss Gamma_state Gamma_control InvGamma State State_m Contr Contr_m DC IDC DCD IDCD distrSS CDF_SS COP_SS distr_b_SS distr_a_SS distr_se_SS CDF_b_SS CDF_a_SS CDF_se_SS compressionIndexesCOP Poly InvCheb Gamma2 nPoly nFullCtrl nRedCtrl nFullMarg nRedMarg nRedStates grid

            println("Saved file at: ", abspath(jld2file))
            println("SAVED")

            # -----------------------------
            # Reference system
            # -----------------------------
            function F_ref(State, State_m, Contr, Contr_m)
                return F_sys_CSC_ref_tvcopula(
                    State, State_m, Contr, Contr_m, StateSS,
                    ControlSS, Gamma_state, Gamma_control, InvGamma,
                    param, grid, SS_stats, DCD, IDCD
                )
            end

            Fss_ref, LHS_ref, RHS_ref, Distr_ref = F_ref(State, State_m, Contr, Contr_m)
            resid_max = maximum(abs.(Fss_ref))
            println(resid_max)

            # -----------------------------
            # SGU solver
            # -----------------------------
            param["overrideEigen"] = true
            hx, gx, F1_aux, F2_aux, F3_aux, F4_aux, param =
                SGU_solver(F_ref, param, grid)

            # -----------------------------
            # Linearized system
            # -----------------------------
            nse = Int(grid["numstates_endo"])
            ns  = Int(grid["numstates"])
            nc  = Int(grid["numcontrols"])
            nsh = Int(grid["numstates_shocks"])

            F1 = vcat(F1_aux[1:nse, :], F1_aux[(ns+1):end, :])
            F2 = vcat(F2_aux[1:nse, :], F2_aux[(ns+1):end, :])
            F3 = vcat(F3_aux[1:nse, :], F3_aux[(ns+1):end, :])
            F4 = vcat(F4_aux[1:nse, :], F4_aux[(ns+1):end, :])

            F11 = F1[:, 1:nse]
            F31 = F3[:, 1:nse]
            F32 = F3[:, (nse+1):end]

            A_ref = hcat(zeros(nse + nc, nse), F2)
            B_ref = hcat(F11, F4)
            C_ref = hcat(F31, zeros(nse + nc, nc))
            E_ref = F32

            hx_aux = hx[1:nse, :]
            H_aux  = vcat(hx_aux, gx)
            H1_aux = H_aux[:, 1:nse]
            H2_aux = H_aux[:, (end - nsh + 1):end]

            P_ref  = hcat(H1_aux, zeros(nse + nc, nc))
            Q_ref  = H2_aux

            cof_ref = hcat(C_ref, B_ref, A_ref)
            J_ref   = E_ref

            SYS_ref = Dict{String,Any}(
                "A_ref"   => A_ref,
                "B_ref"   => B_ref,
                "C_ref"   => C_ref,
                "E_ref"   => E_ref,
                "P_ref"   => P_ref,
                "Q_ref"   => Q_ref,
                "cof_ref" => cof_ref,
                "J_ref"   => J_ref,
            )

            # -----------------------------
            # IRFs
            # -----------------------------
            IRFs_CSC_tvcopula(hx, gx, Xss, Yss, Gamma_state, Gamma_control,
                              param, grid, SS_stats)

            return Dict{String,Any}(
                "SYS_ref"      => SYS_ref,
                "hx"           => hx,
                "gx"           => gx,
                "resid_max"    => resid_max,
                "param_update" => param_update,
                "param"        => param,
            )
        end
    end

    error("Unexpected: loop did not execute.")
end
