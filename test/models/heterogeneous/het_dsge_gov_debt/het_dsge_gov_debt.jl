using DSGE, ModelConstructors, Random
using Test, BenchmarkTools
using JLD2, FileIO, HDF5
import DSGE: klein_transition_matrices, n_model_states, n_backward_looking_states

# What do you want to do?
write_analytical_jacobian = false
write_autodiff_jacobian = false
write_analytical_solution = false
write_autodiff_solution = false
write_analytical_irfs = false
write_autodiff_irfs = false
check_jacobian_indices = true
check_analytical_jacobian = true
check_autodiff_jacobian = true
check_analytical_solution = true
check_autodiff_solution = true
check_analytical_irfs = true
check_autodiff_irfs = true

path = dirname(@__FILE__)

if write_analytical_jacobian
    m = HetDSGEGovDebt(testing_gamma = true, ref_dir = HETDSGEGOVDEBT)
    m <= Setting(:reduce_ell, false)
    m.testing = true # So that it will test against the unnormalized Jacobian
    m[:βstar] = NaN
    steadystate!(m)
    h5open("$path/reference/analytical_jacobian.h5", "w") do file
        write(file, "JJ", DSGE.jacobian(m))
    end
end

if write_autodiff_jacobian
    m = HetDSGEGovDebt(testing_gamma = true, ref_dir = HETDSGEGOVDEBT)
    m <= Setting(:reduce_ell, false)
    m.testing = true # So that it will test against the unnormalized Jacobian
    m[:βstar] = NaN
    steadystate!(m)
    h5open("$path/reference/autodiff_jacobian.h5", "w") do file
        write(file, "JJ", DSGE.jacobian(m, nt′, nt))
    end
end

if check_jacobian_indices
    m = HetDSGEGovDebt(testing_gamma = true, ref_dir = HETDSGEGOVDEBT)
    m <= Setting(:reduce_ell, false)

    file = JLD2.jldopen("$path/reference/indices_jacobian.jld2", "r")
    KFP = read(file, "KFP")
    KKP = read(file, "KKP")
    LRRP = read(file, "LRRP")
    LIIP = read(file, "LIIP")
    LYP = read(file, "LYP")
    LWP = read(file, "LWP")
    LXP = read(file, "LXP")
    BP = read(file, "BP")
    GP = read(file, "GP")
    ZP = read(file, "ZP")
    MUP = read(file, "MUP")
    LAMWP = read(file, "LAMWP")
    LAMFP = read(file, "LAMFP")
    MONP = read(file, "MONP")
    ELLP = read(file, "ELLP")
    RRP = read(file, "RRP")
    IIP = read(file, "IIP")
    TTP = read(file, "TTP")
    WP = read(file, "WP")
    HHP = read(file, "HHP")
    PIP = read(file, "PIP")
    PIWP = read(file, "PIWP")
    LAMP = read(file, "LAMP")
    YP = read(file, "YP")
    XP = read(file, "XP")
    MCP = read(file, "MCP")
    QP = read(file, "QP")
    RKP = read(file, "RKP")
    KF = read(file, "KF")
    KK = read(file, "KK")
    LRR = read(file, "LRR")
    LII = read(file, "LII")
    LY = read(file, "LY")
    LW = read(file, "LW")
    LX = read(file, "LX")
    B = read(file, "B")
    G = read(file, "G")
    Z = read(file, "Z")
    MU = read(file, "MU")
    LAMW = read(file, "LAMW")
    LAMF = read(file, "LAMF")
    MON = read(file, "MON")
    ELL = read(file, "ELL")
    RR = read(file, "RR")
    II = read(file, "II")
    TT = read(file, "TT")
    W = read(file, "W")
    HH = read(file, "HH")
    PI = read(file, "PI")
    PIW = read(file, "PIW")
    LAM = read(file, "LAM")
    Y = read(file, "Y")
    X = read(file, "X")
    MC = read(file, "MC")
    Q = read(file, "Q")
    RK = read(file, "RK")
    TG  = read(file, "TG")
    TGP = read(file, "TGP")
    BG  = read(file, "BG")
    BGP = read(file, "BGP")
    F1  = read(file, "F1")
    F2  = read(file, "F2")
    F5  = read(file, "F5")
    F6  = read(file, "F6")
    F7  = read(file, "F7")
    F8  = read(file, "F8")
    F9  = read(file, "F9")
    F10 = read(file, "F10")
    F11 = read(file, "F11")
    F12 = read(file, "F12")
    F13 = read(file, "F13")
    F14 = read(file, "F14")
    F15 = read(file, "F15")
    F16 = read(file, "F16")
    F17 = read(file, "F17")
    F18 = read(file, "F18")
    F19 = read(file, "F19")
    F20 = read(file, "F20")
    F24 = read(file, "F24")
    F31 = read(file, "F31")
    F32 = read(file, "F32")
    F33 = read(file, "F33")
    F34 = read(file, "F34")
    F35 = read(file, "F35")
    F36 = read(file, "F36")
    F37 = read(file, "F37")
    F38 = read(file, "F38")
    F39 = read(file, "F39")
    F40 = read(file, "F40")
    F41 = read(file, "F41")
    close(file)
    endo = DSGE.augment_model_states(m.endogenous_states_original, DSGE.n_model_states_original(m))

    eq = m.equilibrium_conditions
    @testset "Check indices" begin
        @testset "function blocks which output a function" begin
            @test KFP == endo[:kf′_t]  ## lagged ell function
        end
        @testset "endogenous scalar-valued states:" begin
            @test KKP   == first(endo[:k′_t])
            # @test LRRP  == first(endo[:R′_t1])
            @test LIIP  == first(endo[:i′_t1]) + 1
            @test LYP   == first(endo[:y′_t1]) + 1
            @test LWP   == first(endo[:w′_t1]) + 1
            @test LXP   == first(endo[:I′_t1]) + 1
            @test BG == first(endo[:bg_t]) + 1
            @test BGP == first(endo[:bg′_t]) + 1
        end
        @testset "# exogenous scalar-valued states" begin
            @test BP    == first(endo[:b′_t]) + 1
            @test GP    == first(endo[:g′_t]) + 1
            @test ZP    == first(endo[:z′_t]) + 1
            @test MUP   == first(endo[:μ′_t]) + 1
            @test LAMWP == first(endo[:λ_w′_t]) + 1
            @test LAMFP == first(endo[:λ_f′_t]) + 1
            @test MONP  == first(endo[:rm′_t]) + 1
        end
        @testset "function-valued jumps" begin
            @test ELLP  == endo[:l′_t] .+ 1
        end
        @testset "scalar-valued jumps" begin
            @test RRP   == first(endo[:R′_t])
            @test IIP   == first(endo[:i′_t])
            @test TTP   == first(endo[:t′_t])
            @test WP    == first(endo[:w′_t])
            @test HHP   == first(endo[:L′_t])
            @test PIP   == first(endo[:π′_t])
            @test PIWP  == first(endo[:π_w′_t])
            @test LAMP  == first(endo[:margutil′_t])
            @test YP    == first(endo[:y′_t])
            @test XP    == first(endo[:I′_t])
            @test MCP   == first(endo[:mc′_t])
            @test QP    == first(endo[:Q′_t])
            @test RKP   == first(endo[:capreturn′_t])
            @test KF    == endo[:kf_t]
            @test KK    == first(endo[:k_t])
            # @test LRR   == first(endo[:R_t1])
            @test LII   == first(endo[:i_t1]) + 1
            @test LY    == first(endo[:y_t1]) + 1
            @test LW    == first(endo[:w_t1]) + 1
            @test LX    == first(endo[:I_t1]) + 1
            @test B     == first(endo[:b_t]) + 1
            @test G     == first(endo[:g_t]) + 1
            @test Z     == first(endo[:z_t]) + 1
            @test MU    == first(endo[:μ_t]) + 1
            @test LAMW  == first(endo[:λ_w_t]) + 1
            @test LAMF  == first(endo[:λ_f_t]) + 1
            @test MON   == first(endo[:rm_t]) + 1
            @test ELL   == endo[:l_t] .+ 1
            @test RR    == first(endo[:R_t])
            @test II    == first(endo[:i_t])
            @test TT == first(endo[:t_t])
            @test W == first(endo[:w_t])
            @test HH == first(endo[:L_t])
            @test PI == first(endo[:π_t])
            @test PIW == first(endo[:π_w_t])
            @test LAM == first(endo[:margutil_t])
            @test Y == first(endo[:y_t])
            @test X == first(endo[:I_t])
            @test MC == first(endo[:mc_t])
            @test Q == first(endo[:Q_t])
            @test RK == first(endo[:capreturn_t])
            @test TG == first(endo[:tg_t])
            @test TGP == first(endo[:tg′_t])
        end
        @testset "function blocks" begin
            @test F1  == eq[:eq_euler]
            @test F2  == eq[:eq_kolmogorov_fwd]
        end
        @testset "function blocks which map functions to scalars" begin
            @test F5  == eq[:eq_agg_consumption]
            @test F6  == eq[:eq_lambda]
        end
        @testset "# scalar blocks involving endogenous variables" begin
            @test F7  == eq[:eq_transfers]
            @test F8  == eq[:eq_investment]
            @test F9  == eq[:eq_tobin_q]
            @test F10 == eq[:eq_capital_accumulation]
            @test F11 == eq[:eq_wage_phillips]
            @test F12 == eq[:eq_price_phillips]
            @test F13 == eq[:eq_marginal_cost]
            @test F14 == eq[:eq_gdp]
            @test F15 == eq[:eq_optimal_kl]
            @test F16 == eq[:eq_taylor]
            @test F17 == eq[:eq_fisher]
            @test F18 == eq[:eq_nominal_wage_inflation]
        end
        @testset "lagged variables" begin
            @test F19 == eq[:LR]
            @test F20 == eq[:LI]
            @test F24 == eq[:LY]
            @test F31 == eq[:LW]
            @test F32 == eq[:LX]
        end
        @testset "shocks" begin
            @test F33 == eq[:eq_b]
            @test F34 == eq[:eq_g]
            @test F35 == eq[:eq_z]
            @test F36 == eq[:eq_μ]
            @test F37 == eq[:eq_λ_w]
            @test F38 == eq[:eq_λ_f]
            @test F39 == eq[:eq_rm]
        end
        @testset "new eqns" begin
            @test F40 == eq[:eq_fiscal_rule]
            @test F41 == eq[:eq_g_budget_constraint]
        end
    end
end

if check_analytical_jacobian
    m = HetDSGEGovDebt(testing_gamma = true, ref_dir = HETDSGEGOVDEBT)
    m <= Setting(:reduce_ell, false)
    m.testing = true # So that it will test against the unnormalized Jacobian
    m[:βstar] = NaN
    steadystate!(m)
    JJ = DSGE.jacobian(m)

    analytical_JJ  = h5read("$path/reference/analytical_jacobian.h5", "JJ")
    @testset "Analytical Jacobian of HetDSGEGovDebt" begin
        @test JJ ≈ analytical_JJ
    end
end

if check_autodiff_jacobian
    m = HetDSGEGovDebt(testing_gamma = true, ref_dir = HETDSGEGOVDEBT)
    m <= Setting(:reduce_ell, false)
    m.testing = true # So that it will test against the unnormalized Jacobian
    m[:βstar] = NaN
    steadystate!(m)
    JJ      = DSGE.autodiff_jacobian(m)

    autodiff_JJ  = h5read("$path/reference/autodiff_jacobian.h5", "JJ")
    @testset "Autodiff Jacobian of HetDSGEGovDebt" begin
        @test JJ ≈ autodiff_JJ
    end
end

if write_analytical_solution
    m = HetDSGEGovDebt(testing_gamma = true, ref_dir = HETDSGEGOVDEBT)
    m <= Setting(:reduce_ell, false)
    steadystate!(m)
    gx_analytical, hx_analytical, _ = klein(m; autodiff = false)
    JLD2.jldopen("$path/reference/solve.jld2", true, true, true, IOStream) do file
        write(file, "gx", gx_analytical)
        write(file, "hx", hx_analytical)
    end
end

if write_autodiff_solution
    m = HetDSGEGovDebt(testing_gamma = true, ref_dir = HETDSGEGOVDEBT)
    m <= Setting(:reduce_ell, false)
    steadystate!(m)
    gx_autodiff, hx_autodiff, _ = klein(m; autodiff = true)
    JLD2.jldopen("$path/reference/solve_autodiff.jld2", true, true, true, IOStream) do file
        write(file, "gx", gx_autodiff)
        write(file, "hx", hx_autodiff)
    end
end

if check_analytical_solution
    m = HetDSGEGovDebt(testing_gamma = true, ref_dir = HETDSGEGOVDEBT)
    m <= Setting(:reduce_ell, false)
    steadystate!(m)
    gx_analytical, hx_analytical, _ = klein(m; autodiff = false)

    file = JLD2.jldopen("$path/reference/solve.jld2", "r")
    saved_gx   = read(file, "gx")
    saved_hx   = read(file, "hx")
    close(file)

    @testset "Check analytical solve outputs" begin
        @test saved_gx  ≈ gx_analytical
        @test saved_hx  ≈ hx_analytical
    end
end

if check_autodiff_solution
    m = HetDSGEGovDebt(testing_gamma = true, ref_dir = HETDSGEGOVDEBT)
    m <= Setting(:reduce_ell, false)
    steadystate!(m)
    gx_autodiff, hx_autodiff, _ = klein(m; autodiff = true)

    file = JLD2.jldopen("$path/reference/solve_autodiff.jld2", "r")
    saved_gx   = read(file, "gx")
    saved_hx   = read(file, "hx")
    close(file)

    @testset "Check autodiff solve outputs" begin
        @test saved_gx  ≈ gx_autodiff
        @test saved_hx  ≈ hx_autodiff
    end
end

if write_analytical_irfs
    m = HetDSGEGovDebt(testing_gamma = true, ref_dir = HETDSGEGOVDEBT)
    m <= Setting(:reduce_ell, false)
    steadystate!(m)
    sys = compute_system(m)
    states, obs, pseudo = impulse_responses(m, sys, use_augmented_states = true)
    endo = m.endogenous_states_original
    JLD2.jldopen("$path/reference/irfs.jld2", true, true, true, IOStream) do file
        write(file, "IRFkf", states[endo[:kf′_t], 1:20, 3])
        write(file, "IRFsh", vec(states[endo[:z′_t], 1:20, 3]))
        write(file, "IRFell", states[endo[:l′_t], 1:20, 3])
        write(file, "IRFR", vec(states[endo[:R′_t], 1:20, 3]))
        write(file, "IRFi", vec(states[endo[:i′_t], 1:20, 3]))
        write(file, "IRFW", vec(states[endo[:w′_t], 1:20, 3]))
        write(file, "IRFPI", vec(states[endo[:π′_t], 1:20, 3]))
        write(file, "IRFT", vec(states[endo[:t′_t], 1:20, 3]))
        write(file, "IRFY", vec(states[endo[:y′_t], 1:20, 3]))
        write(file, "IRFPIW", vec(states[endo[:π_w′_t], 1:20, 3]))
        write(file, "IRFX", vec(states[endo[:I′_t], 1:20, 3]))
        write(file, "IRFK", vec(states[endo[:k′_t], 1:20, 3]))
        write(file, "IRFQ", vec(states[endo[:Q′_t], 1:20, 3]))
        write(file, "IRFRK", vec(states[endo[:capreturn′_t], 1:20, 3]))
        write(file, "IRFH", vec(states[endo[:L′_t], 1:20, 3]))
        write(file, "IRFMC", vec(states[endo[:mc′_t], 1:20, 3]))
        write(file, "IRFLAM", vec(states[endo[:margutil′_t], 1:20, 3]))
    end
end

if write_autodiff_irfs
    m = HetDSGEGovDebt(testing_gamma = true, ref_dir = HETDSGEGOVDEBT)
    m <= Setting(:reduce_ell, false)
    m <= Setting(:autodiff, true)
    steadystate!(m)
    sys = compute_system(m)
    states, obs, pseudo = impulse_responses(m, sys, use_augmented_states = true)
    endo = m.endogenous_states_original
    JLD2.jldopen("$path/reference/irfs_autodiff.jld2", true, true, true, IOStream) do file
        write(file, "IRFkf", states[endo[:kf′_t], 1:20, 3])
        write(file, "IRFsh", vec(states[endo[:z′_t], 1:20, 3]))
        write(file, "IRFell", states[endo[:l′_t], 1:20, 3])
        write(file, "IRFR", vec(states[endo[:R′_t], 1:20, 3]))
        write(file, "IRFi", vec(states[endo[:i′_t], 1:20, 3]))
        write(file, "IRFW", vec(states[endo[:w′_t], 1:20, 3]))
        write(file, "IRFPI", vec(states[endo[:π′_t], 1:20, 3]))
        write(file, "IRFT", vec(states[endo[:t′_t], 1:20, 3]))
        write(file, "IRFY", vec(states[endo[:y′_t], 1:20, 3]))
        write(file, "IRFPIW", vec(states[endo[:π_w′_t], 1:20, 3]))
        write(file, "IRFX", vec(states[endo[:I′_t], 1:20, 3]))
        write(file, "IRFK", vec(states[endo[:k′_t], 1:20, 3]))
        write(file, "IRFQ", vec(states[endo[:Q′_t], 1:20, 3]))
        write(file, "IRFRK", vec(states[endo[:capreturn′_t], 1:20, 3]))
        write(file, "IRFH", vec(states[endo[:L′_t], 1:20, 3]))
        write(file, "IRFMC", vec(states[endo[:mc′_t], 1:20, 3]))
        write(file, "IRFLAM", vec(states[endo[:margutil′_t], 1:20, 3]))
    end
end

if check_analytical_irfs
    m = HetDSGEGovDebt(testing_gamma = true, ref_dir = HETDSGEGOVDEBT)
    m <= Setting(:reduce_ell, false)
    steadystate!(m)

    file = jldopen("$path/reference/irfs.jld2", "r")

    IRFkf = read(file, "IRFkf")
    IRFsh = read(file, "IRFsh")
    IRFell = read(file, "IRFell")
    IRFR = read(file, "IRFR")
    IRFi = read(file, "IRFi")
    IRFW = read(file, "IRFW")
    IRFPI = read(file, "IRFPI")
    IRFT = read(file, "IRFT")
    IRFY = read(file, "IRFY")
    IRFPIW = read(file, "IRFPIW")
    IRFX = read(file, "IRFX")
    IRFK = read(file, "IRFK")
    IRFQ = read(file, "IRFQ")
    IRFRK = read(file, "IRFRK")
    IRFH = read(file, "IRFH")
    # IRFc = read(file, "IRFc")
    # IRFC = read(file, "IRFC")
    IRFMC = read(file, "IRFMC")
    IRFLAM = read(file, "IRFLAM")
    # IRFm = read(file, "IRFm")
    close(file)

    sys = compute_system(m)
    states, obs, pseudo = impulse_responses(m, sys, use_augmented_states = true)
    endo = m.endogenous_states_original

    @testset "Check Analytical IRFs" begin
        #Last entry is 3 because it's the third shock, the Z shock
        @test IRFkf ≈ states[endo[:kf′_t], 1:20, 3]
        @test IRFsh ≈ vec(states[endo[:z′_t], 1:20, 3])
        @test IRFell ≈ states[endo[:l′_t], 1:20, 3]
        @test IRFR ≈ vec(states[endo[:R′_t], 1:20, 3])
        @test IRFi ≈ vec(states[endo[:i′_t], 1:20, 3])
        @test IRFW ≈ vec(states[endo[:w′_t], 1:20, 3])
        @test IRFPI ≈ vec(states[endo[:π′_t], 1:20, 3])
        @test IRFT ≈ vec(states[endo[:t′_t], 1:20, 3])
        @test IRFY ≈ vec(states[endo[:y′_t], 1:20, 3])
        @test IRFPIW ≈ vec(states[endo[:π_w′_t], 1:20, 3])
        @test IRFX ≈ vec(states[endo[:I′_t], 1:20, 3])
        @test IRFK ≈ vec(states[endo[:k′_t], 1:20, 3])
        @test IRFQ ≈ vec(states[endo[:Q′_t], 1:20, 3])
        @test IRFRK ≈ vec(states[endo[:capreturn′_t], 1:20, 3])
        @test IRFH ≈ vec(states[endo[:L′_t], 1:20, 3])
        #@test IRFC ≈ vec(obs[m.observables[:obs_consumption], 1:20, 3])
        @test IRFMC ≈ vec(states[endo[:mc′_t], 1:20, 3])
        @test IRFLAM ≈ vec(states[endo[:margutil′_t], 1:20, 3])
        #@test IRFm ≈ vec(states[endo[:margutil′_t], 1:20, 3])
    end
end

if check_autodiff_irfs
    m = HetDSGEGovDebt(testing_gamma = true, ref_dir = HETDSGEGOVDEBT)
    m <= Setting(:reduce_ell, false)
    m <= Setting(:autodiff, true)
    steadystate!(m)

    file = jldopen("$path/reference/irfs_autodiff.jld2", "r")

    IRFkf = read(file, "IRFkf")
    IRFsh = read(file, "IRFsh")
    IRFell = read(file, "IRFell")
    IRFR = read(file, "IRFR")
    IRFi = read(file, "IRFi")
    IRFW = read(file, "IRFW")
    IRFPI = read(file, "IRFPI")
    IRFT = read(file, "IRFT")
    IRFY = read(file, "IRFY")
    IRFPIW = read(file, "IRFPIW")
    IRFX = read(file, "IRFX")
    IRFK = read(file, "IRFK")
    IRFQ = read(file, "IRFQ")
    IRFRK = read(file, "IRFRK")
    IRFH = read(file, "IRFH")
    # IRFc = read(file, "IRFc")
    # IRFC = read(file, "IRFC")
    IRFMC = read(file, "IRFMC")
    IRFLAM = read(file, "IRFLAM")
    # IRFm = read(file, "IRFm")
    close(file)

    sys = compute_system(m)
    states, obs, pseudo = impulse_responses(m, sys, use_augmented_states = true)
    endo = m.endogenous_states_original

    @testset "Check Autodiff IRFs" begin
        #Last entry is 3 because it's the third shock, the Z shock
        @test IRFkf ≈ states[endo[:kf′_t], 1:20, 3]
        @test IRFsh ≈ vec(states[endo[:z′_t], 1:20, 3])
        @test IRFell ≈ states[endo[:l′_t], 1:20, 3]
        @test IRFR ≈ vec(states[endo[:R′_t], 1:20, 3])
        @test IRFi ≈ vec(states[endo[:i′_t], 1:20, 3])
        @test IRFW ≈ vec(states[endo[:w′_t], 1:20, 3])
        @test IRFPI ≈ vec(states[endo[:π′_t], 1:20, 3])
        @test IRFT ≈ vec(states[endo[:t′_t], 1:20, 3])
        @test IRFY ≈ vec(states[endo[:y′_t], 1:20, 3])
        @test IRFPIW ≈ vec(states[endo[:π_w′_t], 1:20, 3])
        @test IRFX ≈ vec(states[endo[:I′_t], 1:20, 3])
        @test IRFK ≈ vec(states[endo[:k′_t], 1:20, 3])
        @test IRFQ ≈ vec(states[endo[:Q′_t], 1:20, 3])
        @test IRFRK ≈ vec(states[endo[:capreturn′_t], 1:20, 3])
        @test IRFH ≈ vec(states[endo[:L′_t], 1:20, 3])
        #@test IRFC ≈ vec(obs[m.observables[:obs_consumption], 1:20, 3])
        @test IRFMC ≈ vec(states[endo[:mc′_t], 1:20, 3])
        @test IRFLAM ≈ vec(states[endo[:margutil′_t], 1:20, 3])
        #@test IRFm ≈ vec(states[endo[:margutil′_t], 1:20, 3])
    end
end
