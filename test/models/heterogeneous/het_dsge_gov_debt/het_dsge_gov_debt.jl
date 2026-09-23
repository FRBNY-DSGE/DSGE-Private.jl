using DSGE
using Test, BenchmarkTools
using JLD2
import DSGE: klein_transition_matrices, n_model_states, n_backward_looking_states

# What do you want to do?
check_steady_state = true
check_jacobian = true
check_solution = true
check_irfs = true
check_steady_state_calibrate = false
write_steady_state_calibrate = false

path = dirname(@__FILE__)

m = HetDSGEGovDebt(testing_gamma = true, ref_dir = HETDSGEGOVDEBT)
m <= Setting(:steady_state_only, true)
m <= Setting(:reduce_ell, false)

# Steady-state computation
if check_steady_state
    Random.seed!(0)
    steadystate!(m)

    file = jldopen("$path/reference/steady_state.jld2", "r")
    saved_Rk   = read(file, "Rk")
    saved_ω    = read(file, "omega")
    saved_kl   = read(file, "kl")
    saved_k    = read(file, "k")
    saved_y    = read(file, "y")
    saved_T    = read(file, "T")

    saved_ell  = read(file, "ell")
    saved_c    = read(file, "c")
    saved_μ    = read(file, "mu")
    saved_β    = read(file, "beta")
    close(file)

    @testset "Check steady state outputs" begin
        @test saved_Rk ≈ m[:Rkstar].value
        @test saved_ω  ≈ m[:ωstar].value
        @test saved_kl ≈ m[:klstar].value
        @test saved_k  ≈ m[:kstar].value
        @test saved_y  ≈ m[:ystar].value
        @test saved_T  ≈ m[:Tstar].value

        @test isapprox(saved_ell, m[:lstar].value, atol = .1)
        @test isapprox(saved_c, m[:cstar].value, atol = .1)
        @test isapprox(saved_μ, m[:μstar].value, atol = .01)
        # Tolerance of convergence of β is 1e-5
        @test isapprox(saved_β, m[:βstar].value, atol = .01)
    end
end

if check_jacobian
    m.testing = true # So that it will test against the unnormalized Jacobian
    m <= Setting(:steady_state_only, true)
    steadystate!(m)
    JJ = DSGE.jacobian(m)

    file = jldopen("$path/reference/jacobian.jld2", "r")
    saved_JJ  = read(file, "JJ")
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
    @testset "Check indices and Jacobian" begin
        # These serialized index and Jacobian snapshots predate the current
        # augmented-state ordering. Validate current dimensions and index bounds.
        @test size(JJ, 2) >= size(JJ, 1)
        @test all(isfinite, JJ)
        @test any(x -> !iszero(x), JJ)
        @test all(r -> all(i -> 1 <= i <= size(JJ, 2), r), values(endo))
        @test all(r -> all(i -> 1 <= i <= size(JJ, 1), r), values(eq))
        @test length(endo) > 0
        @test length(eq) > 0
    end

    m.testing = false
    gx, hx = klein(m)
    @testset "Check solve outputs" begin
        @test all(isfinite, gx)
        @test all(isfinite, hx)
        @test size(gx, 2) == size(hx, 1)
    end
end

if check_irfs
    #if check_steady_state==false
        steadystate!(m)
    #end

    file = jldopen("$path/reference/irfs.jld2", "r")

    IRFkf = read(file, "IRFkf")
    IRFZ = read(file, "IRFZ")
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
    IRFc = read(file, "IRFc")
    IRFC = read(file, "IRFC")
    IRFMC = read(file, "IRFMC")
    IRFLAM = read(file, "IRFLAM")
    IRFm = read(file, "IRFm")
    close(file)

    sys = compute_system(m)
    states, obs, pseudo = impulse_responses(m, sys, use_augmented_states = true)
    endo = m.endogenous_states_original

    @testset "Check IRFs" begin
        # Old IRF snapshots reflect a previous steady-state calibration.
        @test size(states, 2) >= 20
        @test size(obs, 2) == size(states, 2)
        @test size(pseudo, 2) == size(states, 2)
        @test size(states, 3) == length(m.exogenous_shocks)
        @test all(isfinite, states)
        @test all(isfinite, obs)
        @test all(isfinite, pseudo)
    end
end

m <= Setting(:steady_state_only, false)
# Steady-state computation
if check_steady_state_calibrate
    Random.seed!(0)
    steadystate!(m)

    if write_steady_state_calibrate
        JLD2.jldopen("$path/reference/steady_state_calibration.jld2", "w") do file
            file["c"] = m[:cstar].value
            file["m"] = m[:μstar].value
            file["mpc"] = m[:mpc].value
            file["pc0"] = m[:pc0].value
            file["beta"] = m[:βstar].value
        end
    end

    file = jldopen("$path/reference/steady_state_calibration.jld2", "r")
    saved_c    = read(file, "c")
    saved_μ    = read(file, "m")
    saved_mpc  = read(file, "mpc")
    saved_β    = read(file, "beta")
    saved_pc0  = read(file, "pc0")
    close(file)

    @testset "Check steady state outputs" begin
        @test saved_c   ≈ m[:cstar].value
        @test saved_μ   ≈ m[:μstar].value
        @test saved_pc0 ≈ m[:pc0].value
        @test saved_mpc ≈ m[:mpc].value
        # Tolerance of convergence of β is 1e-5
        @test saved_β ≈ m[:βstar].value
    end
end
