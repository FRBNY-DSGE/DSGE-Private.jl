using Test
using MAT
using DSGE

fixture = get(ENV, "KSHAPE_MATLAB_FIXTURE",
              "/data/dsge_data_dir/dsgejl/michael/HANK/sep-imf/matlab_kshape_fixture.mat")
isfile(fixture) || error("MATLAB fixture not found: $fixture")

file = matopen(fixture)
readvar(name) = read(file, name)

grid = readvar("grid")
grid["compressionIndexesCOP"] = Int.(vec(grid["compressionIndexesCOP"]))
for name in ("b", "a", "se")
    grid[name] = vec(grid[name])
end

dcd = Matrix{Float64}.(vec(readvar("DCD")))
idcd = Matrix{Float64}.(vec(readvar("IDCD")))
common = (vec(readvar("Xss")), vec(readvar("Yss")),
          readvar("Gamma_state"), readvar("Gamma_control"), readvar("InvGamma"),
          readvar("param"), grid, readvar("SS_stats"), dcd, idcd)

function evaluate(state, state_minus, control_next, control)
    DSGE.F_sys_CSC_ref_tvcopula(vec(readvar(state)), vec(readvar(state_minus)),
                                vec(readvar(control_next)), vec(readvar(control)),
                                common...)
end

@testset "JuneSTB MATLAB F_sys parity" begin
    base = evaluate("State", "Stateminus", "Controlnext_sparse", "Control_sparse")
    perturbed = evaluate("State2", "Stateminus", "Control2", "Control_sparse")

    @test base[1] ≈ vec(readvar("Difference")) atol=1e-9 rtol=1e-11
    @test base[2] ≈ vec(readvar("LHS")) atol=1e-9 rtol=1e-11
    @test base[3] ≈ vec(readvar("RHS")) atol=1e-9 rtol=1e-11

    for (value, name) in zip(base[5:9],
                             ("c_a_star", "b_a_star", "a_a_star", "c_n_star", "b_n_star"))
        @test value ≈ readvar(name) atol=1e-9 rtol=1e-11
    end

    @test perturbed[1] ≈ vec(readvar("Difference2")) atol=1e-9 rtol=1e-11
    @test perturbed[2] ≈ vec(readvar("LHS2")) atol=1e-9 rtol=1e-11
    @test perturbed[3] ≈ vec(readvar("RHS2")) atol=1e-9 rtol=1e-11
end

close(file)
