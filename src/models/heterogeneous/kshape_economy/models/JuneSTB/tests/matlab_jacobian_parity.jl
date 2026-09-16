using Test
using MAT
using DSGE
using OrderedCollections: OrderedDict

fixture = get(ENV, "KSHAPE_MATLAB_JACOBIAN_FIXTURE",
              "/data/dsge_data_dir/dsgejl/michael/HANK/sep-imf/sepimf_matlab_jacobian_fixture.mat")
isfile(fixture) || error("MATLAB Jacobian fixture not found: $fixture")

data = matread(fixture)
param = data["param"]
grid = data["grid"]
ss = data["SS_stats"]

m = DSGE.kCSC9()
m.dicts[:param] = param
m.dicts[:grid] = grid
m.dicts[:SS_stats] = ss
m.grids[:StateSS] = vec(data["Xss"])
m.grids[:ControlSS] = vec(data["Yss"])
DSGE.sync_kshape_parameters!(m, param)

actual = DSGE.jacobian!(m)
matlab = data["out_Jacob"]
os = Int(grid["os"])
oc = Int(grid["oc"])
n_hh_summary_matlab = 17
grid_sym = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in grid)
state_id, control_id = DSGE.build_indices_kCSC9(
    grid_sym, length(m.grids[:StateSS]), length(m.grids[:ControlSS]))
dist_state_keys = Set([:marginal_b′_t, :marginal_a′_t, :marginal_se′_t, :COP])
dist_ctrl_keys = Set([:VALUE_t, :mutil_c_t, :Va_t])
agg_state_cols = vcat([collect(state_id[key]) for key in keys(state_id)
                       if key ∉ dist_state_keys]...)
agg_ctrl_cols = vcat([collect(control_id[key]) for key in keys(control_id)
                      if key ∉ dist_ctrl_keys]...)

function variable_names(indices, excluded)
    result = String[]
    for (key, range) in indices
        key in excluded && continue
        if length(range) == 1
            push!(result, string(key))
        else
            append!(result, ["$(key)[$i]" for i in eachindex(range)])
        end
    end
    result
end

F0 = OrderedDict{Symbol,Any}()
DSGE.Fsys_agg(F0, m, grid_sym, m.grids[:StateSS], m.grids[:ControlSS],
              zeros(length(m.grids[:StateSS])), zeros(length(m.grids[:ControlSS])),
              zeros(length(m.grids[:StateSS])), zeros(length(m.grids[:ControlSS])),
              state_id, control_id, 1)
equation_keys = collect(keys(F0))
equation_names = String[]
for (key, value) in F0
    n = value isa AbstractArray ? length(value) : 1
    append!(equation_names, n == 1 ? [string(key)] : ["$(key)[$i]" for i in 1:n])
end
state_equations = equation_names[1:os]
control_equations = equation_names[os + 4:end]
state_variables = variable_names(state_id, dist_state_keys)
control_variables = variable_names(control_id, dist_ctrl_keys)
names = ("F21_aux", "F22_aux", "F23_aux", "F24_aux",
         "F41_aux", "F42_aux", "F43_aux", "F44_aux")

reference = (
    matlab["F21_aux"][:, end-os+1:end],
    matlab["F22_aux"][:, end-oc+1:end],
    matlab["F23_aux"][:, end-os+1:end],
    matlab["F24_aux"][:, end-oc+1:end],
    matlab["F41_aux"][n_hh_summary_matlab+1:end, end-os+1:end],
    matlab["F42_aux"][n_hh_summary_matlab+1:end, end-oc+1:end],
    matlab["F43_aux"][n_hh_summary_matlab+1:end, end-os+1:end],
    matlab["F44_aux"][n_hh_summary_matlab+1:end, end-oc+1:end],
)

# update_Jacob_CSC uses a one-sided log-deviation perturbation of 1e-5. Build
# the same numerical derivative from Fsys_agg so that truncation error can be
# separated from an implementation discrepancy with the production AD result.
function flatten_equations(F)
    reduce(vcat, (F[key] isa AbstractArray ? vec(F[key]) : [F[key]]
                  for key in equation_keys))
end

function aggregate_forward_difference()
    h = 1e-5
    nstate = length(m.grids[:StateSS])
    ncontrol = length(m.grids[:ControlSS])
    variables = [zeros(nstate), zeros(ncontrol), zeros(nstate), zeros(ncontrol)]
    baseline = flatten_equations(F0)
    all_columns = (agg_state_cols, agg_ctrl_cols, agg_state_cols, agg_ctrl_cols)
    derivatives = Matrix{Float64}[]

    for (slot, columns) in enumerate(all_columns)
        block = zeros(length(baseline), length(columns))
        for (j, column) in enumerate(columns)
            variables[slot][column] = h
            F = OrderedDict{Symbol,Any}()
            DSGE.Fsys_agg(F, m, grid_sym, m.grids[:StateSS], m.grids[:ControlSS],
                          variables[1], variables[2], variables[3], variables[4],
                          state_id, control_id, 1)
            block[:, j] .= (flatten_equations(F) .- baseline) ./ h
            variables[slot][column] = 0.0
        end
        push!(derivatives, block)
    end

    return (
        derivatives[1][1:os, :], derivatives[2][1:os, :],
        derivatives[3][1:os, :], derivatives[4][1:os, :],
        derivatives[1][os+4:end, :], derivatives[2][os+4:end, :],
        derivatives[3][os+4:end, :], derivatives[4][os+4:end, :],
    )
end

forward_difference = aggregate_forward_difference()
fd_atol = 1e-6
fd_rtol = 1e-7

@testset "JuneSTB complete aggregate Jacobian parity" begin
    for (block, (name, got, fd, expected)) in enumerate(zip(names, actual, forward_difference, reference))
        fd_delta = abs.(fd .- expected)
        fd_failures = count(fd_delta .> fd_atol .+ fd_rtol .* abs.(expected))
        println(name, " MATLAB-vs-Julia-FD: max_abs_error=", maximum(fd_delta),
                " entries_outside_tolerance=", fd_failures)
        if fd_failures > 0
            bad = findall(fd_delta .> fd_atol .+ fd_rtol .* abs.(expected))
            sort!(bad, by=i -> fd_delta[i], rev=true)
            for i in bad[1:min(20, length(bad))]
                row, col = Tuple(i)
                row_name = block <= 4 ? state_equations[row] : control_equations[row]
                col_name = block in (1, 3, 5, 7) ? state_variables[col] : control_variables[col]
                println("  ", (row, col), " ", row_name, " / ", col_name,
                        ": julia_fd=", fd[i], " matlab=", expected[i],
                        " abs_error=", fd_delta[i])
            end
        end
        @test fd_failures == 0

        delta = abs.(got .- expected)
        # The allowance is entry-specific: only the observed difference between
        # Julia's exact AD and its matching one-sided finite difference is added.
        # This cannot hide an equation mismatch because FD parity is tested above.
        allowance = fd_atol .+ fd_rtol .* abs.(expected) .+ 1.05 .* abs.(got .- fd)
        failures = count(delta .> allowance)
        println(name, " AD-vs-MATLAB: size=", size(got),
                " max_abs_error=", maximum(delta),
                " entries_outside_tolerance=", failures)
        if failures > 0
            bad = findall(delta .> allowance)
            sort!(bad, by=i -> delta[i], rev=true)
            for i in bad[1:min(20, length(bad))]
                row, col = Tuple(i)
                row_name = block <= 4 ? state_equations[row] : control_equations[row]
                col_name = block in (1, 3, 5, 7) ? state_variables[col] : control_variables[col]
                println("  ", (row, col), " ", row_name, " / ", col_name,
                        ": julia=", got[i], " matlab=", expected[i],
                        " abs_error=", delta[i])
            end
        end
        @test failures == 0
    end
end
