using Revise
using LinearAlgebra
using OrderedCollections: OrderedDict
using DSGE
using JLD2
using ForwardDiff

#=
include("jacobian.jl")
include("helpers/jachelpers/index2.jl")    
include("helpers/jachelpers/macros2.jl")   
=#

jld2file = "data/XssYss.jld2"
@load jld2file StateSS ControlSS SS_stats grid param

m = mBBQ()

#one time setup to move prev donggyu format to bbl format
grid = Dict{Symbol, Any}(Symbol(k) => grid[k] for k in ["nb", "na", "nse", "ns", "nCOP", "oc", "numstates", "s_dist", "s", "K"])
state_id, control_id = DSGE.build_indices(grid, length(StateSS), length(ControlSS))
ss = DSGE.build_ss(param, SS_stats)

#test state and control controls of zero
State_zero = zeros(length(StateSS))
Control_zero = zeros(length(ControlSS))


F = Dict{Symbol,Any}() #TODO: need to change typing of this



reg = 1
F = DSGE.Fsys_agg(F, m, grid, StateSS, ControlSS, State_zero, Control_zero, ss, state_id, control_id, reg)

# Jacobian of aggregate block
eq_keys = collect(keys(F))
nState  = length(StateSS)
nCtrl   = length(ControlSS)

# Flatten F dict into a vector, handling scalar and vector-valued equations
function flatten_F!(F_vec, F_dict, eq_keys)
    idx = 1
    for k in eq_keys
        v = F_dict[k]
        if v isa AbstractArray
            n = length(v)
            F_vec[idx:idx+n-1] .= v
            idx += n
        else
            F_vec[idx] = v
            idx += 1
        end
    end
end

# Run once to determine flat output size
n_eqs_flat = sum(k -> (F[k] isa AbstractArray ? length(F[k]) : 1), eq_keys)
BA_agg = zeros(n_eqs_flat, nState + nCtrl)

function obj_fnct_agg(F_vec, x)
    F_dict = Dict{Symbol, Any}()
    DSGE.Fsys_agg(F_dict, m, grid, StateSS, ControlSS,
                  x[1:nState], x[nState+1:end],
                  ss, state_id, control_id, reg)
    flatten_F!(F_vec, F_dict, eq_keys)
end

ForwardDiff.jacobian!(BA_agg, obj_fnct_agg, zeros(n_eqs_flat), zeros(nState + nCtrl))

B_agg = BA_agg[:, 1:nState]        # ∂F/∂State_zero
A_agg = BA_agg[:, nState+1:end]    # ∂F/∂Control_zero
