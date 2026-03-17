using LinearAlgebra
using OrderedCollections: OrderedDict
using DSGE
using JLD2
using Revise

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
state_id, control_id = build_indices(grid, length(StateSS), length(ControlSS))
ss = build_ss(param, SS_stats)

#test state and control controls of zero
State_zero = zeros(length(StateSS))
Control_zero = zeros(length(ControlSS))


F = Dict{Symbol,Any}() #TODO: need to change typing of this




F = Fsys_agg(F, State_zero, Control_zero, ss, state_id, control_id, reg = 1)


#F = Fsys_agg(F, X, Xprime, ss, state_id, control_id, reg = 1)
#_jacobian!(m)
