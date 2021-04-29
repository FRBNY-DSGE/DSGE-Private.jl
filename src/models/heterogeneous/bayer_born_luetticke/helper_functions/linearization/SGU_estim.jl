@doc raw"""
    SGU_estim(XSS,A,B,m_par,n_par,indexes_aggr,distrSS;estim)

Calculate the linearized solution to the non-linear difference equations defined
by function [`Fsys`](@ref), while only differentiating with respect to the
aggregate part of the model, [`Fsys_agg()`](@ref).

The partials of the Jacobian belonging to the heterogeneous agent part of the model
are taken from the full-model derivatives provided as arguments, `A` and `B` (computed
by [`SGU()`](@ref)).

# Arguments
- `XSS`: steady state around which the system is linearized
- `A`,`B`: derivative of [`Fsys()`](@ref) with respect to arguments `X` [`B`] and
    `XPrime` [`A`]
- `m_par::ModelParameters`, `n_par::NumericalParameters`: `n_par.sol_algo` determines
    the solution algorithm
- `indexes::IndexStruct`,`indexes_aggr::IndexStructAggr`: access aggregate states and controls by name
- `distrSS::Array{Float64,3}`: steady state joint distribution

# Returns
as in [`SGU()`](@ref)
"""
function SGU_estim(XSSaggr::Array, A::Array, B::Array,
                   θ::NamedTuple, grids::OrderedDict, id::AbstractDict, nt::NamedTuple,
                   eqconds::OrderedDict{Symbol, UnitRange{Int}}, aggr_eqconds::OrderedDict{Symbol, Int},
                   aggr_endo_states::OrderedDict{Symbol, UnitRange{Int}},
                   endo_states::OrderedDict{Symbol, Int}; estim::Bool = true)

    ############################################################################
    # Calculate dericatives of non-lineear difference equation
    ############################################################################

    length_X0   = length(aggr_eqconds) # number of aggregate variables should equal number of aggregate equilibrium conditions
    BA          = ForwardDiff.jacobian(x -> Fsys_agg(x[1:length_X0], x[length_X0+1:end], θ, grids, id, nt, eqconds),
                                       zeros(2 * length_X0))
    Aa          = BA[:, length_X0+1:end] # aggregate A
    Ba          = BA[:, 1:length_X0]     # aggregate B

    @inbounds for (aggr_endo_state_name, aggr_endo_state_i) in aggr_endo_states # endo states are the columns
        for (aggr_eqcond_name, aggr_eqcond_i) in aggr_eqconds # eqconds are the rows
            # So the following line populates A in column major order
            A[first(eqconds[aggr_eqcond_name]), first(endo_states[aggr_endo_state_name])] = Aa[aggr_eqcond_i, aggr_endo_state_i]
            end
        end
    end

    ############################################################################
    # Solve the linearized model: Policy Functions and LOMs
    ############################################################################
    gx, hx, alarm_sgu, nk = SolveDiffEq(A, B, n_par, estim)

    return gx, hx, alarm_sgu, nk, A, B
end
