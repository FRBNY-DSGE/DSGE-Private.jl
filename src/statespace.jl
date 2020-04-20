"""
`Transition{T<:AbstractFloat}`

The transition equation of the state-space model takes the form

    `s_t = TTT*s_{t-1} + RRR*ϵ_t + CCC`

The `Transition` type stores the coefficient `Matrix{T}`s (`TTT`, `RRR`) and constant `Vector{T} CCC`.
"""
mutable struct Transition{T<:AbstractFloat}
    TTT::Matrix{T}
    RRR::Matrix{T}
    CCC::Vector{T}
end
function Transition(TTT::Matrix{T}, RRR::Matrix{T}) where T<:AbstractFloat
    CCC = zeros(eltype(TTT), size(TTT, 1))
    Transition{T}(TTT, RRR, CCC)
end
function Transition(TTT::Matrix{T}, RRR::Matrix{T}, CCC::Matrix{T}) where T<:AbstractFloat
    Transition{T}(TTT, RRR, collect(CCC))
end
function Base.getindex(eq::Transition, d::Symbol)
    if d in (:TTT, :RRR, :CCC)
        return getfield(eq, d)
    else
        throw(KeyError(d))
    end
end

"""
`Measurement{T<:AbstractFloat}`

The measurement equation of the state-space model takes the form

   `y_t = ZZ*s_t + DD + u_t`

where the error `u_t` is the measurement error, which is uncorrelated with the
shocks in the transition equation `ϵ_t`.

### Fields

If `Ns` is the number of states `s_t`, `Ny` is the number of
observables `y_t`, and `Ne` is the number of shocks `ϵ_t`:

- `ZZ`: the `Ny` x `Ns`  measurement matrix
- `DD`: the `Ny` x 1 constant vector
- `QQ`: the `Ne` x `Ne` covariance matrix for the shocks `ϵ_t`
- `EE`: the `Ny` x `Ny` covariance matrix for the measurement error `η_t`
"""
mutable struct Measurement{T<:AbstractFloat}
    ZZ::Matrix{T}
    DD::Vector{T}
    QQ::Matrix{T}
    EE::Matrix{T}
end

function Base.getindex(M::Measurement, d::Symbol)
    if d in (:ZZ, :DD, :QQ, :EE)
        return getfield(M, d)
    else
        throw(KeyError(d))
    end
end

function measurement(m::AbstractDSGEModel, trans::Transition; shocks::Bool=true)
    TTT = trans[:TTT]
    RRR = trans[:RRR]
    CCC = trans[:CCC]
    measurement(m, TTT, RRR, CCC; shocks=shocks)
end

"""
```
PseudoMeasurement{T<:AbstractFloat}
```

The pseudo-measurement equation of the state-space model takes the form

   `x_t = ZZ_pseudo*s_t + DD_pseudo`

### Fields

Let `Ns` be the number of states `s_t` and `Nx` be the number of
pseudo-observables `x_t`:

- `ZZ_pseudo`: the `Nx` x `Ns` pseudo-measurement matrix
- `DD_pseudo`: the `Nx` x 1 constant vector
"""
mutable struct PseudoMeasurement{T<:AbstractFloat}
    ZZ_pseudo::Matrix{T}
    DD_pseudo::Vector{T}
end

function Base.getindex(M::PseudoMeasurement, d::Symbol)
    if d in (:ZZ_pseudo, :DD_pseudo)
        return getfield(M, d)
    else
        throw(KeyError(d))
    end
end

"""
`System{T<:AbstractFloat}`

A mutable struct containing the transition and measurement equations for a
state-space model. The matrices may be directly indexed: `sys[:TTT]`
returns `sys.transition.TTT`, etc.
"""
mutable struct System{T<:AbstractFloat}
    transition::Transition{T}
    measurement::Measurement{T}
    pseudo_measurement::PseudoMeasurement{T}
end

function System(transition::Transition{T}, measurement::Measurement{T}) where T<:AbstractFloat
    # Initialize empty pseudo-measurement equation
    _n_states = size(transition.TTT, 1)
    _n_pseudo = 0
    ZZ_pseudo = zeros(_n_pseudo, _n_states)
    DD_pseudo = zeros(_n_pseudo)
    pseudo_measurement = PseudoMeasurement(ZZ_pseudo, DD_pseudo)

    return System(transition, measurement, pseudo_measurement)
end

function Base.getindex(system::System, d::Symbol)
    if d in (:transition, :measurement, :pseudo_measurement)
        return getfield(system, d)
    elseif d in fieldnames(typeof(system.transition))
        return getfield(system.transition, d)
    elseif d in fieldnames(typeof(system.measurement))
        return getfield(system.measurement, d)
    elseif d in fieldnames(typeof(system.pseudo_measurement))
        return getfield(system.pseudo_measurement, d)
    else
        throw(KeyError(d))
    end
end

function Base.copy(system::System)
    trans = Transition(system[:TTT], system[:RRR], system[:CCC])
    meas  = Measurement(system[:ZZ], system[:DD], system[:QQ], system[:EE])
    pseudo_meas = PseudoMeasurement(system[:ZZ_pseudo], system[:DD_pseudo])
    return System(trans, meas, pseudo_meas)
end


"""
`RegimeSwitchingSystem{N<:Int,T<:AbstractFloat}`

A mutable struct containing the transition and measurement equations for a
state-space model with regime-switching.
The matrices may be directly indexed: `sys[1][:TTT]`
returns `sys.regime[1].transition.TTT`, etc.
"""
mutable struct RegimeSwitchingSystem{T <: Real}
    transitions::Vector{Transition{T}}
    measurements::Vector{Measurement{T}}
    pseudo_measurements::Vector{PseudoMeasurement{T}}

    RegimeSwitchingSystem{T}(transitions, measurements, pseudo_measurements) where {T <: Real} = length(transitions) == length(measurements) ?
        new(transitions, measurements, pseudo_measurements) :
        error("The number of Transition (n = $(length(transitions)))" *
              " and Measurement (n = $(length(measurements))) objects must match.")
end

RegimeSwitchingSystem(transitions::Vector{Transition{T}}, measurements::Vector{Measurement{T}},
                      pseudo_measurements::Vector{PseudoMeasurement{T}}) where {T <: Real} =
                          RegimeSwitchingSystem{T}(transitions, measurements, pseudo_measurements)

function RegimeSwitchingSystem(transitions::Vector{Transition{T}},
                               measurements::Vector{Measurement{T}}) where {T<:AbstractFloat}

    # Initialize empty pseudo-measurement equation
    _n_states = size(transitions[1].TTT, 1)
    _n_pseudo = 0
    ZZ_pseudo = zeros(_n_pseudo, _n_states)
    DD_pseudo = zeros(_n_pseudo)
    pseudo_measurement = PseudoMeasurement(ZZ_pseudo, DD_pseudo)
    pseudo_measurements = [pseudo_measurement for i in 1:length(transitions)]

    return RegimeSwitchingSystem(transitions, measurements, pseudo_measurements)
end

function RegimeSwitchingSystem(systems::Vector{System{T}}) where {T <: Real}
    n_regimes           = length(systems)
    transitions         = Vector{Transition{T}}(undef, n_regimes)
    measurements        = Vector{Measurement{T}}(undef, n_regimes)
    pseudo_measurements = Vector{PseudoMeasurement{T}}(undef, n_regimes)
    for i in 1:n_regimes
        transitions[i]         = systems[i].transition
        measurements[i]        = systems[i].measurement
        pseudo_measurements[i] = systems[i].pseudo_measurement
    end
    return RegimeSwitchingSystem(transitions, measurements, pseudo_measurements)
end

# Retrieve System correspond to a regime
function System(system::RegimeSwitchingSystem, regime::Int)
    return System(system.transitions[regime],
                  system.measurements[regime],
                  system.pseudo_measurements[regime])
end

# Get fields of the RegimeSwitchingSystem
function Base.getindex(system::RegimeSwitchingSystem{T},
                       d::Symbol) where {T<:AbstractFloat}
    if d in (:transitions, :measurements, :pseudo_measurements)
        return getfield(system, d)
    elseif d == :regimes
        return 1:length(system.transitions)
    else
        throw(KeyError(d))
    end
end

# Get a specific regime
function Base.getindex(system::RegimeSwitchingSystem{T},
                       d::Int) where {T<:AbstractFloat}
    if d < 1 || d > length(system.transitions)
        throw(BoundsError(system.transitions,d))
    else
        return System(system, d)
    end
end

# Get specific matrix or type from specific regime
function Base.getindex(system::RegimeSwitchingSystem{T},
                       regime::Int, d::Symbol) where {T <: Real}
    if d == :transition
        system.transitions[regime]
    elseif d == :measurement
        system.measurements[regime]
    elseif d == :pseudo_measurement
        system.pseudo_measurements[regime]
    elseif d in (:TTT, :RRR, :CCC)
        system.transitions[regime][d]
    elseif d in (:ZZ, :DD, :QQ, :EE)
        system.measurements[regime][d]
    elseif d in (:ZZ_pseudo, :DD_pseudo)
        system.pseudo_measurements[regime][d]
    else
        throw(KeyError(d))
    end
end

function n_regimes(system::RegimeSwitchingSystem{T}) where {T<:AbstractFloat}
    return length(system.transitions)
end

function Base.copy(system::RegimeSwitchingSystem{T}) where {T<:AbstractFloat}
    n_reg = n_regimes(system)
    transitions         = Vector{Transition{T}}(undef, n_reg)
    measurements        = Vector{Measurement{T}}(undef, n_reg)
    pseudo_measurements = Vector{PseudoMeasurement{T}}(undef, n_reg)

    for i in 1:length(system[:transitions])
        trans = Transition(system[i,:TTT], system[i,:RRR], system[i,:CCC])
        meas  = Measurement(system[i,:ZZ], system[i,:DD],
                            system[i,:QQ], system[i,:EE])
        pseudo_meas = PseudoMeasurement(system[i,:ZZ_pseudo], system[i,:DD_pseudo])

        transitions[i]         = trans
        measurements[i]        = meas
        pseudo_measurements[i] = pseudo_meas
    end
    return RegimeSwitchingSystem(transitions, measurements, pseudo_measurements)
end

function Base.deepcopy(system::RegimeSwitchingSystem{T}) where {T<:AbstractFloat}
    n_reg = n_regimes(system)
    transitions         = Vector{Transition{T}}(undef, n_reg)
    measurements        = Vector{Measurement{T}}(undef, n_reg)
    pseudo_measurements = Vector{PseudoMeasurement{T}}(undef, n_reg)

    for i in 1:length(system[:transitions])
        trans = Transition(deepcopy(system[i,:TTT]), deepcopy(system[i,:RRR]), deepcopy(system[i,:CCC]))
        meas  = Measurement(deepcopy(system[i,:ZZ]), deepcopy(system[i,:DD]),
                            deepcopy(system[i,:QQ]), deepcopy(system[i,:EE]))
        pseudo_meas = PseudoMeasurement(deepcopy(system[i,:ZZ_pseudo]), deepcopy(system[i,:DD_pseudo]))

        transitions[i]         = trans
        measurements[i]        = meas
        pseudo_measurements[i] = pseudo_meas
    end
    return RegimeSwitchingSystem(transitions, measurements, pseudo_measurements)
end

"""
```
compute_system(m; apply_altpolicy = false)
```

Given the current model parameters, compute the state-space system
corresponding to model `m`. Returns a `System` or `RegimeSwitchingSystem` object.
"""
function compute_system(m::AbstractDSGEModel{T}; apply_altpolicy::Bool = false,
                        verbose::Symbol = :high) where {T <: Real}

    solution_method = get_setting(m, :solution_method)

    regime_switching = haskey(get_settings(m), :regime_switching) ?
        get_setting(m, :regime_switching) : false
    n_regimes        = regime_switching && haskey(get_settings(m), :n_regimes) ?
        get_setting(m, :n_regimes) : 1

    # Solve model
    if regime_switching
        if solution_method == :gensys
            TTTs, RRRs, CCCs = solve(m; apply_altpolicy = apply_altpolicy,
                                     regime_switching = regime_switching,
                                     regimes = 1:n_regimes, verbose = verbose)

            transition_equations = Vector{Transition{T}}(undef, n_regimes)
            for i = 1:n_regimes
                transition_equations[i] = Transition(TTTs[i], RRRs[i], CCCs[i])
            end

            # Infer which measurement and pseudo-measurement equations to use
            type_tuple = (typeof(m), Vector{Matrix{T}}, Vector{Matrix{T}}, Vector{Vector{T}})
            if hasmethod(measurement, type_tuple)
                measurement_equations = measurement(m, TTTs, RRRs, CCCs)
            else
                measurement_equation  = measurement(m, TTTs[1], RRRs[1], CCCs[1])
                measurement_equations = [measurement_equation for i in 1:n_regimes]
            end

            if hasmethod(pseudo_measurement, type_tuple)
                pseudo_measurement_equations = pseudo_measurement(m, TTTs, RRRs, CCCs)
                return RegimeSwitchingSystem(transition_equations,
                                             measurement_equations,
                                             pseudo_measurement_equations)
            else
                return RegimeSwitchingSystem(transition_equations, measurement_equations)
            end
        else
            throw("solution_method $solution_method has not been implemented.")
        end
    else
        if solution_method == :gensys
            TTT, RRR, CCC = solve(m; apply_altpolicy = apply_altpolicy, verbose = verbose)
            transition_equation = Transition(TTT, RRR, CCC)

            # Solve measurement equation
            measurement_equation = measurement(m, TTT, RRR, CCC)

        elseif solution_method == :klein
            # Unpacking the method from solve to hang on to TTT_jump
            if m.spec == "het_dsge"
                TTT_jump, TTT_state, eu = klein(m)
            else
                TTT_jump, TTT_state, eu = klein(m)
            end
            if eu==-1
                throw(KleinError())
            end

            TTT, RRR = klein_transition_matrices(m, TTT_state, TTT_jump)
            CCC = zeros(n_model_states(m))

            if m.spec == "real_bond_mkup"
                GDPeqn = construct_GDPeqn(m, TTT_jump)
                TTT, RRR, CCC = augment_states(m, TTT, TTT_jump, RRR, CCC, GDPeqn)
                # Measurement (needs the additional TTT_jump argument)
                measurement_equation = measurement(m, TTT, TTT_jump, RRR, CCC, GDPeqn)
            elseif m.spec == "het_dsge"
                TTT, RRR, CCC = augment_states(m, TTT, TTT_jump, RRR, CCC)
                measurement_equation = measurement(m, TTT, RRR, CCC)
            else
                TTT, RRR, CCC        = augment_states(m, TTT, RRR, CCC)
                measurement_equation = measurement(m, TTT, RRR, CCC)
            end

            transition_equation = Transition(TTT, RRR, CCC)
        else
            throw("solution_method $solution_method has not been implemented.")
        end
    end

    type_tuple = (typeof(m), Matrix{T}, Matrix{T}, Vector{T})
    if hasmethod(pseudo_measurement, type_tuple)
        # Solve pseudo-measurement equation
        pseudo_measurement_equation = pseudo_measurement(m, TTT, RRR, CCC)
        return System(transition_equation, measurement_equation, pseudo_measurement_equation)
    else
        return System(transition_equation, measurement_equation)
    end
end

"""
```
compute_system(m; apply_altpolicy = false,
               check_system = false, get_system = false,
               get_population_moments = false, use_intercept = false
               verbose = :high)
compute_system(m, data; apply_altpolicy = false,
               check_system = false, get_system = false,
               get_population_moments = false, use_intercept = false
               verbose = :high)
```
Given the current model parameters, compute the DSGE-VAR system
corresponding to model `m`. If a matrix `data` is also passed, then
the VAR is estimated on `data` using the DSGE `m` as a prior
with weight λ.

### Keyword Arguments
* `check_system::Bool`: see `?compute_system` that takes an input `m::AbstractDSGEModel`
    and `system::System`.
* `get_system::Bool`: see Outputs
* `get_population_moments::Bool`: see Outputs
* `use_intercept::Bool`: use an intercept term when computing the OLS estimate of the VAR system.

### Outputs
* If `get_system = true`:
    Returns the updated `system` whose measurement matrices `ZZ`, `DD`, and `QQ` correspond
    to the VAR specifieid by `m`.
* If `get_population_moments = true`:
    Returns the limit cross product matrices that describe the DSGE implied
    population moments between the observables and their lags. If `data` is
    also passed as an input, then the sample population moments are also returned.
* Otherwise:
    Returns `β` and `Σ`, the coefficients and observables covariance matrix of the VAR.
    If `data` is passed in, then `β` and `Σ` are estimated from the data using `m`
    as a prior with weight λ. Otherwise, `β` and `Σ` comprise the VAR approximation
    of the DSGE `m`.
"""
function compute_system(m::AbstractDSGEVARModel{T}; apply_altpolicy::Bool = false,
                        check_system::Bool = false, get_system::Bool = false,
                        get_population_moments::Bool = false, use_intercept::Bool = false,
                        verbose::Symbol = :high) where {T <: Real}

    regime_switching = haskey(get_settings(m), :regime_switching) ?
        get_setting(m, :regime_switching) : false
    n_regimes        = regime_switching && haskey(get_settings(m), :n_regimes) ?
        get_setting(m, :n_regimes) : 1

    dsge = get_dsge(m)
    if regime_switching
        error("Regime switching has not been implemented for a DSGEVAR yet.")
        system = compute_system(dsge; apply_altpolicy = apply_altpolicy,
                                verbose = verbose) # This `system` is really a RegimeSwitchingSystem

        systems = Vector{System{T}}(undef, n_regimes)
        for i in 1:n_regimes
            systems[i] = compute_system(dsge, System(system, i); observables = collect(keys(get_observables(m))),
                                        shocks = collect(keys(get_shocks(m))), check_system = check_system)
        end
        system = RegimeSwitchingSystem(systems) # construct a RegimeSwitchingSystem from underlying systems
    else
        system = compute_system(dsge; verbose = verbose)
        system = compute_system(dsge, system; observables = collect(keys(get_observables(m))),
                                shocks = collect(keys(get_shocks(m))), check_system = check_system)
    end

    if get_system
        return system
    elseif regime_switching
        EEs, MMs = measurement_error(m; regime_switching = regime_switching, n_regimes = n_regimes)
        out = get_population_moments ? Vector{Tuple{3, Matrix{T}}}(undef, n_regimes) :
            Vector{Tuple{2, Matrix{T}}}(undef, n_regimes)

        for i in 1:n_regimes
            out[i] = var_approx_state_space(system[i, :TTT], system[i, :RRR], system[i, :QQ],
                                          system[i, :DD], system[i, :ZZ], EEs[i], MMs[i], n_lags(m);
                                          get_population_moments = get_population_moments,
                                          use_intercept = use_intercept)
        end

        return out
    else
        EE, MM = measurement_error(m)

        return var_approx_state_space(system[:TTT], system[:RRR], system[:QQ],
                                      system[:DD], system[:ZZ], EE, MM, n_lags(m);
                                      get_population_moments = get_population_moments,
                                      use_intercept = use_intercept)
    end
end

function compute_system(m::AbstractDSGEVARModel{T}, data::Matrix{T};
                        apply_altpolicy::Bool = false,
                        check_system::Bool = false, get_system::Bool = false,
                        get_population_moments::Bool = false, use_intercept::Bool = false,
                        verbose::Symbol = :high) where {T<:Real}

    system = compute_system(m; apply_altpolicy = apply_altpolicy,
                            regime_switching = regime_switching, regime = regime,
                            n_regimes = n_regimes, check_system = check_system,
                            get_system = true, use_intercept = use_intercept,
                            verbose = verbose)

    if get_system
        return system
    else
        EE, MM = measurement_error(m)

        lags = n_lags(m)
        YYYY, XXYY, XXXX =
            compute_var_population_moments(data, lags; use_intercept = use_intercept)
        out = var_approx_state_space(system[:TTT], system[:RRR], system[:QQ],
                                     system[:DD], system[:ZZ], EE, MM, n_lags(m);
                                     get_population_moments = true,
                                     use_intercept = use_intercept)

        if get_population_moments
            return out..., YYYY, XXYY, XXXX
        else
            # Compute prior-weighted population moments
            λ = get_λ(m)
            YYYYC = YYYY + λ .* out[1]
            XXYYC = XXYY + λ .* out[2]
            XXXXC = XXXX + λ .* out[3]

            # Draw stationary VAR system
            n_periods = size(data, 2) - lags
            β, Σ =  draw_stationary_VAR(YYYYC, XXYYC, XXXXC,
                                        convert(Int, n_periods + λ * n_periods),
                                        size(data, 1), lags)

            return β, Σ
        end
    end
end

"""
```
compute_system(m::PoolModel{T})
```

Given the current model parameters, compute the state-space system
corresponding to the PoolModel model `m`.

Outputs

```
Φ: state transition function
Ψ: likelihood function, given weights on underlying models (the states) and predictive densities
F_ϵ: structural shock distribution
F_u: likelihood function measurement error distribution
F_λ: initial distribution of λ for state transition function
"""
function compute_system(m::PoolModel{T}; verbose::Symbol = :high) where T<:AbstractFloat
    Φ, F_ϵ, F_λ = transition(m)
    Ψ, F_u = measurement(m)
    return Φ, Ψ, F_ϵ, F_u, F_λ
end

"""
```
compute_system(m::AbstractDSGEModel, system::System;
        observables::Vector{Symbol}, pseudo_observables::Vector{Symbol},
        states::Vector{Symbol}, shocks::Vector{Symbol},
        zero_DD = false, zero_DD_pseudo = false)
```
computes the corresponding transition and measurement
equations specified by the keywords (e.g. `states`, `pseudo_observables`)
using the existing `ZZ`, `ZZ_pseudo`, and `TTT` matrices in `system`.

Note that this function does not update the EE matrix, which is
set to all zeros. To incorporate measurement errors, the user
must specify the EE matrix after applying compute_system.

### Keywords
* `observables`: variables that should be
    entered into the new `ZZ` matrix.
    The `observables` can be both Observables and PseudoObservables,
    but they must be an element of system already
"""
function compute_system(m::AbstractDSGEModel{S}, system::System;
                        observables::Vector{Symbol} = collect(keys(m.observables)),
                        pseudo_observables::Vector{Symbol} =
                        collect(keys(m.pseudo_observables)),
                        states::Vector{Symbol} =
                        vcat(collect(keys(m.endogenous_states)),
                             collect(keys(m.endogenous_states_augmented))),
                        shocks::Vector{Symbol} = collect(keys(m.exogenous_shocks)),
                        zero_DD::Bool = false, zero_DD_pseudo::Bool = false,
                        check_system::Bool = true) where {S<:Real}

    # Set up indices
    oid  = m.observables # observables indices dictionary
    pid  = m.pseudo_observables # pseudo observables indices dictionary
    sid  = m.endogenous_states
    said = m.endogenous_states_augmented # states augmented

    # Find shocks to keep
    shock_inds = map(k -> m.exogenous_shocks[k], shocks)
    Qout = system[:QQ][shock_inds, shock_inds]

    # Find states to keep
    if !issubset(states, vcat(collect(keys(sid)), collect(keys(said))))
        false_states = setdiff(states, vcat(collect(keys(sid)), collect(keys(said))))
        error("The following states in keyword `states` do not exist in `system`: " *
              join(string.(false_states), ", "))
    elseif !isempty(setdiff(vcat(collect(keys(sid)), collect(keys(said))), states))
        which_states = Vector{Int}(undef, length(states))
        for i = 1:length(states)
            which_states[i] = haskey(sid, states[i]) ? sid[states[i]] : said[states[i]]
        end
        Tout = system[:TTT][which_states, which_states]
        Rout = system[:RRR][which_states, shock_inds]
        Cout = system[:CCC][which_states]
    else
        which_states = 1:n_states_augmented(m)
        Tout = copy(system[:TTT])
        Rout = system[:RRR][:, shock_inds]
        Cout = copy(system[:CCC])
    end

    # Compute new ZZ and DD matrices if different observables than current system
    if !isempty(symdiff(observables, collect(keys(oid))))
        Zout = zeros(S, length(observables), size(Tout, 1))
        Dout = zeros(S, length(observables))
        for (i,obs) in enumerate(observables)
            Zout[i, :], Dout[i] = if haskey(oid, obs)
                system[:ZZ][oid[obs], which_states], zero_DD ? zero(S) : system[:DD][oid[obs]]
            elseif haskey(pid, obs)
                system[:ZZ_pseudo][pid[obs], which_states], zero_DD ? zero(S) : system[:DD_pseudo][pid[obs]]
            else
                error("Observable/PseudoObservable $obs cannot be found in the DSGE model $m")
            end
        end
    else
        Zout = copy(system[:ZZ])[:, which_states]
        Dout = zero_DD ? zeros(S, size(Zout, 1)) : system[:DD]
    end
    Eout = zeros(S, length(observables), length(observables)) # measurement errors are set to zero

    # Compute new ZZ_pseudo, DD_pseudo if different pseudo_observables than current system
    if !isempty(symdiff(pseudo_observables, collect(keys(pid))))
        Zpseudoout = zeros(S, length(pseudo_observables), size(Tout, 1))
        Dpseudoout = zeros(S, length(pseudo_observables))
        for (i, pseudoobs) in enumerate(pseudo_observables)
            Zpseudoout[i, :], Dpseudoout[i] = if haskey(oid, pseudoobs)
                system[:ZZ][oid[pseudoobs], which_states], zero_DD_pseudo ?
                    zero(S) : system[:DD][oid[pseudoobs]]
            elseif haskey(pid, pseudoobs)
                system[:ZZ_pseudo][pid[pseudoobs], which_states], zero_DD_pseudo ?
                    zero(S) : system[:DD_pseudo][pid[pseudoobs]]
            else
                error("Observable/PseudoObservable $pseudoobs cannot be found in the DSGE model $m")
            end
        end
    else
        Zpseudoout = copy(system[:ZZ_pseudo])[:, which_states]
        Dpseudoout = zero_DD_pseudo ? zeros(S, size(Zpseudoout, 1)) : copy(system[:DD_pseudo])
    end

    if check_system
        @assert size(Zout, 2) == size(Tout, 1) "Dimension 2 of new ZZ ($(size(Zout,2))) does not match dimension of states ($(size(Tout,1)))."
        @assert size(Qout, 1) == size(Rout, 2) "Dimension 2 of new RRR ($(size(Zout,2))) does not match dimension of shocks ($(size(Qout,1)))."
    end

    # Construct new system
    return System(Transition(Tout, Rout, Cout),
                  Measurement(Zout, Dout, Qout, Eout),
                  PseudoMeasurement(Zpseudoout, Dpseudoout))
end

"""
```
compute_system_function(system::System{S}) where S<:AbstractFloat
```

### Inputs

- `system::System`: The output of compute_system(m), i.e. the matrix outputs from solving a given model, m.

### Outputs

- `Φ::Function`: transition equation
- `Ψ::Function`: measurement equation
- `F_ϵ::Distributions.MvNormal`: shock distribution
- `F_u::Distributions.MvNormal`: measurement error distribution
"""
function compute_system_function(system::System{S}) where S<:AbstractFloat
    # Unpack system
    TTT    = system[:TTT]
    RRR    = system[:RRR]
    CCC    = system[:CCC]
    QQ     = system[:QQ]
    ZZ     = system[:ZZ]
    DD     = system[:DD]
    EE     = system[:EE]

    # Define transition and measurement functions
    @inline Φ(s_t1::Vector{S}, ϵ_t::Vector{S}) = TTT*s_t1 + RRR*ϵ_t + CCC
    @inline Ψ(s_t::Vector{S}) = ZZ*s_t + DD

    # Define shock and measurement error distributions
    nshocks = size(QQ, 1)
    nobs    = size(EE, 1)
    F_ϵ = Distributions.MvNormal(zeros(nshocks), QQ)
    F_u = Distributions.MvNormal(zeros(nobs),    EE)

    return Φ, Ψ, F_ϵ, F_u
end

function zero_system_constants(system::System{S}) where S<:AbstractFloat
    system = copy(system)

    system.transition.CCC = zeros(size(system[:CCC]))
    system.measurement.DD = zeros(size(system[:DD]))
    system.pseudo_measurement.DD_pseudo = zeros(size(system[:DD_pseudo]))

    return system
end

"""
```
var_approx_state_space(TTT, RRR, QQQ, DD, ZZ, EE, MM, p; get_population_moments = false) where {S<:Real}
```
computes the VAR(p) approximation of the linear state space system

```
sₜ = TTT * sₜ₋₁ + RRR * ϵₜ,
yₜ = ZZ * sₜ + DD + uₜ,
```
where the disturbances are assumed to follow
```
ϵₜ ∼ 𝒩 (0, QQ),
uₜ = ηₜ + MM * ϵₜ,
ηₜ ∼ 𝒩 (0, EE).
```
The `MM` matrix implies
```
cov(ϵₜ, uₜ) = QQ * MM'.
```

### Outputs
If `get_population_moments = false`:
* `β`: VAR(p) coefficients
* `Σ`: innovations covariance matrix for the VAR(p) representation
```
yₜ = Xₜβ + μₜ
```
where `Xₜ` appropriately stacks the `p` lags of `yₜ` and `μₜ ∼ 𝒩 (0, Σ)`.

If `get_population_moments = true`: we return the limit cross product matrices.
* `yyyyd`: 𝔼[y,y]
* `XXXXd`: 𝔼[y,X(lag rr)]
* `XXyyd`: 𝔼[X(lag rr),X(lag ll)]

Using these matrices, the VAR(p) representation is given by
```
β = XXXXd \\ XXyyd
Σ = yyyyd - XXyyd' * β
```
"""
function var_approx_state_space(TTT::AbstractMatrix{S}, RRR::AbstractMatrix{S},
                                QQ::AbstractMatrix{S}, DD::AbstractVector{S},
                                ZZ::AbstractMatrix{S}, EE::AbstractMatrix{S},
                                MM::AbstractMatrix{S}, p::Int;
                                get_population_moments::Bool = false,
                                use_intercept::Bool = false) where {S<:Real}

    nobs = size(ZZ, 1)

    HH = EE + MM * QQ * MM'
    VV = QQ * MM'

    ## Compute p autocovariances

    ## Initialize Autocovariances
    GAMM0 = zeros(S, nobs ^ 2, p + 1)
    GA0 =  solve_discrete_lyapunov(TTT, RRR * QQ * RRR')
    Gl   = ZZ * GA0 * ZZ' + ZZ * RRR * VV + (ZZ * RRR * VV)' + HH
    GAMM0[:, 1] = vec(Gl)
    TTl = copy(TTT)
    GA0ZZ = GA0 * ZZ'
    RRRVV = RRR * VV
    for l = 1:p
        Gl = ZZ * TTl * GA0ZZ + ZZ * TTl * RRRVV # ZZ * (TTl * GA0Z) * ZZ' + ZZ * (TTl * RRR * VV)
        GAMM0[:, l+1] = vec(Gl)
        TTl = TTl * TTT
    end

    ## Create limit cross product matrices
    yyyyd = zeros(S, nobs, nobs)
    if use_intercept
        XXXXd = zeros(S, 1 + p * nobs, 1 + p * nobs)
        yyXXd = zeros(S, nobs, 1 + p * nobs)

        XXXXd[1, 1] = 1.
        XXXXd[1, 2:1 + p * nobs] = kron(ones(1, p), DD')
        XXXXd[2:1 + p * nobs, 1] = kron(ones(p), DD)
        yyXXd[:, 1] = DD
    else
        XXXXd = zeros(S, p * nobs, p * nobs)
        yyXXd = zeros(S, nobs, p * nobs)
    end

    yyyyd = reshape(GAMM0[:, 1], nobs, nobs) + DD * DD'

    ## cointadd are treated as the first set of variables in XX
    ## coint    are treated as the second set of variables in XX
    ## composition: cointadd - coint - constant - lags
    shift = use_intercept ? 1 : 0 # for constructing XXXXd, to select the right indices
    for rr = 1:p
        ## E[yy,x(lag rr)]
        yyXXd[:, nobs * (rr - 1) + 1 + shift:nobs * rr + shift] =
            reshape(GAMM0[:, rr + 1], nobs, nobs) + DD * DD'

        ## E[x(lag rr),x(lag ll)]
        for ll = rr:p
            yyyydrrll = reshape(GAMM0[:, ll - rr + 1], nobs, nobs) + DD * DD';
            XXXXd[nobs * (rr - 1) + 1 + shift:nobs * rr + shift,
                  nobs * (ll - 1) + 1 + shift:nobs * ll + shift] = yyyydrrll
            XXXXd[nobs * (ll - 1) + 1 + shift:nobs * ll + shift,
                  nobs * (rr - 1) + 1 + shift:nobs * rr + shift] = yyyydrrll'
        end
    end

    XXyyd = convert(Matrix{S}, yyXXd')

    if get_population_moments
        return yyyyd, XXyyd, XXXXd
    else
        β = \(XXXXd, XXyyd)
        Σ = yyyyd - XXyyd' * β
        Σ += Σ'  # to correct for machine error
        Σ ./= 2. # and guarantee Σ is symmetric
        return β, Σ
    end
end
