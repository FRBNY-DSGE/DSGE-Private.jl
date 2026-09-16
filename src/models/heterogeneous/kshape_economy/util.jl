"""
Some functions have not been updated or adapted to fit with 
mBBQ and are still only compatible with BayerBornLuetticke.
Note, if the type has not been updated, then the method does
not yet exist for mBBQ
"""
# Access functions

# aggregate variable names
get_aggregate_state_variables(m::mBBQ) = m.aggregate_state_variables
get_aggregate_jump_variables(m::mBBQ) = m.aggregate_jump_variables
get_aggregate_endogenous_states(m::mBBQ) = m.aggregate_endogenous_states
get_aggregate_equilibrium_conditions(m::mBBQ) = m.aggregate_equilibrium_conditions
#get_lagged_variables(m::BayerBornLuetticke) = m.state_variables[7:15] # 7 = Y′_tl1, 15 = τ_prog′_t1

#TO-DO: Add types
@inline util(c) = (c .^ (1-θ[:σ_2])) ./ (1-θ[:σ_2])
@inline mutil(c) = 1.0 ./ (c .^ θ[:σ_2])
@inline invutil(u) = ((1 - θ[:σ_2]) .* u) .^ (1 / (1 - θ[:σ_2]))
@inline invmutil(μ) = (1.0 ./ μ) .^ (1 / θ[:σ_2])

#Copula Indices
@inline function get_copula_indices(m::mBBQ)
    get_setting(m, :nb_copula), get_setting(m, :na_copula), get_setting(m, :nse_copula)
end

# Idiosyncratic grid settings
@inline function get_idiosyncratic_dims(m::mBBQ; coarse::Bool = false)
    if coarse
        (get_setting(m, :coarse_nb), get_setting(m, :coarse_na), get_setting(m, :coarse_nse))
    else
        (get_setting(m, :nb), get_setting(m, :na), get_setting(m, :nse))
    end 
end
get_idiosyncratic_gridpts(m::mBBQ) = (m.grids[:b_grid].points, m.grids[:a_grid].points, m.grids[:se_grid].points)
get_idiosyncratic_ndgrids(m::BayerBornLuetticke) = (m.grids[:b_ndgrid], m.grids[:a_ndgrid], m.grids[:se_ndgrid])


# Map lagged variable to steady state name
@inline function _mbbq_parse_endogenous_states(var::Symbol)
    strvar = string(var)
    cutoff_i = if length(strvar) > 3 && strvar[end - 2:end] == "_t1"
        return Symbol(strvar[1:end - 2] * "l1_star")
    else
        return Symbol(strvar[1:end - 1] * "star")
    end 
end

## Saving and loading output from steady state and Jacobian
@inline function save_steadystate(m::BayerBornLuetticke, KSS::T, VmSS::AbstractArray{T, 3}, 
                                  VkSS::AbstractArray{T, 3}, distrSS::AbstractArray{T, 3}) where {T <: Real}
    fp = get_setting(m, :steadystate_output_file)
    if !ispath(dirname(fp))
        mkpath(dirname(fp))
    end
    JLD2.jldopen(fp, true, true, true, IOStream) do file
        write(file, "KSS", KSS)
        write(file, "VmSS", VmSS)
        write(file, "VkSS", VkSS)
        write(file, "distrSS", distrSS)
    end 
    nothing
end
@inline function save_jacobian(m::BayerBornLuetticke{T}) where {T <: Real}
    fp = get_setting(m, :jacobian_output_file)
    if !ispath(dirname(fp))
        mkpath(dirname(fp))
    end
    JLD2.jldopen(fp, true, true, true, IOStream) do file
        write(file, "A", m[:A].value::Matrix{T})
        write(file, "B", m[:B].value::Matrix{T})
    end
    nothing
end

@inline function load_steadystate!(m::mBBQ)
    out = JLD2.jldopen(get_setting(m, :steadystate_output_file), "r")
    populate_from_jld2!(m, out)
    m <= Setting(:compute_full_steadystate, false) # set to false to avoid re-computing full steadystate
    m
end

@inline function load_jacobian!(m::BayerBornLuetticke)
    # loading in a pre-computed Jacobian, so it is assumed
    # we don't need to linearize the heterogeneous block
    m <= Setting(:linearize_heterogeneous_block, false)
    out = JLD2.jldopen(get_setting(m, :jacobian_output_file), "r")
    m[:A] = out["A"]
    m[:B] = out["B"]
    m
end

@inline function compute_and_save_irfs(m::BayerBornLuetticke,T,fp)
    system_main = compute_system(m)
    θ = parameters2namedtuple(m)
    shocks = collect(keys(m.exogenous_shocks))
    shock2deviation_dict = get_setting(m,:shock_to_deviation_dict)
    sd_shocks = zeros(length(shocks))
    ct = 1
    for  i in shocks
       sd_shocks[ct] = θ[shock2deviation_dict[i]]
       ct+=1
    end
    states, pseudo, obs = impulse_responses(m,system_main, T, shocks,sd_shocks)
    JLD2.jldopen(fp,true, true,true,IOStream) do file
      write(file,"states",states)
      write(file,"obs",obs)
      write(file,"pseudo",pseudo)
    end
    nothing
end

