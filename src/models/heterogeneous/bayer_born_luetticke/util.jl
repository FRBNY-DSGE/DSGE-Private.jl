## Access functions

# aggregate variable names
get_aggregate_state_variables(m::BayerBornLuetticke) = m.aggregate_state_variables
get_aggregate_jump_variables(m::BayerBornLuetticke) = m.aggregate_jump_variables
get_aggregate_equilibrium_conditions(m::BayerBornLuetticke) = m.aggregate_equilibrium_conditions
get_lagged_variables(m::BayerBornLuetticke) = m.state_variables[7:15] # 7 = Y′_tl1, 15 = τ_prog′_t1

# Idiosyncratic grid settings
@inline function get_idiosyncratic_dims(m::BayerBornLuetticke; coarse::Bool = false)
    if coarse
        (get_setting(m, :coarse_nm), get_setting(m, :coarse_nk), get_setting(m, :coarse_ny))
    else
        (get_setting(m, :nm), get_setting(m, :nk), get_setting(m, :ny))
    end
end
get_idiosyncratic_gridpts(m::BayerBornLuetticke) = (m.grids[:m_grid].points, m.grids[:k_grid].points, m.grids[:y_grid].points)
get_idiosyncratic_ndgrids(m::BayerBornLuetticke) = (m.grids[:m_ndgrid], m.grids[:k_ndgrid], m.grids[:y_ndgrid])

## Parsing functions

# Map lagged variable to steady state name
@inline function _bbl_parse_endogenous_states(var::Symbol)
    strvar = string(var)
    cutoff_i = if length(strvar) > 3 && strvar[end - 2:end] == "_t1"
        return Symbol(strvar[1:end - 2] * "l1_star")
    else
        return Symbol(strvar[1:end - 1] * "star")
    end
end
