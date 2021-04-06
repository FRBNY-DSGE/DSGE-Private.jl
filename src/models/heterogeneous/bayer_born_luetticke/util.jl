# Constructor functions for grid
function _construct_income_grid_bbl(ymin::T, ymax::T, ny::Int64) where {T <: Real}
    return collect(range(ymin, stop = ymax, length = ny))
end
function _construct_illiquid_asset_grid_bbl(kmin::T, kmax::T, nk::Int64) where {T <: Real}
    return exp.(range(log(kmin + 1.), stop = log(kmax + 1.), length = nk)) .- 1.
end
function _construct_liquid_asset_grid_bbl(mmin::T, mmax::T, nm::Int64) where {T <: Real}
    return exp.(range(0., stop = log(mmax - mmin + 1.), length = nm)) .+ (mmin - 1.)
end

# Access functions
get_aggregate_state_variables(m::BayerBornLuetticke) = m.state_variables[5:end] # first 4 variables are marginals and DCT of distribution
get_aggregate_jump_variables(m::BayerBornLuetticke) = m.jump_variables[3:end] # first 2 variables are DCT of value functions Vm_t and Vk_t
