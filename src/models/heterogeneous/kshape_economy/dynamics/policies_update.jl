"""
    _interp1_linear!(out, xgrid, yvals, xquery)

In-place 1D linear interpolation with linear extrapolation.
Writes interpolated values into `out`.
`xgrid` must be sorted and have same length as `yvals`.
"""
function _interp1_linear!(out::AbstractVector, xgrid::AbstractVector, yvals::AbstractVector, xquery::AbstractVector)
    n = length(xgrid)
    @inbounds for i in eachindex(xquery)
        xq = xquery[i]
        idx = searchsortedlast(xgrid, xq)
        if idx < 1
            idx = 1
        elseif idx >= n
            idx = n - 1
        end
        t = (xq - xgrid[idx]) / (xgrid[idx+1] - xgrid[idx])
        out[i] = yvals[idx] + t * (yvals[idx+1] - yvals[idx])
    end
    return out
end

"""
    _interp1_linear_sortedqueries!(out, xgrid, yvals, xquery)

Like _interp1_linear!, but assumes xquery is sorted increasing.
Runs in O(length(xgrid) + length(xquery)) using a moving pointer.
"""
function _interp1_linear_sortedqueries!(
    out::AbstractVector,
    xgrid::AbstractVector,
    yvals::AbstractVector,
    xquery::AbstractVector
)
    n = length(xgrid)
    j = 1  # pointer into xgrid intervals [j, j+1]

    @inbounds for i in eachindex(xquery)
        xq = xquery[i]

        # clamp interval
        if xq <= xgrid[1]
            j = 1
        elseif xq >= xgrid[end]
            j = n - 1
        else
            while j < n-1 && xgrid[j+1] < xq
                j += 1
            end
        end

        t = (xq - xgrid[j]) / (xgrid[j+1] - xgrid[j])
        out[i] = yvals[j] + t * (yvals[j+1] - yvals[j])
    end
    return out
end


"""
    policies_update(EVb, EVa, Qminus, PIminus, R_cbminus, UU, KK, inc, meshes, grid, param)

Update household policies using the Endogenous Grid Method (EGM).

This function computes optimal consumption and savings policies for households
that can adjust their portfolio (c_a, b_a, a_a) and those that cannot (c_n, b_n).

# Arguments
- `EVb`: Expected marginal value of bonds (nb × na × nse)
- `EVa`: Expected marginal value of illiquid assets (nb × na × nse)
- `Qminus`: Price of capital (scalar)
- `PIminus`: Inflation (scalar)
- `R_cbminus`: Central bank rate (scalar)
- `UU`: Utility scaling factor (scalar, typically 1)
- `KK`: Capital scaling factor (scalar, typically 1)
- `inc`: Dict with income components (labor, dividend, capital, bond, transfer)
- `meshes`: Dict with grid meshes (b, a, se)
- `grid`: Dict with grid parameters (nb, na, nse, b, a)
- `param`: Dict with model parameters (beta_aux, sigma, death_rate, Rprem, b_a_aux)

# Returns
Tuple of 5 arrays (all nb × na × nse):
- `c_a_star`: Optimal consumption when adjusting portfolio
- `b_a_star`: Optimal bond holdings when adjusting portfolio
- `a_a_star`: Optimal illiquid asset holdings when adjusting portfolio
- `c_n_star`: Optimal consumption when not adjusting portfolio
- `b_n_star`: Optimal bond holdings when not adjusting portfolio
"""
function policies_update(EVb, EVa, Qminus, PIminus, R_cbminus, UU, KK, inc, grid, param)

    # Extract grid dimensions
    nb        = get_setting(m, :nb)
    na        = get_setting(m, :na)
    nse       = get_setting(m, :nse)

    # Extract grid vectors (1D grid points)
    grid_b = grid[:b_grid].points
    grid_a = grid[:a_grid].points

    # Extract parameters
    beta_aux  = param[:β]
    sigma     = param[:σ_2]
    death_rate = param[:dr]
    Rprem     = param[:Rprem]
    b_a_aux   = param[:λ_aux]

    # Extract meshes
    meshes_b  = grid[:b_ndgrid]
    meshes_a  = grid[:a_ndgrid]
    meshes_se = grid[:se_ndgrid]

    # Extract income components
    inc_labor    = inc[:labor]
    inc_dividend = inc[:dividend]
    inc_capital  = inc[:capital]
    inc_bond     = inc[:bond]
    inc_transfer = inc[:transfer]

    ## EGM Step 1: Non-adjustment case
    EMU = beta_aux * UU * (1 - death_rate) * reshape(EVb, (nb, na, nse))

    # Linear interpolation (taking into account the financial shock)
    EMU_aux = EMU  # Simplified version as in MATLAB

    c_new = 1.0 ./ (EMU_aux .^ (1.0 / sigma))

    b_star_n = c_new .+ meshes_b .- inc_labor .- inc_dividend .- inc_transfer
    b_star_n = (b_star_n .< 0) .* b_star_n ./ ((R_cbminus + Rprem) / PIminus) .+
               (b_star_n .>= 0) .* b_star_n ./ (R_cbminus / PIminus)
    b_star_n = (1 - death_rate) * b_star_n / b_a_aux

    # Binding constraints check
    binding_constraints = meshes_b .< repeat(b_star_n[1:1, :, :], nb, 1, 1)

    Resource = inc_labor .+ inc_dividend .+ inc_bond .+ inc_transfer
    Resource_aux = reshape(Resource, (nb, na * nse))

    b_star_n_2d = reshape(b_star_n, (nb, na * nse))
    c_n_aux = reshape(c_new, (nb, na * nse))

    # Check monotonicity of b_star_n
    sign_diff = diff(sign.(diff(b_star_n_2d, dims=1)), dims=1)
    if maximum(sum(abs.(sign_diff), dims=1)) != 0
        @warn "non monotone future liquid asset choice encountered"
    end

    c_update = zeros(nb, na * nse)
    b_update = zeros(nb, na * nse)

    # Replace griddedInterpolant loop with direct linear interpolation
    @inbounds for hh in 1:(na * nse)
        xg = @view b_star_n_2d[:, hh]
        yg = @view c_n_aux[:, hh]
        c_col = @view c_update[:, hh]
        #_interp1_linear!(c_col, xg, yg, grid_b)
        _interp1_linear_sortedqueries!(c_col, xg, yg, grid_b)
        for i in 1:nb
            b_update[i, hh] = Resource_aux[i, hh] - c_update[i, hh]
        end
    end

    c_n_star = reshape(c_update, (nb, na, nse))
    b_n_star = reshape(b_update, (nb, na, nse))

    # Apply binding constraints
    c_n_star[binding_constraints] .= Resource[binding_constraints] .- grid_b[1]
    b_n_star[binding_constraints] .= minimum(grid_b)

    # Enforce upper bound on bond holdings
    b_n_star[b_n_star .> grid_b[end]] .= grid_b[end]

    ## EGM Step 2: Find optimal portfolio compositions

    term1 = beta_aux * UU * (1 - death_rate) * reshape(EVa, (nb, na, nse))

    E_return_diff = term1 ./ Qminus .- EMU

    # Check quasi-monotonicity of E_return_diff
    sign_diff_ret = diff(sign.(E_return_diff), dims=1)
    if maximum(sum(abs.(sign_diff_ret), dims=1)) > 2
        @warn "multiple roots of portfolio choice encountered"
    end

    # Find b_a* for given k' that solves the difference equation
    b_a_aux_result = Fastroot(grid_b, E_return_diff)
    b_a_aux_result = max.(b_a_aux_result, grid_b[1])
    b_a_aux_result = min.(b_a_aux_result, grid_b[end])
    b_a_aux_result = reshape(b_a_aux_result, (na, nse))  # b(a')

    ## EGM Step 3: Construct cons_list, res_list, bond_list
    EMU_2d = reshape(EMU, (nb, na * nse))

    # histc equivalent: find bin indices
    idx = zeros(Int, na, nse)
    @inbounds for j in 1:nse
        for i in 1:na
            val = b_a_aux_result[i, j]
            if val <= grid_b[1]
                idx[i, j] = 1
            elseif val >= grid_b[end]
                idx[i, j] = nb - 1
            else
                # Find bin index
                idx[i, j] = searchsortedlast(grid_b, val)
                if idx[i, j] >= nb
                    idx[i, j] = nb - 1
                end
            end
        end
    end

    step = diff(grid_b)
    s = (b_a_aux_result .- grid_b[idx]) ./ step[idx]

    aux_index = (0:(na * nse - 1)) * nb  # aux for linear indices
    aux3 = EMU_2d[idx[:] .+ aux_index[:]]  # vectorized EMU

    # Interpolate EMU
    EMU_star = aux3 .+ s[:] .* (EMU_2d[idx[:] .+ aux_index[:] .+ 1] .- aux3)
    c_a_aux = 1.0 ./ (EMU_star .^ (1.0 / sigma))  # c(k')

    cap_expenditure = inc_capital[1, :, :] / KK  # size na × nse
    auxL = inc_labor[1, :, :]  # size na × nse
    auxLT = inc_transfer[1, :, :]  # size na × nse

    # Resources that lead to capital choice k'
    # = c + b*(a') + q*a' - w*s
    Resource_step3 = c_a_aux .+ b_a_aux_result[:] .+ cap_expenditure[:] .- auxL[:] .- auxLT[:]

    c_a_aux = reshape(c_a_aux, (na, nse))
    Resource_step3 = reshape(Resource_step3, (na, nse))

    # 3.1) Money constraint is not binding, but capital constraint is binding
    b_star_zero = b_a_aux_result[1, :]  # b*(a'=0)

    # Use c_n(a'=0) from constrained problem, when b' is on grid
    aux_c = reshape(c_new[:, 1, :], (nb, nse))
    aux_inc = reshape(inc_labor[1, 1, :] .+ inc_transfer[1, 1, :], (1, nse))

    cons_list = Vector{Vector{Float64}}(undef, nse)
    res_list = Vector{Vector{Float64}}(undef, nse)
    b_list = Vector{Vector{Float64}}(undef, nse)
    a_list = Vector{Vector{Float64}}(undef, nse)

    for j in 1:nse
        # Initialize empty vectors
        cons_list[j] = Float64[]
        res_list[j] = Float64[]
        b_list[j] = Float64[]
        a_list[j] = Float64[]

        # When choosing a' = 0, HHs might still want to choose b' smaller than b*(a'=0)
        if b_star_zero[j] > grid_b[1]
            log_index = grid_b .< b_star_zero[j]
            c_a_cons = aux_c[log_index, j]
            cons_list[j] = c_a_cons
            res_list[j] = c_a_cons .+ grid_b[log_index] .- aux_inc[j]
            b_list[j] = grid_b[log_index]
            a_list[j] = zeros(sum(log_index))
        end
    end

    # Merge lists
    for j in 1:nse
        cons_list[j] = vcat(cons_list[j], c_a_aux[:, j])
        res_list[j] = vcat(res_list[j], Resource_step3[:, j])
        b_list[j] = vcat(b_list[j], b_a_aux_result[:, j])
        a_list[j] = vcat(a_list[j], grid_a)
    end

    ## EGM Step 4: Interpolate to fixed grid
    c_a_star = fill(NaN, (nb * na, nse))
    b_a_star = fill(NaN, (nb * na, nse))
    a_a_star = fill(NaN, (nb * na, nse))

    Resource_grid = reshape(inc_capital .+ inc_bond .+ inc_dividend, (nb * na, nse))
    labor_inc_grid = reshape(inc_labor .+ inc_transfer, (nb * na, nse))

    # Pre-allocate buffer for interpolation
    interp_buf = Vector{Float64}(undef, nb * na)

    for j in 1:nse
        log_index = Resource_grid[:, j] .< res_list[j][1]

        # Check monotonicity of resources
        sign_diff_res = diff(sign.(diff(res_list[j])))
        if maximum(sum(abs.(sign_diff_res))) != 0
            @warn "non monotone resource list encountered"
        end

        # Direct linear interpolation instead of griddedInterpolant
        rg = @view Resource_grid[:, j]
        _interp1_linear!(@view(c_a_star[:, j]), res_list[j], cons_list[j], rg)
        _interp1_linear!(@view(b_a_star[:, j]), res_list[j], b_list[j], rg)
        _interp1_linear!(@view(a_a_star[:, j]), res_list[j], a_list[j], rg)

        # Lowest value of res_list corresponds to b_a'=0 and a_a'=0
        # If resources are smaller than the lowest value of res_list, then HHs consume everything
        c_a_star[log_index, j] .= Resource_grid[log_index, j] .+ labor_inc_grid[log_index, j] .- grid_b[1]
        b_a_star[log_index, j] .= grid_b[1]
        a_a_star[log_index, j] .= 0.0
    end

    c_a_star = reshape(c_a_star, (nb, na, nse))
    b_a_star = reshape(b_a_star, (nb, na, nse))
    a_a_star = reshape(a_a_star, (nb, na, nse))

    # Enforce upper bounds
    b_a_star[b_a_star .> grid_b[end]] .= grid_b[end]
    a_a_star[a_a_star .> grid_a[end]] .= grid_a[end]

    return c_a_star, b_a_star, a_a_star, c_n_star, b_n_star
end
