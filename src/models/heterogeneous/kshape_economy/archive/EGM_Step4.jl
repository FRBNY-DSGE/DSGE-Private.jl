using Interpolations
include("helpers/get_field.jl")

"""
    EGM_Step4(cons_list, res_list, b_list, a_list, inc, param, grid)

Port of EGM_Step4 from MATLAB to Julia.

Obtains consumption, money, and capital policy under adjustment.
The function uses the {(cons_list{j},res_list{j})} as measurement
points. The consumption function in (b,a) can be obtained from
interpolation by using the total resources available at (b,a): R(b,a)=qa+b/pi.

# Arguments
- `cons_list`: Optimal consumption policy (Vector of Vectors, length nse)
- `res_list`: Required resources for cons_list (Vector of Vectors, length nse)
- `b_list`: Bond investment for cons_list (Vector of Vectors, length nse)
- `a_list`: Equity choice for cons_list (Vector of Vectors, length nse)
- `inc`: Income structure containing labor, dividend, capital, bond, transfer
- `param`: Parameter structure containing model parameters
- `grid`: Grid structure containing grid points and dimensions

# Returns
Tuple of 3 outputs:
- `c_a_new`: Update for consumption policy under adjustment (nb × na × nse)
- `b_a_new`: Update for money policy under adjustment (nb × na × nse)
- `a_a_new`: Update for capital policy under adjustment (nb × na × nse)

# Notes
- This function is part of the EGM (Endogenous Grid Method) algorithm
- Uses interpolation to map from endogenous grid to exogenous grid
- Handles binding constraints when resources are below minimum threshold
"""

function EGM_Step4(cons_list, res_list, b_list, a_list, inc, param, grid)
    # Get grid dimensions
    nb_val = Int(get_field(grid, :nb))
    na_val = Int(get_field(grid, :na))
    nse_val = Int(get_field(grid, :nse))
    grid_b = vec(get_field(grid, :b))
    
    # Initialize output arrays
    c_a_new = fill(NaN, nb_val * na_val, nse_val)
    b_a_new = fill(NaN, nb_val * na_val, nse_val)
    a_a_new = fill(NaN, nb_val * na_val, nse_val)
    
    # Get income components
    inc_capital = get_field(inc, :capital)
    inc_bond = get_field(inc, :bond)
    inc_dividend = get_field(inc, :dividend)
    inc_labor = get_field(inc, :labor)
    inc_transfer = get_field(inc, :transfer)
    
    # Resource_grid = reshape(inc.capital + inc.bond + inc.dividend, [grid.nb*grid.na, grid.nse]);
    Resource_grid = reshape(inc_capital .+ inc_bond .+ inc_dividend, nb_val * na_val, nse_val)
    
    # labor_inc_grid = reshape(inc.labor + inc.transfer, [grid.nb*grid.na, grid.nse]);
    labor_inc_grid = reshape(inc_labor .+ inc_transfer, nb_val * na_val, nse_val)
    
    for j = 1:nse_val
        # Ensure lists are 1D vectors (MATLAB cell arrays might be 2D when loaded)
        res_list_j = vec(res_list[j])
        cons_list_j = vec(cons_list[j])
        b_list_j = vec(b_list[j])
        a_list_j = vec(a_list[j])
        
        # log_index = Resource_grid(:,j)<res_list{j}(1);
        log_index = Resource_grid[:, j] .< res_list_j[1]
        
        # Create interpolation objects
        # MATLAB's griddedInterpolant allows extrapolation, so we need to handle out-of-bounds manually
        itp_cons = interpolate((res_list_j,), cons_list_j, Gridded(Linear()))
        itp_bond = interpolate((res_list_j,), b_list_j, Gridded(Linear()))
        itp_equity = interpolate((res_list_j,), a_list_j, Gridded(Linear()))
        
        # Interpolate for all points, handling out-of-bounds with linear extrapolation
        res_min = res_list_j[1]
        res_max = res_list_j[end]
        
        for i in 1:(nb_val * na_val)
            res_val = Resource_grid[i, j]
            
            if res_val < res_min
                # Below minimum: use constraint binding logic (will be overwritten below)
                # Use linear extrapolation: extend from first two points
                if length(res_list_j) >= 2
                    slope_cons = (cons_list_j[2] - cons_list_j[1]) / (res_list_j[2] - res_list_j[1])
                    slope_bond = (b_list_j[2] - b_list_j[1]) / (res_list_j[2] - res_list_j[1])
                    slope_equity = (a_list_j[2] - a_list_j[1]) / (res_list_j[2] - res_list_j[1])
                    c_a_new[i, j] = cons_list_j[1] + slope_cons * (res_val - res_min)
                    b_a_new[i, j] = b_list_j[1] + slope_bond * (res_val - res_min)
                    a_a_new[i, j] = a_list_j[1] + slope_equity * (res_val - res_min)
                else
                    c_a_new[i, j] = cons_list_j[1]
                    b_a_new[i, j] = b_list_j[1]
                    a_a_new[i, j] = a_list_j[1]
                end
            elseif res_val > res_max
                # Above maximum: use linear extrapolation from last two points
                if length(res_list_j) >= 2
                    slope_cons = (cons_list_j[end] - cons_list_j[end-1]) / (res_list_j[end] - res_list_j[end-1])
                    slope_bond = (b_list_j[end] - b_list_j[end-1]) / (res_list_j[end] - res_list_j[end-1])
                    slope_equity = (a_list_j[end] - a_list_j[end-1]) / (res_list_j[end] - res_list_j[end-1])
                    c_a_new[i, j] = cons_list_j[end] + slope_cons * (res_val - res_max)
                    b_a_new[i, j] = b_list_j[end] + slope_bond * (res_val - res_max)
                    a_a_new[i, j] = a_list_j[end] + slope_equity * (res_val - res_max)
                else
                    c_a_new[i, j] = cons_list_j[end]
                    b_a_new[i, j] = b_list_j[end]
                    a_a_new[i, j] = a_list_j[end]
                end
            else
                # Within bounds: interpolate normally
                c_a_new[i, j] = itp_cons(res_val)
                b_a_new[i, j] = itp_bond(res_val)
                a_a_new[i, j] = itp_equity(res_val)
            end
        end
        
        # Lowest value of res_list corresponds to b_a'=b(1) and a_a'=0.
        # Any resources on grid smaller than res_list imply that HHs consume all
        # resources plus income.
        # When both constraints are binding:
        # c_a_new(log_index,j) = Resource_grid(log_index,j) + labor_inc_grid(log_index,j)-grid.b(1);
        # b_a_new(log_index,j) = grid.b(1);
        # a_a_new(log_index,j) = 0;
        c_a_new[log_index, j] = Resource_grid[log_index, j] .+ labor_inc_grid[log_index, j] .- grid_b[1]
        b_a_new[log_index, j] .= grid_b[1]
        a_a_new[log_index, j] .= 0.0
    end
    
    # c_a_new = reshape(c_a_new,[grid.nb, grid.na, grid.nse]);
    # b_a_new = reshape(b_a_new,[grid.nb, grid.na, grid.nse]);
    # a_a_new = reshape(a_a_new,[grid.nb, grid.na, grid.nse]);
    c_a_new = reshape(c_a_new, nb_val, na_val, nse_val)
    b_a_new = reshape(b_a_new, nb_val, na_val, nse_val)
    a_a_new = reshape(a_a_new, nb_val, na_val, nse_val)
    
    return (c_a_new, b_a_new, a_a_new)
end

