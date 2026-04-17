using Interpolations

# Assumes all inputs (inc, param, grid) are Dicts with String keys
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
    # Get grid dimensions - direct Dict access
    nb_val = Int(grid["nb"])
    na_val = Int(grid["na"])
    nse_val = Int(grid["nse"])
    grid_b = vec(grid["b"])
    
    # Initialize output arrays
    c_a_new = fill(NaN, nb_val * na_val, nse_val)
    b_a_new = fill(NaN, nb_val * na_val, nse_val)
    a_a_new = fill(NaN, nb_val * na_val, nse_val)
    
    # Get income components - direct Dict access
    inc_capital = inc["capital"]
    inc_bond = inc["bond"]
    inc_dividend = inc["dividend"]
    inc_labor = inc["labor"]
    inc_transfer = inc["transfer"]
    
    # Resource_grid = reshape(inc.capital + inc.bond + inc.dividend, [grid.nb*grid.na, grid.nse]);
    Resource_grid = reshape(inc_capital .+ inc_bond .+ inc_dividend, nb_val * na_val, nse_val)
    
    # labor_inc_grid = reshape(inc.labor + inc.transfer, [grid.nb*grid.na, grid.nse]);
    labor_inc_grid = reshape(inc_labor .+ inc_transfer, nb_val * na_val, nse_val)
    
    for j = 1:nse_val
        # Ensure lists are 1D vectors (MATLAB cell arrays might be 2D when loaded)
        res_list_j  = vec(res_list[j])
        cons_list_j = vec(cons_list[j])
        b_list_j    = vec(b_list[j])
        a_list_j    = vec(a_list[j])

        # MATLAB's griddedInterpolant sorts knots automatically; Julia requires sorted unique knots
        perm = sortperm(res_list_j)
        res_list_j  = res_list_j[perm]
        cons_list_j = cons_list_j[perm]
        b_list_j    = b_list_j[perm]
        a_list_j    = a_list_j[perm]
        # deduplicate
        uniq = [true; diff(res_list_j) .> 0]
        res_list_j  = res_list_j[uniq]
        cons_list_j = cons_list_j[uniq]
        b_list_j    = b_list_j[uniq]
        a_list_j    = a_list_j[uniq]
        
        # log_index = Resource_grid(:,j)<res_list{j}(1);
        log_index = Resource_grid[:, j] .< res_list_j[1]
        
        # when at most one constraint binds:
        # cons = griddedInterpolant(res_list{j},cons_list{j});
        # c_a_new(:,j) = cons(Resource_grid(:,j));
        itp_cons = interpolate((res_list_j,), cons_list_j, Gridded(Linear()))
        itp_cons_ext = extrapolate(itp_cons, Line())
        c_a_new[:, j] = itp_cons_ext.(Resource_grid[:, j])
        
        # bond = griddedInterpolant(res_list{j},b_list{j});
        # b_a_new(:,j) = bond(Resource_grid(:,j));
        itp_bond = interpolate((res_list_j,), b_list_j, Gridded(Linear()))
        itp_bond_ext = extrapolate(itp_bond, Line())
        b_a_new[:, j] = itp_bond_ext.(Resource_grid[:, j])
        
        # equity = griddedInterpolant(res_list{j},a_list{j});
        # a_a_new(:,j) = equity(Resource_grid(:,j));
        itp_equity = interpolate((res_list_j,), a_list_j, Gridded(Linear()))
        itp_equity_ext = extrapolate(itp_equity, Line())
        a_a_new[:, j] = itp_equity_ext.(Resource_grid[:, j])
        
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

