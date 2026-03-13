using LinearAlgebra
# Assumes all inputs (grid, inc, param) are Dicts with String keys
"""
    EGM_Step3(EMU, grid, inc, b_a_star, c_n_aux, param)

Port of EGM_Step3 from MATLAB to Julia.

Returns the resources (res_list), consumption (cons_list) and bond investment (b_list) 
for given equity choice (a_list). For a' = 0, there doesn't need to be a unique 
corresponding b*. This function gets a list of consumption choices and resources for 
bond choices b'<b* (b_list) and equity choice a'=0 (a_list) and merges them with 
consumption choices and resources that are obtained if equity constraint does not bind next period.

# Arguments
- `EMU`: Expected marginal utility (3D array: nb × na × nse)
- `grid`: Grid structure containing grid points and dimensions
- `inc`: Income structure containing labor, dividend, capital, bond, transfer
- `b_a_star`: Optimal bond holdings for each equity choice (na × nse)
- `c_n_aux`: Consumption in t as a function of t+1 grid (constrained version) (nb × na × nse)
- `param`: Parameter structure containing model parameters

# Returns
Tuple of 6 outputs:
- `c_star`: Optimal consumption policy as function of a', s (both constraints do not bind) (na × nse)
- `Resource`: Required resource for c_star (na × nse)
- `cons_list`: Optimal consumption policy if a) only a'>=0 binds and b) both constraints do not bind (Vector of Vectors, length nse)
- `res_list`: Required resources for cons_list (Vector of Vectors, length nse)
- `b_list`: Bond investment for cons_list (Vector of Vectors, length nse)
- `a_list`: Equity choice for cons_list (Vector of Vectors, length nse)

# Notes
- This function is part of the EGM (Endogenous Grid Method) algorithm
- Handles both binding and non-binding constraints
"""


function EGM_Step3(EMU, grid, inc, b_a_star, c_n_aux, param)
    # 1) Constraints for bond and equity are not binding
    # EMU = reshape(EMU,[grid.nb, grid.na*grid.nse]);
    nb_val = Int(grid["nb"])
    na_val = Int(grid["na"])
    nse_val = Int(grid["nse"])
    
    EMU = reshape(EMU, nb_val, na_val * nse_val)
    
    # Interpolation of psi-function at b*(b,a)
    # [~,idx] = histc(b_a_star,grid.b);
    # Ensure grid.b is a 1D vector (MATLAB might return row vector)
    grid_b = vec(grid["b"])
    _, idx = genweight(b_a_star, grid_b)
    
    # idx(b_a_star<=grid.b(1)) = 1;
    idx[b_a_star .<= grid_b[1]] .= 1
    
    # idx(b_a_star>=grid.b(end)) = grid.nb-1;
    idx[b_a_star .>= grid_b[end]] .= nb_val - 1
    
    # step = diff(grid.b);
    step = diff(grid_b)
    
    # s = (b_a_star-grid.b(idx))./step(idx);
    s = (b_a_star .- grid_b[idx]) ./ step[idx]
    
    # aux_index = (0:(grid.na*grid.nse)-1)*grid.nb;
    aux_index = (0:(na_val * nse_val) - 1) .* nb_val
    
    # aux3 = EMU(idx(:)+aux_index(:));
    # MATLAB linear indexing: EMU(idx(:)+aux_index(:))
    # In Julia 2D: EMU[idx[i], i] where i is the column index
    idx_flat = vec(idx)
    aux3 = [EMU[idx_flat[i], i] for i in 1:length(idx_flat)]
    
    # Interpolate EMU over b*, b-dim is dropped
    # EMU_star = aux3 + s(:).*(EMU(idx(:)+aux_index(:)+1)-aux3);
    aux3_next = [EMU[min(idx_flat[i] + 1, nb_val), i] for i in 1:length(idx_flat)]
    EMU_star = aux3 .+ vec(s) .* (aux3_next .- aux3)
    
    # c_star = 1./(EMU_star.^(1/param.sigma));
    sigma_val = param["sigma"]
    c_star = 1.0 ./ (EMU_star.^(1.0 / sigma_val))
    
    # a_investment = shiftdim(inc.capital(1,:,:));
    # shiftdim removes singleton dimensions - in Julia, indexing [1, :, :] already gives 2D
    inc_capital = inc["capital"]
    a_investment = inc_capital[1, :, :]  # Already 2D, no need for dropdims
    
    # auxL = shiftdim(inc.labor(1,:,:));
    inc_labor = inc["labor"]
    auxL = inc_labor[1, :, :]  # Already 2D, no need for dropdims
    
    # auxLT = shiftdim(inc.transfer(1,:,:));
    inc_transfer = inc["transfer"]
    auxLT = inc_transfer[1, :, :]  # Already 2D, no need for dropdims
    
    # Resources that lead to equity choice a'
    # Resource = c_star + b_a_star(:) + a_investment(:) - auxL(:) - auxLT(:);
    Resource = c_star .+ vec(b_a_star) .+ vec(a_investment) .- vec(auxL) .- vec(auxLT)
    
    # c_star = reshape(c_star, [grid.na grid.nse]);
    c_star = reshape(c_star, na_val, nse_val)
    Resource = reshape(Resource, na_val, nse_val)
    
    # 2) Bond constraint is not binding, but equity constraint is binding
    # b_star_zero = shiftdim(b_a_star(1,:));
    # b_a_star is (na, nse), so b_a_star[1, :] is already 1D
    b_star_zero = vec(b_a_star[1, :])  # Ensure 1D vector
    
    # Use consumption at a'=0 from constrained problem, when b' is on grid
    # aux_c = reshape(c_n_aux(:,1,:),[grid.nb grid.nse]);
    aux_c = reshape(c_n_aux[:, 1, :], nb_val, nse_val)
    
    # aux_inc = reshape(inc.labor(1,1,:)+inc.transfer(1,1,:),[1 grid.nse]);
    aux_inc = reshape(inc_labor[1, 1, :] + inc_transfer[1, 1, :], 1, nse_val)
    
    # cons_list = cell(grid.nse,1);
    cons_list = Vector{Vector{Float64}}(undef, nse_val)
    res_list = Vector{Vector{Float64}}(undef, nse_val)
    b_list = Vector{Vector{Float64}}(undef, nse_val)
    a_list = Vector{Vector{Float64}}(undef, nse_val)
    
    for j = 1:nse_val
        # When choosing zero equity, HHs might still want to choose bond 
        # holdings smaller than b*(a'=0)
        if b_star_zero[j] > grid_b[1]
            # Calculate consumption policies, when HHs chooses bond holdings
            # lower than b*(a'=0) and equity holdings a'=0 and save then in cons_list
            # log_index = grid.b<b_star_zero(j);
            log_index = grid_b .< b_star_zero[j]
            
            # aux_c is the consumption policy under no adj (fix a'=0), for for m'<m_a*(a'=0)
            # c_k_cons = aux_c(log_index,j);
            c_k_cons = aux_c[log_index, j]
            
            # cons_list{j} = c_k_cons;
            cons_list[j] = c_k_cons
            
            # res_list{j} = grid.b(log_index)' + c_k_cons - aux_inc(j);
            res_list[j] = grid_b[log_index] .+ c_k_cons .- aux_inc[j]
            
            # b_list{j} = grid.b(log_index)';
            b_list[j] = grid_b[log_index]
            
            # a_list{j} = zeros(sum(log_index),1);
            a_list[j] = zeros(sum(log_index))
        else
            # Initialize empty lists if condition not met
            cons_list[j] = Float64[]
            res_list[j] = Float64[]
            b_list[j] = Float64[]
            a_list[j] = Float64[]
        end
    end
    
    # Merge lists
    # c_star = reshape(c_star,[grid.na grid.nse]);
    # b_a_star = reshape(b_a_star,[grid.na grid.nse]);
    # Resource = reshape(Resource,[grid.na grid.nse]);
    # Note: c_star and Resource are already reshaped above
    b_a_star_reshaped = reshape(b_a_star, na_val, nse_val)
    
    grid_a = vec(grid["a"])  # Ensure 1D vector
    
    for j = 1:nse_val
        # cons_list{j} = [cons_list{j};c_star(:,j)];
        cons_list[j] = [cons_list[j]; vec(c_star[:, j])]
        
        # res_list{j} = [res_list{j};Resource(:,j)];
        res_list[j] = [res_list[j]; vec(Resource[:, j])]
        
        # b_list{j} = [b_list{j};b_a_star(:,j)];
        b_list[j] = [b_list[j]; vec(b_a_star_reshaped[:, j])]
        
        # a_list{j} = [a_list{j};grid.a'];
        # grid.a' in MATLAB is transpose, but we want column vector
        a_list[j] = [a_list[j]; grid_a]  # grid_a is already a column vector
    end
    
    return (c_star, Resource, cons_list, res_list, b_list, a_list)
end

