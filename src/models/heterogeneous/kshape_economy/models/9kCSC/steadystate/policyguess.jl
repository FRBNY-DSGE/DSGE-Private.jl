"""
    policyguess(meshes, WW, RR, RBRB, param, grid)

Compute initial policy guesses for consumption and continuation values.

This function computes initial guesses for:
- Consumption when adjusting capital (c_a_guess)
- Consumption when not adjusting capital (c_n_guess)
- Marginal continuation value of holding capital (psi_guess)
- Income components (inc)

# Arguments
- `meshes`: Mesh structure with fields `se`, `a`, `b` (grid meshes)
- `WW`: Wage rate (scalar or array)
- `RR`: Return on capital (scalar or array)
- `RBRB`: Return on bonds (scalar or array)
- `param`: Parameter structure with fields `death_rate`, `q`, `b_a_aux`, `LT`
- `grid`: Grid structure with fields `nb`, `na`, `nse`

# Returns
Tuple of 4 outputs:
- `c_a_guess`: Consumption guess when adjusting capital (nb × na × nse)
- `c_n_guess`: Consumption guess when not adjusting capital (nb × na × nse)
- `psi_guess`: Initial guess for marginal continuation value of holding capital (nb × na × nse)
- `inc`: Income structure with fields `labor`, `dividend`, `capital`, `bond`, `transfer`
"""

# Assumes all inputs (meshes, param, grid) are Dicts with String keys
function policyguess(meshes, WW, RR, RBRB, param, grid)
    
    # Get grid dimensions
    nb_val = Int(grid["nb"])
    na_val = Int(grid["na"])
    nse_val = Int(grid["nse"])
    
    # Get mesh arrays - direct Dict access
    meshes_se = meshes["se"]
    meshes_a = meshes["a"]
    meshes_b = meshes["b"]
    
    # Get parameters - direct Dict access
    death_rate = param["death_rate"]
    q = param["q"]
    b_a_aux = param["b_a_aux"]
    LT = param["LT"]
    
    # Income
    inc = Dict()
    
    # inc.labor = WW.*meshes.se;
    inc["labor"] = WW .* meshes_se.^(1 - param["tau_P"])
    
    # inc.dividend = (RR+param.death_rate/(1-param.death_rate)*param.q).*meshes.a;
    inc["dividend"] = (RR .+ death_rate / (1 - death_rate) * q) .* meshes_a
    
    # inc.capital = param.q*meshes.a;
    inc["capital"] = q .* meshes_a
    
    # inc.bond = RBRB/(1-param.death_rate)*param.b_a_aux.*meshes.b;
    inc["bond"] = RBRB / (1 - death_rate) * b_a_aux .* meshes_b
    
    # inc.transfer = param.LT*ones(grid.nb,grid.na,grid.nse);
    inc["transfer"] = LT * ones(nb_val, na_val, nse_val)
    
    # Consumption guess
    # c_a_guess = inc.labor + inc.dividend + inc.capital + max(inc.bond,0) + inc.transfer;
    c_a_guess = inc["labor"] .+ inc["dividend"] .+ inc["capital"] .+ max.(inc["bond"], 0) .+ inc["transfer"]
    
    # c_n_guess = inc.labor + inc.dividend + max(inc.bond,0) + inc.transfer;
    c_n_guess = inc["labor"] .+ inc["dividend"] .+ max.(inc["bond"], 0) .+ inc["transfer"]
    
    # Initially guessed marginal continuation value of holding capital;
    # psi_guess = zeros([grid.nb grid.na grid.nse]);
    psi_guess = zeros(nb_val, na_val, nse_val)
    
    return (c_a_guess, c_n_guess, psi_guess, inc)
end

