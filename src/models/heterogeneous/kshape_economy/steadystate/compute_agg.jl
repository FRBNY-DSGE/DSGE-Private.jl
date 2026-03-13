# Assumes all inputs (meshes, grid, param) are Dicts with String keys
"""
    compute_agg(meshes, grid, param)

Port of compute_agg.m from MATLAB to Julia.

Computes aggregate variables for the steady state equilibrium, including:
- Labor market variables (employment, unemployment, vacancies, matching)
- Production and factor prices (wages, rental rates)
- Financial variables (profits, returns, bank variables)
- Distribution variables (wage distribution, returns)

# Arguments
- `meshes`: Dictionary containing mesh grids for b, a, se (bonds, equity, idiosyncratic productivity)
- `grid`: Grid structure containing grid points and dimensions
- `param`: Parameter structure containing model parameters

# Returns
Tuple of 27 outputs:
- `mc`: Marginal cost
- `L`: Aggregate labor
- `n`: Labor supply per worker
- `r_l`: Labor rental rate
- `r_k`: Capital rental rate
- `Profit`: Total profit
- `v`: Utilization rate
- `J`: Value of matching (vector)
- `u`: Unemployment rate
- `V`: Vacancies
- `M`: Matches
- `f`: Job finding rate
- `w`: Wage
- `WW`: After-tax wage/benefit (3D array)
- `RR`: After-tax equity return
- `RBRB`: After-tax interest rate (3D array)
- `Y`: Output
- `P_SE`: Transition matrix for employment states
- `param`: Updated parameter structure
- `Lambda`: Discount factor
- `grid`: Updated grid structure
- `THETA`: Leverage ratio
- `NWb`: Net worth of banks
- `Profit_FI`: Financial intermediary profit
- `Ab`: Bank equity holdings
- `Bb`: Family deposits at FIs
- `Cb`: Family consumption

# Notes
- This function modifies `param` and `grid` in place
- Requires `find_alpha` function to be available
"""



function compute_agg(meshes, grid, param)
    
    v = param["v"]  # ss utilization rate of 0.75
    
    # Handle ss_obj_old - it might not exist or might be wrapped
    ss_obj_old = param["ss_obj_old"]
    if ss_obj_old === nothing || (ss_obj_old isa Dict && !haskey(ss_obj_old, :param_old) && !haskey(ss_obj_old, "param_old"))
        # If ss_obj_old doesn't exist, create dummy values
        param_old = param
        SS_stats_old = Dict()
        grid_old = grid
    else
        # Access using bracket notation - works for both DictWrapper and Dict
        param_old = if ss_obj_old isa Dict
            haskey(ss_obj_old, :param_old) ? ss_obj_old[:param_old] : ss_obj_old["param_old"]
        else
            ss_obj_old.param_old
        end
        SS_stats_old = if ss_obj_old isa Dict
            haskey(ss_obj_old, :SS_stats_old) ? ss_obj_old[:SS_stats_old] : ss_obj_old["SS_stats_old"]
        else
            ss_obj_old.SS_stats_old
        end
        grid_old = if ss_obj_old isa Dict
            haskey(ss_obj_old, :grid_old) ? ss_obj_old[:grid_old] : ss_obj_old["grid_old"]
        else
            ss_obj_old.grid_old
        end
    end
    
    Lambda = param["beta"]  # No longer need a discount factor trick
    
    u = param["u"]    # target
    v_f = param["v_f"]  # target
    
    Eshare = param["Eshare"]
    N = (1 - u) * (1 - Eshare)
    U = u * (1 - Eshare)
    
    in_val = param["in"]
    N_tilde = (1 - in_val) * N
    U_tilde = (1 - in_val) * U + param["out"] * Eshare
    
    lambda_val = param["lambda"]
    M = (lambda_val + in_val - lambda_val * in_val) * N
    V = M / v_f
    
    f = M / (lambda_val * N_tilde + U_tilde)
    
    # P_SS3 = kron([1-param.lambda+param.lambda*f,param.lambda*(1-f);f,1-f],eye(grid.ns));
    kron_matrix = [1-lambda_val+lambda_val*f lambda_val*(1-f); f 1-f]
    ns_val = Int(grid["ns"])  # Convert to Int for matrix dimensions
    P_SS3 = kron(kron_matrix, Matrix(1.0I, ns_val, ns_val))
    
    # P_SS3 = [P_SS3 repmat(0, [grid.nse-1 1])];
    P_SS3 = [P_SS3 zeros(size(P_SS3, 1), 1)]
    
    # lastrow = [zeros(1,2*grid.ns),1];
    # ns_val already defined above
    lastrow = [zeros(1, 2*ns_val) 1]
    
    # P_SS3 = [P_SS3; lastrow];
    P_SS3 = [P_SS3; lastrow]
    
    set_field!(param, :P_SS3, P_SS3)
    
    # P_SE = param.P_SS2*param.P_SS3;
    P_SE = param["P_SS2"] * P_SS3
    
    # grid.se_dist = ones(grid.nse,1)/grid.nse;
    nse_val = Int(grid["nse"])  # Convert to Int for array dimensions
    set_field!(grid, :se_dist, ones(nse_val, 1) / nse_val)
    
    # for kkk = 1:100000
    #     grid.se_dist = P_SE'*grid.se_dist;
    # end
    for kkk = 1:100000
        se_dist_val = grid["se_dist"]
        set_field!(grid, :se_dist, P_SE' * se_dist_val)
    end
    
    set_field!(param, :P_SE, P_SE)
    
    # grid.se_bar = grid.s*grid.se_dist(1:grid.ns)/(sum(grid.se_dist(1:grid.ns)));
    s_val = grid["s"]
    se_dist_val = grid["se_dist"]
    # ns_val already defined above as Int
    se_bar_val = dot(s_val, se_dist_val[1:ns_val]) / sum(se_dist_val[1:ns_val])
    set_field!(grid, :se_bar, se_bar_val)
    
    # alpha_init = log(1);
    # options = optimset('disp','none','TolFun',1e-15,'TolX',1e-15);
    # [out,~,exitflag_alpha] = fsolve(@(x) find_alpha(x,V,v_f,N_tilde,U_tilde,param),alpha_init,options);
    alpha_init = log(1.0)
    sol = nlsolve((F, x) -> F[1] = find_alpha(x[1], V, v_f, N_tilde, U_tilde, param), [alpha_init], 
                  show_trace=false, ftol=1e-15, xtol=1e-15)
    
    if !sol.f_converged && !sol.x_converged
        @warn "Could not find a steady state parameter alpha..."
    end
    
    set_field!(param, :alpha, exp(sol.zero[1]))
    
    # mc = (param.eta-1)/param.eta;
    eta_val = param["eta"]
    mc = (eta_val - 1) / eta_val
    
    n = param["n"]
    
    # L = N*grid.se_bar*n;  % aggregate labor
    se_bar_val = grid["se_bar"]
    L = N * se_bar_val * n
    
    # Y = (v*grid.K).^param.theta*L^param.theta_2;
    K_val = grid["K"]
    theta_val = param["theta"]
    theta_2_val = param["theta_2"]
    Y = (v * K_val)^theta_val * L^theta_2_val
    
    # r_l = mc*param.theta_2*Y/L;
    r_l = mc * theta_2_val * Y / L
    
    # r_k = mc*param.theta*Y/(v*grid.K);
    r_k = mc * theta_val * Y / (v * K_val)
    
    # param.LT = param.LT_ratio*Y;
    set_field!(param, :LT, param["LT_ratio"] * Y)
    
    # w_bar = param.wmarkdown*r_l;
    w_bar = param["wmarkdown"] * r_l
    
    w = w_bar
    
    set_field!(param, :w_bar, w_bar)
    
    # J = inv(eye(grid.ns)-param.beta_L*(1-param.death_rate)*(1-param.lambda)*(1-param.in)*param.P_SS)*(r_l-param.w_bar)*n*(grid.s');
    ns_val = Int(grid["ns"])  # Convert to Int for matrix dimensions
    I_ns = Matrix(1.0I, ns_val, ns_val)
    beta_L_val = param["beta_L"]
    death_rate_val = param["death_rate"]
    P_SS_val = param["P_SS"]
    w_bar_val = param["w_bar"]
    s_val = grid["s"]
    J = inv(I_ns - beta_L_val * (1 - death_rate_val) * (1 - lambda_val) * (1 - in_val) * P_SS_val) * 
        (r_l - w_bar_val) * n * s_val'
    
    J_aux1 = inv(I_ns - beta_L_val * (1 - death_rate_val) * (1 - lambda_val) * (1 - in_val) * P_SS_val) * 
             (r_l - w_bar_val) * n * s_val'
    J_aux2 = inv(I_ns - beta_L_val * (1 - death_rate_val) * (1 - lambda_val) * (1 - in_val) * P_SS_val) * 
             n * s_val'
    J_aux3 = inv(I_ns - beta_L_val * (1 - death_rate_val) * (1 - lambda_val) * (1 - in_val) * P_SS_val) * 
             (r_l - w_bar_val) * n * s_val'
    
    # J_bar = J'*grid.s_dist;
    s_dist_val = grid["s_dist"]
    J_bar = first(J' * s_dist_val)  # Extract scalar from 1x1 matrix
    J_bar1 = first(J_aux1' * s_dist_val)  # Extract scalar from 1x1 matrix
    J_bar2 = first(J_aux2' * s_dist_val)  # Extract scalar from 1x1 matrix
    J_bar3 = first(J_aux3' * s_dist_val)  # Extract scalar from 1x1 matrix
    
    set_field!(param, :J_bar1, J_bar1)
    set_field!(param, :J_bar2, J_bar2)
    set_field!(param, :J_bar3, J_bar3)
    
    # param.delta_ss = r_k*v/param.delta_1;
    set_field!(param, :delta_ss, r_k * v / param["delta_1"])
    
    # param.fix_L = (J_bar1-param.iota*V/M)/J_bar2;
    iota_val = param["iota"]
    set_field!(param, :fix_L, (J_bar1 - iota_val * V / M) / J_bar2)
    
    # J = inv(eye(grid.ns)-param.beta_L*(1-param.death_rate)*(1-param.lambda)*(1-param.in)*param.P_SS)*((r_l-param.fix_L-param.w_bar)*n*(grid.s'));
    fix_L_val = param["fix_L"]
    J = inv(I_ns - beta_L_val * (1 - death_rate_val) * (1 - lambda_val) * (1 - in_val) * P_SS_val) * 
        ((r_l - fix_L_val - w_bar_val) * n * s_val')
    
    # J_bar = J'*grid.s_dist;
    J_bar = first(J' * s_dist_val)  # Extract scalar from 1x1 matrix
    
    # Profit_L = (r_l-param.fix_L - param.w_bar)*L - param.iota*V;
    Profit_L = (r_l - fix_L_val - w_bar_val) * L - iota_val * V
    
    # Profit_K = (r_k*v - param.delta_ss)*grid.K;
    delta_ss_val = param["delta_ss"]
    Profit_K = (r_k * v - delta_ss_val) * K_val
    
    # param.fix = Y - mc*Y + Profit_L + Profit_K - SS_stats_base.Profit;
    # Handle SS_stats_base access - try both Dict and struct access
    SS_profit = if SS_stats_base isa Dict
        haskey(SS_stats_base, :Profit) ? SS_stats_base[:Profit] : 
        (haskey(SS_stats_base, "Profit") ? SS_stats_base["Profit"] : 0.0)
    else
        try
            getproperty(SS_stats_base, :Profit)
        catch
            0.0
        end
    end
    set_field!(param, :fix, Y - mc * Y + Profit_L + Profit_K - SS_profit)
    
    # param.fix_ratio = param.fix/Y;
    fix_val = param["fix"]
    set_field!(param, :fix_ratio, fix_val / Y)
    
    # Profit_int = Y - mc*Y - param.fix;
    Profit_int = Y - mc * Y - fix_val
    
    # Profit = Profit_int + Profit_L + Profit_K;
    Profit = Profit_int + Profit_L + Profit_K
    
    # r_a = (1-param.tau_a)*(1-param.Eratio-param.b_share)*Profit/grid.K;
    tau_a_val = param["tau_a"]
    Eratio_val = param["Eratio"]
    b_share_val = param["b_share"]
    r_a = (1 - tau_a_val) * (1 - Eratio_val - b_share_val) * Profit / K_val
    
    # Lincome = grid.se_bar*(1-param.u)*(1-param.Eshare)*param.n*param.w_bar;
    u_val = param["u"]
    Lincome = se_bar_val * (1 - u_val) * (1 - Eshare) * n * w_bar_val
    
    # UB = param.xi/(1+param.xi)*grid.se(grid.ns+1:2*grid.ns)*grid.se_dist(grid.ns+1:2*grid.ns)*param.n*param.w_bar;
    xi_val = param["xi"]
    se_val = grid["se"]
    UB = xi_val / (1 + xi_val) * dot(se_val[ns_val+1:2*ns_val], se_dist_val[ns_val+1:2*ns_val]) * n * w_bar_val
    
    # TAX_revenue = param.tau_w*(Lincome+UB) + param.tau_a*Profit;
    tau_w_val = param["tau_w"]
    TAX_revenue = tau_w_val * (Lincome + UB) + tau_a_val * Profit
    
    # B_gov = param.B_gov_ratio*Y;
    B_gov = param["B_gov_ratio"] * Y
    
    # B_F = param.B_F_ratio*B_gov;
    B_F = param["B_F_ratio"] * B_gov
    
    # param.MMF_cont_ratio = 0.01*Y/TAX_revenue;
    set_field!(param, :MMF_cont_ratio, 0.01 * Y / TAX_revenue)
    
    # MMF_cont = param.MMF_cont_ratio*TAX_revenue;
    MMF_cont_ratio_val = param["MMF_cont_ratio"]
    MMF_cont = MMF_cont_ratio_val * TAX_revenue
    
    # param.psi = (1-param.tau_w)*param.w_bar*param.n^param.xi;
    set_field!(param, :psi, (1 - tau_w_val) * w_bar_val * n^xi_val)
    
    # the bankers family
    # R = param.R_cb/param.pi_cb;  % ss real rate
    R = param["R_cb"] / param["pi_cb"]
    
    # R_a = (param.q+r_a)/param.q;  % growth rate of return on the equity
    q_val = param["q"]
    R_a = (q_val + r_a) / q_val
    
    # Ab = param.Ab_ratio*grid.K;  % ss equity holdings of banks
    Ab = param["Ab_ratio"] * K_val
    
    # THETA = param.THETA;
    THETA = param["THETA"]
    
    # param.omega = (1-param.theta_b*((R_a-param.b_a_aux2*R)*THETA+param.b_a_aux2*R))/THETA;
    theta_b_val = param["theta_b"]
    b_a_aux2_val = param["b_a_aux2"]
    omega_val = (1 - theta_b_val * ((R_a - b_a_aux2_val * R) * THETA + b_a_aux2_val * R)) / THETA
    set_field!(param, :omega, omega_val)
    
    # zz = (R_a-param.b_a_aux2*R)*THETA + param.b_a_aux2*R;
    zz = (R_a - b_a_aux2_val * R) * THETA + b_a_aux2_val * R
    
    # xx = zz;
    xx = zz
    
    # ee = (1-param.theta_b)*param.Lambda_bank*param.b_a_aux2*R/(1-param.theta_b*zz*param.Lambda_bank);
    Lambda_bank_val = param["Lambda_bank"]
    ee = (1 - theta_b_val) * Lambda_bank_val * b_a_aux2_val * R / (1 - theta_b_val * zz * Lambda_bank_val)
    
    # vv = ((1-param.theta_b)*(R_a-param.b_a_aux2*R)*param.Lambda_bank)/(1-param.theta_b*xx*param.Lambda_bank);
    vv = ((1 - theta_b_val) * (R_a - b_a_aux2_val * R) * Lambda_bank_val) / (1 - theta_b_val * xx * Lambda_bank_val)
    
    # param.DELTA = vv + ee/THETA;
    set_field!(param, :DELTA, vv + ee / THETA)
    
    # NWb = Ab/THETA;
    NWb = Ab / THETA
    
    # Profit_FI = (1-param.theta_b)*((R_a-param.b_a_aux2*R)*THETA+param.b_a_aux2*R)*NWb-param.omega*param.q*Ab;
    Profit_FI = (1 - theta_b_val) * ((R_a - b_a_aux2_val * R) * THETA + b_a_aux2_val * R) * NWb - omega_val * q_val * Ab
    
    # Bb = param.Bb_ratio*B_FI;
    # Note: B_FI is not explicitly computed in MATLAB code, but it's (THETA-1)*NWb
    B_FI = (THETA - 1) * NWb
    Bb = param["Bb_ratio"] * B_FI
    
    # Cb = (MMF_cont + (param.b_a_aux2*R-1)*Bb);
    Cb = (MMF_cont + (b_a_aux2_val * R - 1) * Bb)
    
    # param.tau_tilde_a = param.tau_w*param.Eratio*Profit/(Profit_FI+param.Eratio*Profit);
    set_field!(param, :tau_tilde_a, tau_w_val * Eratio_val * Profit / (Profit_FI + Eratio_val * Profit))
    
    # [meshes.b, meshes.a, meshes.se] = ndgrid(grid.b,grid.a,grid.se);
    # Note: meshes should already have these, but we update them here
    # Assuming ndgrid function is available (from steady_state.jl or similar)
    # For now, we'll assume meshes is already set up correctly
    # If needed, uncomment:
    # meshes.b, meshes.a, meshes.se = ndgrid(grid.b, grid.a, grid.se)
    
    # auxWW = ones(grid.nb,grid.na,grid.nse);
    nb_val = Int(grid["nb"])  # Convert to Int for array dimensions
    na_val = Int(grid["na"])  # Convert to Int for array dimensions
    # nse_val already defined above as Int
    auxWW = ones(nb_val, na_val, nse_val)
    
    # auxWW(:,:,grid.ns+1:end) = zeros(grid.nb,grid.na,grid.ns+1);
    auxWW[:, :, (ns_val+1):end] .= 0.0
    
    # auxWW = auxWW.*(meshes.se.^(param.b_aux-1));
    # Handle meshes access - assume Dict
    meshes_se = meshes["se"]
    b_aux_val = param["b_aux"]
    auxWW = auxWW .* (meshes_se.^(b_aux_val - 1))
    
    # NW = param.xi/(1+param.xi)*n*param.w_bar;
    NW = xi_val / (1 + xi_val) * n * w_bar_val
    
    # WW_aux = (1-param.tau_w)*(NW*ones(grid.nb,grid.na,grid.nse));
    WW_aux = (1 - tau_w_val) * (NW * ones(nb_val, na_val, nse_val))
    
    # WW = (1-param.tau_w)*(NW*ones(grid.nb,grid.na,grid.nse) + param.b_share*Profit/((1-param.u)*(1-param.Eshare)*grid.s2_bar).*auxWW);
    s2_bar_val = grid["s2_bar"]
    WW = (1 - tau_w_val) * (NW * ones(nb_val, na_val, nse_val) .+ b_share_val * Profit / ((1 - u_val) * (1 - Eshare) * s2_bar_val) .* auxWW)
    
    # WW(:,:,end) = (1-param.tau_w)*(param.Eratio*Profit)/param.Eshare*ones(grid.nb,grid.na);
    WW[:, :, end] .= (1 - tau_w_val) * (Eratio_val * Profit) / Eshare
    
    # RR = r_a;  % After-tax equity return
    RR = r_a
    
    # RBRB = param.R_cb/param.pi_cb + (meshes.b<0).*(param.Rprem/param.pi_cb);  % After-tax interest rate
    # Handle meshes access - assume Dict
    meshes_b = meshes["b"]
    R_cb_val = param["R_cb"]
    pi_cb_val = param["pi_cb"]
    Rprem_val = param["Rprem"]
    RBRB = R_cb_val / pi_cb_val .+ (meshes_b .< 0) .* (Rprem_val / pi_cb_val)
    
    return mc, L, n, r_l, r_k, Profit, v, J, u, V, M, f, w, WW, RR, RBRB, Y, P_SE, param, Lambda, grid, THETA, NWb, Profit_FI, Ab, Bb, Cb
end

