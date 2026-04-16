function out = post_density_covid_Kshape_MCMC_v2(INIT,prior_info,data_est,ZLB_duration_1,ZLB_duration_2,Jacob_base,Jacob_ZLB,Jacob_afterGR,SS_stats,param,grid,mu_dist,Value,mutil_c,Va,Copula,Fss_ZLB,H_aux)

tic;

% global prior_info data_1 Jacob_base Jacob_ZLB SS_stats param grid mu_dist Value mutil_c Va Copula Fss_ZLB

param_update.kappa     = INIT(1 );
param_update.rho_w     = INIT(2 );
param_update.d         = 1-INIT(3);
param_update.phi       = INIT(4 )*10;
param_update.phi_pi    = INIT(5 );
param_update.phi_u     = INIT(6 );
param_update.rho_BB    = INIT(7 );
param_update.rho_B     = INIT(8 );
param_update.rho_D     = INIT(9 ) ;
param_update.rho_G     = INIT(10) ;
param_update.rho_R     = INIT(11) ;
param_update.rho_Z     = INIT(12) ;
param_update.rho_eta   = INIT(13) ;
param_update.rho_iota  = INIT(14) ;
param_update.sig_D     = INIT(15)/100 ;
param_update.sig_G     = INIT(16)/100 ;
param_update.sig_R     = INIT(17)/100 ;
param_update.sig_Z     = INIT(18)/100 ;
param_update.sig_eta   = INIT(19)/100 ;
param_update.sig_iota  = INIT(20)/100 ;
param_update.sig_w     = INIT(21)/100 ;
param_update.sig_BB    = INIT(22)/100 ;
param_update.GAMMA     = INIT(23);
param_update.sig_B_F   = INIT(24)/100;
param_update.rho_B_F   = INIT(25);
param_update.iota      = INIT(26);

param_update.eps_w     = 1;
param_update.phi_x     = 0;
param_update.rho_PSI_W = 0;
param_update.varphi    = 0;
param_update.alpha_lk  = 0;
param_update.theta_b   = 0.97;

param_update.rho_X_QE   = 0.95;
param_update.phi_pi_QE  = 10;
param_update.phi_u_QE   = 10;
                
param_update.gamma_B   = 0;
param_update.gamma_pi  = 0;
param_update.gamma_Y   = 0;
        
param_update.rho_MP    = 0;
param_update.gamma_pi  = 0;
param_update.gamma_T   = 0;
        
param_update.rho_PSI_RP = 0;
param_update.sig_RP     = 0.1/100;
param_update.Calvo      = 0.7;

param.QE     = 'QE';
param.regime = 'Taylor';

update_ss_v5;

ss_check = 1;

if param.fix < 0
    
    ss_check = 0;
    
elseif param.fix2 < 0
    
    ss_check = 0;
        
end


if  param_update.GAMMA >= 0 && ss_check > 0 && param_update.eps_w > 0 && param_update.iota > 0 && param_update.phi_pi > 1 && param_update.phi_u > 0 && abs(param_update.rho_BB) < 1 && abs(param_update.rho_B) < 1 && abs(param_update.rho_D) < 1 && abs(param_update.rho_G) < 1  && abs(param_update.rho_Z) < 1 && abs(param_update.rho_PSI_RP) < 1 && abs(param_update.rho_eta) < 1 && abs(param_update.rho_iota) < 1 && param_update.sig_D > 0 && param_update.sig_G > 0 && param_update.sig_R > 0  && param_update.sig_Z > 0 && param_update.sig_RP > 0 && param_update.sig_eta > 0 && param_update.sig_iota > 0 && param_update.sig_w > 0 && param_update.sig_BB > 0
    
    % save('INIT','INIT');
    
    parameters_agg_EST_v3;
    
    state_reduc_covid;
    
    % fprintf('kappa     : %1.3f \n', param.kappa);
    % fprintf('GAMMA     : %1.3f \n', param.GAMMA);
    % fprintf('rho_w     : %1.3f \n', param.rho_w);
    % fprintf('d         : %1.3f \n', 1-param.d  );
    % fprintf('eps_w     : %1.3f \n', param.eps_w);    
    % fprintf('theta_b   : %1.3f \n', param.theta_b);
    % fprintf('iota      : %1.3f \n', param.iota);
    % fprintf('alpha_lk  : %1.3f \n', param.alpha_lk);
    % fprintf('phi_pi    : %1.3f \n', param.phi_pi);
    % fprintf('phi_u     : %1.3f \n', param.phi_u);
    % fprintf('rho_R     : %1.3f \n', param.rho_R);
    % fprintf('rho_Z     : %1.3f \n', param.rho_Z);
    % fprintf('rho_BB    : %1.3f \n', param.rho_BB);
    % fprintf('rho_B     : %1.3f \n', param.rho_B);
    % fprintf('rho_D     : %1.3f \n', param.rho_D);
    % fprintf('rho_PSI_W : %1.3f \n', param.rho_PSI_W);
    % fprintf('rho_B_F   : %1.3f \n', param.rho_B_F);
    % fprintf('phi       : %1.3f \n', param.phi);
    % fprintf('rho QE    : %1.3f \n', param.rho_PSI_RP);
    % fprintf('phi x     : %1.3f \n', param.phi_x);
    % fprintf('gamma B   : %1.3f \n', param.gamma_B);
    % fprintf('gamma pi  : %1.3f \n', param.gamma_pi);
    % fprintf('gammaY    : %1.3f \n', param.gamma_Y);
    
    SIGMA_full    = eye(grid.numstates_shocks);
    
    SIGMA_full(1 ,1 ) = 1;
    SIGMA_full(2 ,2 ) = param.sig_B_F.^2;
    SIGMA_full(3 ,3 ) = param.sig_BB.^2;
    SIGMA_full(4 ,4 ) = param.sig_Z.^2;
    SIGMA_full(5 ,5 ) = param.sig_G.^2;
    SIGMA_full(6 ,6 ) = param.sig_D.^2;
    SIGMA_full(7 ,7 ) = param.sig_R.^2;
    SIGMA_full(8 ,8 ) = param.sig_iota.^2;
    SIGMA_full(9 ,9 ) = param.sig_eta.^2;
    SIGMA_full(10,10) = param.sig_w.^2;

    SIGMA    = eye(grid.numstates_shocks-1);
    
    SIGMA(1 ,1 ) = param.sig_B_F.^2;
    SIGMA(2 ,2 ) = param.sig_BB.^2;
    SIGMA(3 ,3 ) = param.sig_Z.^2;
    SIGMA(4 ,4 ) = param.sig_G.^2;
    SIGMA(5 ,5 ) = param.sig_D.^2;
    SIGMA(6 ,6 ) = param.sig_R.^2;
    SIGMA(7 ,7 ) = param.sig_iota.^2;
    SIGMA(8 ,8 ) = param.sig_eta.^2;
    SIGMA(9 ,9 ) = param.sig_w.^2;
    
    SIGMA_ZLB    = eye(grid.numstates_shocks-2);
    
    SIGMA_ZLB(1 ,1 ) = param.sig_B_F.^2;
    SIGMA_ZLB(2 ,2 ) = param.sig_BB.^2;
    SIGMA_ZLB(3 ,3 ) = param.sig_Z.^2;
    SIGMA_ZLB(4 ,4 ) = param.sig_G.^2;
    SIGMA_ZLB(5 ,5 ) = param.sig_D.^2;
    SIGMA_ZLB(6 ,6 ) = param.sig_iota.^2;
    SIGMA_ZLB(7 ,7 ) = param.sig_eta.^2;
    SIGMA_ZLB(8 ,8 ) = param.sig_w.^2;
    
    % F_ref = @(a,b,c,d)F_sys_ref(a,b,c,d,Xss,Yss,Gamma_state,Gamma_control,InvGamma,param,grid,SS_stats,Copula,param.P_SE);
    % F_ZLB = @(a,b,c,d)F_sys_ZLB_QE(a,b,c,d,Xss,Yss,Gamma_state,Gamma_control,InvGamma,param,grid,SS_stats,Copula,param.P_SE);
    
    % [Fss_ZLB,LHS_ZLB,RHS_ZLB,Distr_ZLB] = F_ZLB(State,State_m,Contr,Contr_m);
    
    param.overrideEigen = true;  % Warning appears, but critical Eigenvalue shifted
    
    indexes_v2;
    
    % out_Jacob         = compute_Jacob_base_v1(param,grid,SS_stats,idx,Xss,Yss);
    % out_Jacob_ZLB     = compute_Jacob_GR_v1(param,grid,SS_stats,idx,Xss,Yss);
    % out_Jacob_afterGR = compute_Jacob_afterGR_v1(param,grid,SS_stats,idx,Xss,Yss);

    out_Jacob         = compute_Jacob_covid_v1(param,grid,SS_stats,idx,Xss,Yss);
    
    %save
    %save('TESTIDX.mat', 'idx');
    %save('TESTJACOB.mat', 'out_Jacob');
    
    %assert(false);
    
    out_Jacob_ZLB     = compute_Jacob_covid_v1_GR(param,grid,SS_stats,idx,Xss,Yss);
    
    %save('TESTIDX2.mat', 'idx');
    %save('TESTJACOBZLB.mat', 'out_Jacob_ZLB');
    
    %assert(false);
    out_Jacob_afterGR = compute_Jacob_covid_v1_afterGR(param,grid,SS_stats,idx,Xss,Yss);
    % out_Jacob_afterGR = compute_Jacob_base_v3(param,grid,SS_stats,idx,Xss,Yss);
    
    [hx,gx,F1_aux,F2_aux,F3_aux,F4_aux,param,indicator_1]                                                 = SGU_EST_ref_v2(param,grid,Jacob_base   ,idx,out_Jacob);
    % [F1_ZLB_aux,F2_ZLB_aux,F3_ZLB_aux,F4_ZLB_aux,param]                                                   = SGU_EST_ZLB_v2(param,grid,Jacob_ZLB    ,idx,out_Jacob_ZLB);
    % [hx_afterGR,gx_afterGR,F1_afterGR_aux,F2_afterGR_aux,F3_afterGR_aux,F4_afterGR_aux,param,indicator_2] = SGU_EST_ref_v2(param,grid,Jacob_afterGR,idx,out_Jacob_afterGR);
    [F1_ZLB_aux,F2_ZLB_aux,F3_ZLB_aux,F4_ZLB_aux,param]                                                   = SGU_EST_ZLB_v2(param,grid,Jacob_base    ,idx,out_Jacob_ZLB);
    [hx_afterGR,gx_afterGR,F1_afterGR_aux,F2_afterGR_aux,F3_afterGR_aux,F4_afterGR_aux,param,indicator_2] = SGU_EST_ref_v2(param,grid,Jacob_base    ,idx,out_Jacob_afterGR);
    
    indicator = indicator_1*indicator_2;

    if indicator == 1
        
        F1 = [F1_aux(1:grid.numstates_endo,:);F1_aux(grid.numstates+1:end,:)];
        F2 = [F2_aux(1:grid.numstates_endo,:);F2_aux(grid.numstates+1:end,:)];
        F3 = [F3_aux(1:grid.numstates_endo,:);F3_aux(grid.numstates+1:end,:)];
        F4 = [F4_aux(1:grid.numstates_endo,:);F4_aux(grid.numstates+1:end,:)];
        
        F11 = F1(:,1:grid.numstates_endo);
        F12 = F1(:,grid.numstates_endo+1:end);
        
        F31 = F3(:,1:grid.numstates_endo);
        F32 = F3(:,grid.numstates_endo+1:end);
        
        save('TESTIDX.mat', 'idx');
        
        % Construct a coefficient matrix
        
        A_ref = [zeros(grid.numstates_endo+grid.numcontrols,grid.numstates_endo), F2];
        B_ref = [F11, F4];
        C_ref = [F31, zeros(grid.numstates_endo+grid.numcontrols,grid.numcontrols)];
        E_ref = F32;
        
        % Constuct P, Q
        
        hx_aux = hx(1:grid.numstates_endo,:);
        HH_aux = [hx_aux;gx];
        H1_aux = HH_aux(:,1:grid.numstates_endo);
        H2_aux = HH_aux(:,end-grid.numstates_shocks+1:end);
        
        P_ref  = [H1_aux,zeros(grid.numstates_endo+grid.numcontrols,grid.numcontrols)];
        Q_ref  = H2_aux;
        
        cof_ref = [C_ref,B_ref,A_ref];
        J_ref   = E_ref;
        
        F1_ZLB = [F1_ZLB_aux(1:grid.numstates_endo,:);F1_ZLB_aux(grid.numstates+1:end,:)];
        F2_ZLB = [F2_ZLB_aux(1:grid.numstates_endo,:);F2_ZLB_aux(grid.numstates+1:end,:)];
        F3_ZLB = [F3_ZLB_aux(1:grid.numstates_endo,:);F3_ZLB_aux(grid.numstates+1:end,:)];
        F4_ZLB = [F4_ZLB_aux(1:grid.numstates_endo,:);F4_ZLB_aux(grid.numstates+1:end,:)];
        
        F11_ZLB = F1_ZLB(:,1:grid.numstates_endo);
        F12_ZLB = F1_ZLB(:,grid.numstates_endo+1:end);
        
        F31_ZLB = F3_ZLB(:,1:grid.numstates_endo);
        F32_ZLB = F3_ZLB(:,grid.numstates_endo+1:end);
        
        % Construct a coefficient matrix
        
        A_ZLB = [zeros(grid.numstates_endo+grid.numcontrols,grid.numstates_endo), F2_ZLB];
        B_ZLB = [F11_ZLB, F4_ZLB];
        C_ZLB = [F31_ZLB, zeros(grid.numstates_endo+grid.numcontrols,grid.numcontrols)];
        E_ZLB = F32_ZLB;
        D_ZLB = [Fss_ZLB(1:grid.numstates_endo,:);Fss_ZLB(grid.numstates+1:end,:)];
        
        cof_ZLB = [C_ZLB,B_ZLB,A_ZLB];
        J_ZLB   = E_ZLB;

        F1_afterGR = [F1_afterGR_aux(1:grid.numstates_endo,:);F1_afterGR_aux(grid.numstates+1:end,:)];
        F2_afterGR = [F2_afterGR_aux(1:grid.numstates_endo,:);F2_afterGR_aux(grid.numstates+1:end,:)];
        F3_afterGR = [F3_afterGR_aux(1:grid.numstates_endo,:);F3_afterGR_aux(grid.numstates+1:end,:)];
        F4_afterGR = [F4_afterGR_aux(1:grid.numstates_endo,:);F4_afterGR_aux(grid.numstates+1:end,:)];
        
        F11_afterGR = F1_afterGR(:,1:grid.numstates_endo);
        F12_afterGR = F1_afterGR(:,grid.numstates_endo+1:end);
        
        F31_afterGR = F3_afterGR(:,1:grid.numstates_endo);
        F32_afterGR = F3_afterGR(:,grid.numstates_endo+1:end);
        
        % Construct a coefficient matrix
        
        A_afterGR = [zeros(grid.numstates_endo+grid.numcontrols,grid.numstates_endo), F2];
        B_afterGR = [F11_afterGR, F4_afterGR];
        C_afterGR = [F31_afterGR, zeros(grid.numstates_endo+grid.numcontrols,grid.numcontrols)];
        E_afterGR = F32_afterGR;
        
        % Constuct P, Q
        
        hx_afterGR_aux = hx_afterGR(1:grid.numstates_endo,:);
        HH_afterGR_aux = [hx_afterGR_aux;gx_afterGR];
        H1_afterGR_aux = HH_afterGR_aux(:,1:grid.numstates_endo);
        H2_afterGR_aux = HH_afterGR_aux(:,end-grid.numstates_shocks+1:end);
        
        P_afterGR  = [H1_afterGR_aux,zeros(grid.numstates_endo+grid.numcontrols,grid.numcontrols)];
        Q_afterGR  = H2_afterGR_aux;
        
        cof_afterGR = [C_afterGR,B_afterGR,A_afterGR];
        J_afterGR   = E_afterGR;
        
        ZLB_indicator     = exp(data_est(5,:)')*param.R_cb < 1+1e-8;
        
        [num_obs,num_data_periods] = size(data_est);

        afterGR_indicator         = zeros(num_data_periods,1);
        afterGR_indicator(97:end) = 1;
        
        % [num_obs,~] = size(data_est);
        
        % num_data_periods = 80;
        
        shock_series = zeros(grid.numstates_shocks,num_data_periods);
        
        grid.num_data = 10;
                
        HQ = H_aux*Q_ref;

        HQ_afterGR = H_aux*Q_afterGR;
        
        H_lik = H_aux(1:grid.num_data-1,:);
        
        Q_lik = Q_ref(:,2:end);

        Q_afterGR_lik = Q_afterGR(:,2:end);

        H_ZLB = zeros(grid.num_data-2,grid.num_endo);

        H_ZLB(1:4,:)   = H_aux(1:4,:);
        H_ZLB(5:end,:) = H_aux(6:end-1,:);
        
        l_terms = zeros(num_data_periods,1);
        
        nvars = grid.num_endo;
        
        decrulea = P_ref;
        decruleb = Q_ref;
        
        Cbarmat = cof_ref(:,1:nvars);
        Bbarmat = cof_ref(:,nvars+1:2*nvars);
        Abarmat = cof_ref(:,2*nvars+1:3*nvars);

        decrulea_afterGR = P_afterGR;
        decruleb_afterGR = Q_afterGR;
        
        Cbarmat_afterGR = cof_afterGR(:,1:nvars);
        Bbarmat_afterGR = cof_afterGR(:,nvars+1:2*nvars);
        Abarmat_afterGR = cof_afterGR(:,2*nvars+1:3*nvars);
        
        % cofstar contains the system for the model when the constraint binds
        
        Cstarbarmat = cof_ZLB(:,1:nvars);
        Bstarbarmat = cof_ZLB(:,nvars+1:2*nvars);
        Astarbarmat = cof_ZLB(:,2*nvars+1:3*nvars);
        
        % No ZLB No OccBin
        
        ZLB_num_1 = 1;
        ZLB_num_2 = 1;

        unique_EZLB_duration_1 = unique(ZLB_duration_1);

        num_EZLB_duration_1 = length(unique_EZLB_duration_1);

        Ps_aux = zeros(grid.num_endo,grid.num_endo,num_EZLB_duration_1);
        Ds_aux = zeros(grid.num_endo,num_EZLB_duration_1);
        Es_aux = zeros(grid.num_endo,grid.numstates_shocks,num_EZLB_duration_1);

        for kkk = 1:num_EZLB_duration_1

            Tmax = unique_EZLB_duration_1(kkk);
            
            OccBin_one_cons_GR_v1;
            
            Ps_aux(:,:,kkk) = P_1;
            Ds_aux(:,kkk) = D_1;
            Es_aux(:,:,kkk) = E_1;

        end

        unique_EZLB_duration_2 = unique(ZLB_duration_2);

        num_EZLB_duration_2 = length(unique_EZLB_duration_2);

        Ps_afterGR_aux = zeros(grid.num_endo,grid.num_endo,num_EZLB_duration_2);
        Ds_afterGR_aux = zeros(grid.num_endo,num_EZLB_duration_2);
        Es_afterGR_aux = zeros(grid.num_endo,grid.numstates_shocks,num_EZLB_duration_2);

        for kkk = 1:num_EZLB_duration_2

            Tmax = unique_EZLB_duration_2(kkk);
            
            OccBin_one_cons_covid_v1;
            
            Ps_afterGR_aux(:,:,kkk) = P_1;
            Ds_afterGR_aux(:,kkk)   = D_1;
            Es_afterGR_aux(:,:,kkk) = E_1;

        end

        X_init = zeros(grid.num_endo,1);

        for tt = 1:num_data_periods
            
            % for tt = 1:1
            
            
            X_data = data_est(:,tt) ;

            if ZLB_indicator(tt) < 1 && afterGR_indicator(tt) == 0
                
                % eps_t = inv(HQ)*(X_data - H_aux*P_ref*X_init);
                eps_t = (HQ)\(X_data - H_aux*P_ref*X_init);

                shock_series(:,tt) = eps_t;
                
                X_init = P_ref*X_init + Q_ref*eps_t;
                
                eps_t_lik = eps_t(2:end);
                
                l_terms(tt) = -1/2*eps_t'*inv(SIGMA_full)*eps_t - log(abs(det(HQ)));
                % l_terms(tt) = -1/2*eps_t_lik'*inv(SIGMA)*eps_t_lik - log(abs(det(H_lik*Q_lik)));
                % l_terms(tt) = -1/2*eps_t'*inv(SIGMA)*eps_t - log(abs(det(HQ)));
                
            elseif ZLB_indicator(tt) == 1 && afterGR_indicator(tt) == 0

                Tmax = ZLB_duration_1(ZLB_num_1);

                index_Tmax = sum(unique_EZLB_duration_1<=Tmax);
                
                X_data_aux    = X_data;
                X_data_aux(5) = - D_ZLB(grid.numstates-grid.os+R_cb_ind);
                
                % OccBin_one_cons_exo_ZLB_v1;

                P_1 = Ps_aux(:,:,index_Tmax);
                D_1 = Ds_aux(:,index_Tmax);
                E_1 = Es_aux(:,:,index_Tmax);

                HE  = H_aux*E_1;
    
                eps_t = (HE)\(X_data_aux-H_aux*P_1*X_init - H_aux*D_1);
    
                history      = zeros(nvars,2);
                history(:,1) = X_init;
    
                history(:,2) = P_1*history(:,1) + D_1 + E_1*eps_t;
                
                shock_series(:,tt) = eps_t;
                
                X_init = history(:,2);
                
                eps_t_lik = [eps_t(2:5);eps_t(7:end)];
                
                % E_lik = [E(:,2:5),E(:,7:end)];
                E_lik = [E_1(:,2:5),E_1(:,7:end)];
                
                l_terms(tt) = -1/2*eps_t'*inv(SIGMA_full)*eps_t - log(abs(det(HE)));
                % l_terms(tt) = -1/2*eps_t_lik'*inv(SIGMA_ZLB)*eps_t_lik - log(abs(det(H_ZLB*E_lik)));

                ZLB_num_1 = ZLB_num_1 + 1;
                
                clear P_1 D_1 E_1

            elseif ZLB_indicator(tt) < 1 && afterGR_indicator(tt) == 1

                % eps_t = inv(HQ)*(X_data - H_aux*P_ref*X_init);
                eps_t = (HQ_afterGR)\(X_data - H_aux*P_afterGR*X_init);

                shock_series(:,tt) = eps_t;
                
                X_init = P_afterGR*X_init + Q_afterGR*eps_t;
                
                eps_t_lik = eps_t(2:end);
                
                l_terms(tt) = -1/2*eps_t'*inv(SIGMA_full)*eps_t - log(abs(det(HQ_afterGR)));
                % l_terms(tt) = -1/2*eps_t_lik'*inv(SIGMA)*eps_t_lik - log(abs(det(H_lik*Q_afterGR_lik)));
                % l_terms(tt) = -1/2*eps_t'*inv(SIGMA)*eps_t - log(abs(det(HQ)));

            elseif ZLB_indicator(tt) == 1 && afterGR_indicator(tt) == 1

                Tmax = ZLB_duration_2(ZLB_num_2);

                index_Tmax = sum(unique_EZLB_duration_2<=Tmax);
                
                X_data_aux    = X_data;
                X_data_aux(5) = - D_ZLB(grid.numstates-grid.os+R_cb_ind);
                
                % OccBin_one_cons_exo_ZLB_v1;

                P_1 = Ps_afterGR_aux(:,:,index_Tmax);
                D_1 = Ds_afterGR_aux(:,index_Tmax);
                E_1 = Es_afterGR_aux(:,:,index_Tmax);

                HE  = H_aux*E_1;
    
                eps_t = (HE)\(X_data_aux-H_aux*P_1*X_init - H_aux*D_1);
    
                history      = zeros(nvars,2);
                history(:,1) = X_init;
    
                history(:,2) = P_1*history(:,1) + D_1 + E_1*eps_t;
                
                shock_series(:,tt) = eps_t;
                
                X_init = history(:,2);
                
                eps_t_lik = [eps_t(2:5);eps_t(7:end)];
                
                % E_lik = [E(:,2:5),E(:,7:end)];
                E_lik = [E_1(:,2:5),E_1(:,7:end)];
                
                l_terms(tt) = -1/2*eps_t'*inv(SIGMA_full)*eps_t - log(abs(det(HE)));
                % l_terms(tt) = -1/2*eps_t_lik'*inv(SIGMA_ZLB)*eps_t_lik - log(abs(det(H_ZLB*E_lik)));

                ZLB_num_2 = ZLB_num_2 + 1;
                
                clear P_1 D_1 E_1
                
            end
            
        end
        
        log_lik = - num_data_periods/2*log(det(SIGMA)) + sum(l_terms);
        
        log_lik = - log_lik; % should be minimized
        
        log_prior = prior_density_covid_v10(INIT,prior_info);
        
        % out = log_lik + log_prior + log_EZLB_prior_mass;
        out = log_lik + log_prior;
        
        
    elseif indicator == 0
        
        
        out = 4e+2;
        
        Tmaxs = 0;
        error_check = 0;
        
    end
    
else
    
    out = 6e+2;
    error_check = 0;
    Tmaxs = 0;
    
    indicator = 0;
     
end

esp_time = toc;
% disp('-----------------------------------------------');
% fprintf('Elapsed time     : %1.3f \n', esp_time);
% fprintf('Log-likelihood   : %5.6f \n', -out);
% % fprintf('# of errors      : %5.2f \n', sum(error_check));
% fprintf('# of e ZLB dur   : %5.2f \n', sum(ZLB_duration_1)+sum(ZLB_duration_2));
% disp('-----------------------------------------------');

% if indicator > 0
    
%     shock_mean = mean(abs(shock_series),2);
%     disp('-----------------------------------------------');
%     fprintf('0. QE                : %1.4f \n', shock_mean(1));
%     fprintf('1. Profit            : %1.4f \n', shock_mean(2));
%     fprintf('2. Risk premium      : %1.4f \n', shock_mean(3));
%     fprintf('3. TFP               : %1.4f \n', shock_mean(4));
%     fprintf('4. Gov purchase      : %1.4f \n', shock_mean(5));
%     fprintf('5. Liquidity prep    : %1.4f \n', shock_mean(6));
%     fprintf('6. MP                : %1.4f \n', shock_mean(7));
%     fprintf('7. MEI (IST)         : %1.4f \n', shock_mean(8));
%     fprintf('8. Price mark-up     : %1.4f \n', shock_mean(9));
%     fprintf('9. Wage mark-up      : %1.4f \n', shock_mean(10));
%     disp('-----------------------------------------------');
%     fprintf('Corr 1 & 2           : %1.4f \n', corr(shock_series(1,:)',shock_series(2,:)'));
%     fprintf('Corr 1 & 3           : %1.4f \n', corr(shock_series(1,:)',shock_series(3,:)'));
%     fprintf('Corr 2 & 3           : %1.4f \n', corr(shock_series(2,:)',shock_series(3,:)'));
%     fprintf('Corr 2 & 7           : %1.4f \n', corr(shock_series(2,:)',shock_series(7,:)'));
%     fprintf('Corr 3 & 7           : %1.4f \n', corr(shock_series(3,:)',shock_series(7,:)'));
%     fprintf('Corr 5 & 7           : %1.4f \n', corr(shock_series(5,:)',shock_series(7,:)'));
%     fprintf('Corr 7 & 8           : %1.4f \n', corr(shock_series(7,:)',shock_series(8,:)'));
%     disp('-----------------------------------------------');
    
% end






