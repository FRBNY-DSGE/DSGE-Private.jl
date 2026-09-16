function [hx,gx,F1,F2,F3,F4,param,indicator] = SGU_EST_ref_v2(param,grid,Jacob_base,idx,Jacob_update)
% This funtion solves for a competitive equilibrium defined as a zero of the
% function F which is written in Schmitt-Grohé Uribe form using the algorithm
% suggested  in Schmit-Grohé and Uribe (2004): "Solving dynamic general equilibrium models
% using a second-order approximation to the policy function"
%
%now do numerical linearization
State       = zeros(grid.numstates,1);
State_m     = State;
Contr       = zeros(grid.numcontrols,1);
Contr_m     = Contr;
% tic
% [Fb,~,~]  = F(State,State_m,Contr,Contr_m);%Fsys(State,State,Contr,Contr,Xss,Yss,Gamma_state,Gamma_control,InvGamma,par,mpar,grid,targets);
% toc
%Use Schmitt Gohe Uribe Algorithm
% E[x' u']' =inv(A)*B*[x u]'
% A = [dF/dx' dF/du'], B =[dF/dx dF/du]
% A = [F1 F2]; B=[F3 F4]

F1_aux = zeros(grid.numstates+grid.numcontrols,grid.numstates);   % Tomorrows states do not affect error on controls and have unit effect on state error
F2_aux = zeros(grid.numstates+grid.numcontrols,grid.numcontrols); % Jacobian wrt tomorrow's controls (TO BE FILLED)
F3_aux = zeros(grid.numstates+grid.numcontrols,grid.numstates);   % Jacobian wrt today's states (TO BE FILLED)
%Today's Value functions do not affect error on states and have unit effect
%on Value function error (LAST TWO COLUMNS TO BE FILLED: Aggregate Prices)
F4_aux = [zeros(grid.numstates,grid.numcontrols);eye(grid.numcontrols,grid.numcontrols)];

F1Xnext = Jacob_base.F1Xnext;
F1Ynext = Jacob_base.F1Ynext;
F1X     = Jacob_base.F1X;
F1Y     = Jacob_base.F1Y;

F21next = Jacob_base.F21next;
F21     = Jacob_base.F21;
F24next = Jacob_base.F24next;
F24     = Jacob_base.F24;

F4Xnext = Jacob_base.F4Xnext;
F4Ynext = Jacob_base.F4Ynext;
F4X     = Jacob_base.F4X;
F4Y     = Jacob_base.F4Y;

F31next = Jacob_base.F31next;
F32next = Jacob_base.F32next;
F33next = Jacob_base.F33next;
F3Xnext = Jacob_base.F3Xnext;
F41next = Jacob_base.F41next;
F42next = Jacob_base.F42next;
F43next = Jacob_base.F43next;

F51next = Jacob_base.F51next;
F53next = Jacob_base.F53next;
F61next = Jacob_base.F61next;
F3Ynext = Jacob_base.F3Ynext;
F5Xnext = Jacob_base.F5Xnext;
F5Ynext = Jacob_base.F5Ynext;
F64next = Jacob_base.F64next;

F31     = Jacob_base.F31;
F32     = Jacob_base.F32;
F51     = Jacob_base.F51;
F61     = Jacob_base.F61;

F3Y     = Jacob_base.F3Y;
F54     = Jacob_base.F54;
F64     = Jacob_base.F64;

F5X     = Jacob_base.F5X;
F5Y     = Jacob_base.F5Y;


% out = compute_Jacob(param,grid,SS_stats,idx);

F1 = [F1Xnext; ...
      Jacob_update.F21_aux; ... 
      F4Xnext; ...
      F5Xnext; ...
        Jacob_update.F41_aux(end-grid.oc_agg+1:end,:)];

F3 = [F1X; ...
        Jacob_update.F23_aux; ...
      F4X; ...
      F5X; ...
        Jacob_update.F43_aux(end-grid.oc_agg+1:end,:)];

F2 = [F1Ynext; ...
        Jacob_update.F22_aux; ...
      F4Ynext;
      F5Ynext;
        Jacob_update.F42_aux(end-grid.oc_agg+1:end,:)];

F4 = [F1Y; ...
        Jacob_update.F24_aux;...
      F4Y; ...
      F5Y; ...
        Jacob_update.F44_aux(end-grid.oc_agg+1:end,:)];


% F1 = [F1Xnext; ...
%       F4Xnext; ...
%       F51next, F1_aux(end-grid.oc+1:end-grid.oc+grid.oc_summary,grid.numstates-grid.os+1:grid.numstates-grid.os+grid.os_agg), F53next; ...
%       % Jacob_base.F1_aux(end-grid.oc+1:end-grid.oc+grid.oc_summary,:); ...
%       F61next, F1_aux(end-grid.oc_agg+1:end,grid.numstates-grid.os+1:grid.numstates)  ];

% F3 = [F1X; ...
%       F21, F3_aux(grid.numstates-grid.os+1:grid.numstates-grid.os+grid.os_agg,grid.numstates-grid.os+1:grid.numstates); ...
%       F31, F32, F3_aux(grid.numstates-grid.os_process+1:grid.numstates,grid.numstates-grid.os_process+1:grid.numstates); ...
%       F4X; ...
%       F51, F3_aux(end-grid.oc+1:end-grid.oc_agg,grid.numstates-grid.os+1:grid.numstates); ...
%       % Jacob_base.F3_aux(end-grid.oc+1:end-grid.oc+grid.oc_summary,:); ...
%       F61, F3_aux(end-grid.oc_agg+1:end,grid.numstates-grid.os+1:grid.numstates)  ];

% F2 = [F1Ynext; ...
%       F24next, F2_aux(grid.numstates-grid.os+1:grid.numstates-grid.os+grid.os_agg,end-grid.oc+1:end); ...
%       F3Ynext;
%       F4Ynext;
%       F5Ynext;
%       F64next, F2_aux(end-grid.oc_agg+1:end,end-grid.oc+1:end)];

% F4 = [F1Y; ...
%       F24, F4_aux(grid.numstates-grid.os+1:grid.numstates-grid.os+grid.os_agg,end-grid.oc+1:end); ...
%       F3Y; ...
%       F4Y; ...
%       F54, F4_aux(end-grid.oc+1:end-grid.oc_agg,end-grid.oc+1:end); ...
%       % Jacob_base.F4_aux(end-grid.oc+1:end-grid.oc_agg,:); ...
%       F64, F4_aux(end-grid.oc_agg+1:end,end-grid.oc+1:end)];

    

% F1 = Jacob_base.F1_aux;
% F2 = Jacob_base.F2_aux;
% F3 = Jacob_base.F3_aux;
% F4 = Jacob_base.F4_aux;


%% Schmidt-Grohé and Uribe code based on qz decomposition of the system
%  Code adapted from SGUs online ressources
% toc

[s, t, Q, Z] = qz(full([F1,F2]),full(-[F3, F4]));

relev = abs(diag(s))./abs(diag(t));
ll    = sort(relev);
slt   = relev>=1;
nk    = sum(slt); % Number of state Variables based on Eigenvalues

indicator = 1;

if nk>grid.numstates
    if param.overrideEigen
        warning(['The Equilibrium is Locally Indeterminate, critical eigenvalue shifted to: ' num2str(ll(end-grid.numstates))])
        slt = relev>ll(end-grid.numstates);
        nk  = sum(slt);
    else
        error(['No Local Equilibrium Exists, last eigenvalue: ' num2str(ll(end-grid.numstates))])
        %return
    end
    indicator = 0;
    gx = 0;
    hx = 0;
elseif nk<grid.numstates
    if param.overrideEigen
        warning(['No Local Equilibrium Exists, critical eigenvalue shifted to: ' num2str(ll(end-grid.numstates))])
        slt = relev>ll(end-grid.numstates);
        nk  = sum(slt);
    else
        error(['No Local Equilibrium Exists, last eigenvalue: ' num2str(ll(end-grid.numstates))])
        %  return
    end
    indicator = 0;
    gx = 0;
    hx = 0;
end

[s,t,~,Z] = ordqz(s,t,Q,Z,slt);

z21=Z(nk+1:end,1:nk);
z11=Z(1:nk,1:nk);
s11=s(1:nk,1:nk);
t11=t(1:nk,1:nk);

%Checks

% disp(rank(z11));
if indicator > 0
if rank(z11)<nk
    warning('invertibility condition violated')
    indicator = 0;
    gx = 0;
    hx = 0;
else
    z11i=z11\eye(nk);
    gx=real(z21*z11i);
    hx=real(z11*(s11\t11)*z11i);    
end
% toc
end
end
