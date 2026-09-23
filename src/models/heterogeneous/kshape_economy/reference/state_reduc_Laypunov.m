
Dindex = grid.Dindex;
Vindex = grid.Vindex;

further_compress_critS = 1e-9;   % threshold for copula eigenvectors
further_compress_critC = 1e-9;   % threshold for value function eigenvectors

[PRightStates, PRightAll, n_states_r, n_controls_r, StateCOVAR, ControlCOVAR] = ...
    compute_reduction(hx, gx, grid, Dindex, Vindex, ...
                      further_compress_critS, further_compress_critC);

% From your dimensions:
n_states       = grid.numstates;          % 251
n_states_endo  = grid.numstates_endo;     % 236
n_states_shock = grid.numstates_shocks;   % 15  (= 251 - 236)
n_controls     = grid.numcontrols;        % 431

% PRightAll has rows ordered as: [states(1:251); controls(1:431)]
% P_ref    has rows ordered as:  [endo_states(1:236); controls(1:431)]
%
% Drop the shock-state rows (rows 237:251 in the state block of PRightAll)
keep_rows = [1:n_states_endo, ...                    % endo state rows: 1:236
             n_states+1 : n_states+n_controls];      % control rows:  252:682

PRightAll_Pref = PRightAll(keep_rows, :);            % 667 x 258  ✓

% Full-state projection
hx_r = PRightStates' * hx * PRightStates;   % (157 x 157)
gx_r = gx * PRightStates;      

% Now the reductions are dimensionally consistent:
P_r = PRightAll_Pref' * P_ref * PRightAll_Pref;      % 258 x 258  ✓
Q_r = PRightAll_Pref' * Q_ref;                       % 258 x 15   ✓
% H_r = H * PRightAll_Pref;                            % n_y x 258  ✓

