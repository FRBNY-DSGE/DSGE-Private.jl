function [PRightStates, PRightAll, n_states_r, n_controls_r, StateCOVAR, ControlCOVAR] = ...
    compute_reduction(hx, gx, grid, Dindex, Vindex, ...
                      further_compress_critS, further_compress_critC, shock_sigmas)
% COMPUTE_REDUCTION  BBL-style prior-based model reduction for HANK models.
%
% Mirrors Luetticke's Julia compute_reduction.jl, using your SGU_solver.m
% outputs (hx, gx) directly.
%
% The projection matrices (PRightStates, PRightAll) are computed ONCE at a
% fixed baseline and held fixed during estimation. shock_sigmas only affects
% which subspace is selected, but the subspace is structurally determined by
% hx, so any reasonable positive shock variances (including ones) work fine.
%
% -------------------------------------------------------------------------
% INPUTS (required)
% -------------------------------------------------------------------------
%   hx          [numstates x numstates]
%               Law of motion for states from SGU_solver: S_{t+1} = hx * S_t
%               = lr.LOMstate in Luetticke's Julia code.
%
%   gx          [numcontrols x numstates]
%               State-to-control map from SGU_solver: C_t = gx * S_t
%               = lr.State2Control in Luetticke's Julia code.
%
%   grid        Struct with fields:
%                 .numstates        total states (endogenous + shocks)
%                 .numstates_endo   endogenous states only
%                 .numstates_shocks exogenous shock states
%                 .numcontrols      number of controls
%
%   Dindex      [vector] indices into STATE vector (1..numstates) for the
%               copula DCT coefficients. = sr.indexes.COP in Julia.
%
%   Vindex      [vector] indices into CONTROL vector (1..numcontrols) for
%               value function DCT coefficients (Vm and Vk).
%               = [sr.indexes.Vm; sr.indexes.Vk] in Julia.
%
%   further_compress_critS  scalar (e.g. 1e-5)
%               Keep copula eigenvector i if |lambda_i| > critS * max|lambda|.
%
%   further_compress_critC  scalar (e.g. 1e-5)
%               Same threshold for value function eigenvectors.
%
% -------------------------------------------------------------------------
% INPUTS (optional)
% -------------------------------------------------------------------------
%   shock_sigmas  [numstates_shocks x 1]  (default: ones)
%               Standard deviations of structural shocks in the order they
%               appear in the last grid.numstates_shocks columns of hx.
%
%               IMPORTANT: Because the reduction is run once at a fixed
%               baseline, using ones (identity shock covariance) is fully
%               valid. The subspace selected is structurally determined by
%               hx — i.e., which directions of the idiosyncratic space are
%               reachable from aggregate shocks at all — not by exact shock
%               magnitudes. BBL use the prior mode, but ones works equally
%               well in practice.
%
% -------------------------------------------------------------------------
% OUTPUTS
% -------------------------------------------------------------------------
%   PRightStates  [numstates x n_states_r]
%                 Right projection for the state vector.
%                 Reduced state: S_r_t = PRightStates' * S_t
%
%   PRightAll     [(numstates+numcontrols) x (n_states_r+n_controls_r)]
%                 Right projection for the full [states; controls] vector.
%
%   n_states_r    Number of retained state dimensions.
%   n_controls_r  Number of retained control dimensions.
%
% -------------------------------------------------------------------------
% APPLYING THE REDUCTION
% -------------------------------------------------------------------------
%   shock_cols = (grid.numstates - grid.numstates_shocks + 1):grid.numstates;
%
%   P_r  = PRightStates' * hx * PRightStates;    % [n_states_r x n_states_r]
%   Q_r  = PRightStates' * hx(:, shock_cols);    % [n_states_r x n_shocks]
%   gx_r = gx * PRightStates;                    % [numcontrols x n_states_r]
%
%   % Recover full state from reduced state: S_t ≈ PRightStates * S_r_t
% =========================================================================

n_states   = grid.numstates;
n_controls = grid.numcontrols;
n_shocks   = grid.numstates_shocks;

% -------------------------------------------------------------------------
% Handle optional shock_sigmas — default to ones
% -------------------------------------------------------------------------
if nargin < 8 || isempty(shock_sigmas)
    shock_sigmas = ones(n_shocks, 1);
    fprintf(['compute_reduction: shock_sigmas not supplied. ' ...
             'Using identity covariance (ones).\n' ...
             '  This is valid: the reduction subspace depends on the\n' ...
             '  structure of hx, not the exact shock magnitudes.\n\n']);
end

% -------------------------------------------------------------------------
% STEP 1: Build shock covariance SCov [numstates x numstates]
%
% Shock states occupy the LAST n_shocks positions in the state vector,
% matching H2_aux = H_aux(:, end-numstates_shocks+1:end) in your code.
% -------------------------------------------------------------------------
SCov = zeros(n_states, n_states);
shock_idx = (n_states - n_shocks + 1) : n_states;
for i = 1:n_shocks
    SCov(shock_idx(i), shock_idx(i)) = shock_sigmas(i)^2;
end

% -------------------------------------------------------------------------
% STEP 2: Discrete Lyapunov equation for long-run state covariance
%
%   hx * StateCOVAR * hx' - StateCOVAR + SCov = 0
%
% MATLAB dlyap(A,Q) solves A*X*A' - X + Q = 0.
% Equivalent to lyapd(lr.LOMstate, SCov) in Julia.
% -------------------------------------------------------------------------
StateCOVAR = dlyap(hx, SCov);                        % [numstates x numstates]

% -------------------------------------------------------------------------
% STEP 3: Propagate to control covariance
%
%   C_t = gx * S_t  =>  Var(C_t) = gx * Var(S_t) * gx'
%
% Equivalent to lr.State2Control * StateCOVAR * lr.State2Control' in Julia.
% -------------------------------------------------------------------------
ControlCOVAR = gx * StateCOVAR * gx';
ControlCOVAR = (ControlCOVAR + ControlCOVAR') / 2;  % enforce symmetry

% -------------------------------------------------------------------------
% STEP 4: Eigen-decompose the COPULA block of StateCOVAR
%
% Dindex indexes into the STATE vector (1..numstates).
% -------------------------------------------------------------------------
CovD = StateCOVAR(Dindex, Dindex);
[evecS, evalS_mat] = eig(CovD);
evalS = real(diag(evalS_mat));
keepD = abs(evalS) > max(abs(evalS)) * further_compress_critS;

fprintf('Copula block:   %d / %d eigenvectors retained (critS = %.0e)\n', ...
    sum(keepD), numel(evalS), further_compress_critS);

% -------------------------------------------------------------------------
% STEP 5: Eigen-decompose the VALUE FUNCTION block of ControlCOVAR
%
% Vindex indexes into the CONTROL vector (1..numcontrols).
% ControlCOVAR is (numcontrols x numcontrols), so Vindex used directly.
% -------------------------------------------------------------------------
CovV = ControlCOVAR(Vindex, Vindex);
[evecC, evalC_mat] = eig(CovV);
evalC = real(diag(evalC_mat));
keepV = abs(evalC) > max(abs(evalC)) * further_compress_critC;

fprintf('Value fn block: %d / %d eigenvectors retained (critC = %.0e)\n', ...
    sum(keepV), numel(evalC), further_compress_critC);

% -------------------------------------------------------------------------
% STEP 6: Build PRightStates  [numstates x n_states_r]
%
% Identity matrix with Dindex block replaced by eigenvectors.
% Columns for dropped eigenvectors (keepD = false) are removed.
% Equivalent to n_par.PRightStates in Julia.
% -------------------------------------------------------------------------
PRightStates_aux = eye(n_states);
PRightStates_aux(Dindex, Dindex) = evecS;

keep_states = true(n_states, 1);
keep_states(Dindex(~keepD)) = false;

PRightStates = PRightStates_aux(:, keep_states);     % [numstates x n_states_r]
n_states_r   = sum(keep_states);

% -------------------------------------------------------------------------
% STEP 7: Build PRightAll  [ntotal x ntotal_r]
%
% Full stacked [states; controls] projection.
% Vindex shifts by numstates to address the control block of the full vector.
% Equivalent to n_par.PRightAll in Julia.
% -------------------------------------------------------------------------
n_total     = n_states + n_controls;
Vindex_full = n_states + Vindex;                     % shift into full-vector space

PRightAll_aux = eye(n_total);
PRightAll_aux(Dindex,      Dindex)      = evecS;
PRightAll_aux(Vindex_full, Vindex_full) = evecC;

keep_all = true(n_total, 1);
keep_all(Dindex(~keepD))      = false;
keep_all(Vindex_full(~keepV)) = false;

PRightAll    = PRightAll_aux(:, keep_all);           % [ntotal x ntotal_r]
n_controls_r = n_controls - sum(~keepV);

% -------------------------------------------------------------------------
% Summary
% -------------------------------------------------------------------------
fprintf('\nReduction summary:\n');
fprintf('  States:   %d  ->  %d\n', n_states,   n_states_r);
fprintf('  Controls: %d  ->  %d\n', n_controls, n_controls_r);
fprintf('  Total:    %d  ->  %d  (%.1f%% reduction)\n', ...
    n_total, n_states_r + n_controls_r, ...
    100 * (1 - (n_states_r + n_controls_r) / n_total));

end
