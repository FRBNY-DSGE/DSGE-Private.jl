% Add paths
function [] = metropolis_hastings_speed(node)

path = '~/dsge/workspace/dirtyCode990/';
dirs = {[path, 'data/']; [path, 'dsgesolv/']; [path, 'estimation/']; [path, 'forecast/']; ...
        [path, 'initialization/']; [path, 'kalman/']; [path, 'toolbox/']; [path, 'plotting/']};
addpath(dirs{:});
spath = 'save/';

% Load variables
load('~/.julia/v0.3/DSGE/test/estimate/metropolis_hastings.mat', 'randvecs', 'randvals');
load('pre_mh_state.mat');

% Set some parameters
cc0 = 0.01;
cc = 0.09;

% Call metropolis_hastings

testing = 1;

tic;
metropolis_hastings(testing, spath, params, mspec, YY, YY0, YYcoint0,...
    cc0, cc, sigscale, nobs,nlags,nvar,npara,trspec,pmean,pstdd, ...
    pshape,TTT, RRR,CCC,valid,para_mask,coint, ...
    cointadd,cointall,nant, args_nant_antlags, bounds, para_fix, ...
    sigpropdim, sigproplndet, sigpropinv, ...
    randvecs, randvals);
time_elapsed = toc;

%% As a check, read in the parameter matrix to make sure it matches

% !! This is directly copied from metropolis_hastings. Must change in order
% to read in files correctly! Bad form but don't want to mess with the
% function.

nblocks = 22;
nsim = 100;
ntimes = 5;
nburn = nsim*2;

num1 = nblocks*nsim*npara*4;            % number of bites per file
numb1 = nburn*npara*4;                  % number of bites to discard

% read in saved parameter draws
infile_params = [spath,'/params'];
fid = fopen(infile_params,'r');
status = fseek(fid,numb1,'bof');


% Reshape parameter draws into a matrix so can calculate covariance
theta = [];
while ( ftell(fid) < num1 )
    theta_add = fread(fid,[npara,nsim],'single')';
    theta = [theta; theta_add];
end

% Save as a Matfile
save('save/parameter_check','theta','-single');
fclose('all');

disp(['node: ', node, ' seconds: ', num2str(time_elapsed)]);
exit;

end
