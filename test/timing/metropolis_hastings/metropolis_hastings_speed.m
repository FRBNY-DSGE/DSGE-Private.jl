% Add paths
path = '~/dsge/cleanCode990/';
dirs = {[path, 'data/']; [path, 'dsgesolv/']; [path, 'estimation/']; [path, 'forecast/']; ...
        [path, 'initialization/']; [path, 'kalman/']; [path, 'toolbox/']; [path, 'plotting/']};
addpath(dirs{:});
spath = 'save/'

% Load variables
load('~/.julia/v0.3/DSGE/test/estimate/metropolis_hastings.mat');
load('~/.julia/v0.3/DSGE/test/estimate/sigscale.mat');
load('mh_initialize_complete.mat')

% Set some parameters
YY = YYall
cc0 = 0.01
cc = 0.09

% Call metropolis_hastings
testing = 1
iterations = 1;
%iterations = 1000;
tic;
for i = 1:iterations;
  metropolis_hastings(testing, spath, params, mspec, YY, YY0, YYcoint0,...
      cc0, cc, sigscale, nobs,nlags,nvar,npara,trspec,pmean,pstdd, ...
      pshape, TTT_old,RRR_old,CCC_old,valid_old,para_mask,coint, ...
      cointadd,cointall,nant, args_nant_antlags, bounds, para_fix, ...
      sigpropdim, sigproplndet, sigpropinv, ...
      randvecs, randvals)
end;
time_elapsed = toc;

disp(time_elapsed);
exit;
