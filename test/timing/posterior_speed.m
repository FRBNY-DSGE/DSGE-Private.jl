% Add paths
path = '/data/dsge_data_dir/proc/dsge/pearl/cleanCode990/';
dirs = {[path, 'data/']; [path, 'dsgesolv/']; [path, 'estimation/']; [path, 'forecast/']; ...
        [path, 'initialization/']; [path, 'kalman/']; [path, 'toolbox/']; [path, 'plotting/']};
addpath(dirs{:});

% Load variables
load('posterior.mat')

% Call objfcndsge
iterations = 1000;
tic;
for i = 1:iterations;
  objfcndsge(para, YY, YY0, nobs, nlags, nvar, mspec, npara, trspec, pmean, ...
      pstdd, pshape, para_mask, para_fix, marglh, coint, cointadd, cointall, ...
      YYcoint0, 0, args_nant_antlags{:});
end
time_elapsed = toc;
disp([num2str(iterations), ' calls to objfcndsge executed in ', num2str(time_elapsed), ' seconds']);
exit;