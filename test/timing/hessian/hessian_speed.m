function [] = hessian_speed(node)

% Add paths
path = '/data/dsge_data_dir/proc/dsge/pearl/cleanCode990/';
dirs = {[path, 'data/']; [path, 'dsgesolv/']; [path, 'estimation/']; [path, 'forecast/']; ...
        [path, 'initialization/']; [path, 'kalman/']; [path, 'toolbox/']; [path, 'plotting/']};
addpath(dirs{:});

% Load variables
load('hessian.mat');

tic;
hessizero('objfcndsge', [params, para_mask], 0, ...
    YY, YY0, nobs, nlags, nvar, mspec, npara, trspec, pmean, pstdd, pshape, para_mask, ...
    para_fix, marglh, coint, cointadd, cointall, YYcoint0, MIN, args_nant_antlags{:});
time_elapsed = toc;

disp(['node: ', node, ' seconds: ',  num2str(time_elapsed)]);
exit;

end