function [] = kalman_speed(node)
% Add paths
path = '/data/dsge_data_dir/proc/dsge/pearl/cleanCode990/';
dirs = {[path, 'data/']; [path, 'dsgesolv/']; [path, 'estimation/']; [path, 'forecast/']; ...
        [path, 'initialization/']; [path, 'kalman/']; [path, 'toolbox/']; [path, 'plotting/']};
addpath(dirs{:});

% Load variables
load('kalman.mat');

% Call Kalman filter
iterations = 1000;
tic;
for i = 1:iterations;
  [L, zend, Pend] = kalcvf2NaN(data, lead, a, F, b, H, variance, z0, vz0);
end;
time_elapsed = toc;

disp(['node: ', node, ' seconds: ',  num2str(time_elapsed)]);
exit;

end