% Add paths
path = '/data/dsge_data_dir/proc/dsge/pearl/cleanCode990/';
dirs = {[path, 'data/']; [path, 'dsgesolv/']; [path, 'estimation/']; [path, 'forecast/']; ...
        [path, 'initialization/']; [path, 'kalman/']; [path, 'toolbox/']; [path, 'plotting/']};
addpath(dirs{:});

% Load variables
load('gensys.mat');

% Call gensys
iterations = 10;
tic;
for i = 1:iterations;
  gensys(G0, G1, C, PSI, PIE, 1+1e-6);
end;
time_elapsed = toc;

disp([num2str(iterations), ' calls to gensys executed in ', num2str(time_elapsed), ' seconds']);