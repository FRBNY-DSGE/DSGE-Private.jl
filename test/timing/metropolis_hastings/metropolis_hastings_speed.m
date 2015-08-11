% Add paths
path = '~/dsge/cleanCode990/';
dirs = {[path, 'data/']; [path, 'dsgesolv/']; [path, 'estimation/']; [path, 'forecast/']; ...
        [path, 'initialization/']; [path, 'kalman/']; [path, 'toolbox/']; [path, 'plotting/']};
addpath(dirs{:});

% Load variables
load('~/.julia/v0.3/DSGE/test/estimate/metropolis_hastings.mat');
load('~/.julia/v0.3/DSGE/test/estimate/sigscale.mat');
load('~/.julia/v0.3/DSGE/test/estimate/mh_initialize.mat') ÐI#ÐI

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
  metropolis_hastings(testing)
end;
time_elapsed = toc;

disp(time_elapsed);
exit;
