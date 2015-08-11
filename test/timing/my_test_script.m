% Skeleton code to do matlab timing test
% Use tic...toc construct to be able to control number of iterations. Subject
% to change if desired.
%
% Created 2015-08-10 MJS
addpath('/path/to/gensys');
load('args_for_gensys.mat');
nIterations=100000;

tic
for i=1:nIterations
  [out...] = gensys(args...);
end

fprintf(1, '%9.2f', toc);
