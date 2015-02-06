% dev.mat contains test data:
%
%     y: [1x76834 single]
%  cost: [3x76834 single]
%     x: [1326x76834 single]
%     z: [1x76834 single]
% score: [3x76834 single]
%  pred: {1x1700 cell}
%    w1: [20000x1327 single]
%    w2: [3x20001 single]
%
% Note that the first column of w is for bias.

if 0 
msg('Loading dev.mat');
tic;load dev.mat;toc;                   % 2.67s
end

if 0 %%% CPU forw
msg('Creating cpu net');
l1=relu('w', dev.w1, 'bias',1);
l2=soft('w', dev.w2, 'bias',1);
net={l1,l2};
msg('CPU forward fullbatch');
tic;cpu_y = forward(net, dev.x);toc;    % 17.76s
save cpu_y cpu_y
msg('CPU forward batch=1000');
tic;cpu_y2 = forward(net, dev.x, 1000);toc; % 22.98s
assert(isequal(cpu_y, cpu_y2))
msg('CPU-dev.score maxdiff=%g', max(abs(dev.score(:)-cpu_y(:)))) % 3.05e-5
end

if 0 %%% GPU forw
msg('Creating gpu net');
gnet = copynet(net, 'gpu');
msg('GPU forward batch=10000');
tic;gpu_y = forward(gnet, dev.x, 10000);toc; % 2.72s
save gpu_y gpu_y
msg('GPU forward batch=1000');
tic;gpu_y2 = forward(gnet, dev.x, 1000);toc; % 2.51s
assert(isequal(gpu_y, gpu_y2))
msg('GPU forward batch=100');
tic;gpu_y3 = forward(gnet, dev.x, 100);toc; % 4.32s
assert(isequal(gpu_y, gpu_y3))
msg('GPU-CPU maxdiff=%g', max(abs(gpu_y(:)-cpu_y(:)))) % 2.5e-4
end

