msg('Loading dev.mat');
tic;load dev.mat;toc;                   % 2.67s
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
