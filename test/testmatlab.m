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

% testidx = []

if any(testidx == 0) %%% Load data
    path('../matlab', path);
    msg('Loading dev.h5');
    tic;
    dev.x = h5read('dev.h5', '/x');
    dev.ytest = h5read('dev.h5', '/y');
    [~,dev.ylabels] = max(h5read('dev.h5', '/ygold'));
    dev.w1 = [h5read('dev.h5', '/b1'), h5read('dev.h5', '/w1')];
    dev.w2 = [h5read('dev.h5', '/b2'), h5read('dev.h5', '/w2')];
    toc;  % 2.39s
    msg('Creating cpu net');
    l1=relu('w', dev.w1, 'bias',1);
    l2=soft('w', dev.w2, 'bias',1);
    net0={l1,l2};
    x10k=dev.x(:,1:10000);
    y10k=dev.ylabels(:,1:10000);
    x100=dev.x(:,1:100);
    y100=dev.ylabels(:,1:100);
end

if any(testidx == 1) %%% forward
    net = copynet(net0, 'cpu');
    msg('CPU forward fullbatch');
    tic;cpu_y = forward(net, dev.x);toc;    % 17.76s, -0.8778, -14.1189, 6.1067
    disp(cpu_y(:,1));
    msg('CPU forward batch=1000');
    tic;cpu_y2 = forward(net, dev.x, 1000);toc; % 21.07s
    assert(isequal(cpu_y, cpu_y2))
    gnet = copynet(net0, 'gpu');
    msg('GPU forward batch=10000');
    tic;gpu_y = gather(forward(gnet, dev.x, 10000));toc; % 2.58s
    msg('GPU forward batch=1000');
    tic;gpu_y2 = gather(forward(gnet, dev.x, 1000));toc; % 2.43s
    assert(isequal(gpu_y, gpu_y2))
    msg('GPU forward batch=100');
    tic;gpu_y3 = gather(forward(gnet, dev.x, 100));toc; % 4.32s
    assert(isequal(gpu_y, gpu_y3))
    msg('CPU-dev.y maxdiff=%g', max(abs(dev.ytest(:)-cpu_y(:)))) % 3.05176e-5
    msg('GPU-dev.y maxdiff=%g', max(abs(dev.ytest(:)-gpu_y(:)))) % 2.5177e-4
    msg('GPU-CPU maxdiff=%g', max(abs(gpu_y(:)-cpu_y(:)))) % 2.5177e-4
    msg('Saving forward.h5');
    delete('forward.h5');
    h5save('forward.h5', '/cpu_y', cpu_y);
    h5save('forward.h5', '/gpu_y', gpu_y);
    %% clear gnet net
end

if any(testidx == 2) %%% forwback10k
    net = copynet(net0, 'cpu');
    msg('CPU forwback 10k');
    tic; forwback(net, x10k, y10k);toc;    % 6.95s
    gnet = copynet(net0, 'gpu');
    msg('GPU forwback 10k');
    tic; forwback(gnet, x10k, y10k);toc;    % 0.73s
    cnet = copynet(gnet, 'cpu');
    msg('dw1 maxdiff=%g', max(abs(net{1}.dw(:) - cnet{1}.dw(:)))); % 2.68819e-05
    msg('dw2 maxdiff=%g', max(abs(net{2}.dw(:) - cnet{2}.dw(:)))); % 4.09782e-08
    msg('Saving forwback10k.h5');
    delete('forwback10k.h5');
    h5save('forwback10k.h5', '/dw1', cnet{1}.dw(:,2:end));
    h5save('forwback10k.h5', '/dw2', cnet{2}.dw(:,2:end));
    h5save('forwback10k.h5', '/db1', cnet{1}.dw(:,1));
    h5save('forwback10k.h5', '/db2', cnet{2}.dw(:,1));
    %% clear net gnet cnet
end

if any(testidx == 3) %%% gradient check
    net = copynet(net0, 'cpu');
    msg('CPU gradient 100');
    tic; gradient(net, x100, y100, 100),toc;    % 19.60s, 3.1335e-10
    gnet = copynet(net0, 'gpu');
    msg('GPU gradient 100');
    tic; gradient(gnet, x100, y100, 100),toc;    % 3.20s, 2.6866e-10
end


if any(testidx == 4) %%% single epoch train with batch=100
    net = copynet(net0, 'cpu');
    for l=1:numel(net) net{l}.learningRate=0.01; end
    gnet = copynet(net, 'gpu');
    msg('CPU train one epoch: batch=100 lr=0.01');
    tic;train(net, dev.x, dev.ylabels, 'epochs', 1, 'batch', 100);toc;
    msg('GPU train one epoch: batch=100 lr=0.01');
    tic;train(gnet, dev.x, dev.ylabels, 'epochs', 1, 'batch', 100);toc;
    cnet = copynet(gnet, 'cpu');
    msg('dw1 maxdiff=%g', max(abs(net{1}.dw(:) - cnet{1}.dw(:)))); % 161.71s
    msg('dw2 maxdiff=%g', max(abs(net{2}.dw(:) - cnet{2}.dw(:)))); % 20.27s
    msg('Saving train01.h5');
    delete('train01.h5');
    h5save('train01.h5', '/dw1', cnet{1}.dw(:,2:end));
    h5save('train01.h5', '/dw2', cnet{2}.dw(:,2:end));
    h5save('train01.h5', '/db1', cnet{1}.dw(:,1));
    h5save('train01.h5', '/db2', cnet{2}.dw(:,1));
end

%%% momentum
%%% nesterov
%%% adagrad
%%% maxnorm
%%% L1
%%% L2
%%% dropout forw
%%% dropout back
%%% dropout gradient check
%%% all with no-bias
%%% all with cpu/gpu
%%% compare with caffe
