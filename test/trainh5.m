function trainh5(varargin)
    DEFAULT_LEARNING_RATE=0.01;
    for i=1:2:numel(varargin) o.(varargin{i}) = varargin{i+1}; end
    assert(isfield(o,'x'));
    assert(isfield(o,'y'));
    assert(isfield(o,'out'));
    assert(isfield(o,'net'));
    if ~isfield(o,'learningRate') o.learningRate = DEFAULT_LEARNING_RATE; end
    fprintf(2, 'Reading... ');
    x = h5read(o.x, '/data');
    if ~isfield(o,'batch') o.batch = size(x,2); end
    ymat = h5read(o.y, '/data');
    [~, yvec] = max(ymat);
    o.net = strsplit(o.net, ',');
    plist = {'learningRate','adagrad','nesterov','momentum','dropout','maxnorm','L1','L2'};
    for i=1:numel(o.net) 
        net{i} = h5read_layer(o.net{i}); 
        for j=1:numel(plist)
            p = plist{j};
            if isfield(o, p)
                net{i}.(p) = o.(p);
            end
        end
    end
    if gpuDeviceCount > 0
        net = copynet(net, 'gpu');
    end
    fprintf(2, 'train... ');
    if (isfield(o,'iters'))
        x = x(:,1:o.iters*o.batch);
        yvec = yvec(1:o.iters*o.batch);
    end
    tic; train(net, x, yvec, 'batch', o.batch, 'epochs', 1);
    fprintf(2, '%g seconds... saving... ', toc);
    net = copynet(net, 'cpu');
    for i=1:numel(net)
        fname = sprintf('%s%d.h5', o.out, i);
        if exist(fname, 'file')
            try delete(fname); catch; end;
        end
        h5write_layer(fname, net{i});
    end
    fprintf(2, 'done\n');
end

