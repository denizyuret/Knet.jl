function predict(varargin)
    batch = 128;
    assert(nargin >= 3);
    fprintf(2, 'Reading... ');
    x = h5read(varargin{1}, '/data');
    for i=2:(numel(varargin)-1)
        net{i-1} = h5read_layer(varargin{i});
    end
    if gpuDeviceCount > 0
        net = copynet(net, 'gpu');
    end
    fprintf(2, 'predicting... ');
    tic; y = gather(forward(net, x, batch)); 
    fprintf(2, '%g seconds\n', toc);
    try delete(varargin{end});
    catch; end;
    h5save(varargin{end}, '/data', y);
end

