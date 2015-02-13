function predict(varargin)
    batch = 128;
    assert(nargin >= 3);
    x = h5read(varargin{1}, '/data');
    for i=2:(numel(varargin)-1)
        net{i-1} = h5read_layer(varargin{i});
    end
    net = copynet(net, 'gpu');
    tic; y = gather(forward(net, x, batch)); 
    fprintf(2, '%g seconds\n', toc);
    try delete(varargin{end});
    catch; end;
    h5save(varargin{end}, '/data', y);
end

