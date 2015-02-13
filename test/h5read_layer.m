% TODO: Finish implementing the other fields
% TODO: Finish implementing h5write_layer

function layer = h5read_layer(fname)
    w = h5read(fname, '/w');
    b = h5read(fname, '/b');
    type = h5readatt(fname, '/', 'type');
    if type == 1
        layer = relu('w', [b,w], 'bias', 1);
    elseif type == 2
        layer = soft('w', [b,w], 'bias', 1);
    end
end
