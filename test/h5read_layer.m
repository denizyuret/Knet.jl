% TODO: Finish implementing the other fields
% TODO: Finish implementing h5write_layer

function layer = h5read_layer(fname)
    info = h5info(fname, '/');
    w = h5read(fname, '/w');
    if ismember('b', {info.Datasets.Name}) 
        b = h5read(fname, '/b'); 
    end
    if (ismember('fy', {info.Attributes.Name}))
        if (strcmp(h5readatt(fname, '/', 'fy'), 'relu'))
            layer = relu('w', [b,w], 'bias', 1);
        else
            layer = soft('w', [b,w], 'bias', 1);
        end
    else
        layer = soft('w', [b,w], 'bias', 1);
    end
end
