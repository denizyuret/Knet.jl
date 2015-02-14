function h5write_layer(fname, l)
    h5save(fname, '/w', l.w(:,2:end));
    h5save(fname, '/b', l.w(:,1));
    h5save(fname, '/dw', l.dw(:,2:end));
    h5save(fname, '/db', l.dw(:,1));
    if isa(l, 'relu')
        h5writeatt(fname, '/', 'type', int32(1));
    elseif isa(l, 'soft')
        h5writeatt(fname, '/', 'type', int32(2));
    end
end
