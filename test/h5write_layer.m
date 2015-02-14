function h5write_layer(fname, l)
    assert(l.bias == 1);
    plist = { 'w', 'dw', 'dw1', 'dw2' };
    for i=1:numel(plist)
        p = plist{i};
        if ~isempty(l.(p))
            h5save(fname, ['/' p], l.(p)(:,2:end));
            h5save(fname, ['/' strrep(p,'w','b')], l.(p)(:,1));
        end
    end

    plist = { 'adagrad', 'nesterov' };
    for i=1:numel(plist)
        p = plist{i};
        if ~isempty(l.(p))
            h5writeatt(fname, '/', p, int32(l.(p)));
        end
    end

    plist = { 'learningRate', 'momentum', 'dropout', 'maxnorm', 'L1', 'L2' };
    for i=1:numel(plist)
        p = plist{i};
        if ~isempty(l.(p))
            h5writeatt(fname, '/', p, single(l.(p)));
        end
    end

    if isa(l, 'relu')
        h5writeatt(fname, '/', 'type', int32(1));
    elseif isa(l, 'soft')
        h5writeatt(fname, '/', 'type', int32(2));
    end
end

