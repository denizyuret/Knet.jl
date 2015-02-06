function g = copynet(n, xform)
    for i=1:numel(n)
        g{i} = n{i}.copy;
    end
    if nargin < 2 return; end
    for i=1:numel(n)
        names = fieldnames(g{i});
        for j=1:numel(names)
            nj = names{j};
            fj = g{i}.(nj);
            if strcmp(xform, 'cpu')
                if isa(fj, 'gpuArray')
                    g{i}.(nj) = gather(fj);
                end
            elseif strcmp(xform, 'gpu')
                if isa(fj, 'numeric') && (numel(fj) > 1)
                    g{i}.(nj) = gpuArray(fj);
                end
            elseif strcmp(xform, 'rnd')
                if (isa(fj, 'numeric') || isa(fj, 'gpuArray')) && (numel(fj) > 1)
                    g{i}.(nj) = 0.01 * randn(size(fj), 'like', fj);
                end
            end
        end
    end
end
