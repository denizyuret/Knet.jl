function dx = forwback(net, x, y)
    for l=1:numel(net)
        x = net{l}.forw(x);
    end
    for l=numel(net):-1:2
        y = net{l}.back(y);
    end
    net{1}.back(y);
end
