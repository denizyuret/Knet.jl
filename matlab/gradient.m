function [diff, cmp, li, dnet] = gradient(net, x, y, tries, epsilon)
% checks the gradient calculation of the network by comparing it to
% numerical estimates of the gradient.

    if nargin < 5 epsilon = 1e-5; end
    if nargin < 4 tries = 10; end

    dnet = dblnet(net);
    dblx = double(x);
    dbly = double(y);

    netforw(dnet, dblx, dbly);
    netback(dnet, dbly);
    % at this point we have net{l}.dx contain the backprop gradients
    % now let's compare with numerical estimates

    cmp = []; li = [];

    for t=1:tries
        l = randi(numel(dnet));
        if rand > 0.5 % pick a nonzero derivative at least half the time
            nz = find(dnet{l}.dw ~= 0);
            i = nz(randi(numel(nz)));
        else
            i = randi(numel(dnet{l}.dw));
        end
        wi_save = dnet{l}.w(i);
        dnet{l}.w(i) = wi_save + epsilon;
        loss2 = netforw(dnet, dblx, dbly);
        dnet{l}.w(i) = wi_save - epsilon;
        loss1 = netforw(dnet, dblx, dbly);
        dnet{l}.w(i) = wi_save;
        cmp = [cmp; dnet{l}.dw(i), (loss2-loss1)/(2 * epsilon)];
        li = [li; l i];
    end

    diff = norm(cmp(:,2)-cmp(:,1))/norm(cmp(:,2)+cmp(:,1));
end

function loss = netforw(net, x, y)
    s = rng(1); % to make sure we get the same dropout
    L = numel(net);
    a = x;
    for l=1:L
        if net{l}.dropout
            net{l}.xmask = (rand(size(a)) > net{l}.dropout);
            a = a .* net{l}.xmask * (1/(1-net{l}.dropout));
        end
        a = net{l}.forw(a);
    end
    loss = net{L}.loss(a, y);
    rng(s);
end

function netback(net, y)
    d = y;
    for l=numel(net):-1:2
        d = net{l}.back(d);
    end
    net{1}.back(d);
end

function dnet = dblnet(net)
    for i=1:numel(net)
        dnet{i} = net{i}.copy;
        dnet{i}.w = double(dnet{i}.w);
    end
end
