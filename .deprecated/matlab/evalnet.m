function [accuracy, loss] = evalnet(net, x, y, batch)
    if nargin < 4
        batch = 1000;
    end
    p = forward(net, x, batch);
    accuracy = net{end}.accuracy(p, y);
    loss = net{end}.loss(p, y);
end
