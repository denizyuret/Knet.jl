classdef soft < layer
    methods

        function y = fforw(l, y)
        % Let's return the linear output (i.e. unnormalized log
        % probabilities) here for speed, do softmax in fback.
        end

        function dy = fback(l, dy)
        % softmax is the last layer and takes a vector of correct
        % classes as dy to start backprop.
            m = numel(dy);
            dy = full(sparse(double(gather(dy)), 1:m, 1, size(l.y,1), m));
            dy(:) = (softmax(gather(l.y)) - dy) / m;
        end

        function nll = loss(l, y, labels)
            probs = softmax(gather(y));
            py = probs(sub2ind(size(probs), labels, 1:numel(labels)));
            nll = -mean(log(max(py, realmin(class(py)))));
        end

        function acc = accuracy(l, y, labels)
            [~,maxp] = max(y);
            acc = mean(maxp == labels);
        end

        function l = soft(varargin)
            l = l@layer(varargin{:});
        end

    end
end
