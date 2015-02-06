classdef layer < matlab.mixin.Copyable
    
    properties
        w               % weight matrix
        bias            % boolean: whether first column of w is for bias

        learningRate    % learning rate
        momentum        % momentum
        adagrad         % boolean indicating adagrad trick
        nesterov        % boolean indicating nesterov trick
        dropout         % probability of dropping inputs
        maxnorm         % parameter for maxnorm regularization
        L1              % parameter for L1 regularization
        L2              % parameter for L2 regularization

        dw1             % moving average of gradients for momentum
        dw2             % sum of squared gradients for adagrad
    end
    properties (Transient = true)
        dw              % gradient of parameters
        x,y		% last input and output
        xmask 		% input mask for dropout
        xones           % row of ones to use for bias
    end
    methods
        
        function y = fforw(l, y)
        % fforw is the activation function, sigmoid by default, override to
        % get other types of units.
 
            y(:) = 1 ./ (1 + exp(-y));
        end

        function dy = fback(l, dy)
        % fback multiplies its input with the derivative of the
        % activation function fforw.

            dy(:) = dy .* l.y .* (1 - l.y);
        end


        function y = forw(l, x)
        % forw transforms input x to output y using the linear
        % transformation followed by the activation function. 

            if l.bias
                if size(x, 2) ~= size(l.xones, 2)
                    l.xones = 1 + 0 * x(1,:);
                end
                l.x = [l.xones; x];
            else
                l.x = x;
            end
            l.y = l.fforw(l.w * l.x);
            y = l.y;
        end

        function dx = back(l, dy)
        % back transforms the loss gradient with respect to output
        % dy to the gradient with respect to input dx.

            dy = l.fback(dy);
            if isempty(l.dw)
                l.dw = 0 * l.w;
            end
            l.dw(:) = dy * l.x';
            if nargout > 0
                dx = l.w' * dy;
                if l.bias
                    dx = dx(2:end,:);
                end
                if ~isempty(l.xmask)
                    dx(:) = dx .* l.xmask * (1/(1-l.dropout));
                    l.xmask = [];
                end
            end
        end

        function x = drop(l, x)
        % Drop each element of the input x with probability
        % l.dropout.
            l.xmask = (l.randlike(x) > l.dropout);
            x(:) = x .* l.xmask * (1/(1-l.dropout));
        end

        function y = randlike(l, x)
            if isa(x, 'gpuArray')
                y = gpuArray.rand(size(x), classUnderlying(x));
            else
                y = rand(size(x), class(x));
            end
        end

        function update(l)
            if l.L1
                l.dw(:) = l.dw + l.L1 * sign(l.w);
            end
            if l.L2
                l.dw(:) = l.dw + l.L2 * l.w;
            end
            if l.adagrad
                if ~isempty(l.dw2)
                    l.dw2(:) = l.dw .* l.dw + l.dw2;
                else
                    l.dw2 = l.dw .* l.dw;
                end
                l.dw(:) = l.dw ./ (1e-8 + sqrt(l.dw2));
            end
            if ~isempty(l.learningRate)
                l.dw(:) = l.learningRate * l.dw;
            end
            if l.momentum
                if ~isempty(l.dw1)
                    l.dw1(:) = l.dw + l.momentum * l.dw1;
                else
                    l.dw1 = l.dw;
                end
                if l.nesterov
                    l.dw(:) = l.dw + l.momentum * l.dw1;
                else
                    l.dw(:) = l.dw1;
                end
            end

            l.w(:) = l.w - l.dw;

            if l.maxnorm
                norms = sqrt(sum(l.w.^2, 2));
                if any(norms > l.maxnorm)
                    scale = min(l.maxnorm ./ norms, 1);
                    l.w(:) = bsxfun(@times, l.w, scale);
                end
            end
        end

        function l = layer(varargin)
            for i=1:2:numel(varargin)
                l.(varargin{i}) = varargin{i+1};
            end
        end

    end % methods
end % classdef
