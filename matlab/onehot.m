classdef onehot < layer

    properties
        n1	% first n1 rows of input are indices into onehotdict
    end
    
    methods

        function y = forw(l, x, varargin)
        % x(n,m) is the data matrix where each column represents one instance.
        % The first n1 rows of an instance are column indices into the
        % embedding matrix l.w(d,v). y(d*n1+(n-n1),m) should contain the
        % d-dimensional embeddings for the first l.n1 x rows followed by
        % a copy of the remaining n-l.n1 rows.

            d = size(l.w, 1);
            n = size(x, 1);
            m = size(x, 2);
            y = l.zeroslike(d*l.n1 + (n-l.n1), m, l.w);
            for i=1:l.n1
                y((i-1)*d+1:i*d,:) = l.w(:,x(i,:));
            end
            y(l.n1*d+1:end,:) = x(l.n1+1:end,:);
            l.x = x;
        end

        function back(l, dy)
            dy = gather(dy);	% faster on cpu?
            d = size(l.w, 1);
            x = reshape(l.x(1:l.n1,:), 1, []);
            dy = reshape(dy(1:d*l.n1,:), d, []);
            dw = l.zeroslike(size(l.w,1), size(l.w,2), dy);
            for i=1:numel(x)
                dw(:,x(i)) = dw(:,x(i)) + dy(:,i);
            end
            if isa(l.w, 'gpuArray')
                l.dw = gpuArray(dw);
            end
        end

        function l = onehot(varargin)
            l = l@layer(varargin{:});
            if isempty(l.n1)
                error('Must specify n1 (number of onehot entries).');
            end
        end
        
        function y = zeroslike(l, rows, cols, var)
            if isa(var, 'gpuArray')
                y = zeros(rows, cols, classUnderlying(var), 'gpuArray');
            else
                y = zeros(rows, cols, class(var));
            end
        end
    end % methods
end % classdef
