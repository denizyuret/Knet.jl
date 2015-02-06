classdef relu < layer
    methods

        function y = fforw(l, y)
            y(:) = y .* (y > 0);
        end

        function dy = fback(l, dy)
            dy(:) = dy .* (l.y > 0);
        end

        function l = relu(varargin)
            l = l@layer(varargin{:});
        end

    end
end
