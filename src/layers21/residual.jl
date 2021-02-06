export Residual


struct Residual
    layers
    activation
    function Residual(layers...; activation=nothing)
        if length(layers) == 0
            error("Usage: Residual(l1, [l2=identity, ls...]; activation)")
        elseif length(layers) == 1
            layers = Any[layers[1], identity]
        else
            layers = Any[layers...]
        end
        new(layers, activation)
    end
end


function (r::Residual)(x)
    y = r.layers[1](x)
    for i in 2:length(r.layers)
        y = y + r.layers[i](x)
    end
    r.activation === nothing ? y : r.activation.(y)
end
