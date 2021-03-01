export Residual


struct Residual
    blocks
    activation
    function Residual(blocks...; activation=nothing)
        if length(blocks) == 0
            error("Usage: Residual(l1, [l2=identity, ls...]; activation)")
        elseif length(blocks) == 1
            blocks = Any[blocks[1], identity]
        else
            blocks = Any[blocks...]
        end
        new(blocks, activation)
    end
end


function (r::Residual)(x)
    y = r.blocks[1](x)
    for i in 2:length(r.blocks)
        y = y + r.blocks[i](x)
    end
    r.activation === nothing ? y : r.activation(y)
end
