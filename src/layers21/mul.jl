export Mul

struct Mul
    blocks
    activation
    function Mul(blocks...; activation=nothing)
        new(blocks, activation)
    end
end

function (r::Mul)(x)
    y = r.blocks[1](x)
    for i in 2:length(r.blocks)
        y = y .* r.blocks[i](x)
    end
    r.activation === nothing ? y : r.activation(y)
end
