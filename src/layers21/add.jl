export Add

struct Add
    blocks
    activation
    function Add(blocks...; activation=nothing)
        new(blocks, activation)
    end
end

function (r::Add)(x)
    y = r.blocks[1](x)
    for i in 2:length(r.blocks)
        y = y .+ r.blocks[i](x)
    end
    r.activation === nothing ? y : r.activation(y)
end
