export Block
import Base: getindex, setindex!, length, size, firstindex, lastindex, push!, pop!, pushfirst!, popfirst!, append!, prepend!


struct Block
    layers
    name                        # only used for display
    Block(ls...; name=nothing) = new(Any[ls...], name)
end


function (s::Block)(x)
    for l in s.layers
        x = l(x)
    end
    return x
end


# AbstractArray interface
function getindex(s::Block, i)
    if length(i) == 1
        getindex(s.layers, i)
    else
        name=(s.name === nothing ? nothing : "$(s.name)[$i]")
        Block(getindex(s.layers, i)...; name)
    end
end

setindex!(s::Block, v, i) = setindex!(s.layers, v, i)
length(s::Block) = length(s.layers)
size(s::Block) = size(x.layers)
firstindex(s::Block) = 1
lastindex(s::Block) = length(s)

# Dequeue interface
push!(s::Block, x...) = (push!(s.layers, x...); s)
pop!(s::Block) = pop!(s.layers)
pushfirst!(s::Block, x...) = (pushfirst!(s.layers, x...); s)
popfirst!(s::Block) = popfirst!(s.layers)
append!(s::Block, t::Block) = (append!(s.layers, t.layers); s)
append!(s::Block, t) = (append!(s.layers, t); s)
prepend!(s::Block, t::Block) = (prepend!(s.layers, t.layers); s)
prepend!(s::Block, t) = (prepend!(s.layers, t); s)

# TODO: iterator interface
