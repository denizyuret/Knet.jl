export Sequential
import Base: getindex, setindex!, length, size, firstindex, lastindex, push!, pop!, pushfirst!, popfirst!, append!, prepend!


struct Sequential
    layers
    name                        # only used for display
    Sequential(ls...; name=nothing) = new(Any[ls...], name)
end


function (s::Sequential)(x)
    for l in s.layers
        x = l(x)
    end
    return x
end


# AbstractArray interface
function getindex(s::Sequential, i)
    if length(i) == 1
        getindex(s.layers, i)
    else
        name=(s.name === nothing ? nothing : "$(s.name)[$i]")
        Sequential(getindex(s.layers, i)...; name)
    end
end

setindex!(s::Sequential, v, i) = setindex!(s.layers, v, i)
length(s::Sequential) = length(s.layers)
size(s::Sequential) = size(x.layers)
firstindex(s::Sequential) = 1
lastindex(s::Sequential) = length(s)

# Dequeue interface
push!(s::Sequential, x...) = (push!(s.layers, x...); s)
pop!(s::Sequential) = pop!(s.layers)
pushfirst!(s::Sequential, x...) = (pushfirst!(s.layers, x...); s)
popfirst!(s::Sequential) = popfirst!(s.layers)
append!(s::Sequential, t::Sequential) = (append!(s.layers, t.layers); s)
append!(s::Sequential, t) = (append!(s.layers, t); s)
prepend!(s::Sequential, t::Sequential) = (prepend!(s.layers, t.layers); s)
prepend!(s::Sequential, t) = (prepend!(s.layers, t); s)

# TODO: iterator interface
