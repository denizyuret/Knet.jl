# Implementation is taken from [Flux.jl](https://github.com/FluxML/Flux.jl)
"""
    Chain(layers...)
Chain multiple layers / functions together, so that they are called in sequence
on a given input.
```julia
m = Chain(x -> x^2, x -> x+1)
m(5) == 26
m = Chain(Dense(input=10, output=5), Dense(input=5, output=2))
x = rand(10)
m(x) == m[2](m[1](x))
```
`Chain` also supports indexing and slicing, e.g. `m[2]` or `m[1:end-1]`.
`m[1:3](x)` will calculate the output of the first three layers.
"""
struct Chain{T<:Tuple}
   layers::T
   Chain(xs...) = new{typeof(xs)}(xs)
end

children(c::Chain) = c.layers
mapchildren(f, c::Chain) = Chain(f.(c.layers)...)

applychain(::Tuple{}, x) = x
applychain(fs::Tuple, x) = applychain(Base.tail(fs), first(fs)(x))

(c::Chain)(x) = applychain(c.layers, x)

Base.getindex(c::Chain, i::AbstractArray) = Chain(c.layers[i]...)
Base.getindex(c::Chain, i::Integer) = c.layers[i]
Base.getindex(c::Chain, ::Colon) = c
Base.length(c::Chain) = length(c.layers)
Base.iterate(c::Chain) = Base.iterate(c.layers)
Base.iterate(c::Chain, state) = Base.iterate(c.layers, state)

function Base.show(io::IO, c::Chain)
  print(io, "Chain(")
  join(io, c.layers, ", ")
  print(io, ")")
end
