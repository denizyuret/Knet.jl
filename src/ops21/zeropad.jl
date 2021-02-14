export zeropad
using AutoGrad


# cut-and-paste method: see Knet/prof/zeropad for alternatives and timing

function zeropad(x, p; y=nothing, fillzero=true)
    p = padtuple(p, size(x))
    n = ndims(x)
    d = ntuple(i->(i < n-1 ? size(x,i)+sum(p[i]) : size(x,i)), n)
    c = ntuple(i->(i < n-1 ? (p[i][1]+1:p[i][1]+size(x,i)) : Colon()), n)
    y === nothing ? y = similar(x,d) : @assert typeof(y)===typeof(x) && size(y)==d
    fillzero && fill!(y, 0)
    y[c...] .= x
    return y
end


# Gradients

function ∇zeropad(dy, p)
    p = padtuple(p, size(dy))
    n = ndims(dy)
    c = ntuple(i->(i < n-1 ? (p[i][1]+1:size(dy,i)-p[i][2]) : Colon()), n)
    dy[c...]
end

@primitive1 zeropad(x,p...;o...),dy,y  ∇zeropad(dy,p...)
@primitive1 ∇zeropad(dy,p...),ddx,dx  zeropad(ddx,p...)


# padtuple turns Int/Vector/Tuple to ((a,b),(c,d)) format

# Check if the right ((a,b),(c,d)) format
padtuple(p::NTuple{P,NTuple{2,Int}}, d::Dims{D}) where {P,D} = (@assert P == D-2; p)

# Main conversions
padtuple(p::Int, d::Dims{D}) where {D} = ntuple(i->(p,p), D-2)
padtuple(p::Vector{Int}, d::Dims{D}) where {D} = (@assert length(p) == D-2; ntuple(i->(p[i],p[i]), D-2))
padtuple(p::NTuple{P,Int}, d::Dims{D}) where {P,D} = (@assert P == D-2; ntuple(i->(p[i],p[i]), P))

# Handle other integer types
padtuple(p::Integer, d) = padtuple(Int(p), d)
padtuple(p::Vector{<:Integer}, d) = padtuple(Int.(p), d)
padtuple(p::NTuple{P,<:Integer}, d) where {P} = padtuple(Int.(p), d)
padtuple(p::NTuple{P,NTuple{2,<:Integer}}, d) where {P} = padtuple(ntuple(i->Int.(p[i]),P), d)


# symmetric padding can be handled by the convolution
symmetric(p::Integer) = true
symmetric(p::Vector{<:Integer}) = true
symmetric(p::NTuple{P,<:Integer}) where {P} = true
symmetric(p::NTuple{P,NTuple{2,<:Integer}}) where {P} = all(x[1]==x[2] for x in p)
