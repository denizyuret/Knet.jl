import Base.Broadcast: BroadcastStyle, Style, broadcastable, broadcasted, Broadcasted, materialize!
import Base: copyto!
using AutoGrad: Value

## Broadcasting:

# Both f.(x...) and broadcast(f,x...) turn into materialize(broadcasted(::BroadcastStyle,f,x...)).

# Any call involving KnetArray should be unfused: (see AutoGrad/src/core.notes)
broadcasted(::Style{KnetArray}, f, args...) = f(Bcasted.(args)...).value

# The following should set the style for any call that involves a KnetArray:
BroadcastStyle(::Type{<:KnetArray}) = Style{KnetArray}()
broadcastable(x::KnetArray) = x  # This is necessary for the style stuff to work, default definition `collect(x)` turns x into Array.

# Make sure the KnetArray style overrides others except the AutoGrad.Value style:
BroadcastStyle(k::Style{KnetArray}, s::BroadcastStyle) = k
BroadcastStyle(k::Style{KnetArray}, v::Style{Value}) = v

# We use a different Bcasted type than AutoGrad to avoid infinite loops:
struct Bcasted{T}; value::T; end

# This fixes (x .- log.(sum(exp.(x),dims=:))) where log.(::Number) gives a Broadcasted object
Bcasted(x::Broadcasted) = Bcasted(copy(x))

# For broadcasting Knet primitives the following needs to be defined (see unary.jl, binary.jl)
# f(x::Bcasted) = broadcasted(f, x.value) |> Bcasted
# broadcasted(f,x::Bcasted) = broadcasted(f, x.value) |> Bcasted

# The following fixes in-place assignment operations:

function copyto!(a::KnetArray,b::Broadcasted{S,X,F,T}) where {S,X,F<:typeof(identity),T}
    b = b.args[1]
    if size(a) == size(b)
        copyto!(a,b)
    else
        fill!(a,0)
        a .+= convert(KnetArray, b)
    end
    return a
end

function copyto!(a::KnetArray,b::Broadcasted{S,X,F,T}) where {S,X,F<:typeof(identity),T<:Tuple{<:Number}}
    fill!(a,b.args[1])
end

function copyto!(a::SubArray{A,B,C,D,E},b::Broadcasted{S,X,F,T}) where {A,B,C<:KnetArray,D,E,S,X,F<:typeof(identity),T<:Tuple{<:Number}}
    setindex!(a.parent, b.args[1], a.indices...)
end

function copyto!(a::SubArray{A,B,C,D,E},b::Broadcasted{S,X,F,T}) where {A,B,C<:KnetArray,D,E,S,X,F<:typeof(identity),T<:Tuple{<:Any}}
    setindex!(a.parent, b.args[1], a.indices...)
end

# copyto!(a::SubArray{T,N,P,I,L},b::Broadcasted) where {T,N,P<:KnetArray,I,L} = setindex!(a.parent, copy(b), a.indices...)
# copyto!(a::SubArray{T,N,P,I,L},b::Broadcasted{<:Broadcast.AbstractArrayStyle{0}}) where {T,N,P<:KnetArray,I,L} = (if !isempty(b); setindex!(a.parent, first(b), a.indices...); end)

