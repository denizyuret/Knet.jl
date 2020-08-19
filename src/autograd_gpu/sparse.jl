import AutoGrad: Sparse, zeroslike
import LinearAlgebra: axpy!
import Base: +, -, copyto!
using Knet.KnetArrays: DevArray
using Base.Broadcast: Broadcasted


Sparse(a::DevArray{T,N},v::Vector{Any},i::Vector{Any}) where {T,N} = Sparse{T,N}(a,v,i)
zeroslike(a::DevArray)=zero(a)
axpy!(a, x::Sparse, y::DevArray) = addto!(y, a*x)
+(a::DevArray, s::Sparse) = addto!(copy(a), s)
+(s::Sparse, a::DevArray) = addto!(copy(a), s)
-(a::DevArray, s::Sparse) = addto!(copy(a), -s)
-(s::Sparse, a::DevArray) = addto!(-a, s)


function copyto!(a::DevArray, bc::Broadcasted{S,A,F,X}) where
    {S, A, F <: Union{typeof(+),typeof(-)}, X <: Tuple{Any,Sparse}}
    (b,c) = bc.args
    if !(size(a) == size(b) == size(c.container))
        a .= bc.f.(b, full(c))
        return a
    end
    a === b || copyto!(a, b)
    F <: typeof(-) && (c = -c)
    addto!(a, c)
    return a
end
