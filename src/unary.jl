# unary.jl: Unary Array->Array operations.
# uses unary_ops from ops.jl.

using SpecialFunctions
import Base.Broadcast: broadcasted

function unary_op(f, j=f, o...)
    J=Symbol(j)
    for S in (32,64)
        T = Symbol("Float$S")
        F = "$(f)_$S"
        @eval begin
            function broadcasted(::typeof($J),x::KnetArray{$T})
                y = similar(x)
                @knet8($F,(Cint,Ptr{$T},Ptr{$T}),length(y),x,y)
                return y
            end
        end
    end
end

# Define some common operations as primitives for efficiency:
# 1. Avoid creating intermediate arrays
# 2. Avoid taking derivatives of intermediate operations

for (f,g,y,dx) in
    ((:invx, :invxback, :(one(T)/xi), :(-yi*yi*dyi)),
     (:relu, :reluback, :(max(zero(T),xi)), :(ifelse(yi>0,dyi,zero(T)))),
     (:elu,  :eluback,  :(xi >= 0 ? xi : exp(xi)-1), :(yi >= 0 ? dyi : dyi * (1+yi))),
     (:tanx, :tanhback, :(tanh(xi)), :(dyi*(one(T)-yi*yi))),
     (:sigm, :sigmback, 
      # Numerically stable implementation from
      # http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick
      :(if xi>=0; z=exp(-xi); one(T)/(one(T)+z); else; z=exp(xi); z/(one(T)+z); end),
      :(dyi*yi*(one(T)-yi))),
     )
    @eval begin
        $f(xi::T) where {T<:Number}=$y
        $g(dyi::T,yi::T) where {T<:Number}=$dx
        function broadcasted(::typeof($f),x::Array{T}) where {T<:AbstractFloat}
            y = similar(x)
            @inbounds for i=1:length(y)
                xi = x[i]
                y[i] = $y
            end
            return y
        end
        function broadcasted(::typeof($g),dy::Array{T},y::Array{T}) where {T<:AbstractFloat}
            dx = similar(dy)
            @inbounds for i=1:length(dx)
                yi = y[i]
                dyi = dy[i]
                dx[i] = $dx
            end
            return dx
        end
        @primitive $f(x),dy,y $g.(dy,y)
    end
end

"`invx(x) = (1./x)`" invx
"`relu(x) = max(0,x)`" relu
"`sigm(x) = (1./(1+exp(-x)))`" sigm

# To avoid conflict with AutoGrad:
# TODO: test this in Julia6, do we need to fix broadcast_func(tanh)?
import Base: tanh
@primitive tanh(x::Array),dy,y     tanhback.(dy,y)
@primitive tanh(x::KnetArray),dy,y tanhback.(dy,y)
@primitive tanhback(dy,y),ddx  ddx.*(1 .- y.*y)  ddx.*(-2 .* dy.*y)

# Unary plus and minus
import Base: +, -

broadcasted(::typeof(+), a::KnetArray)=a
+(a::KnetArray)=a
-(a::KnetArray)=broadcasted(-,a)

# Identity: Issue #335
broadcasted(::typeof(identity), a::KnetArray)=a

for f in unary_ops
    if !isa(f,Tuple); f=(f,); end
    unary_op(f...)
end

# Unbroadcasted zero works on arrays: this moved to karray.jl
# import Base: zero
# zero(x::KnetArray)=zero.(x)
