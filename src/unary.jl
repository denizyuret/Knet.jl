# unary.jl: Unary Array->Array operations.
# uses unary_ops from ops.jl.

using SpecialFunctions
import Base.Broadcast: broadcasted

function unary_op(f, j=f, o...)
    J=Symbol(j)
    M = which(@__MODULE__, J)
    @eval begin
        ($M).$J(x::Bcasted) = bcasted($J, x.value) |> Bcasted
        broadcasted(::typeof($J),x::Bcasted) = bcasted($J, x.value) |> Bcasted
    end
    for S in (32,64)
        T = Symbol("Float$S")
        F = "$(f)_$S"
        @eval begin
            function broadcasted(::typeof($J),x::KnetArray{$T})
                y = similar(x)
                @knet8($F,(Cint,Ptr{$T},Ptr{$T}),length(y),x,y)
                return y
            end
            bcasted(f::typeof($J),x::KnetArray{$T}) = broadcasted(f,x)
        end
    end
end

# Constants for selu activation from https://arxiv.org/abs/1706.02515
const λ01 = (1-erfc(1/sqrt(2))*sqrt(exp(1)))*sqrt(2pi)*(2*erfc(sqrt(2))*exp(2)+pi*erfc(1/sqrt(2))^2*exp(1)-2*(2+pi)*erfc(1/sqrt(2))*sqrt(exp(1))+pi+2)^(-0.5)
const α01 = -sqrt(2/pi)/(erfc(1/sqrt(2))*exp(1/2)-1)
const λα01 = λ01 * α01

# Define some common operations as primitives for efficiency:
# 1. Avoid creating intermediate arrays
# 2. Avoid taking derivatives of intermediate operations

for (f,g,y,dx) in
    ((:invx, :invxback, :(one(T)/xi), :(-yi*yi*dyi)),
     (:relu, :reluback, :(max(zero(T),xi)), :(ifelse(yi>0,dyi,zero(T)))),
     (:selu, :seluback, :(xi >= 0 ? T(λ01)*xi : T(λα01)*(exp(xi)-1)), :(yi >= 0 ? dyi * T(λ01) : dyi * (yi + T(λα01)))),
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
"`sigm(x) = (1./(1+exp(-x)))`" sigm

"""
    relu(x)
Return `max(0,x)`.

Reference: Rectified Linear Units Improve Restricted Boltzmann Machines (https://icml.cc/Conferences/2010/abstracts.html#432).
"""
relu

"""
    elu(x)

Return `(x > 0 ? x : exp(x)-1)`.

Reference: Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs) (https://arxiv.org/abs/1511.07289).
"""
elu

"""
    selu(x)

Return `λ01 * (x > 0 ? x : α01 * (exp(x)-1))` where `λ01=1.0507009873554805` and `α01=1.6732632423543778`.

Reference: Self-Normalizing Neural Networks (https://arxiv.org/abs/1706.02515).
"""
selu

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
