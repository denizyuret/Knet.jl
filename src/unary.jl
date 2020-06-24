# unary.jl: Unary Array->Array operations.
# uses unary_ops from ops.jl.

using SpecialFunctions
import Base.Broadcast: broadcasted
import NNlib: relu, selu, elu, gelu

function unary_op(f, j=f, o...)
    J=Symbol(j)
    M = which(@__MODULE__, J)
    for S in (32,64)
        T = Symbol("Float$S")
        F = "$(f)_$S"
        @eval begin
            function broadcasted(::typeof($J),x::KnetArray{$T})
                y = similar(x)
                @knet8($F,(Cint,Ptr{$T},Ptr{$T}),length(y),x,y)
                return y
            end
            # Bcasted methods
            ($M).$J(x::Bcasted{<:KnetArray{$T}}) = broadcasted($J, x.value) |> Bcasted
            broadcasted(::typeof($J),x::Bcasted{<:KnetArray{$T}}) = broadcasted($J, x.value) |> Bcasted
        end
    end
    @eval begin # so we do not trigger some default Base implementation
        ($M).$J(x::Bcasted) = throw(MethodError($J,(x,)))
        broadcasted(::typeof($J),x::Bcasted) = throw(MethodError($J,(x,)))
    end
end

# Constants for selu activation from https://arxiv.org/abs/1706.02515
const λ01 = (1-erfc(1/sqrt(2))*sqrt(exp(1)))*sqrt(2pi)*(2*erfc(sqrt(2))*exp(2)+pi*erfc(1/sqrt(2))^2*exp(1)-2*(2+pi)*erfc(1/sqrt(2))*sqrt(exp(1))+pi+2)^(-0.5)
const α01 = -sqrt(2/pi)/(erfc(1/sqrt(2))*exp(1/2)-1)
const λα01 = λ01 * α01

# Constant for gelu activation function from https://arxiv.org/pdf/1606.08415v3.pdf
const GConstant01 = sqrt(2/pi)
const GConstant02 = 0.044715 * sqrt(2/pi)
const GConstant03 = GConstant01 / 2

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
    (:gelu, :geluback, 
    # 0.5x(1+tanh(0.0356774 x^3 + 0.797885 x))
    :(T(0.5)*xi*( one(T) + tanh( (T(GConstant02)*xi^3) + (T(GConstant01)*xi) ) )),
    # 0.5 tanh(0.0356774 x^3 + 0.797885 x) + (0.0535161 x^3 + 0.398942 x) sech^2(0.0356774 x^3 + 0.797885 x) + 0.5
    :(dyi*(T(0.5)*tanh(T(GConstant02)*xi^3 + T(GConstant01)*xi) + (T(0.0535161)*xi^3 + T(GConstant03)*xi)*(sech(T(GConstant02)*xi^3 + T(GConstant01)*xi))^2 + T(0.5)))),
    )

    @eval begin
        $f(xi::T) where {T<:Number}=$y
        $g(dyi::T,yi::T,xi::T) where {T<:Number}=$dx
        function broadcasted(::typeof($f),x::Array{T}) where {T<:AbstractFloat}
            y = similar(x)
            @inbounds for i=1:length(y)
                xi = x[i]
                y[i] = $y
            end
            return y
        end
        function broadcasted(::typeof($g),dy::Array{T},y::Array{T},x::Array{T}) where {T<:AbstractFloat}
            dx = similar(dy)
            @inbounds for i=1:length(dx)
                yi = y[i]
                xi = x[i]
                dyi = dy[i]
                dx[i] = $dx
            end
            return dx
        end
        @primitive $f(x),dy,y $g.(dy,y,x)
    end
end

"`invx(x) = 1/x`" invx
"`sigm(x) = 1/(1+exp(-x))`" sigm

"""
    relu(x)
Return `max(0,x)`.

References: 
* [Nair and Hinton, 2010](https://icml.cc/Conferences/2010/abstracts.html#432). Rectified Linear Units Improve Restricted Boltzmann Machines. ICML.
* [Glorot, Bordes and Bengio, 2011](http://proceedings.mlr.press/v15/glorot11a). Deep Sparse Rectifier Neural Networks. AISTATS.
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

"""
    gelu(x)

Return `0.5 * x * (1 + tanh( √(2/π) * (0.044715 x^3 + x) ))`

Reference: Gaussian Error Linear Units (https://arxiv.org/pdf/1606.08415v3.pdf).
"""
gelu

# To avoid conflict with AutoGrad:
import Base: tanh
@primitive tanh(x::Array),dy,y     tanhback.(dy,y)
@primitive tanh(x::KnetArray),dy,y tanhback.(dy,y)

# For 2nd derivatives:
@primitive tanhback(dy,y),ddx  ddx.*(1 .- y.*y)  ddx.*(-2 .* dy.*y)
@primitive reluback(dy,y),ddx  ddx.*(y.>0)       nothing
@primitive invxback(dy,y),ddx  ddx.*(-y.*y)      ddx.*(-2 .* dy .* y)
@primitive sigmback(dy,y),ddx  ddx.*y.*(1 .- y)  ddx.*dy.*(1 .- 2 .* y)
@primitive seluback(dy,y),ddx  ddx.*((y.>=0).*λ01.+(y.<0).*(y.+λα01))  ddx.*dy.*(y.<0)
@primitive  eluback(dy,y),ddx  ddx.*((y.>=0).+(y.<0).*(y.+1))          ddx.*dy.*(y.<0)

# Unary plus and minus
import Base: +, -

broadcasted(::typeof(+), a::KnetArray)=a
+(a::KnetArray)=a
-(a::KnetArray)=broadcasted(-,a)

# Identity: Issue #335
broadcasted(::typeof(identity), a::KnetArray)=a

# TODO: SpecialFunctions loggamma update #486.
# SpecialFunctions 0.8: lgamma(x::Real)` is deprecated, use `(logabsgamma(x))[1]` instead. 
# Other alternative is loggamma, throws a DomainError if gamma(x) is negative.
# However neither are equivalent to cuda lgamma.
# commenting this out until we implement a better solution.
# if !isdefined(SpecialFunctions, :loggamma) && isdefined(SpecialFunctions, :lgamma)
#     loggamma(x) = lgamma(x)
# end
# if isdefined(SpecialFunctions, :loggamma) && !isdefined(SpecialFunctions, :lgamma)
#     lgamma(x) = loggamma(x)
# end
# if isempty(methods(loggamma,(AutoGrad.Value,))) # Should be fixed after AutoGrad 1.1.4
#     @primitive loggamma(x),dy,y (dy.*(digamma.(x)))
# end

# Unbroadcasted zero works on arrays: this moved to karray.jl
# import Base: zero
# zero(x::KnetArray)=zero.(x)

for f in unary_ops
    if !isa(f,Tuple); f=(f,); end
    unary_op(f...)
end

function unary_op_with_int_degree(f, j=f, o...)
    J=Symbol(j)
    M = which(@__MODULE__, J)
    for S in (32,64)
        T = Symbol("Float$S")
        F = "$(f)_$S"
        @eval begin
            function broadcasted(::typeof($J),d::Integer,x::KnetArray{$T})
                y = similar(x)
                @knet8($F,(Cint,Cint,Ptr{$T},Ptr{$T}),length(y),d,x,y)
                return y
            end
            # Bcasted methods
            ($M).$J(d,x::Bcasted{<:KnetArray{$T}}) = broadcasted($J, d, x.value) |> Bcasted
            broadcasted(::typeof($J),d,x::Bcasted{<:KnetArray{$T}}) = broadcasted($J, d, x.value) |> Bcasted
        end
    end
    @eval begin # so we do not trigger some default Base implementation
        ($M).$J(d,x::Bcasted) = throw(MethodError($J,(x,)))
        broadcasted(::typeof($J),d,x::Bcasted) = throw(MethodError($J,(x,)))
    end
end

for f in unary_ops_with_int_degree
    if !isa(f,Tuple); f=(f,); end
    unary_op_with_int_degree(f...)
end
