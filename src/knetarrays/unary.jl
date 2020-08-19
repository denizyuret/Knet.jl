import Base: +, -, abs, abs2, acos, acosh, asin, asinh, atan, atanh, ceil, cos, cosh, exp, exp10, exp2, expm1, floor, log, log10, log1p, log2, one, round, sign, sin, sinh, sqrt, tan, tanh, trunc, zero
import Base.Math: cbrt, cospi, sinpi
import SpecialFunctions: besselj, besselj0, besselj1, bessely, bessely0, bessely1, digamma, erf, erfc, erfcinv, erfcx, erfinv, gamma, lgamma, trigamma
import Base.Broadcast: broadcasted
using Knet.LibKnet8: @knet8, unary_ops, unary_ops_with_int_degree
# include("broadcast.jl") ## Bcasted

# unary.jl: Unary Array->Array operations.

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

for f in unary_ops
    if !isa(f,Tuple); f=(f,); end
    f[1] ∈ ("elu","relu","selu","sigm") && continue
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

# Unary plus and minus

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

