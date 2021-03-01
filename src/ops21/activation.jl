export elu, gelu, hardsigmoid, hardswish, relu, selu, sigm, swish, tanh_
using AutoGrad: AutoGrad, @primitive1, @zerograd
using SpecialFunctions: erf

# AutoGrad does not support broadcasted functions with keywords (https://github.com/denizyuret/AutoGrad.jl/issues/124)
# In Ops21 we will discourage use of broadcasted notation for activations and use f(x::Array) instead of f.(x::Array)

"""
    elu(x)

Return `(x > 0 ? x : exp(x)-1)`.

Reference: Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs) (https://arxiv.org/abs/1511.07289).

"""
elu(x::T) where {T<:Real} = (x >= 0 ? x : exp(x)-T(1))
eluback(x::T,y::T,dy::T) where {T<:Real} = (y >= 0 ? dy : dy * (T(1)+y))

elu(x::T) where {T<:Array} = elu.(x)
eluback(x::T, y::T, dy::T) where {T<:Array} = eluback.(x, y, dy)

@primitive1 elu(x),dy,y  eluback(x,y,dy)
#@primitive1 eluback(x::Real,y::Real,dy::Real),ddx  (ddx.*((y.>=0).+(y.<0).*(y.+1)))  (ddx.*dy.*(y.<0))
@zerograd eluback(x,y,dy)       # TODO: second derivatives


"""
    gelu(x)

Return `x * Φ(x)` where `Φ(x)` is the cumulative distribution function for the Gaussian
distribution.

References:
* [Hendrycks and Gimpel 2016](https://arxiv.org/abs/1606.08415) Gaussian Error Linear Units (GELUs)

"""
gelu(x::T) where {T<:Real} = x * (1 + erf(x/T(sqrt(2)))) / 2
geluback(x::T,y::T,dy::T) where {T<:Real} = dy * ((1 + erf(x/T(sqrt(2)))) / 2 + x*exp(-x^2/2)/T(sqrt(2pi)))

gelu(x::T) where {T<:Array} = gelu.(x)
geluback(x::T, y::T, dy::T) where {T<:Array} = geluback.(x, y, dy)

@primitive1 gelu(x),dy,y  geluback(x,y,dy)
# @primitive1 geluback(x::Real,y::Real,dy::Real),ddx,dx  (x2=x.^2; ddx.*dy.*(exp.(-x2 ./ 2).*(2 .- x2)./eltype(x)(sqrt(2pi))))  (ddx .* dx ./ dy)
@zerograd geluback(x,y,dy)

# Other possible approximations given in the paper:
# gelu(x::T) where T = 0.5*x*(1 + tanh(T(sqrt(2/pi))*(x + T(0.044715)*x^3)))
# gelu(x::T) where T = x*sigm(T(1.702)*x)


"""
    hardsigmoid(x)

Return `(x <= -3 ? 0 : x >= 3 ? 1 : (x+3)/6)`.

References:
* [Howard+ 2019](https://arxiv.org/abs/1905.02244) Searching for MobileNetV3

"""
hardsigmoid(x::T) where {T<:Real} = (x <= T(-3) ? T(0) : x >= T(3) ? T(1) : (x+3)/6)
hardsigmoidback(x::T,y::T,dy::T) where {T<:Real} = dy * (x <= T(-3) ? T(0) : x >= T(3) ? T(0) : T(1/6))

hardsigmoid(x::T) where {T<:Array} = hardsigmoid.(x)
hardsigmoidback(x::T, y::T, dy::T) where {T<:Array} = hardsigmoidback.(x, y, dy)

@primitive1 hardsigmoid(x),dy,y  hardsigmoidback(x,y,dy)
# @primitive1 hardsigmoidback(x::Real,y::Real,dy::Real),ddx,dx  ??
@zerograd hardsigmoidback(x,y,dy)


"""
    hardswish(x)

Return `(x <= -3 ? 0 : x >= 3 ? x : x*(x+3)/6)`.

References:
* [Howard+ 2019](https://arxiv.org/abs/1905.02244) Searching for MobileNetV3

"""
hardswish(x::T) where {T<:Real} = (x <= T(-3) ? T(0) : x >= T(3) ? x : x*(x+3)/6)
hardswishback(x::T,y::T,dy::T) where {T<:Real} = dy * (x <= T(-3) ? T(0) : x >= T(3) ? T(1) : (2x+3)/6)

hardswish(x::T) where {T<:Array} = hardswish.(x)
hardswishback(x::T, y::T, dy::T) where {T<:Array} = hardswishback.(x, y, dy)

@primitive1 hardswish(x),dy,y  hardswishback(x,y,dy)
# @primitive1 hardswishback(x::Real,y::Real,dy::Real),ddx,dx  ??
@zerograd hardswishback(x,y,dy)


"""
    relu(x; max_value=Inf, negative_slope=0, threshold=0)

Return the following:
    
    (x >= max_value ? max_value :
     x >= threshold ? x :
     negative_slope * (x - threshold))
"""
function relu(x::T; max_value=Inf, negative_slope=0, threshold=0) where {T<:Real}
    (x >= max_value ? oftype(x, max_value) :
     x >= threshold ? x :
     negative_slope == 0 ? zero(T) :
     negative_slope * (x - oftype(x, threshold)))
end

function reluback(x::T,y::T,dy::T; max_value=Inf, negative_slope=0, threshold=0) where {T<:Real}
    dy * (x >= max_value ? zero(T) :
          x >= threshold ? one(T) :
          oftype(x, negative_slope))
end

relu(x::T; o...) where {T<:Array} = relu.(x; o...)
reluback(x::T, y::T, dy::T; o...) where {T<:Array} = reluback.(x, y, dy; o...)

@primitive1 relu(x; o...),dy,y   reluback(x,y,dy; o...)
#@primitive1 reluback(x::Real,y::Real,dy::Real; o...),ddx,dx  zero(dy)
@zerograd reluback(x,y,dy; o...)


"""
    selu(x)

Return `λ01 * (x > 0 ? x : α01 * (exp(x)-1))` where `λ01=1.0507009873554805` and `α01=1.6732632423543778`.

Reference: Self-Normalizing Neural Networks (https://arxiv.org/abs/1706.02515).
"""
selu(x::T) where {T<:Real} = (x >= 0 ? T(λ01)*x : T(λα01)*(exp(x)-T(1)))
seluback(x::T,y::T,dy::T) where {T<:Real} = (y >= 0 ? dy * T(λ01) : dy * (y + T(λα01)))

selu(x::T) where {T<:Array} = selu.(x)
seluback(x::T, y::T, dy::T) where {T<:Array} = seluback.(x, y, dy)

@primitive1 selu(x),dy,y seluback(x,y,dy)
# @primitive1 seluback(x::Real,y::Real,dy::Real),ddx  (T=eltype(y); ddx.*((y.>=0).*T(λ01).+(y.<0).*(y.+T(λα01))))  (ddx.*dy.*(y.<0))
@zerograd seluback(x,y,dy)

const λ01 = 1.0507009873554805  # (1-erfc(1/sqrt(2))*sqrt(exp(1)))*sqrt(2pi)*(2*erfc(sqrt(2))*exp(2)+pi*erfc(1/sqrt(2))^2*exp(1)-2*(2+pi)*erfc(1/sqrt(2))*sqrt(exp(1))+pi+2)^(-0.5)
const α01 = 1.6732632423543778  # -sqrt(2/pi)/(erfc(1/sqrt(2))*exp(1/2)-1)
const λα01 = 1.7580993408473773 # λ01 * α01


"""
    sigm(x)

Return `1/(1+exp(-x))`.

Reference: Numerically stable sigm implementation from http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick.
"""
sigm(x::T) where {T<:Real} = (x >= 0 ? T(1)/(T(1)+exp(-x)) : (z=exp(x); z/(T(1)+z)))
sigmback(x::T,y::T,dy::T) where {T<:Real} = (dy*y*(T(1)-y))

sigm(x::T) where {T<:Array} = sigm.(x)
sigmback(x::T, y::T, dy::T) where {T<:Array} = sigmback.(x, y, dy)

@primitive1 sigm(x),dy,y  sigmback(x,y,dy)
#@primitive1 sigmback(x::Real,y::Real,dy::Real),ddx  ddx.*y.*(1 .- y)  ddx.*dy.*(1 .- 2 .* y)
@zerograd sigmback(x,y,dy)


"""
    swish(x)

Return `x * sigm(x)`.

References:
* Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Searching for activation functions. arXiv preprint arXiv:1710.05941.
"""
swish(x::T) where {T<:Real} = x * sigm(x)
swishback(x::T,y::T,dy::T) where {T<:Real} = dy * (x >= 0 ? (z=exp(-x);(z*x+z+1)/((z+1)^2)) : (z=exp(x);(z*(x+z+1)/((z+1)^2))))

swish(x::T) where {T<:Array} = swish.(x)
swishback(x::T, y::T, dy::T) where {T<:Array} = swishback.(x, y, dy)

@primitive1 swish(x),dy,y  swishback(x,y,dy)
# @primitive1 swishback(x::Real,y::Real,dy::Real),ddx,dx  (x2=x.^2; ddx.*dy.*(exp.(-x2 ./ 2).*(2 .- x2)./eltype(x)(sqrt(2pi))))  (ddx .* dx ./ dy)
@zerograd swishback(x,y,dy)


"""
    tanh_(x)

Return `tanh(x)`. The difference is `tanh_(x::Array)` broadcasts instead of computing a Matrix function.
"""
tanh_(x::T) where {T<:Real} = tanh(x)
tanh_back(x::T,y::T,dy::T) where {T<:Real} = dy*(1 - abs2(y))

tanh_(x::T) where {T<:Array} = tanh_.(x)
tanh_back(x::T, y::T, dy::T) where {T<:Array} = tanh_back.(x, y, dy)

@primitive1 tanh_(x),dy,y  tanh_back(x,y,dy)
# @primitive1 tanh_back(x::Real,y::Real,dy::Real),ddx,dx  (x2=x.^2; ddx.*dy.*(exp.(-x2 ./ 2).*(2 .- x2)./eltype(x)(sqrt(2pi))))  (ddx .* dx ./ dy)
@zerograd tanh_back(x,y,dy)
