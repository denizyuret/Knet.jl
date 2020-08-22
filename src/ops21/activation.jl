export gelu                     # new in Ops21
export elu, relu, selu, sigm    # from Ops20
using Knet.Ops20: elu, relu, selu, sigm
using AutoGrad: AutoGrad, @primitive


"""
    gelu(x)

Return `x * Φ(x)` where `Φ(x)` is the cumulative distribution function for the Gaussian
distribution.

References:
* [Hendrycks and Gimpel 2016](https://arxiv.org/abs/1606.08415) Gaussian Error Linear Units (GELUs)

"""
gelu(x::T) where T = x * (1 + erf(x/T(sqrt(2)))) / 2
geluback(x::T,dy::T) where T = dy * ((1 + erf(x/T(sqrt(2)))) / 2 + x*exp(-x^2/2)/T(sqrt(2pi)))
@primitive gelu(x),dy,y  geluback.(x,dy)
@primitive geluback(x,dy),ddx,dx  (x2=x.^2; ddx.*dy.*(exp.(-x2 ./ 2).*(2 .- x2)./eltype(x)(sqrt(2pi))))  (ddx .* dx ./ dy)

# Other possible approximations given in the paper:
# gelu(x::T) where T = 0.5*x*(1 + tanh(T(sqrt(2/pi))*(x + T(0.044715)*x^3)))
# gelu(x::T) where T = x*sigm(T(1.702)*x)
