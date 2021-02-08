export elu, gelu, relu, selu, sigm, swish
import Knet.Ops20
using AutoGrad: AutoGrad, @primitive
using SpecialFunctions: erf

const elu  = Ops20.elu
const relu = Ops20.relu
const selu = Ops20.selu
const sigm = Ops20.sigm

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


"""
    swish(x)

Return `x * sigm(x)`.

References:
* Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Searching for activation functions. arXiv preprint arXiv:1710.05941.
"""
swish(x::T) where T = x * sigm(x)
swishback(x::T,dy::T) where T = dy * (x >= 0 ? (z=exp(-x);(z*x+z+1)/((z+1)^2)) : (z=exp(x);(z*(x+z+1)/((z+1)^2))))
