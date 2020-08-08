export elu, relu, selu, sigm
using AutoGrad: AutoGrad, @primitive


"""
    elu(x)

Return `(x > 0 ? x : exp(x)-1)`.

Reference: Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs) (https://arxiv.org/abs/1511.07289).
"""
elu(x::T) where T = (x >= 0 ? x : exp(x)-T(1))
eluback(dy::T,y::T) where T = (y >= 0 ? dy : dy * (T(1)+y))
@primitive elu(x),dy,y eluback.(dy,y)
@primitive eluback(dy,y),ddx  (ddx.*((y.>=0).+(y.<0).*(y.+1)))  (ddx.*dy.*(y.<0))


"""
    relu(x)
Return `max(0,x)`.

References: 
* [Nair and Hinton, 2010](https://icml.cc/Conferences/2010/abstracts.html#432). Rectified Linear Units Improve Restricted Boltzmann Machines. ICML.
* [Glorot, Bordes and Bengio, 2011](http://proceedings.mlr.press/v15/glorot11a). Deep Sparse Rectifier Neural Networks. AISTATS.
"""
relu(x::T) where T = max(x,T(0))
reluback(dy::T,y::T) where T = (y>0 ? dy : T(0))
@primitive relu(x),dy,y reluback.(dy,y)
@primitive reluback(dy,y),ddx  (ddx.*(y.>0))  nothing


"""
    selu(x)

Return `λ01 * (x > 0 ? x : α01 * (exp(x)-1))` where `λ01=1.0507009873554805` and `α01=1.6732632423543778`.

Reference: Self-Normalizing Neural Networks (https://arxiv.org/abs/1706.02515).
"""
selu(x::T) where T = (x >= 0 ? T(λ01)*x : T(λα01)*(exp(x)-T(1)))
seluback(dy::T,y::T) where T = (y >= 0 ? dy * T(λ01) : dy * (y + T(λα01)))
@primitive selu(x),dy,y seluback.(dy,y)
@primitive seluback(dy,y),ddx  (T=eltype(y); ddx.*((y.>=0).*T(λ01).+(y.<0).*(y.+T(λα01))))  (ddx.*dy.*(y.<0))
const λ01 = 1.0507009873554805  # (1-erfc(1/sqrt(2))*sqrt(exp(1)))*sqrt(2pi)*(2*erfc(sqrt(2))*exp(2)+pi*erfc(1/sqrt(2))^2*exp(1)-2*(2+pi)*erfc(1/sqrt(2))*sqrt(exp(1))+pi+2)^(-0.5)
const α01 = 1.6732632423543778  # -sqrt(2/pi)/(erfc(1/sqrt(2))*exp(1/2)-1)
const λα01 = 1.7580993408473773 # λ01 * α01


"""
    sigm(x)

Return `1/(1+exp(-x))`.

Reference: Numerically stable sigm implementation from http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick.
"""
sigm(x::T) where T = (x >= 0 ? T(1)/(T(1)+exp(-x)) : (z=exp(x); z/(T(1)+z)))
sigmback(dy::T,y::T) where T = (dy*y*(T(1)-y))
@primitive sigm(x),dy,y  sigmback.(dy,y)
@primitive sigmback(dy,y),ddx  ddx.*y.*(1 .- y)  ddx.*dy.*(1 .- 2 .* y)
