export elu, relu, selu, sigm
import Base.Broadcast: broadcasted
using AutoGrad: AutoGrad, @primitive


# Constants for selu activation from https://arxiv.org/abs/1706.02515
const λ01 = 1.0507009873554805  # (1-erfc(1/sqrt(2))*sqrt(exp(1)))*sqrt(2pi)*(2*erfc(sqrt(2))*exp(2)+pi*erfc(1/sqrt(2))^2*exp(1)-2*(2+pi)*erfc(1/sqrt(2))*sqrt(exp(1))+pi+2)^(-0.5)
const α01 = 1.6732632423543778  # -sqrt(2/pi)/(erfc(1/sqrt(2))*exp(1/2)-1)
const λα01 = 1.7580993408473773 # λ01 * α01


for (f,g,y,dx) in
    ((:elu,  :eluback,  :(xi >= 0 ? xi : exp(xi)-T(1)), :(yi >= 0 ? dyi : dyi * (T(1)+yi))),
     (:relu, :reluback, :(max(T(0),xi)), :(ifelse(yi>0,dyi,T(0)))),
     (:selu, :seluback, :(xi >= 0 ? T(λ01)*xi : T(λα01)*(exp(xi)-T(1))), :(yi >= 0 ? dyi * T(λ01) : dyi * (yi + T(λα01)))),
     # Numerically stable sigm implementation from http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick
     (:sigm, :sigmback, :(xi >= 0 ? T(1)/(T(1)+exp(-xi)) : (z=exp(xi); z/(T(1)+z))), :(dyi*yi*(T(1)-yi))),
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


# For 2nd derivatives:
@primitive  eluback(dy,y),ddx  ddx.*((y.>=0).+(y.<0).*(y.+1))          ddx.*dy.*(y.<0)
@primitive reluback(dy,y),ddx  ddx.*(y.>0)       nothing
@primitive seluback(dy,y),ddx  ddx.*((y.>=0).*λ01.+(y.<0).*(y.+λα01))  ddx.*dy.*(y.<0)
@primitive sigmback(dy,y),ddx  ddx.*y.*(1 .- y)  ddx.*dy.*(1 .- 2 .* y)


"""
    elu(x)

Return `(x > 0 ? x : exp(x)-1)`.

Reference: Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs) (https://arxiv.org/abs/1511.07289).
"""
elu


"""
    relu(x)
Return `max(0,x)`.

References: 
* [Nair and Hinton, 2010](https://icml.cc/Conferences/2010/abstracts.html#432). Rectified Linear Units Improve Restricted Boltzmann Machines. ICML.
* [Glorot, Bordes and Bengio, 2011](http://proceedings.mlr.press/v15/glorot11a). Deep Sparse Rectifier Neural Networks. AISTATS.
"""
relu


"""
    selu(x)

Return `λ01 * (x > 0 ? x : α01 * (exp(x)-1))` where `λ01=1.0507009873554805` and `α01=1.6732632423543778`.

Reference: Self-Normalizing Neural Networks (https://arxiv.org/abs/1706.02515).
"""
selu


"""
    sigm(x)

Return `1/(1+exp(-x))`.
"""
sigm

