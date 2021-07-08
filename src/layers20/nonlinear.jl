"""
    NonAct()
    (l::NonAct)(x) = x

Identity activation does nothing
"""
struct NonAct <: Activation end
@inline (l::NonAct)(x) = x

"""
    ReLU()
    (l::ReLU)(x) = max(0,x)

Rectified Linear Unit function.
"""
struct ReLU <: Activation end
@inline (l::ReLU)(x) = relu.(x)

"""
    Sigm()
    (l::Sigm)(x) = sigm(x)

Sigmoid function
"""
struct Sigm <: Activation end
@inline (l::Sigm)(x) = sigm.(x)

"""
    Tanh()
    (l::Tanh)(x) = tanh(x)

Tangent hyperbolic function
"""
struct Tanh <: Activation end
@inline (l::Tanh)(x) = tanh.(x)


"""
    ELU()
    (l::ELU)(x) = elu(x) -> Computes x < 0 ? exp(x) - 1 : x

Exponential Linear Unit nonlineariy.
"""
struct ELU <: Activation end
@inline (l::ELU)(x) = elu.(x)

"""
    LeakyReLU(α=0.2)
    (l::LeakyReLU)(x) -> Computes x < 0 ? α*x : x
"""
mutable struct LeakyReLU <: Activation
    α::AbstractFloat
    LeakyReLU(alpha::AbstractFloat=0.2) = new(alpha)
end
@inline (l::LeakyReLU)(x) = relu.(x) .+ l.α*min.(0,x) # relu.(x) .+ l.α*relu.(-x) ?

#FIXME: x*x*x -> x.^3 after Knet new version
mutable struct GeLU <: Activation
end
"""
    gelu(x) = 0.5x*(1 + tanh(√(2/π)*(x + 0.044715x^3)))
[Gaussian Error Linear Unit](https://arxiv.org/pdf/1606.08415.pdf)
activation function.
"""
function (::GeLU)(x)
    T = eltype(x)
    λ = T(√(2/π))
    α = T(0.044715)
    h = T(0.5)
    return h .* x .* (one(T) .+ tanh.(λ .* (x .+ α * x .* x .* x)))
end
"""
    Dropout(p=0)

Dropout Layer. `p` is the droput probability.
"""
mutable struct Dropout <: Activation
    p::Real
end
Dropout(;p=0) = Dropout(p)
@inline (l::Dropout)(x) = dropout(x,l.p)

"""
    LogSoftMax(dims=:)
    (l::LogSoftMax)(x)

Treat entries in x as as unnormalized log probabilities and return normalized log probabilities.

dims is an optional argument, if not specified the normalization is over the whole x, otherwise the normalization is performed over the given dimensions. In
particular, if x is a matrix, dims=1 normalizes columns of x, dims=2 normalizes rows of x.
"""
struct LogSoftMax <: Activation
    dims::Union{Integer,Colon}
end
LogSoftMax(;dims=:) = LogSoftMax(dims)
@inline (l::LogSoftMax)(x) = logp(x;dims=l.dims)

"""
    SoftMax(dims=:)
    (l::SoftMax)(x)

Treat entries in x as as unnormalized scores and return softmax probabilities.

dims is an optional argument, if not specified the normalization is over the whole x, otherwise the normalization is performed over the given dimensions. In
particular, if x is a matrix, dims=1 normalizes columns of x, dims=2 normalizes rows of x.
"""
struct SoftMax <: Activation
    dims::Union{Integer,Colon}
end
SoftMax(;dims=:) = SoftMax(dims)
@inline (l::SoftMax)(x) = softmax(x;dims=l.dims)

"""
    LogSumExp(dims=:)
    (l::LogSumExp)(x)

  Compute log(sum(exp(x);dims)) in a numerically stable manner.

  dims is an optional argument, if not specified the summation is over the whole x, otherwise the summation is performed over the given dimensions. In particular if x
  is a matrix, dims=1 sums the columns of x, dims=2 sums the rows of x.
"""
struct LogSumExp <: Activation
    dims::Union{Integer,Colon}
end
LogSumExp(;dims=:) = LogSumExp(dims)
@inline (l::LogSumExp)(x) = logsumexp(x;dims=l.dims)


"""
    normalise(x; dims=1)
    Normalises x to mean 0 and standard deviation 1, across the dimensions given by dims. Defaults to normalising over columns.
"""
function normalise(x; dims=1)
  μ′ = mean(x, dims = dims)
  σ′ = std(x, dims = dims, mean = μ′, corrected=false)
  return (x .- μ′) ./ σ′
end
