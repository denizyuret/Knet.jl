"""
    Multiply(input=inputDimension, output=outputDimension, winit=xavier, atype=KnetLayers.arrtype)

Creates a matrix multiplication layer based on `inputDimension` and `outputDimension`.
    (m::Multiply)(x::AbstractArray{T,N}; keepsize=true)= m.w * x

By default parameters initialized with xavier, you may change it with `winit` argument.

# Keywords
* `input=inputDimension`: input size
* `output=outputDimension`: output size
* `winit=xavier`: weight initialization distribution
* `atype=KnetLayers.arrtype` : underlying array type for parameters.
   Default arrtype=KnetArray{Float32} if running on a GPU device, otherwise Array{Float32}.
   see `KnetLayers.settype!`
* `keepsize=true`: When `N=3` or higher dimensional arrays supplied, `Multiply` squeezes the
                   dimensions except the first dimension in order to perform multiplication.
                   It reshapes the output to recover input sizeality `N`.
                   However, one may obtain 2D vesion by setting keepsize=false

"""
mutable struct Multiply{P} <: Layer
    weight::P
end
Multiply(;input::Int, output::Int, winit=xavier, atype=arrtype) = Multiply(param(output, input; init=winit, atype=atype))
@inline (m::Multiply)(x::AbstractArray{<:Integer}) = m.weight[:,x] # Lookup (EmbedLayer)

# TODO: Find a faster (or compound) way for tensor-product
function (m::Multiply)(x; keepsize=true)
    if ndims(x) > 2
        s = size(x)
        y = m.weight * reshape(x, s[1], prod(s[2:end]))
        return (keepsize ? reshape(y, size(y, 1), s[2:end]...) : y)
    else
        return m.weight * x
    end
end
Base.show(io::IO,m::Multiply{P}) where P = print(io,Multiply{P},"(input=",size(m.weight,2)," output=",size(m.weight,1),")")
@inline Base.size(m::Multiply) = size(m.weight)

"""
    Embed(input=inputSize, output=embedSize, winit=xavier, atype=KnetLayers.arrtype)
Creates an embedding layer according to given `inputSize` and `embedSize`.
By default parameters initialized with xavier, change it with `winit` argument.

    (m::Embed)(x::Array{T}) where T<:Integer
    (m::Embed)(x::AbstractArray{T,N}; keepsize=true) # see `Multiply`


Embed objects are callable with an input which is either and integer array
(one hot encoding) or an AbstractArray{T,N}
For N-dimensional array, `size(x,1)==inputSize`, see `Multiply`

# Keywords

* `input=inputDimension`: input size
* `output=embeddingDimension`: output size
* `winit=xavier`: weight initialization distribution
* `atype=KnetLayers.arrtype` : underlying array type for parameters.
   Default arrtype=KnetArray{Float32} if running on a GPU device, otherwise Array{Float32}.
   see `KnetLayers.settype!`
"""
const Embed = Multiply

"""
    Bias(sizes..., atype=KnetLayers.arrtype, winit=zeros)
Creates a bias layer according to given `sizes`
By default parameters initialized with zeros, change it with `winit` argument.
`Bias` layer can be initialized with Bias(nothing) to make it ineffective.

    (m::Bias)(x) = m.b .+ x
    (m::Bias{Nothing})(x) = x
# Keywords

* `winit=zeros`: weight initialization distribution
* `atype=KnetLayers.arrtype` : underlying array type for parameters.
   Default arrtype=KnetArray{Float32} if running on a GPU device, otherwise Array{Float32}.
   see `KnetLayers.settype!`
"""
struct Bias{T}
    b::T
end
@inline (m::Bias)(x) = m.b .+ x
@inline (m::Bias{Nothing})(x) = x
#Bias(size::Int;atype=arrtype, winit=zeros, o...) = Bias(param(size;atype=atype, init=winit,o...))
Bias(sizes::Int...;atype=arrtype, winit=zeros, o...) = Bias(param(sizes...;atype=atype, init=winit, o...))
Bias() = Bias(nothing)
Base.show(io::IO,m::Bias{T}) where T = print(io,Bias{T},"(length=",length(m.b),")")
@inline Base.size(m::Bias) = size(b)
"""
    Linear(input=inputSize, output=outputSize, winit=xavier, binit=zeros, atype=KnetLayers.arrtype)

Creates and linear layer according to given `inputSize` and `outputSize`.
`Linear` consists of a `Multiply` and a `Bias` layer.

    (m::Linear)(x) = m.bias(m.mult(x))

# Keywords
* `input=inputSize`   input size
* `output=outputSize` output size
* `winit=xavier`: weight initialization distribution
* `bias=zeros`: bias initialization distribution. It can be `nothing` for unbiased `Linear` layer.
* `atype=KnetLayers.arrtype` : underlying array type for parameters.
   Default arrtype=KnetArray{Float32} if running on a GPU device, otherwise Array{Float32}.
   see `KnetLayers.settype!`
"""
mutable struct Linear <: Layer
    mult::Multiply
    bias::Bias
end
function Linear(;input::Int, output::Int, winit=xavier, binit=zeros, atype=arrtype)
    Linear(Multiply(input=input, output=output, winit=winit, atype=atype),Bias(output, winit=binit, atype=atype))
end
@inline (m::Linear)(x) = m.bias(m.mult(x))
@inline Base.size(m::Linear) = size(m.mult)
"""
    Dense(input=inputSize, output=outputSize, activation=ReLU(), winit=xavier, binit=zeros, atype=KnetLayers.arrtype)
Creates a dense layer according to given `input` and `output` sizes.
If activation is `nothing`, it acts like a `Linear` Layer. see `Linear`

    (m::Dense{Nothing})(x) = m.linear(x)
    (m::Dense{<:Activation})(x)= m.activation(m.linear(x))

# Keywords
* `input=inputSize`   input size
* `output=outputSize` output size
* `winit=xaiver`: weight initialization distribution
* `bias=zeros`:   bias initialization distribution
* `activation=ReLU()`  an  activation layer, or nothing for a `Linear` equivalent layer.
* `atype=KnetLayers.arrtype` : underlying array type for parameters.
   Default arrtype=KnetArray{Float32} if running on a GPU device, otherwise Array{Float32}.
   see `KnetLayers.settype!`
"""
mutable struct Dense{T<:ActOrNothing} <: Layer
    linear::Linear
    activation::T
end

function Dense(;input::Int, output::Int, activation::ActOrNothing=ReLU(), winit=xavier, binit=zeros, atype=arrtype)
    Dense(Linear(input=input, output=output, winit=winit, binit=binit, atype=atype), activation)
end
@inline (m::Dense{Nothing})(x) = m.linear(x)
@inline (m::Dense{<:Activation})(x)= m.activation(m.linear(x))

Base.show(io::IO, x::Dense) = print(io,typeof(x),"(",x.linear,")")
@inline Base.size(m::Dense) = size(m.linear)
#TO-DO: Remove after the issue is resolved:
#https://github.com/denizyuret/Knet.jl/issues/418
"""
    BatchNorm(channels:Int;options...)
    (m::BatchNorm)(x;training=false) #forward run
# Options
* `momentum=0.1`: A real number between 0 and 1 to be used as the scale of
 last mean and variance. The existing running mean or variance is multiplied by
 (1-momentum).
* `mean=nothing': The running mean.
* `var=nothing`: The running variance.
* `meaninit=zeros`: The function used for initialize the running mean.
* `varinit=ones`: The function used for initialize the running variance of batchnorm
* `dataType=eltype(KnetLayers.arrtype)` : element type ∈ {Float32,Float64} for parameters.
* `usegpu=KnetLayers.arrtype <: KnetArray`
# Keywords
* `training`=nothing: When training is true, the mean and variance of x are used and moments
 argument is modified if it is provided. When training is false, mean and variance
 stored in the moments argument are used. Default value is true when at least one
 of x and params is AutoGrad.Value, false otherwise.
"""
mutable struct BatchNorm <: Layer
    params
    moments::Knet.Ops20.BNMoments
end

function BatchNorm(channels::Int;  usegpu = arrtype <: KnetArray, dataType=eltype(arrtype), o...)
    w = bnparams(dataType,channels)
    m = bnmoments(;o...)
    p = usegpu ? Param(KnetArray(w)) : Param(w)
    BatchNorm(p,m)
end
@inline (m::BatchNorm)(x;o...) = batchnorm(x,m.moments,m.params;o...)
Base.show(io::IO,x::BatchNorm) where P = print(io,BatchNorm,"(",x.params,", ",x.moments,")")


"""
    Diagonal(D::Integer)
Creates an element-wise linear transformation layer with learnable
vectors `w` and `b`:
    y = w .* x .+ b
The input `x` must be a array where `size(x, 1) == D`.
"""
struct Diagonal{T} <: Layer
  w::T
  b::T
end

Diagonal(D::Integer; winit = ones, binit = zeros, atype=arrtype) =
  Diagonal(param(D; init=winit, atype=atype),
           param(D; init=binit, atype=atype))

function (a::Diagonal)(x)
  w, b = a.w, a.b
  return a.w.*x .+ a.b
end

Base.length(l::Diagonal) = length(l.w)

function Base.show(io::IO, l::Diagonal)
  print(io, "Diagonal(", length(l), ")")
end


"""
    LayerNorm(h::Integer)
A [normalisation layer](https://arxiv.org/pdf/1607.06450.pdf) designed to be
used with recurrent hidden states of size `h`. Normalises the mean/stddev of
each input before applying a per-neuron gain/bias.
"""
struct LayerNorm{T} <: Layer
  diag::Diagonal{T}
end

LayerNorm(h::Integer) =
  LayerNorm(Diagonal(h))


(a::LayerNorm)(x) = a.diag(normalise(x))

function Base.show(io::IO, l::LayerNorm)
  print(io, "LayerNorm(", length(l.diag), ")")
end
