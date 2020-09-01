module Layers20
import CUDA, Knet
using Knet.KnetArrays
using Knet.Train20
using Knet.Ops20
import Knet.FileIO_gpu: _ser, JLDMODE


"""
    Used for setting default underlying array type for layer parameters.

    settype!(t::T) where T<:Type{KnetArray{V}} where V <: AbstractFloat = CUDA.functional() ? (global arrtype=t) : error("No GPU available")
    settype!(t::T) where T<:Type{Array{V}} where V <: AbstractFloat = (global arrtype=t)
    settype!(t::Union{Type{KnetArray},Type{Array}}) = settype!(t{Float32})

# Example
```julia
julia> KnetLayers.settype!(KnetArray) # on a GPU machine
KnetArray{Float32}
```
"""
settype!(t::T) where T<:Type{KnetArray{V}} where V <: AbstractFloat = CUDA.functional() ? (global arrtype=t) : error("No GPU available")
settype!(t::T) where T<:Type{Array{V}} where V <: AbstractFloat = (global arrtype=t)
settype!(t::Union{Type{KnetArray},Type{Array}}) = settype!(t{Float32})
arrtype = Array{Float32}

include("core.jl");
include("primitive.jl");   export Bias, Multiply, Embed, Linear, Dense, BatchNorm, Diagonal, LayerNorm
include("nonlinear.jl");   export NonAct, ReLU,Sigm,Tanh,LeakyReLU,ELU, Dropout, LogSoftMax, SoftMax, LogSumExp, GeLU
include("loss.jl");        export CrossEntropyLoss, BCELoss, LogisticLoss, SigmoidCrossEntropyLoss
include("cnn.jl");         export Pool,UnPool,DeConv,Conv
include("special.jl");     export MLP
include("rnn.jl");         export RNN,SRNN,LSTM,GRU,RNNOutput,PadRNNOutput,PadSequenceArray
include("chain.jl");       export Chain
include("attention.jl");   export MultiheadAttention
include("transformer.jl"); export Transformer, TransformerDecoder, PositionEmbedding, TransformerModel

function __init__()
    global arrtype = CUDA.functional() ? KnetArray{Float32} : Array{Float32}
end

end # module
