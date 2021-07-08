import Base.Iterators: product
####
#### Output Structure
####

"""
    struct RNNOutput
        y
        hidden
        memory
        indices
    end

RNN layer outputs are always `RNNOutput`
`hidden`,`memory` and `indices` may be nothing depending on the keyword arguments you used in forward.

`y` is last hidden states of each layer. `size(y)=(H/2H,[B,T])`.
If you use unequal length instances in a batch input, `y` becomes 2D array `size(y)=(H/2H,sum_of_sequence_lengths)`.
See `indices` below to get correct time outputs for a specific instance or
see  `PadRNNOutput` to pad whole output.
`h` is the hidden states in each timesstep. `size(h) = h,B,L/2L`

`c` is the hidden states in each timesstep. `size(h) = h,B,L/2L`

`indices` is corresponding instace indices for your `RNNOutput.y`. You may call `yi = y[:,indices[i]]`.
"""
struct RNNOutput{T,V,Z,S<:VVecOrNothing}
    y::T
    hidden::V
    memory::Z
    indices::S
end

function Base.show(io::IO,R::RNNOutput{T,V,Z,S}) where {T,V,Z,S}
    print(io,"RNNOutput(y: $T$(size(R.y))")
    V !== Nothing && print(io,", hidden : $V$(size(R.hidden))")
    Z !== Nothing && print(io,", memory : $Z$(size(R.memory))")
    S !== Nothing && print(io,", indices: $S$(size(R.indices))")
    print(io,")")
end

####
#### RNN Types
####
"""
    SRNN(;input=inputSize, hidden=hiddenSize, activation=:relu, options...)
    LSTM(;input=inputSize, hidden=hiddenSize, options...)
    GRU(;input=inputSize, hidden=hiddenSize, options...)

    (1) (l::T)(x; kwargs...) where T<:AbstractRNN
    (2) (l::T)(x::Array{Int}; batchSizes=nothing, kwargs...) where T<:AbstractRNN
    (3) (l::T)(x::Vector{Vector{Int}}; sorted=false, kwargs...) where T<:AbstractRNN

All RNN layers has above forward run(1,2,3) functionalities.

(1) `x` is an AbstractArray{T,N} with size equals d,[B,T]

(2) For this, an RNN with embedding layer is needed. see `embed` option below.
`x` is an integer array and inputs corresponds one-hot indices.
One can give 2D array for minibatching as rows corresponds to one instance, or
One can give 1D array with minibatching by specifying batch batchSizes argument.
see `Knet.rnnforw` for the details of batchSizes.

(3) For this, RNN with embedding layer is needed.
`x` is an vector of integer vectors. Every integer vector corresponds to an
instance. It automatically batches inputs for the user. It is better to give inputs as sorted.
If your inputs sorted you can make `sorted` argument true to increase performance.

see `RNNOutput` for output details

# options

* `embed=nothing`: embedding size or and embedding layer
* `numLayers=1`: Number of RNN layers.
* `bidirectional=false`: Create a bidirectional RNN if `true`.
* `dropout=0`: Dropout probability. Ignored if `numLayers==1`.
* `skipInput=false`: Do not multiply the input with a matrix if `true`.
* `dataType=eltype(KnetLayers.arrtype)`: Data type to use for weights. Default is Float32.
* `algo=0`: Algorithm to use, see CUDNN docs for details.
* `seed=0`: Random number seed for dropout. Uses `time()` if 0.
* `winit=xavier`: Weight initialization method for matrices.
* `binit=zeros`: Weight initialization method for bias vectors.
* `usegpu=(KnetLayers.arrtype <: KnetArray)`: GPU used by default if one exists.

# Keywords

* hx=nothing : initial hidden states
* cx=nothing : initial memory cells
* hy=false   : if true returns h
* cy=false   : if true returns c

"""
AbstractRNN

for layer in (:SRNN, :LSTM, :GRU)
    layername=string(layer)
    @eval begin

        mutable struct $layer{P,E<:LayerOrNothing} <: AbstractRNN{P,E}
            embedding::E
            params::P
            specs::RNN
            gatesview::DictOrNothing
        end

        @inline (m::$layer)(x,h...;o...) = RNNOutput(_forw(m,x,h...;o...)...)
        # FIXME: activation input is not compatible with rest of the package
        function $layer(;input::Integer, hidden::Integer, embed=nothing, activation=:relu,
                         usegpu=(arrtype <: KnetArray), dataType=eltype(arrtype), o...)
            embedding,inputSize = _getEmbed(input,embed)
            rnnType = $layer==SRNN ? activation : Symbol(lowercase($layername))
            r = RNN(inputSize, hidden; rnnType=rnnType, atype=(usegpu ? KnetArray{dataType} : Array{dataType}), o...)
            $layer(embedding,r.w,r,gatesview($layer,r))
        end
    end
end

###
### Internal Utils
###
function Base.show(io::IO, m::AbstractRNN{P,E}) where {P,E}
    r = m.specs
    embedSize,inputSize = m.embedding !== nothing ? size(m.embedding.weight) : (nothing,r.inputSize)
    print(io, ("SRNN(ReLU){","SRNN(Tanh){","LSTM{","GRU{")[r.mode+1], "$P, $E}(input=", inputSize, ",hidden=", r.hiddenSize)
    if embedSize!=nothing; print(io,",embed=",embedSize); end
    if r.direction == 1; print(io, ",bidirectional"); end
    if r.numLayers > 1; print(io, ",layers=", r.numLayers); end
    if r.dropout > 0; print(io, ",dropout=", r.dropout); end
    if r.inputMode == 1; print(io, ",skipinput"); end
    if r.dataType != Float32; print(io, ',', r.dataType); end
    print(io, ')')
end

@inline _getEmbed(input::Integer,embed::Nothing) = (nothing,input)
@inline _getEmbed(input::Integer,embed::Embed) =
    size(embed.weight,2) == input ? (embed,size(embed.weight,1)) : error("dimension mismatch in embeddings")
@inline _getEmbed(input::Integer,embed::Integer) = (Embed(input=input,output=embed),embed)

gate_mappings(::Type{<:SRNN}) = Dict(:h=>(1,2))
gate_mappings(::Type{<:GRU})  = Dict(:r=>(1,4),:u=>(2,5),:n=>(3,6))
gate_mappings(::Type{<:LSTM}) = Dict(:i=>(1,5),:f=>(2,6),:n=>(3,7),:o=>(4,8))
const input_mappings = Dict(:i=>1,:h=>2)
const param_mappings = Dict(:w=>1,:b=>2)

gatesview(T::Type{<:AbstractRNN},r::RNN) =
    Dict((Symbol(ty,ih,g,l,d),rnnparam(r, (r.direction+1)*(l-1)+d+1, id[ihid], param; useview=true))
          for (l,d,(g,id),(ih,ihid),(ty,param)) in
              product(1:r.numLayers,0:r.direction,gate_mappings(T),
                      input_mappings,param_mappings))

# Saves from unnecessary memory taken by gatesview
function _ser(r::T, s::IdDict, m::typeof(JLDMODE)) where T <: AbstractRNN
    if !haskey(s,r)
        if r.gatesview !== nothing
            s[r] = T(_ser(r.embedding,s,m), _ser(r.params,s,m), _ser(r.specs,s,m), nothing)
        else
            s[r] = T(_ser(r.embedding,s,m), _ser(r.params,s,m), _ser(r.specs,s,m), gatesview(T,s[r.specs]))
        end
    end
    return s[r]
end

####
#### User Utils
####
"""

    PadSequenceArray(batch::Vector{Vector{T}}; direction=:Right, pad=0) where T<:Integer

Pads a batch of integer arrays with zeros

```julia
julia> PadSequenceArray([[1,2,3],[1,2],[1]])
3×3 Array{Int64,2}:
 1  2  3
 1  2  0
 1  0  0

 julia> PadSequenceArray([[1,2,3],[1,2],[1]], direction=:Left)
 3×3 Array{Int64,2}:
  1  2  3
  0  1  2
  0  0  1
```

"""
function PadSequenceArray(batch::Vector{Vector{T}}; direction=:Right, pad=0) where T<:Integer
    B      = length(batch)
    lngths = length.(batch)
    Tmax   = maximum(lngths)
    padded = Array{T}(undef,B,Tmax)
    @inbounds for n = 1:B
        if direction == :Right
            padded[n,1:lngths[n]] = batch[n]
            padded[n,lngths[n]+1:end] .= pad
        else direction == :Left
            padded[n,1:end-lngths[n]] .= pad
            padded[n,end-lngths[n]+1:end] = batch[n]
        end
    end
    return padded
end

# FIXME: long
"""
    PadRNNOutput(s::RNNOutput)
Pads a RNNOutput if it is produced by unequal length batches
`size(s.y)=(H/2H,sum_of_sequence_lengths)` becomes `(H/2H,B,Tmax)`
"""
@inline PadRNNOutput(s::RNNOutput{<:Any,<:Any,<:Any,Nothing}) = s,nothing

function PadRNNOutput(s::RNNOutput)
    d = size(s.y,1)
    B = length(s.indices)
    lngths = length.(s.indices)
    Tmax = maximum(lngths)
    mask = trues(d,B,Tmax)
    cw = []
    @inbounds for i=1:B
        y1 = s.y[:,s.indices[i]]
        df = Tmax-lngths[i]
        if df > 0
            mask[:,:,end-df+1:end] .= false
            pad = fill!(arrtype(undef,d*df),zero(eltype(arrtype)))
            ypad = reshape(cat1d(y1,pad),d,Tmax) # hcat(y1,kpad)
            push!(cw,ypad)
        else
            push!(cw,y1)
        end
    end
    RNNOutput(reshape(vcat(cw...),d,B,Tmax),s.hidden,s.memory,nothing),mask
end

"""
    _pack_sequence(batch::Vector{Vector{T}}) where T<:Integer

Packs unequal length of sequence batches for cuDNN format.
It return a tuple of tokens and batchSizes.
`tokens` consists of all input tokens in the order of cuDNN format. So, length(tokens) == sum(length,batch)
`batchSizes` keeps the information of how many instance available for a time sequence. So, lengh(bathSizes) = maximum(length,batch)

#Example
```julia
julia> _pack_sequence([[1,2,3,4],[7,6],[8,5],[9,10]])
(tokens = [1, 7, 8, 9, 2, 6, 5, 10, 3, 4], batchSizes = [4, 4, 1, 1])
```
"""
function _pack_sequence(batch::Vector{Vector{T}}) where T<:Integer
    B      = length(batch)
    Lmax   = length(first(batch))
    tokens = zeros(Int,sum(length,batch))
    bsizes = zeros(Int,Lmax)
    i = 1
    @inbounds for t = 1:Lmax
        bs = 0
        @inbounds for k = 1:B
            if t<=length(batch[k])
                tokens[i] = batch[k][t]
                bs += 1
                i  += 1
            end
        end
        bsizes[t] = bs
    end
    return (tokens=tokens,batchSizes=bsizes)
end

"""
    _batchSizes2indices(batchSizes::Vector{<:Integer})


    Finds the indices of the `tokens` belongs to for each instance.
    see `_pack_sequence` for details
"""
@inline _batchSizes2indices(batchSizes) =
    map(1:batchSizes[1]) do i
        @inbounds [i;i.+cumsum(filter(x->(x>=i),batchSizes)[1:end-1])]
    end

# FIXME: long
function _forw(rnn::AbstractRNN{<:Any,<:Embed},batch::Vector{Vector{T}},h...;
               sorted=issorted(batch,by=length),o...) where T<:Integer

    if all(length.(batch).==length(batch[1]))
        return _forw(rnn,permutedims(cat(batch...;dims=2),(2,1)),h...;o...)
    end

    if !sorted
        v     = sortperm(batch; by=length, rev=true, alg=MergeSort)
        batch = batch[v]
    end

    tokens, bsizes = _pack_sequence(batch)
    inds = _batchSizes2indices(bsizes)
    y,hidden,memory,_ = _forw(rnn,tokens,h...;batchSizes=bsizes,o...)

    if !sorted
        rev  = invperm(v)
        return y,_sort3D(hidden,rev),_sort3D(memory,rev),inds[rev]
    end
    return y,hidden,memory,inds
end

@inline function _forw(specs::RNN,params,x,h...;o...)
    y,hidden,memory,_ = rnnforw(specs,params,x,h...;o...)
    return y,hidden,memory,nothing
end

@inline _forw(rnn::AbstractRNN{<:Any,Nothing},x,h...;o...) =
    _forw(rnn.specs,rnn.params,x,h...;o...)

@inline _forw(rnn::AbstractRNN,x,h...;o...) =
    _forw(rnn.specs, rnn.params, rnn.embedding(x),h...;o...)

@inline function _forw(rnn::AbstractRNN{<:Any,<:Embed}, seq::Vector{T}, h...; batchSizes=nothing, o...) where T<:Integer
     inp = batchSizes===nothing ? reshape(seq,1,length(seq)) : seq
    _forw(rnn.specs, rnn.params, rnn.embedding(inp), h...; batchSizes=batchSizes,o...)
end

@inline _forw(rnn::AbstractRNN{<:Any,<:Nothing}, seq::Array{T},h...;o...) where T<:Integer =
    error("Integer inputs can only be used with RNNs that has embeddings.")

@inline _sort3D(hidden::Nothing,inds) = nothing
@inline _sort3D(hidden::AbstractArray,inds) = hidden[:,inds,:]
@inline _sort3D(h::KnetArray,inds)  =
    reshape(cat1d(map(i->h[:,:,i][:,inds],1:size(h,3))...),size(h))
