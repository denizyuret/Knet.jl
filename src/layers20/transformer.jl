abstract type AbstractTransformer <: Layer end

# struct PwFFN
#     din::Dense
#     dout::Dense
# end
#
# @treelike PwFFN
#
#
# "just a wrapper for two dense layer."
# PwFFN(size::Int, h::Int, act = relu) = PwFFN(
#     Dense(input=size, output=h, activation=act),
#     Dense(h, size)
# )
#
# function (pw::PwFFN)(x)::AbstractMatrix
#     # size(x) == (dims, seq_len)
#     pw.dout(pw.din(x))
# end

struct Transformer <: AbstractTransformer
    mh::MultiheadAttention
    mhn::LayerNorm
    pw::MLP
    pwn::LayerNorm
    drop::Dropout
end



"""
    Transformer(size::Int, head::Int, ps::Int; future::Bool = true, act = relu, pdrop = 0.1)
    Transformer(size::Int, head::Int, hs::Int, ps::Int; future::Bool = true, act = relu, pdrop = 0.1)
Transformer layer.
`size` is the input size. if `hs` is not specify, use `div(size, head)` as the hidden size of multi-head attention.
`ps` is the hidden size & `act` is the activation function of the positionwise feedforward layer.
When `future` is `false`, the k-th token can't see the j-th tokens where j > k. `pdrop` is the dropout rate.
"""
function Transformer(size::Int, head::Int, ps::Int; future::Bool = true, act = relu, pdrop = 0.1)
    rem(size, head) != 0 && error("size not divisible by head")
    Transformer(size, head, div(size, head), ps;future=future, act=act, pdrop=pdrop)
end

Transformer(size::Int, head::Int, hs::Int, ps::Int; future::Bool = true, act = ReLU(), pdrop = 0.1) = Transformer(
    MultiheadAttention(head, size, hs, size; future=future, pdrop=pdrop),
    LayerNorm(size),
    MLP(size, ps, size; activation=act),
    LayerNorm(size),
    Dropout(pdrop),
)

function (t::Transformer)(x, mask=nothing)
    a = t.mh(x, x, x; mask=mask)
    a = t.drop(a)
    res_a = x .+ a
    if ndims(x) == 3
        insize = size(res_a)
        res_a = reshape(res_a, insize[1], :)
    end
    res_a = t.mhn(res_a)
    pwffn = t.pw(res_a)
    pwffn = t.drop(pwffn)
    res_pwffn = res_a .+ pwffn
    res_pwffn = t.pwn(res_pwffn)
    if ndims(x) == 3
        res_pwffn = reshape(res_pwffn, :, Base.tail(insize)...)
    end
    res_pwffn
end

function Base.show(io::IO, t::Transformer)
    hs = div(size(t.mh.iqproj)[1], t.mh.head)
    h, ps = size(t.pw.layers[end])

    print(io, "Transformer(")
    print(io, "head=$(t.mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "pwffn_size=$(ps), ")
    print(io, "size=$(h)")
    print(io, ", dropout=$(t.drop.p))")
    print(io, ")")
end

struct TransformerDecoder <: AbstractTransformer
    mh::MultiheadAttention
    mhn::LayerNorm
    imh::MultiheadAttention
    imhn::LayerNorm
    pw::MLP
    pwn::LayerNorm
    drop::Dropout
end


"""
    TransformerDecoder(size::Int, head::Int, ps::Int; act = ReLU(), pdrop = 0.1)
    TransformerDecoder(size::Int, head::Int, hs::Int, ps::Int; act = ReLU(), pdrop = 0.1)
TransformerDecoder layer. Decode the value from a Encoder.
`size` is the input size. if `hs` is not specify, use `div(size, head)` as the hidden size of multi-head attention.
`ps` is the hidden size & `act` is the activation function of the positionwise feedforward layer.
`pdrop` is the dropout rate.
"""
function TransformerDecoder(size::Int, head::Int, ps::Int; act = ReLU(), pdrop = 0.1)
    rem(size, head) != 0 && error("size not divisible by head")
    TransformerDecoder(size, head, div(size, head), ps; act=act, pdrop=pdrop)
end

TransformerDecoder(size::Int, head::Int, hs::Int, ps::Int; act = ReLU(), pdrop = 0.1) = TransformerDecoder(
    MultiheadAttention(head, size, hs, size; future=false, pdrop=pdrop),
    LayerNorm(size),
    MultiheadAttention(head, size, hs, size; future=true, pdrop=pdrop),
    LayerNorm(size),
    MLP(size, ps, size; activation=act),
    LayerNorm(size),
    Dropout(pdrop),
)

function (td::TransformerDecoder)(x, m, mask=nothing)
    a = td.mh(x,x,x)
    a = td.drop(a)
    res_a = x .+ a
    res_a = td.mhn(res_a)

    ia = td.imh(res_a, m, m, mask=mask)
    ia = td.drop(ia)
    res_ia = res_a .+ ia
    if ndims(x) == 3
        insize = size(res_ia)
        res_ia = reshape(res_ia, insize[1], :)
    end
    res_ia = td.imhn(res_ia)
    pwffn = td.pw(res_ia)
    pwffn = td.drop(pwffn)
    res_pwffn = res_ia .+ pwffn
    res_pwffn = td.pwn(res_pwffn)
    if N == 3
        res_pwffn = reshape(res_pwffn, :, Base.tail(insize)...)
    end
    res_pwffn
end

function Base.show(io::IO, td::TransformerDecoder)
    hs = div(size(td.imh.iqproj)[1], td.imh.head)
    h, ps = size(td.pw.layers[end])

    print(io, "TransformerDecoder(")
    print(io, "head=$(td.mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "pwffn_size=$(ps), ")
    print(io, "size=$(h)")
    print(io, ", dropout=$(td.drop.p))")
end

mutable struct PositionEmbedding <: Layer
    trainable::Bool
    embedding::Embed
end
get_value(e::PositionEmbedding, name::Symbol, xs::NamedTuple) = pe(first(xs))

function PE(size, pos, i::Int)
    if rem(i, 2) == 0
        sin(pos/1e4^(i/size))
    else
        cos(pos/1e4^((i-1)/size))
    end
end

function PositionEmbedding(size::Int, max_len::Int = 1024; trainable::Bool = false, atype=arrtype)
    if trainable
        embedding = param(size, max_len; atype=atype, init=randn)
    else
        embedding = Matrix{Float64}(undef, size, max_len)
        for l = 1:max_len
            map!(i->PE(size, l, i), selectdim(embedding, 2, l), 1:size)
        end
        embedding = convert(atype,embedding)
    end
    PositionEmbedding(trainable, Embed(embedding))
end

function (pe::PositionEmbedding)(len::Int)
    max_len = size(pe.embedding)[2]
    if len > max_len
       if pe.trainable
           error("position embedding length exceeded")
       else
           over = similar(weight, D, len)
           copyto!(over, 1, weight, 1, length(weight))
           for l = cur_len+1:len
               over[:,l] .=  convert(atype, map(i->PE(D, l, i), 1:D))
           end
           pe.embedding = Embed(over)
       end
   end
   return pe.embedding.weight[:,1:len]
end

(pe::PositionEmbedding)(x::AbstractArray{Int}) = pe(size(x, 1))
(pe::PositionEmbedding)(x) = x .+ pe(size(x, 2))

function Base.show(io::IO, pe::PositionEmbedding)
    s, max_len = size(pe.embedding)
    if pe.trainable
        print(io, "PositionEmbedding($(s), max_len=$(max_len))")
    else
        print(io, "PositionEmbedding($(s))")
    end
end


struct TransformerModel{E, T <: AbstractTransformer, C} <: Layer
    embed::E
    transformers::T
    classifer::C
end

TransformerModel(embed, transformers) = TransformerModel(embed, transformers, identity)

function Base.show(io::IO, model::TransformerModel)
    print(io, "TransformerModel{")
    print(io, typeof(model.transformers))
    print(io, "}(")
    print(io, model.embed)
    print(io, ", ")
    print(io, model.transformers)
    if model.classifer !== identity
        print(io, ", ")
        print(io, model.classifer)
    end
    print(io, ")")
end
