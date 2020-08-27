export Embed
using Knet: atype
using AutoGrad: Param

"""
    Embed(embedsize, vocabsize; atype, init)
    Embed(weights)

References:
* torch.nn.Embed
* tf.keras.layers.Embed
"""
struct Embed; w;
    Embed(w)=new(w isa Param ? w : Param(w))
end

function Embed(embedsize::Integer, vocabsize::Integer; atype=atype(), init=𝑵(1))
    Embed(convert(atype, init(embedsize,vocabsize)))
end

function (l::Embed)(x)
    l.w[:,x]
end

# function (l::Embed)(x::MaskedArray)
#     a = l(x.array)
#     m = (x.mask === nothing ? nothing : reshape(x.mask, 1, size(x.mask)...))
#     MaskedArray(a, m)
# end
