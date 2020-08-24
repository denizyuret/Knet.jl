export Embedding

"""
    Embedding(vocabsize, embedsize)
    Embedding(weights)

References:
* torch.nn.Embedding
* tf.keras.layers.Embedding
"""
struct Embedding; w; end

function Embedding(vocabsize,embedsize)
    Embedding(param(embedsize,vocabsize))
end

function (l::Embedding)(x)
    l.w[:,x]
end

# function (l::Embedding)(x::MaskedArray)
#     a = l(x.array)
#     m = (x.mask === nothing ? nothing : reshape(x.mask, 1, size(x.mask)...))
#     MaskedArray(a, m)
# end
