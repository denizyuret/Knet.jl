export Embed

struct Embed; w; end

function Embed(vocabsize,embedsize)
    Embed(param(embedsize,vocabsize))
end

function (l::Embed)(x)
    l.w[:,x]
end

# function (l::Embed)(x::MaskedArray)
#     a = l(x.array)
#     m = (x.mask === nothing ? nothing : reshape(x.mask, 1, size(x.mask)...))
#     MaskedArray(a, m)
# end
