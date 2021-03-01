export SqueezeExcitation
using Knet.Ops20: pool

struct SqueezeExcitation
    block
    SqueezeExcitation(l...) = new(Sequential(l...))
end

function (s::SqueezeExcitation)(x)
    y = pool(x; mode=1, window=size(x)[1:end-2])
    y = s.block(y)
    return x .* y
end
