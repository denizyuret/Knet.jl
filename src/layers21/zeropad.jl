export ZeroPad
using Knet.Ops21: zeropad

mutable struct ZeroPad; padding; xsize; y; end

ZeroPad(p) = ZeroPad(p, nothing, nothing)

function (z::ZeroPad)(x)
    if size(x) !== z.xsize || typeof(x) !== typeof(z.y)
        z.y = zeropad(x, z.padding)
        z.xsize = size(x)
    else
        z.y = zeropad(x, z.padding; y=z.y, fillzero=false)
    end
    return z.y
end
