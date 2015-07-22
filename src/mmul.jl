type Mmul <: Layer; w; x; y; dx; dy; Mmul(p::KUparam)=new(p); end

Mmul(d...; o...)=Mmul(KUparam(d...; o...))
Mmul(n::Integer; o...)=Mmul(n, 0; o...)

param(l::Mmul)=l.w

function forw(l::Mmul, x; o...)
    (y, w, x) = initforw(l, x; o...)
    A_mul_B!(y, w, x) # y = w * x
    return y
end

function back(l::Mmul, dy; returndx=true, o...)
    (dw, dy, x) = initback(l, dy)
    A_mul_Bt!(dw, dy, x)        # dw = dy * x'
    if returndx
        (dx, w, dy) = initbackx(l, dy)
        At_mul_B!(dx, w, dy)    # dx = w' * dy
    end
end

function initforw(l::Mmul, x; predict=false, o...)
    l.x = x
    (xrows, xcols) = size2(l.x)
    (wrows, wcols) = size(l.w)
    if isempty(l.w) 
        nz(l.w,:init,nothing) || (l.w.init = initgaussian)
        wcols=xrows
        init(l.w, eltype(x), (wrows, wcols))
    end
    @assert ndims(l.w) == 2
    @assert eltype(l.w) == eltype(x)
    @assert xrows == wcols
    dsimilar!(l, :y, l.x, (wrows, xcols))
    return ((predict && nz(l.w, :average, false)) ?
            (l.y, l.w.avg, l.x) :
            (l.y, l.w.arr, l.x))
end

function initback(l::Mmul, dy)
    @assert issimilar(dy, l.y)
    l.dy = dy
    similar!(l.w, :diff, l.w.arr)
    return (l.w.diff, dy, l.x)
end

function initbackx(l::Mmul, dy)
    similar!(l, :dx, l.x)
    return (l.dx, l.w.arr, dy)
end
