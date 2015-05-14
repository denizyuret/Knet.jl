type Mmul <: Layer; w; x; y; dx; dy; Mmul()=new(); end
Mmul(w::Param)=(l=Mmul();l.w=w;l)
Mmul(w;a...)=Mmul(Param(w;a...))
Mmul(d::Integer...;a...)=Mmul(Param(randn(d)*0.01;a...))

update(l::Mmul)=update(l.w)
setparam!(l::Mmul,k,v)=setparam!(l.w,k,v)

function forw(l::Mmul, x; o...)
    initforw(l, x)
    @into! l.y = l.w.data * l.x
    return l.y
end

function back(l::Mmul, dy; dx=true, o...)
    initback(l, dy, dx)
    @into! l.w.diff = l.dy * l.x'
    if dx
        @into! l.dx = l.w.data' * l.dy
        return l.dx
    end
end

function initforw(l::Mmul, x)
    @assert ndims(l.w.data) == 2
    (wrows, wcols) = size(l.w.data)
    xcols = (ndims(x) == 1 ? 1 : size(x, ndims(x)))
    xrows = div(length(x),xcols)
    @assert xrows == wcols
    l.x = (size(x)==(xrows,xcols) ? x : reshape(x, xrows, xcols))
    chksize(l, :y, l.w.data, (wrows, xcols))
end

function initback(l::Mmul, dy, dx)
    @assert size(dy) == size(l.y)
    l.dy = dy
    chksize(l.w, :diff, l.w.data)
    dx && chksize(l, :dx, l.x)
end
