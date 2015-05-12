type Mmul <: Layer; w::Param; x; y; dx; dy; Mmul(w)=new(w); end
Mmul(w::Array;a...)=Mmul(Param(w;a...))
Mmul(w::CudaArray;a...)=Mmul(Param(w;a...))
Mmul(d::Integer...;a...)=Mmul(Param(float32(randn(d)*0.01));a...)

update(l::Mmul)=update(l.w)

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
    (wrows, wcols) = size(l.w.data)
    (xrows, xcols) = (wcols, size(x, ndims(x)))
    if ((ndims(x)==2) && (size(x,1)==xrows))
        l.x = x
    else
        @assert isa(x, Tensor)
        @assert length(x)/xcols == xrows
        l.x = reinterpret(eltype(x), x.data, (xrows, xcols))
    end
    chksize(l, :y, l.w.data, (wrows, xcols))
end

function initback(l::Mmul, dy, dx)
    @assert size(dy) == size(l.y)
    l.dy = dy
    chksize(l.w, :diff, l.w.data)
    dx && chksize(l, :dx, l.x)
end
