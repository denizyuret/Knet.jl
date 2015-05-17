using Base.LinAlg.BLAS: gemm!

type Mmul <: Layer; w; x; y; dx; dy; Mmul(w::Param)=new(w); end
Mmul(w;o...)=Mmul(Param(w;o...))
Mmul(d::Integer...;o...)=Mmul(Param(randn(d)*0.01;o...))

copy(l::Mmul;o...)=Mmul(copy(l.w;o...))
update(l::Mmul;o...)=update(l.w)
setparam!(l::Mmul; o...)=setparam!(l.w; o...)

function forw(l::Mmul, x; o...)
    initforw(l, x)
    (x0,x1)=(zero(eltype(l.x)), one(eltype(l.x)))
    gemm!('N','N',x1,l.w.data,l.x,x0,l.y) # l.y = l.w.data * l.x
    return l.y
end

function back(l::Mmul, dy; returndx=true, o...)
    initback(l, dy, returndx)
    (x0,x1)=(zero(eltype(l.x)), one(eltype(l.x)))
    gemm!('N','T',x1,l.dy,l.x,x0,l.w.diff) # l.w.diff = l.dy * l.x'
    returndx && gemm!('T','N',x1,l.w.data,l.dy,x0,l.dx) # l.dx = l.w.data' * l.dy
end

function initforw(l::Mmul, x)
    @assert ndims(l.w.data) == 2
    (wrows, wcols) = size(l.w.data)
    xcols = (ndims(x) == 1 ? 1 : size(x, ndims(x)))
    xrows = div(length(x),xcols)
    @assert xrows == wcols
    l.x = (size(x)==(xrows,xcols) ? x : reshape(x, xrows, xcols))
    similar!(l, :y, l.w.data, (wrows, xcols))
end

function initback(l::Mmul, dy, returndx)
    @assert issimilar(dy, l.y)
    l.dy = dy
    similar!(l.w, :diff, l.w.data)
    returndx && similar!(l, :dx, l.x)
end
