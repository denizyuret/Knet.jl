type Mmul <: Layer; w; x; y; dx; dy; Mmul(p::KUparam)=new(p); end
param(l::Mmul)=l.w
default_init(::Type{Mmul})=initgaussian

Mmul(d...; init=default_init(Mmul), o...)=Mmul(KUparam(d...; init=init, o...))
Mmul(n::Integer)=Mmul(KUparam(n,0)) # cannot specify init here

function forw(l::Mmul, x; o...)
    initforw(l, x)
    A_mul_B!(l.y, l.w, l.x) # l.y = l.w * l.x
    return l.y
end

function back(l::Mmul, dy; returndx=true, o...)
    initback(l, dy, returndx)
    A_mul_Bt!(diff(l.w), l.dy, l.x)                # l.dw = l.dy * l.x'
    returndx && (At_mul_B!(l.dx, l.w, l.dy); l.dx) # l.dx = l.w' * l.dy
end

function initforw(l::Mmul, x)
    l.x = x
    (xrows, xcols) = size2(l.x)
    (wrows, wcols) = size(l.w)
    isempty(l.w) && (wcols=xrows; l.w=KUparam(eltype(x), (wrows, wcols); init=default_init(Mmul)))
    @assert ndims(l.w) == 2
    @assert eltype(l.w) == eltype(x)
    @assert xrows == wcols
    dsimilar!(l, :y, l.x, (wrows, xcols))
end

function initback(l::Mmul, dy, returndx)
    @assert issimilar(dy, l.y)
    l.dy = dy
    initdiff(l.w)
    returndx && similar!(l, :dx, l.x)
end

