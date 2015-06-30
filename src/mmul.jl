type Mmul <: Layer; w; x; y; dx; dy; n;
    Mmul(d...; init=initgaussian, o...)=new(Param(d...; init=init, o...))
    Mmul(n::Integer)=(l=new();l.n=n;l)
end

# copy(l::Mmul;o...)=Mmul(copy(l.w;o...))
update(l::Mmul;o...)=update(l.w)
setparam!(l::Mmul; o...)=setparam!(l.w; o...)

function forw(l::Mmul, x; o...)
    initforw(l, x)
    (x0,x1)=(zero(eltype(l.x)), one(eltype(l.x)))
    A_mul_B!(l.y, l.w.data, l.x)                # l.y = l.w.data * l.x
    return l.y
end

function back(l::Mmul, dy; returndx=true, o...)
    initback(l, dy, returndx)
    (x0,x1)=(zero(eltype(l.x)), one(eltype(l.x)))
    A_mul_Bt!(l.w.diff, l.dy, l.x)              # l.w.diff = l.dy * l.x'
    returndx && At_mul_B!(l.dx, l.w.data, l.dy)	# l.dx = l.w.data' * l.dy
end

function initforw(l::Mmul, x)
    (xrows,xcols) = size2(x)
    l.x = (size(x)==(xrows,xcols) ? x : reshape(x, xrows, xcols))
    isdefined(l,:w) || (l.w = Param(eltype(x), l.n, xrows; init=initgaussian))
    (wrows, wcols) = size2(l.w.data)
    @assert ndims(l.w.data) == 2
    @assert typeof(l.w.data) == typeof(l.x)
    @assert eltype(l.w.data) == eltype(l.x)
    @assert xrows == wcols
    similar!(l, :y, l.w.data, (wrows, xcols))
end

function initback(l::Mmul, dy, returndx)
    @assert issimilar(dy, l.y)
    l.dy = dy
    similar!(l.w, :diff, l.w.data)
    returndx && similar!(l, :dx, l.x)
end

