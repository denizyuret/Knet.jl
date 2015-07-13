type Mmul <: Layer; w; x; y; dx; dy;
    Mmul(d...; init=initgaussian, o...)=new(Param(d...; init=init, o...))
    Mmul(n::Integer)=new(Param(n,0))
end

# copy(l::Mmul;o...)=Mmul(copy(l.w;o...))
update(l::Mmul;o...)=update(l.w)
setparam!(l::Mmul; o...)=setparam!(l.w; o...)

# The input could be a tensor.  In which case perform internal calculations in 2D:
mat2d(x)=(ndims(x)==2 ? x : reshape(x, size2(x)))

function forw(l::Mmul, x; o...)
    initforw(l, x)
    A_mul_B!(l.y, l.w.data, mat2d(l.x)) # l.y = l.w.data * l.x
    return l.y
end

function back(l::Mmul, dy; returndx=true, o...)
    initback(l, dy, returndx)
    A_mul_Bt!(l.w.diff, l.dy, mat2d(l.x)) # l.w.diff = l.dy * l.x'
    returndx && (At_mul_B!(mat2d(l.dx), l.w.data, l.dy); l.dx)	# l.dx = l.w.data' * l.dy
end

function initforw(l::Mmul, x)
    l.x = x
    x2 = mat2d(x)
    (xrows, xcols) = size(x2)
    (wrows, wcols) = size(l.w.data)
    wcols==0 && (wcols=xrows; l.w.data = initgaussian((gpu()?CudaDynArray:Array)(eltype(x),wrows,wcols)))
    @assert ndims(l.w.data) == 2
    @assert typeof(l.w.data) == typeof(x2) "typeof(l.w.data)=$(typeof(l.w.data)) typeof(x2)=$(typeof(x2))"
    @assert eltype(l.w.data) == eltype(x)
    @assert xrows == wcols
    similar!(l, :y, l.w.data, (wrows, xcols))
end

function initback(l::Mmul, dy, returndx)
    @assert issimilar(dy, l.y)
    l.dy = dy
    similar!(l.w, :diff, l.w.data)
    returndx && similar!(l, :dx, l.x)
end

