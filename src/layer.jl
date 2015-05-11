# Layer: Fully connected layer.  All fields are optional in the
# constructor, operations corresponding to uninitialized fields are
# not performed.

type Layer <: AbstractLayer
    w; b; f; fx; dw; db; pw; pb; x; y; dx; dy; dropout; xdrop; 
    Layer(;a...)=setparam!(new();a...)
end

# TODO: Do we still need all this?  Deprecate unused ones.
atype(w) = (usegpu ? CudaArray(w) : w)
Layer(w; o...) = (l=Layer(); l.w=atype(w); setparam!(l; o...); l)
Layer(w, b; o...) = (l=Layer(); l.w=atype(w); l.b=atype(b); setparam!(l; o...); l)
Layer(f::Function, w; o...) = (l=Layer(w; o...); l.f=f; l)
Layer(f::Function, w, b; o...) = (l=Layer(w,b; o...); l.f=f; l)
Layer(c::Integer, r::Integer; bias=true, o...) = (w=float32(randn(r,c)*0.01);bias ? Layer(w,zeros(Float32, r, 1);o...) : Layer(w; o...))
Layer(f::Function, c::Integer, r::Integer; o...) = (l=Layer(c,r;o...); l.f=f; l)

function forw(l::Layer, x, apply_fx=true)
    initforw(l, x)  # sets l.x
    isdefined(l,:fx) && apply_fx && l.fx(l,l.x)
    isdefined(l,:w) ? (@into! l.y = l.w * l.x) : (l.y = l.x)
    isdefined(l,:b) && (@in1! l.y .+ l.b)
    isdefined(l,:f) && l.f(l,l.y)
    return l.y
end

function initforw(l, x)
    l.x = matrixwithrows(x, size(l.w, 2))
    chksize(l, :y, l.w, (size(l.w,1),size(l.x,2)))
end

function back(l::Layer, dy, return_dx=true)
    initback(l, dy, return_dx)  # sets l.dy
    isdefined(l,:f) && l.f(l,l.y,l.dy)
    isdefined(l,:b) && sum!(l.db,l.dy)
    isdefined(l,:w) && (@into! l.dw = l.dy * l.x')
    return_dx || return
    isdefined(l,:w) ? (@into! l.dx = l.w' * l.dy) : (l.dx = l.dy)
    isdefined(l,:fx) && l.fx(l,l.x,l.dx)
    return l.dx
end

function initback(l, dy, return_dx)
    @assert size(dy) == size(l.y)
    l.dy = dy
    isdefined(l,:b) && chksize(l, :db, l.b)
    isdefined(l,:w) && chksize(l, :dw, l.w)
    return_dx && isdefined(l,:w) && chksize(l, :dx, l.x)
end

function matrixwithrows(x,xrows)
    (ndims(x)==2) && (size(x,1)==xrows) && return x
    @assert isa(x, Tensor)
    xcols = size(x, ndims(x))
    @assert length(x)/xcols == xrows
    reinterpret(eltype(x), x.data, (xrows, xcols))
end

