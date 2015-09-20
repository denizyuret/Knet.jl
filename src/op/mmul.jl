type Mmul <: Op; w; x; ybuf; dx; Mmul(p::KUparam)=new(p); end

Mmul(d...; o...)=Mmul(KUparam(d...; o...))
Mmul(n::Integer; o...)=Mmul(n, 0; o...)

params(l::Mmul)=Any[l.w]
ninputs(l::Mmul)=1
overwrites(l::Mmul)=false
back_reads_x(l::Mmul)=true
back_reads_y(l::Mmul)=false

# TODO: consolidate part of this code in common with conv.jl
# TODO: upgrade conv.jl, pool.jl, CUDNN.jl to the new CUDNN library

function forw(l::Mmul, x; y=nothing, o...)
    l.x = x
    (y, w, x) = initforw(l, x, y; o...)
    A_mul_B!(y, w, x) # y = w * x
end

forw(l::Mmul, ::Void; o...)=nothing

function initforw(l::Mmul, x, y; train=true, o...)
    (xrows, xcols) = size2(x)
    (wrows, wcols) = size(l.w)
    if isempty(l.w) 
        nz(l.w,:init,nothing) || (l.w.init = randn!; l.w.initp = (0,0.01))
        wcols=xrows
        init(l.w, eltype(x), (wrows, wcols))
    end
    ndims(l.w) == 2 || error("ndims(w)!=2")
    eltype(l.w) == eltype(x) || error("eltype mismatch")
    xrows == wcols || error("xrows!=wcols")
    y == nothing && (y = dsimilar!(l, :ybuf, x, (wrows, xcols)))
    atype(x) == atype(y) || error("atype mismatch")
    eltype(x) == eltype(y) || error("eltype mismatch")
    size(y) == (wrows, xcols) || error("ysize mismatch: $(size(y)) $((wrows,xcols))")
    return ((!train && nz(l.w, :average, false)) ?
            (y, l.w.avg, x) :
            (y, l.w.arr, x))
end

function back(l::Mmul, dy; dx=nothing, x=l.x, incr=false, returndx=true, o...)
    (dy == nothing || x == nothing) && (return nothing) # TODO: are we sure about x?
    initback(l, dy, x, incr)
    if incr
        A_mul_Bt!(l.w.inc, dy, x)
        axpy!(1, l.w.inc, l.w.diff)
    else
        A_mul_Bt!(l.w.diff, dy, x)    # dw = dy * x'
    end
    if returndx
        dx = initbackx(l,x,dx)
        At_mul_B!(dx, l.w.arr, dy)    # dx = w' * dy
    end
end

function initback(l::Mmul, dy, x, incr)
    atype(dy) == atype(x) || error("atype mismatch")
    eltype(dy) == eltype(x) || error("eltype mismatch")
    size(dy) == ysize(l,x) || error("ysize mismatch")
    similar!(l.w, :diff, l.w.arr)
    incr && similar!(l.w, :inc, l.w.arr)
end

function initbackx(l::Mmul, x, dx)
    dx == nothing && (dx = similar!(l, :dx, x))
    issimilar(dx,x) || error("Gradient mismatch")
    return dx
end

function ysize(l::Mmul,x)
    (xrows,xcols) = size2(x)
    (wrows,wcols) = size(l.w)
    wcols==0 || wcols==xrows || error("Bad input size")
    return (wrows,xcols)
end

ysize(l::Mmul, ::Void)=nothing
