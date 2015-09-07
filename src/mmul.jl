type Mmul <: Layer; w; x; y; dx; Mmul(p::KUparam)=new(p); end

Mmul(d...; o...)=Mmul(KUparam(d...; o...))
Mmul(n::Integer; o...)=Mmul(n, 0; o...)

param(l::Mmul)=l.w
overwrites(l::Mmul)=false
back_reads_x(l::Mmul)=true
back_reads_y(l::Mmul)=false


function forw(l::Mmul, x; y=nothing, o...)
    (y, w, x) = initforw(l, x, y; o...)
    A_mul_B!(y, w, x) # y = w * x
end

forw(l::Mmul, ::Void; o...)=nothing

function initforw(l::Mmul, x, y; predict=false, o...)
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
    # if a y keyword argument has been specified, try using that
    # otherwise try l.y which is the array from the previous call
    # if the size is wrong try resizing the array
    y != nothing && (l.y = y)
    dsimilar!(l, :y, l.x, (wrows, xcols)) # TODO: this may end up not using user supplied y!
    return ((predict && nz(l.w, :average, false)) ?
            (l.y, l.w.avg, l.x) :
            (l.y, l.w.arr, l.x))
end

function back(l::Mmul, dy; dx=nothing, x=l.x, incr=false, returndx=true, o...)
    @assert issimilar(dy, l.y)
    initback(l, incr, returndx, dx)
    if incr
        A_mul_Bt!(l.w.inc, dy, x)
        axpy!(1, l.w.inc, l.w.diff)
    else
        A_mul_Bt!(l.w.diff, dy, x)        # dw = dy * x'
    end
    if returndx
        At_mul_B!(l.dx, l.w.arr, dy)    # dx = w' * dy
    end
end

function initback(l::Mmul, incr, returndx, dx)
    similar!(l.w, :diff, l.w.arr)
    incr && similar!(l.w, :inc, l.w.arr)
    if returndx
        dx != nothing && (l.dx=dx)
        similar!(l, :dx, l.x)
    end
end

# function initback(l::Mmul, dy, x, incr)
#     @assert issimilar(dy, l.y)
#     x != nothing && (l.x = x)
#     if !incr
#         initdiff(l.w)
#         return (l.w.diff, dy, l.x)
#     else
#     end
# end

# function initbackx(l::Mmul, dy, dx)
#     dx != nothing && (l.dx = dx)
#     similar!(l, :dx, l.x)
#     return (l.dx, l.w.arr, dy)
# end

    # if a y keyword argument has been specified, try using that
    # otherwise try l.y which is the array from the previous call
    # if the size is wrong try resizing the array
    # y != nothing && (l.y = y)
    