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

forw(l::Mmul, ::Void; o...)=nothing

function forw(l::Mmul, x; y=nothing, o...)
    l.x = x
    (y, w, x) = initforw(l, x, y; o...)
    A_mul_B!(y, w, x) # y = w * x
end

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
    # atype(x) == atype(y) || error("atype mismatch")
    eltype(x) == eltype(y) || error("eltype mismatch")
    size(y) == (wrows, xcols) || error("ysize mismatch: $(size(y)) $((wrows,xcols))")
    return ((!train && nz(l.w, :average, false)) ?
            (y, l.w.avg, x) :
            (y, l.w.arr, x))
end

# dw = dy * x'
# dx = w' * dy
function back(l::Mmul, dy; dx=nothing, x=l.x, incr=false, returndx=true, o...)
    if !isdefined(l.w,:diff)
        if x != nothing && issparse(x)
            l.w.diff = CudaSparseMatrixCSR(spzeros(eltype(x), size(l.w)...))
        else
            l.w.diff = fill!(similar(l.w.arr),0)
        end
    end
    if dy == nothing || x == nothing
        !incr && fill!(l.w.diff,0)
    elseif incr
        !isdefined(l.w,:inc) && (l.w.inc = similar(l.w.diff))
        A_mul_Bt!(l.w.inc, dy, x)
        axpy!(1, l.w.inc, l.w.diff)
    else
        A_mul_Bt!(l.w.diff, dy, x)
    end
    if returndx
        if dy == nothing
            error()
        else
            dx == nothing && (dx = similar!(l, :dx, dy, size(x))) # x can be sparse, dy/dx always dense
            At_mul_B!(dx, l.w.arr, dy)
        end
        return dx
    end
end

function ysize(l::Mmul,x)
    (xrows,xcols) = size2(x)
    (wrows,wcols) = size(l.w)
    wcols==0 || wcols==xrows || error("Bad input size")
    return (wrows,xcols)
end

ysize(l::Mmul, ::Void)=nothing


### DEAD CODE:

# ;; This buffer is for notes you don't want to save, and for Lisp evaluation.
# ;; If you want to create a file, visit that file with C-x C-f,
# ;; then enter the text in that file's own buffer.

# x(ndims,ninst)  x(ninst,ndims)
# y  =  w * x	y  =  x * w
# dw = dy * x'    dw = x' * dy
# dx = w' * dy    dx = dy * w'

# function forwT(l::Mmul, xT::CudaSparseMatrixCSR; y=nothing, o...)
#     # xT has been transposed!
#     @show map(summary, (l.w, xT, y))
#     l.x = xT
#     (y, yT, w, xT) = initforwT(l, xT, y; o...)
#     CUSPARSE.csrmm2!('N','T',1f0,xT,w,0f0,yT,'O') # yT = xT * w'
#     CUBLAS.geam!('T','T',1f0,yT,0f0,yT,y) # y = yT' need temp variable :(
# end

# function initforwT(l::Mmul, xT::CudaSparseMatrixCSR, y; train=true, o...)
#     (xcols, xrows) = size2(xT)  # xT has been transposed!
#     (wrows, wcols) = size(l.w)
#     if isempty(l.w) 
#         nz(l.w,:init,nothing) || (l.w.init = randn!; l.w.initp = (0,0.01))
#         wcols=xrows
#         init(l.w, eltype(xT), (wrows, wcols))
#     end
#     ndims(l.w) == 2 || error("ndims(w)!=2")
#     eltype(l.w) == eltype(xT) || error("eltype mismatch")
#     xrows == wcols || error("xrows!=wcols")

#     y == nothing && (y = dsimilar!(l, :ybuf, xT, (wrows, xcols)))
#     yT = dsimilar!(l, :ytbuf, xT, (xcols, wrows))
#     # atype(x) == atype(y) || error("atype mismatch")
#     eltype(xT) == eltype(y) == eltype(yT) || error("eltype mismatch")
#     size(y) == (wrows, xcols) || error("ysize mismatch: $(size(y)) $((wrows,xcols))")
#     size(yT) == (xcols, wrows) || error("yTsize mismatch: $(size(yT)) $((xcols,wrows))")
#     return ((!train && nz(l.w, :average, false)) ?
#             (y, yT, l.w.avg, xT) :
#             (y, yT, l.w.arr, xT))
# end

#     if isa(x, CudaSparseMatrixCSR)
#         returndx && error("Cannot return dx for sparse x") # TODO: are we sure?
#         return backT(l,dy,x,incr;o...)
#     end
# # sparse back with xT
# function backT(l::Mmul, dy, xT::CudaSparseMatrixCSR, incr; o...)
#     # dw = dy * x'
#     dyS = sparse(dy)            # TODO: get rid of alloc
#     isdefined(l.w, :diff) || (l.w.diff = CudaSparseMatrixCSR(spzeros(eltype(xT), size(l.w)...)))
#     if incr
#         isdefined(l.w, :inc) || (l.w.inc = similar(l.w.diff))
#         CUSPARSE.gemm!('N','N',dy,xT,l.w.inc,'O')
#         axpy!(1, l.w.inc, l.w.diff) # TODO: check impl, sparsity patterns may be the same
#     else
#         CUSPARSE.gemm!('N','N',dy,xT,l.w.diff,'O')
#     end
# end

# function ysize(l::Mmul,x::CudaSparseMatrixCSR)
#     # x is transposed
#     (xcols,xrows) = size2(x)
#     (wrows,wcols) = size(l.w)
#     wcols==0 || wcols==xrows || error("Bad input size")
#     return (wrows,xcols)
# end


# function initbackd(l::Mmul, dy, x)
#     atype(dy) == atype(x) || error("atype mismatch")
#     eltype(dy) == eltype(x) || error("eltype mismatch")
#     size(dy) == ysize(l,x) || error("ysize mismatch")
#     similar!(l.w, :diff, l.w.arr)
#     incr && similar!(l.w, :inc, l.w.arr)
# end

# function initbacki(l::Mmul, dy, x, incr)
#     atype(dy) == atype(x) || error("atype mismatch")
#     eltype(dy) == eltype(x) || error("eltype mismatch")
#     size(dy) == ysize(l,x) || error("ysize mismatch")
#     similar!(l.w, :diff, l.w.arr)
#     incr && similar!(l.w, :inc, l.w.arr)
# end

# function initbackx(l::Mmul, x, dx)
#     dx == nothing && (dx = similar!(l, :dx, x))
#     issimilar(dx,x) || error("Gradient mismatch")
#     return dx
# end

