# TODO: generalize to N-D
# TODO: cpu implementation

type Conv <: Layer; w; x; y; dx; dy; Conv(p::KUparam)=new(p); end

Conv(d...; o...)=Conv(KUparam(d...; o...))
Conv(nout::Integer, width::Integer; o...)=Conv(KUparam(width, width, 0, nout; o...))

param(l::Conv)=l.w

function forw(l::Conv, x; o...)
    initforw(l,x)
    cudnnConvolutionForward(l.x, l.w, l.y)
    return l.y
end

function initforw(l::Conv, x)
    xchannels = size(x)[end-1]  # x dims are (x1, x2, ..., channels, images)
    wsize = [size(l.w)...]
    if isempty(l.w) 
        nz(l.w,:init,nothing) || (l.w.init = initxavier)
        wsize[end-1]=xchannels
        init(l.w, eltype(x), tuple(wsize...))
    end
    @assert eltype(x) == eltype(l.w) "$(eltype(x)) != $(eltype(l.w))"
    @assert ndims(x) == ndims(l.w)
    @assert xchannels == wsize[end-1]
    similar!(l, :y, x, cudnnGetConvolutionNdForwardOutputDim(x, l.w))
    l.x = x
end

function back(l::Conv, dy; returndx=true, o...)
    initback(l, dy, returndx)
    cudnnConvolutionBackwardFilter(l.x, l.dy, l.w)
    returndx && cudnnConvolutionBackwardData(l.w, l.dy, l.dx)
end

function initback(l::Conv, dy, returndx)
    @assert issimilar(dy, l.y)
    # @assert eltype(dy) == eltype(l.y)
    # l.dy = (size(dy) == size(l.y) ? dy : reshape(dy, size(l.y)))
    l.dy = dy
    initdiff(l.w)
    returndx && similar!(l, :dx, l.x)
end


# Make things work with KUdense

CUDNN.cudnnGetConvolutionNdForwardOutputDim(x::KUdense, w::KUparam)=cudnnGetConvolutionNdForwardOutputDim(x.arr, w.arr)
CUDNN.cudnnConvolutionForward(x::KUdense, w::KUparam, y::KUdense)=(cudnnConvolutionForward(x.arr, w.arr, y.arr);y)
CUDNN.cudnnConvolutionBackwardFilter(x::KUdense, dy::KUdense, w::KUparam)=(cudnnConvolutionBackwardFilter(x.arr, dy.arr, w.diff);w)
CUDNN.cudnnConvolutionBackwardData(w::KUparam, dy::KUdense, dx::KUdense)=(cudnnConvolutionBackwardData(w.arr, dy.arr, dx.arr);dx)

# Make things work with CPU (for now)

CUDNN.cudnnGetConvolutionNdForwardOutputDim(x::Array, w::Array)=cudnnGetConvolutionNdForwardOutputDim(CudaArray(x),CudaArray(w))
CUDNN.cudnnConvolutionForward(x::Array, w::Array, y::Array)=(y1=CudaArray(y);cudnnConvolutionForward(CudaArray(x), CudaArray(w), y1);copy!(y,1,y1,1,length(y)))
CUDNN.cudnnConvolutionBackwardFilter(x::Array, dy::Array, w::Array)=(w1=CudaArray(w);cudnnConvolutionBackwardFilter(CudaArray(x), CudaArray(dy), w1); copy!(w,1,w1,1,length(w)))
CUDNN.cudnnConvolutionBackwardData(w::Array, dy::Array, dx::Array)=(dx1=CudaArray(dx);cudnnConvolutionBackwardData(CudaArray(w), CudaArray(dy), dx1); copy!(dx,1,dx1,1,length(dx)))



# DEAD CODE:

# function forw(l::Conv, x; o...)
#     # a = Knet.Atype
#     # Knet.atype(CudaDynArray)
#     y = forw(copy(l), CudaDynArray(x); o...)
#     # Knet.atype(a)
#     l.x = x
#     l.y = to_host(y)
# end

# function back(l::Conv, dy; o...)
#     # error("CPU conv not implemented")
#     a = Knet.Atype
#     Knet.atype(CudaDynArray)
#     ll = copy(l); ll.y = CudaDynArray(l.y); ll.x = CudaDynArray(l.x)
#     dx = back(ll, CudaDynArray(dy); o...)
#     Knet.atype(a)
#     l.dy = dy
#     l.w.diff = to_host(ll.w.diff)
#     l.dx = to_host(dx)
# end

# else # if GPU
# warn("No cpu conv")
# # Use the default behavior (error):
# ## forw(l::Conv,x;o...)=(l.x=l.y=x)
# ## back(l::Conv,dy;o...)=(l.w.diff=l.w.arr; l.dx=l.dy=dy)
# end # if GPU

