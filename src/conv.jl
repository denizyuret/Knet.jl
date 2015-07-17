# TODO: generalize to N-D
# TODO: cpu implementation

type Conv <: Layer; w; x; y; dx; dy; n; Conv(p::KUparam)=new(p); end
param(l::Conv)=l.w
default_init(::Type{Conv})=initxavier
Conv(d...; init=default_init(Conv), o...)=Conv(KUparam(d...; init=init, o...))
Conv(nout::Integer, width::Integer)=Conv(KUparam(width, width, 0, nout))


if GPU

function forw(l::Conv, x::KUdense{CudaArray}; o...)
    initforw(l,x)
    cudnnConvolutionForward(l.x.arr, l.w.arr, l.y.arr)
    return l.y
end

function initforw(l::Conv, x::KUdense{CudaArray})
    xchannels = size(x)[end-1]  # x dims are (x1, x2, ..., channels, images)
    wsize = [size(l.w)...]
    isempty(l.w) && (wsize[end-1]=xchannels; l.w=KUparam(eltype(x), tuple(wsize...); init=default_init(Conv)))
    @assert eltype(x) == eltype(l.w) "$(eltype(x)) != $(eltype(l.w))"
    @assert ndims(x) == ndims(l.w)
    @assert xchannels == wsize[end-1]
    similar!(l, :y, x, cudnnGetConvolutionNdForwardOutputDim(x.arr, l.w.arr))
    l.x = x
end

function back(l::Conv, dy::KUdense{CudaArray}; returndx=true, o...)
    initback(l, dy, returndx)
    cudnnConvolutionBackwardFilter(l.x.arr, l.dy.arr, l.w.diff)
    returndx && cudnnConvolutionBackwardData(l.w.arr, l.dy.arr, l.dx.arr)
end

function initback(l::Conv, dy::KUdense{CudaArray}, returndx)
    @assert issimilar(dy, l.y)
    # @assert eltype(dy) == eltype(l.y)
    # l.dy = (size(dy) == size(l.y) ? dy : reshape(dy, size(l.y)))
    initdiff(l.w)
    returndx && similar!(l, :dx, l.x)
end

# function forw(l::Conv, x; o...)
#     # a = KUnet.Atype
#     # KUnet.atype(CudaDynArray)
#     y = forw(copy(l), CudaDynArray(x); o...)
#     # KUnet.atype(a)
#     l.x = x
#     l.y = to_host(y)
# end

# function back(l::Conv, dy; o...)
#     # error("CPU conv not implemented")
#     a = KUnet.Atype
#     KUnet.atype(CudaDynArray)
#     ll = copy(l); ll.y = CudaDynArray(l.y); ll.x = CudaDynArray(l.x)
#     dx = back(ll, CudaDynArray(dy); o...)
#     KUnet.atype(a)
#     l.dy = dy
#     l.w.diff = to_host(ll.w.diff)
#     l.dx = to_host(dx)
# end

else # if GPU
warn("No cpu conv")
# Use the default behavior (error):
## forw(l::Conv,x;o...)=(l.x=l.y=x)
## back(l::Conv,dy;o...)=(l.w.diff=l.w.arr; l.dx=l.dy=dy)
end # if GPU

