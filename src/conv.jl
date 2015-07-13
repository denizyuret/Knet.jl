# TODO: generalize to N-D
# TODO: cpu implementation

type Conv <: Layer; w; x; y; dx; dy; n;
    Conv(d...; init=initxavier, o...)=new(Param(d...; init=init, o...))
    Conv(nout::Integer, width::Integer)=new(Param(width, width, 0, nout))
end

update(l::Conv; o...)=update(l.w; o...)
setparam!(l::Conv; o...)=setparam!(l.w; o...)

if GPU

function forw(l::Conv, x::AbstractCudaArray; o...)
    initforw(l,x)
    cudnnConvolutionForward(l.x, l.w.data, l.y)
end

function initforw(l::Conv, x::AbstractCudaArray)
    xchannels = size(x)[end-1]  # x dims are (x1, x2, ..., channels, images)
    w = l.w.data
    wsize = [size(w)...]
    if isempty(w)
        wsize[end-1]=xchannels
        w = l.w.data = initxavier((gpu()?CudaDynArray:Array)(eltype(x), wsize...))
    end
    @assert eltype(x) == eltype(w) "$(eltype(x)) != $(eltype(w))"
    @assert ndims(x) == ndims(w)
    @assert xchannels == wsize[end-1]
    similar!(l, :y, x, cudnnGetConvolutionNdForwardOutputDim(x, w))
    l.x = x
end

function back(l::Conv, dy::AbstractCudaArray; returndx=true, o...)
    initback(l, dy, returndx)
    cudnnConvolutionBackwardFilter(l.x, l.dy, l.w.diff)
    returndx && cudnnConvolutionBackwardData(l.w.data, l.dy, l.dx)
end

function initback(l::Conv, dy::AbstractCudaArray, returndx)
    @assert eltype(dy) == eltype(l.y)
    l.dy = (size(dy) == size(l.y) ? dy : reshape(dy, size(l.y)))
    similar!(l.w, :diff, l.w.data)
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
## back(l::Conv,dy;o...)=(l.w.diff=l.w.data; l.dx=l.dy=dy)
end # if GPU

