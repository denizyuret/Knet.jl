# TODO: generalize to N-D
# TODO: cpu implementation
# DONE: get rid of Float32
# DONE: add ConvolutionDescriptor if needed
# TODO: add xavier init

type Conv <: Layer; w; x; y; dx; dy; n;
    Conv(d...; o...)=new(Param(d...; o...))
    Conv(n::Integer)=(l=new();l.n=n)
end

# copy(l::Conv; o...)=Conv(copy(l.w; o...))
update(l::Conv; o...)=update(l.w; o...)
setparam!(l::Conv; o...)=setparam!(l.w; o...)

if GPU

function forw(l::Conv, x::AbstractCudaArray; o...)
    initforw(l,x)
    cudnnConvolutionForward(l.x, l.w.data, l.y)
end

function initforw(l::Conv, x::AbstractCudaArray)
    isdefined(l,:w) || (l.w = Param(eltype(x), tuple(fill(l.n,ndims(x))...); init=initrand))
    w = l.w.data
    @assert eltype(x) == eltype(w)
    @assert ndims(x) == ndims(w)
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

