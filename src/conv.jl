# TODO: generalize to N-D
# TODO: cpu implementation
# TODO: get rid of Float32
# TODO: add ConvolutionDescriptor if needed
# TODO: add xavier init

type Conv <: Layer; w::Param; x; y; dx; dy; Conv(w::Param)=new(w); end
Conv(w; o...)=Conv(Param(w; o...))
Conv(d::Integer...; o...)=Conv(Param(randn(d)*0.01; o...))

copy(l::Conv; o...)=Conv(copy(l.w; o...))
update(l::Conv; o...)=update(l.w; o...)
setparam!(l::Conv; o...)=setparam!(l.w; o...)
forw(l::Conv, x; o...)=error("CPU conv not implemented")
back(l::Conv, dy; o...)=error("CPU conv not implemented")

if GPU

function forw(l::Conv, x::CudaArray; o...)
    initforw(l, x)
    cudnnConvolutionForward(l.x, l.w.data, l.y)
end

function initforw(l::Conv, x::CudaArray)
    l.x = x
    similar!(l, :y, l.x, cudnnGetConvolutionNdForwardOutputDim(l.x, l.w.data))
end

function back(l::Conv, dy::CudaArray; returndx=true, o...)
    initback(l, dy, returndx)
    cudnnConvolutionBackwardFilter(l.x, l.dy, l.w.diff)
    returndx && cudnnConvolutionBackwardData(l.w.data, l.dy, l.dx)
end

function initback(l::Conv, dy::CudaArray, returndx)
    if (size(dy) == size(l.y))
        l.dy = dy
    else
        @assert length(dy) == length(l.y)
        l.dy = reshape(dy, size(l.y))
    end
    similar!(l.w, :diff, l.w.data)
    returndx && similar!(l, :dx, l.x)
end

end # if GPU

