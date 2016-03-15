type Conv <: Op; padding; stride; upscale; mode; algorithm; workSpace; workSpaceSizeInBytes; alpha; beta;
    function Conv(; padding=0, stride=1, upscale=1, mode=CUDNN_CONVOLUTION,
                  algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                  workSpace=nothing, workSpaceSizeInBytes=0, alpha=1.0, beta=0.0,
                  o...)
        new(padding, stride, upscale, mode, algorithm, workSpace, workSpaceSizeInBytes, alpha, beta)
    end
end

ninputs(::Conv)=2
canoverwrite(::Conv)=false
back_reads_x(::Conv)=true
back_reads_y(::Conv)=false

function forw(c::Conv, w, x, y; o...)
    if w == nothing
        error("Uninitialized filter")
    elseif x == nothing
        return nothing
    end
    ws = (c.workSpace == nothing ? C_NULL : c.workSpace)
    cudnnConvolutionForward(x, w, y; padding=c.padding, stride=c.stride, upscale=c.upscale, mode=c.mode,
                            algorithm=c.algorithm, workSpace=ws, workSpaceSizeInBytes=c.workSpaceSizeInBytes,
                            alpha=c.alpha, beta=c.beta)
    gpusync()
    return y
end

function back(c::Conv, dy, dw, dx; x=nothing, o...)
    dw == nothing && dx == nothing && return
    dw != nothing && (x[2] != nothing ? cudnnConvolutionBackwardFilter(x[2], dy, dw; padding=c.padding, stride=c.stride, upscale=c.upscale, mode=c.mode) : fillsync!(dw,0))
    dx != nothing && (x[1] != nothing ? cudnnConvolutionBackwardData(x[1], dy, dx; padding=c.padding, stride=c.stride, upscale=c.upscale, mode=c.mode) : error("Uninitialized filter"))
    gpusync()
end

# x: (x1,x2...,C,N)
# w: (w1,w2...,C,K)
# y: (y1,y2...,K,N)
# If padding=0 and stride=1: yi=xi-wi+1
# In general we have: yi = 1 + (xi + 2*padding - wi) / stride

function infersize(c::Conv,w,x,y)
    if w==x==y==nothing
        nothing
    elseif w==nothing || x==nothing || y==nothing
        n = (w!=nothing ? length(w) : x!=nothing ? length(x) : y!=nothing ? length(y) : error())
        w == nothing && (w = ntuple(i->0, n))
        x == nothing && (x = ntuple(i->0, n))
        y == nothing && (y = ntuple(i->0, n))
        infersize(c,w,x,y)
    else
        s = (isa(c.stride,  Integer) ? ntuple(i->c.stride,  length(y)-2) : c.stride)
        p = (isa(c.padding, Integer) ? ntuple(i->c.padding, length(y)-2) : c.padding)
        length(w) == length(x) == length(y) == length(s)+2 == length(p)+2 || throw(DimensionMismatch())
        w = [w...]; x = [x...]; y = [y...]
        for i=1:length(s)
            if w[i] > 0 && x[i] > 0 && y[i] > 0
                y[i] == 1 + div(x[i] + 2*p[i] - w[i], s[i]) || throw(DimensionMismatch())
            elseif w[i] > 0 && x[i] > 0 && y[i] == 0
                y[i] = 1 + div(x[i] + 2*p[i] - w[i], s[i])
            elseif w[i] > 0 && x[i] == 0 && y[i] > 0 && s[i] == 1
                x[i] = y[i] - 1 - 2*p[i] + w[i]
            end
        end
        n = length(x)
        equate!(x,n-1,w,n-1)
        equate!(x,n,y,n)
        equate!(w,n,y,n-1)
        w = tuple(w...); x = tuple(x...); y = tuple(y...)
        (w,x,y)
    end
end

function equate!(a,i,b,j)
    a[i] == b[j] ? nothing :
    a[i] == 0 ? a[i] = b[j] :
    b[j] == 0 ? b[j] = a[i] :
    throw(DimensionMismatch())
end

### DEAD CODE

# TODO: generalize to N-D
# TODO: cpu implementation
# TODO: upgrade to new cudnn version
# TODO: upgrade to new knet interface

# type Conv <: Op; w; x; ybuf; dx; Conv(p::KUparam)=new(p); end

# Conv(d...; o...)=Conv(KUparam(d...; o...))
# Conv(nout::Integer, width::Integer; o...)=Conv(KUparam(width, 0, nout; o...))

# params(l::Conv)=Any[l.w]
# ninputs(::Conv)=1
# overwrites(::Conv)=false
# back_reads_x(::Conv)=true
# back_reads_y(::Conv)=false

# # TODO: this unnecessarily allocates w and y
# ysize(l::Conv, x)=(isempty(l.w) && initforw(l,x,nothing); cudnnGetConvolutionNdForwardOutputDim(x,l.w))

# function forw(l::Conv, x; y=nothing, o...)
#     l.x = x
#     y = initforw(l,x,y)
#     cudnnConvolutionForward(x, l.w, y)
# end

# function back(l::Conv, dy; dx=nothing, x=l.x, incr=false, returndx=true, o...)
#     initback(l, dy, x, incr)
#     if incr
#         cudnnConvolutionBackwardFilter(x, dy, l.w.inc)
#         axpy!(1, l.w.inc, l.w.diff)
#     else
#         cudnnConvolutionBackwardFilter(x, dy, l.w.diff)
#     end
#     if returndx
#         dx = initbackx(l,x,dx)
#         cudnnConvolutionBackwardData(l.w, dy, dx)
#     end
# end

# function initback(l::Conv, dy, x, incr)
#     atype(dy) == atype(x) || error("atype mismatch")
#     eltype(dy) == eltype(x) || error("eltype mismatch")
#     size(dy) == ysize(l,x) || error("ysize mismatch")
#     similar!(l.w, :diff, l.w.arr)
#     incr && similar!(l.w, :inc, l.w.arr)
# end

# function initbackx(l::Conv, x, dx)
#     dx == nothing && (dx = similar!(l, :dx, x))
#     issimilar(dx,x) || error("Gradient mismatch")
#     return dx
# end

# # TODO: We should split up the w and y parts and share with Mmul

# function initforw(l::Conv, x, y)
#     n = ndims(x)
#     c = size(x)[n-1]  # x dims are (x1, x2, ..., channels, images)
#     if isempty(l.w) 
#         nz(l.w,:init,nothing) || (l.w.init = xavier!)
#         r = size(l.w, 1)
#         o = size(l.w, ndims(l.w))
#         wsize = ntuple(i->(i<n-1 ? r : i==n-1 ? c : o), n)
#         init(l.w, eltype(x), wsize)
#     end
#     eltype(x) == eltype(l.w) || "$(eltype(x)) != $(eltype(l.w))"
#     n == ndims(l.w) || error("ndims mismatch")
#     c == size(l.w)[n-1] || error("channel mismatch")
#     ys = ysize(l,x)
#     y == nothing && (y = similar!(l, :ybuf, x, ys))
#     typeof(y) == typeof(x) || error("Type mismatch")
#     size(y) == ys || error("Size mismatch")
#     return y
# end

# xavier!(a)=(fanin = length(a) / (size(a)[end]); scale = sqrt(3 / fanin); rand!(a, -scale, scale); a)

# # Make things work with KUdense

# import CUDNN: cudnnGetConvolutionNdForwardOutputDim, cudnnConvolutionForward, cudnnConvolutionBackwardFilter, cudnnConvolutionBackwardData

# cudnnGetConvolutionNdForwardOutputDim(x::KUdense, w::KUparam)=cudnnGetConvolutionNdForwardOutputDim(x.arr, w.arr)
# cudnnConvolutionForward(x::KUdense, w::KUparam, y::KUdense)=(cudnnConvolutionForward(x.arr, w.arr, y.arr);y)
# cudnnConvolutionBackwardFilter(x::KUdense, dy::KUdense, w::BaseArray)=(cudnnConvolutionBackwardFilter(x.arr, dy.arr, w);w)
# cudnnConvolutionBackwardData(w::KUparam, dy::KUdense, dx::KUdense)=(cudnnConvolutionBackwardData(w.arr, dy.arr, dx.arr);dx)

# Make things work with CPU (for now)

# cudnnGetConvolutionNdForwardOutputDim(x::Array, w::Array)=cudnnGetConvolutionNdForwardOutputDim(CudaArray(x),CudaArray(w))
# cudnnConvolutionForward(x::Array, w::Array, y::Array)=(y1=CudaArray(y);cudnnConvolutionForward(CudaArray(x), CudaArray(w), y1);copysync!(y,1,y1,1,length(y)))
# cudnnConvolutionBackwardFilter(x::Array, dy::Array, w::Array)=(w1=CudaArray(w);cudnnConvolutionBackwardFilter(CudaArray(x), CudaArray(dy), w1); copysync!(w,1,w1,1,length(w)))
# cudnnConvolutionBackwardData(w::Array, dy::Array, dx::Array)=(dx1=CudaArray(dx);cudnnConvolutionBackwardData(CudaArray(w), CudaArray(dy), dx1); copysync!(dx,1,dx1,1,length(dx)))


#     else

#     @assert length(w) == length(x) == length(y)
#     nd = length(x)
#     x = [x...]
#     w = [w...]
#     x[nd-1] == 0 && (x[nd-1] = w[nd-1])
#     w[nd-1] == 0 && (w[nd-1] = x[nd-1])
#     @assert x[nd-1] == w[nd-1]
#     y = zeros(x)
#     for i=1:nd-2
#         w[i] > 0 && x[i] > 0 && (y[i] = x[i]-w[i]+1)
#     end
#     y[nd-1] = w[nd]
#     y[nd] = x[nd]
#     return (tuple(w...), tuple(x...), tuple(y...))
# end

# # TODO: write doc
# """
# @knet function conv(w, x; padding=0, stride=1, upscale=1, mode=CUDNN_CONVOLUTION)
# """    
# function conv(w, x, y; padding=0, stride=1, upscale=1, mode=CUDNN_CONVOLUTION)
#     @assert in(mode, (CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION))
#     (Conv(padding, stride, upscale, mode), w, x, y)
# end

# using CUDNN: cudnnConvolutionDescriptor_t
