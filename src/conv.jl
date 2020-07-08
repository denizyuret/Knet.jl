using LinearAlgebra: lmul!
using LinearAlgebra.BLAS: gemm!
using NNlib

"""

    conv4(w, x; kwargs...)

Execute convolutions or cross-correlations using filters specified
with `w` over tensor `x`.

Currently KnetArray{Float32/64,4/5} and Array{Float32/64,4} are
supported as `w` and `x`.  If `w` has dimensions `(W1,W2,...,I,O)` and
`x` has dimensions `(X1,X2,...,I,N)`, the result `y` will have
dimensions `(Y1,Y2,...,O,N)` where

    Yi=1+floor((Xi+2*padding[i]-Wi)/stride[i])

Here `I` is the number of input channels, `O` is the number of output
channels, `N` is the number of instances, and `Wi,Xi,Yi` are spatial
dimensions.  `padding` and `stride` are keyword arguments that can be
specified as a single number (in which case they apply to all
dimensions), or an array/tuple with entries for each spatial
dimension.

# Keywords

* `padding=0`: the number of extra zeros implicitly concatenated at the start and at the end of each dimension.
* `stride=1`: the number of elements to slide to reach the next filtering window.
* `dilation=1`: dilation factor for each dimension.
* `mode=0`: 0 for convolution and 1 for cross-correlation.
* `alpha=1`: can be used to scale the result.
* `handle`: handle to a previously created cuDNN context. Defaults to a Knet allocated handle.

"""
function conv4(w::KnetArray{T},x::KnetArray{T}; handle=cudnnhandle(), alpha=1,
               o...) where {T} # padding=0, stride=1, dilation=1, mode=0
    beta=0 # nonzero beta does not make sense when we create y
    y = similar(x, cdims(w,x;o...))
    (algo,workSpace) = conv4_algo(w, x, y; handle=handle, o...)
    @cudnn(cudnnConvolutionForward,
          (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,   UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
          handle,Ref(T(alpha)),TD(x),x,FD(w),w,CD(w,x;o...),algo,workSpace,bytes(workSpace),Ref(T(beta)),TD(y),y)
    return y
end

function conv4x(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T}; handle=cudnnhandle(), alpha=1,
                   o...) where {T} # padding=0, stride=1, dilation=1, mode=0
    beta = 0
    dx = similar(x)
    (algo,workSpace) = conv4x_algo(w,x,dy,dx; handle=handle, o...)
    if cudnnVersion >= 4000
        @cudnn(cudnnConvolutionBackwardData,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),FD(w),w,TD(dy),dy,CD(w,x;o...),algo,workSpace,bytes(workSpace),Ref(T(beta)),TD(dx),dx)
    elseif cudnnVersion >= 3000
        @cudnn(cudnnConvolutionBackwardData_v3,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),FD(w),w,TD(dy),dy,CD(w,x;o...),algo,workSpace,bytes(workSpace),Ref(T(beta)),TD(dx),dx)
    else
        @cudnn(cudnnConvolutionBackwardData,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,       Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),FD(w),w,TD(dy),dy,CD(w,x;o...),Ref(T(beta)),TD(dx),dx)
    end
    return dx
end

function conv4w(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T}; handle=cudnnhandle(), alpha=1,
                   o...) where {T} # padding=0, stride=1, dilation=1, mode=0
    beta = 0
    dw = similar(w)
    (algo,workSpace) = conv4w_algo(w,x,dy,dw;handle=handle,o...)
    if cudnnVersion >= 4000
        @cudnn(cudnnConvolutionBackwardFilter,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),TD(x),x,TD(dy),dy,CD(w,x;o...),algo,workSpace,bytes(workSpace),Ref(T(beta)),FD(dw),dw)
    elseif cudnnVersion >= 3000
        @cudnn(cudnnConvolutionBackwardFilter_v3,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),TD(x),x,TD(dy),dy,CD(w,x;o...),algo,workSpace,bytes(workSpace),Ref(T(beta)),FD(dw),dw)
    else
        @cudnn(cudnnConvolutionBackwardFilter,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,       Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),TD(x),x,TD(dy),dy,CD(w,x;o...),Ref(T(beta)),FD(dw),dw)
    end
    return dw
end


@primitive conv4(w,x; o...),dy  conv4w(w,x,dy;o...)  conv4x(w,x,dy;o...)
@zerograd  conv4x(w,x,dy;o...)
@zerograd  conv4w(w,x,dy;o...)


"""

    pool(x; kwargs...)

Compute pooling of input values (i.e., the maximum or average of
several adjacent values) to produce an output with smaller height
and/or width.

Currently 4 or 5 dimensional KnetArrays with `Float32` or `Float64`
entries are supported.  If `x` has dimensions `(X1,X2,...,I,N)`, the
result `y` will have dimensions `(Y1,Y2,...,I,N)` where

    Yi=1+floor((Xi+2*padding[i]-window[i])/stride[i])

Here `I` is the number of input channels, `N` is the number of
instances, and `Xi,Yi` are spatial dimensions.  `window`, `padding`
and `stride` are keyword arguments that can be specified as a single
number (in which case they apply to all dimensions), or an array/tuple
with entries for each spatial dimension.

# Keywords:

* `window=2`: the pooling window size for each dimension.
* `padding=0`: the number of extra zeros implicitly concatenated at the start and at the end of each dimension.
* `stride=window`: the number of elements to slide to reach the next pooling window.
* `mode=0`: 0 for max, 1 for average including padded values, 2 for average excluding padded values.
* `maxpoolingNanOpt=0`: Nan numbers are not propagated if 0, they are propagated if 1.
* `alpha=1`: can be used to scale the result.
* `handle`: Handle to a previously created cuDNN context. Defaults to a Knet allocated handle.

"""
function pool(x::KnetArray{T}; handle=cudnnhandle(), alpha=1,
                 o...) where {T} # window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=0
    y = similar(x, pdims(x; o...))
    beta = 0
    @cudnn(cudnnPoolingForward,
          (Cptr, Cptr,      Ptr{T},    Cptr,Ptr{T},Ptr{T},   Cptr,Ptr{T}),
          handle,PD(x;o...),Ref(T(alpha)),TD(x),x,    Ref(T(beta)),TD(y),y)
    return y
end

function poolx(x::KnetArray{T},y::KnetArray{T},dy::KnetArray{T}; handle=cudnnhandle(), alpha=1, mode=0,
                  o...) where {T} # window=2, padding=0, stride=window, maxpoolingNanOpt=0
    dx = similar(x)
    beta = 0
    @cudnn(cudnnPoolingBackward,
          (Cptr,Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Ptr{T},Cptr,Ptr{T}),
          handle,PD(x;mode=mode,o...),Ref(T(alpha)),TD(y),y,TD(dy),dy,TD(x),x,Ref(T(beta)),TD(dx),dx)
    return dx
end

@primitive pool(x;o...),dy,y  poolx(x,y,dy;o...)
@zerograd  poolx(x,y,dy;o...)


### CPU convolution using NNlib

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

function conv4(w::AbstractArray{T,N}, x::AbstractArray{T,N};
               padding=0, stride=1, dilation=1, mode=0, alpha=1) where {T,N}
    stride = expand(Val(N-2), stride)
    padding = expand(Val(N-2), padding)
    dilation = expand(Val(N-2), dilation)
    cdims = DenseConvDims(x, w; stride = stride, padding = padding, dilation = dilation, flipkernel = (mode!=0))
    y = conv(x, w, cdims)
    alpha == 1 ? y : lmul!(alpha, y)
end

function conv4w(w::AbstractArray{T,N},x::AbstractArray{T,N},dy::AbstractArray{T,N};
                padding=0, stride=1, dilation=1, mode=0, alpha=1) where {T,N}
    stride = expand(Val(N-2), stride)
    padding = expand(Val(N-2), padding)
    dilation = expand(Val(N-2), dilation)
    cdims = DenseConvDims(x, w; stride = stride, padding = padding, dilation = dilation, flipkernel = (mode!=0))
    dw = ∇conv_filter(x, dy, cdims)
    alpha == 1 ? dw : lmul!(alpha, dw)
end

function conv4x(w::AbstractArray{T,N},x::AbstractArray{T,N},dy::AbstractArray{T,N};
                padding=0, stride=1, dilation=1, mode=0, alpha=1) where {T,N}
    stride = expand(Val(N-2), stride)
    padding = expand(Val(N-2), padding)
    dilation = expand(Val(N-2), dilation)
    cdims = DenseConvDims(x, w; stride = stride, padding = padding, dilation = dilation, flipkernel = (mode!=0))
    dx = ∇conv_data(dy, w, cdims)
    alpha == 1 ? dx : lmul!(alpha, dx)
end

# TODO: handle alpha, maxpoolingNanOpt, mode
# TODO: pool(ka(x), stride=1) gives error

function pool(x::AbstractArray{T,N}; handle=nothing,
              alpha=1, mode=0, window=2, padding=0, stride=window, maxpoolingNanOpt=0) where {T,N}
    window = expand(Val(N-2), window)
    stride = expand(Val(N-2), stride)
    padding = expand(Val(N-2), padding)
    pdims = PoolDims(x, window; padding = padding, stride = stride)
    y = (mode == 0 ? maxpool(x, pdims) :
         mode == 1 ? meanpool(x, pdims) :
         # mode == 2 ? meanpool(x, pdims) : ## CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING is missing in NNlib Issue #218
         error("mode=$mode is not supported for CPU pool."))
    alpha == 1 ? y : lmul!(alpha, y)
end

function poolx(x::AbstractArray{T,N},y::AbstractArray{T,N},dy::AbstractArray{T,N}; handle=nothing,
               alpha=1, mode=0, window=2, padding=0, stride=window, maxpoolingNanOpt=0) where {T,N}
    if alpha != 1
        y = y ./ T(alpha)
    end
    window = expand(Val(N-2), window)
    stride = expand(Val(N-2), stride)
    padding = expand(Val(N-2), padding)
    pdims = PoolDims(x, window; padding = padding, stride = stride)
    dx = (mode == 0 ? ∇maxpool(dy, y, x, pdims) :
          mode == 1 ? ∇meanpool(dy, y, x, pdims) :
          # mode == 2 ? ∇meanpool(dy, y, x, pdims) : ## CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING is missing in NNlib Issue #218
          error("mode=$mode is not supported for CPU pool."))
    alpha == 1 ? dx : lmul!(alpha, dx)
end


"""

Unpooling; `reverse` of pooling. 

TODO: Does not work correctly for every window, padding, mode combination. Test before use.

    x == pool(unpool(x;o...); o...)

"""
function unpool(x; window=2, alpha=1, o...) # padding=0, stride=window, mode=0, maxpoolingNanOpt=0
    w = prod(psize(window,x))
    y = similar(x,updims(x; window=window, o...))
    poolx(y,x,x.*w; o..., window=window, mode=1, alpha=1/alpha)
end

function unpoolx(dy; window=2, alpha=1, o...) # padding=0, stride=window, mode=0, maxpoolingNanOpt=0
    w = prod(psize(window,dy))
    pool(dy; o..., window=window, mode=1, alpha=1/alpha) * w
end

# @primitive unpool(x;o...),dy,y -pool(-dy;o...)
@primitive  unpool(x;o...),dy,y  unpoolx(dy;o...)


"""

    y = deconv4(w, x; kwargs...)

Simulate 4-D deconvolution by using _transposed convolution_ operation. Its forward pass is
equivalent to backward pass of a convolution (gradients with respect to input
tensor). Likewise, its backward pass (gradients with respect to input tensor) is equivalent
to forward pass of a convolution. Since it swaps forward and backward passes of convolution
operation, padding and stride options belong to output tensor. See [this
report](https://arxiv.org/abs/1603.07285) for further explanation.

Currently KnetArray{Float32/64,4} and Array{Float32/64,4} are supported as `w` and `x`.  If
`w` has dimensions `(W1,W2,...,O,I)` and `x` has dimensions `(X1,X2,...,I,N)`, the result
`y` will have dimensions `(Y1,Y2,...,O,N)` where

Yi = Wi+stride[i]*(Xi-1)-2*padding[i]

Here I is the number of input channels, O is the number of output channels, N is the number
of instances, and Wi,Xi,Yi are spatial dimensions. padding and stride are keyword arguments
that can be specified as a single number (in which case they apply to all dimensions), or an
array/tuple with entries for each spatial dimension.

# Keywords

* `padding=0`: the number of extra zeros implicitly concatenated at the start and at the end of each dimension.
* `stride=1`: the number of elements to slide to reach the next filtering window.
* `mode=0`: 0 for convolution and 1 for cross-correlation.
* `alpha=1`: can be used to scale the result.
* `handle`: handle to a previously created cuDNN context. Defaults to a Knet allocated handle.

"""
function deconv4(w,x; o...)
    y = similar(x,dcdims(w,x;o...))
    return conv4x(w,y,x;o...)
end

function deconv4w(w,x,dy; o...)
    return conv4w(w,dy,x;o...)
end

function deconv4x(w,x,dy; o...)
    return conv4(w,dy;o...)
end


@primitive deconv4(w,x; o...),dy,y  deconv4w(w,x,dy; o...)  deconv4x(w,x,dy; o...)
@zerograd deconv4w(w,x,dy; o...)
@zerograd deconv4x(w,x,dy; o...)

# cudnn descriptors

mutable struct TD; ptr; end
TD(a::KnetArray{T}) where {T} = TD(T,size(a))
TD(T::Type, dims::Integer...) = TD(T, dims)
function TD(T::Type, dims)
    d = Cptr[0]
    @cudnn(cudnnCreateTensorDescriptor,(Ptr{Cptr},),d)
    n = length(dims)
    sz = [Cint(dims[i]) for i=n:-1:1]
    st = similar(sz); st[n] = 1
    for i=(n-1):-1:1; st[i] = st[i+1] * sz[i+1]; end
    @cudnn(cudnnSetTensorNdDescriptor,
          (Cptr,UInt32,Cint,Ptr{Cint},Ptr{Cint}),
          d[1], DT(T), n, sz, st)
    td = TD(d[1])
    finalizer(x->@cudnn(cudnnDestroyTensorDescriptor,(Cptr,),x.ptr), td)
    return td
end

mutable struct FD; ptr; end
FD(a::KnetArray{T}) where {T}=FD(T,size(a))
FD(T::Type, dims::Integer...) = FD(T,dims)
function FD(T::Type, dims)
    d = Cptr[0]
    @cudnn(cudnnCreateFilterDescriptor,(Ptr{Cptr},),d)
    n = length(dims)
    sz = [Cint(dims[i]) for i=n:-1:1]
    if cudnnVersion >= 5000
        @cudnn(cudnnSetFilterNdDescriptor,
              (Cptr,UInt32,UInt32,Cint,Ptr{Cint}),
              d[1], DT(T), 0,     n,   sz)
    elseif cudnnVersion >= 4000
        @cudnn(cudnnSetFilterNdDescriptor_v4,
              (Cptr,UInt32,UInt32,Cint,Ptr{Cint}),
              d[1], DT(T), 0,     n,   sz)
    else
        @cudnn(cudnnSetFilterNdDescriptor,
              (Cptr,UInt32,Cint,Ptr{Cint}),
              d[1], DT(T),    n,   sz)
    end
    fd = FD(d[1])
    finalizer(x->@cudnn(cudnnDestroyFilterDescriptor,(Cptr,),x.ptr), fd)
    return fd
end

mutable struct CD; ptr
    function CD(w::KnetArray,x::KnetArray; padding=0, stride=1, dilation=1, mode=0, upscale=nothing)
        upscale !== nothing && error("upscale is deprecated, please use dilation instead.")
        d = Cptr[0]
        @cudnn(cudnnCreateConvolutionDescriptor,(Ptr{Cptr},),d)
        nd = ndims(x)-2
        if cudnnVersion >= 4000
            @cudnn(cudnnSetConvolutionNdDescriptor,
                  (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},UInt32,UInt32),
                  d[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(dilation,nd),mode,DT(x))
        elseif cudnnVersion > 3000 # does not work when cudnnVersion==3000
            @cudnn(cudnnSetConvolutionNdDescriptor_v3,
                  (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},UInt32,UInt32),
                  d[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(dilation,nd),mode,DT(x))
        else
            @cudnn(cudnnSetConvolutionNdDescriptor,
                  (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},UInt32),
                  d[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(dilation,nd),mode)
        end
        cd = new(d[1])
        finalizer(x->@cudnn(cudnnDestroyConvolutionDescriptor,(Cptr,),x.ptr),cd)
        return cd
    end
end

mutable struct PD; ptr
    function PD(x::KnetArray; window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=0)
        d = Cptr[0]
        @cudnn(cudnnCreatePoolingDescriptor,(Ptr{Cptr},),d)
        nd = ndims(x)-2
        if cudnnVersion >= 5000
            @cudnn(cudnnSetPoolingNdDescriptor,
                  (Cptr,UInt32,UInt32,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),
                  d[1],mode,maxpoolingNanOpt,nd,cdsize(window,nd),cdsize(padding,nd),cdsize(stride,nd))
        elseif cudnnVersion >= 4000
            @cudnn(cudnnSetPoolingNdDescriptor_v4,
                  (Cptr,UInt32,UInt32,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),
                  d[1],mode,maxpoolingNanOpt,nd,cdsize(window,nd),cdsize(padding,nd),cdsize(stride,nd))
        else
            @cudnn(cudnnSetPoolingNdDescriptor,
                  (Cptr,UInt32,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),
                  d[1],mode,nd,cdsize(window,nd),cdsize(padding,nd),cdsize(stride,nd))
        end
        pd = new(d[1])
        finalizer(x->@cudnn(cudnnDestroyPoolingDescriptor,(Cptr,),x.ptr), pd)
        return pd
    end
end

import Base: unsafe_convert
unsafe_convert(::Type{Cptr}, td::TD)=td.ptr
unsafe_convert(::Type{Cptr}, fd::FD)=fd.ptr
unsafe_convert(::Type{Cptr}, cd::CD)=cd.ptr
unsafe_convert(::Type{Cptr}, pd::PD)=pd.ptr

# fill and reverse Cint array with padding etc. for cudnn calls
function cdsize(w, nd)
    if isa(w,Number)
        fill(Cint(w),nd)
    elseif length(w)==nd
        [ Cint(w[nd-i+1]) for i=1:nd ]
    else
        throw(DimensionMismatch("$w $nd"))
    end
end

# convert padding etc. size to an Int array of the right dimension
function psize(p, x)
    nd = ndims(x)-2
    if isa(p,Number)
        fill(Int(p),nd)
    elseif length(p)==nd
        collect(Int,p)
    else
        throw(DimensionMismatch("psize: $p $nd"))
    end
end

DT(::KnetArray{Float32})=Cint(0)
DT(::KnetArray{Float64})=Cint(1)
DT(::KnetArray{Float16})=Cint(2)
DT(::Type{Float32}) = Cint(0)
DT(::Type{Float64}) = Cint(1)
DT(::Type{Float16}) = Cint(2)

# outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*dilation)+1) )/convolutionStride;

function cdims(w,x; padding=0, stride=1, dilation=1, o...)
    N = ndims(x)
    ntuple(N) do i
        if i < N-1
            pi = (if isa(padding,Number); padding; else padding[i]; end)
            si = (if isa(stride,Number); stride; else stride[i]; end)
            di = (if isa(dilation,Number); dilation; else dilation[i]; end)
            1 + (size(x,i) + 2*pi - (((size(w,i)-1)*di)+1)) ÷ si
        elseif i == N-1
            size(w,N)
        else # i == N
            size(x,N)
        end
    end
end

function dcdims(w,x; padding=0, stride=1, dilation=1, o...)
    # TODO: handle dilation here
    dilation != 1 && error("deconv4 cannot handle dilation!=1 yet.")
    N = ndims(x)
    @assert size(x,N-1) == size(w,N)
    ntuple(N) do i
        if i < N-1
            pi = (if isa(padding,Number); padding; else padding[i]; end)
            si = (if isa(stride,Number); stride; else stride[i]; end)
            # di = (if isa(dilation,Number); dilation; else dilation[i]; end)
            si*(size(x,i)-1) + size(w,i) - 2*pi
        elseif i == N-1
            size(w,N-1)
        else
            size(x,N)
        end
    end
end

function pdims(x; window=2, padding=0, stride=window, o...)
    N = ndims(x)
    ntuple(N) do i
        if i < N-1
            wi = (if isa(window,Number); window; else window[i]; end)
            pi = (if isa(padding,Number); padding; else padding[i]; end)
            si = (if isa(stride,Number); stride; else stride[i]; end)
            1 + div(size(x,i) + 2*pi - wi, si)
        else
            size(x,i)
        end
    end
end

function updims(x; window=2, padding=0, stride=window, o...)
    window = psize(window,x)
    stride = psize(stride,x)
    padding = psize(padding,x)
    N = ndims(x)
    ntuple(N) do i
        if i < N-1
            (size(x,i)-1)*stride[i]+window[i]-2*padding[i]
        else
            size(x,i)
        end
    end
end

# convolution padding size that preserves the input size when filter size is odd and stride=1
padsize(w)=ntuple(i->div(size(w,i)-1,2), ndims(w)-2)


## Utilities to find a fast algorithm

struct cudnnConvolutionFwdAlgoPerf_t
    algo::Cint
    status::Cint
    time::Cfloat
    memory::Csize_t
    determinism::Cint
    mathType::Cint
    r1::Cint; r2::Cint; r3::Cint
end

const CUDNN_MAX_FIND = 100      # How many times can we call FindAlgorithm
const requestedAlgoCount = 10
const returnedAlgoCount = Cint[0]
const perfResults = Array{cudnnConvolutionFwdAlgoPerf_t}(undef,requestedAlgoCount)
bytes(x::KnetArray{T}) where {T}=length(x)*sizeof(T)

# This seems to cover a reasonable subset of the available algorithms
# The user can set this to 0 for a more memory-tight execution
maxWorkspaceSize(w,x,y) = min(gpufree() ÷ 10, bytes(x) * 100)

const conv4_algos = Dict()
function conv4_algo(w::KnetArray{T}, x::KnetArray{T}, y::KnetArray{T}; handle=cudnnhandle(), o...) where {T}
    global conv4_algos, requestedAlgoCount, returnedAlgoCount, perfResults
    key = (T,size(w),size(x),o...)
    if haskey(conv4_algos, key)
        p = conv4_algos[key]
        return (p.algo, cudnnWorkSpace(p.memory))
    elseif length(conv4_algos) >= CUDNN_MAX_FIND
        return (0, cudnnWorkSpace())
    else
        workSpace = KnetArray{UInt8}(undef, maxWorkspaceSize(w,x,y))
        wd, xd, yd, cd = FD(w), TD(x), TD(y), CD(w,x;o...)
        @cudnn(cudnnFindConvolutionForwardAlgorithmEx,
              (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Ptr{Cint},Cptr,Cptr,Csize_t),
              handle,xd,x,wd,w,cd,yd,y,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,bytes(workSpace))
        workSpace = nothing; Knet.gc(); GC.gc()
        p = perfChoose(perfResults, returnedAlgoCount[1])
        conv4_algos[key] = p
        return (p.algo, cudnnWorkSpace(p.memory))
    end
end

const conv4w_algos = Dict()
function conv4w_algo(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T},dw::KnetArray{T}; handle=cudnnhandle(), o...) where {T}
    global conv4w_algos, requestedAlgoCount, returnedAlgoCount, perfResults
    key = (T,size(w),size(x),o...)
    if haskey(conv4w_algos, key)
        p = conv4w_algos[key]
        return (p.algo, cudnnWorkSpace(p.memory))
    elseif length(conv4w_algos) >= CUDNN_MAX_FIND
        return (0, cudnnWorkSpace())
    else
        workSpace = KnetArray{UInt8}(undef, maxWorkspaceSize(w,x,dy))
        wd, xd, yd, cd = FD(dw), TD(x), TD(dy), CD(w,x;o...)
        @cudnn(cudnnFindConvolutionBackwardFilterAlgorithmEx,
              (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Ptr{Cint},Cptr,Cptr,Csize_t),
              handle,xd,x,yd,dy,cd,wd,dw,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,bytes(workSpace))
        workSpace = nothing; Knet.gc(); GC.gc()
        p = perfChoose(perfResults, returnedAlgoCount[1])
        conv4w_algos[key] = p
        return (p.algo, cudnnWorkSpace(p.memory))
    end
end

const conv4x_algos = Dict()
function conv4x_algo(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T},dx::KnetArray{T}; handle=cudnnhandle(), o...) where {T}
    global conv4x_algos, requestedAlgoCount, returnedAlgoCount, perfResults
    key = (T,size(w),size(x),o...)
    if haskey(conv4x_algos, key)
        p = conv4x_algos[key]
        return (p.algo, cudnnWorkSpace(p.memory))
    elseif length(conv4x_algos) >= CUDNN_MAX_FIND
        return (0, cudnnWorkSpace())
    else
        workSpace = KnetArray{UInt8}(undef, maxWorkspaceSize(w,x,dy))
        wd, xd, yd, cd = FD(w), TD(dx), TD(dy), CD(w,x;o...)
        @cudnn(cudnnFindConvolutionBackwardDataAlgorithmEx,
              (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Ptr{Cint},Cptr,Cptr,Csize_t),
              handle,wd,w,yd,dy,cd,xd,dx,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,bytes(workSpace))
        workSpace = nothing; Knet.gc(); GC.gc()
        p = perfChoose(perfResults, returnedAlgoCount[1])
        conv4x_algos[key] = p
        return (p.algo, cudnnWorkSpace(p.memory))
    end
end


function perfChoose(ps, n)
    global CUDNN_WORKSPACE_MAXSIZE
    if n==ps
        warn("returnedAlgoCount==requestedAlgoCount")
    end
    (ibest,mbest,tbest) = (0,Inf,Inf)
    for i = 1:n
        # These metrics are written in a sorted fashion where the first element has the lowest compute time.
        if ps[i].status == 0 && ps[i].memory < mbest && ps[i].time < tbest * 1.1
            (ibest,mbest,tbest) = (i,ps[i].memory,ps[i].time)
        end
    end
    if ibest == 0; error("No good algo found."); end
    return ps[ibest]
end

# Fresh workspace for every op is safer:
cudnnWorkSpace(len=0)=KnetArray{UInt8}(undef,len)
