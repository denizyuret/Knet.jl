using CUDA
import ..Ops20: conv4, conv4x, conv4w, pool, poolx

function conv4(w::KnetArray{T},x::KnetArray{T}; handle=CUDNN.handle(), alpha=1,
               o...) where {T} # padding=0, stride=1, dilation=1, mode=0, group=1
    beta=0 # nonzero beta does not make sense when we create y
    y = similar(x, cdims(w,x;o...))
    (algo,workSpace) = conv4_algo(w, x, y; handle=handle, o...)
    algo = CUDNN.cudnnConvolutionFwdAlgo_t(algo)
    CUDNN.cudnnConvolutionForward(handle,Ref(T(alpha)),TD(x),x,FD(w),w,CD(w,x;o...),algo,workSpace,bytes(workSpace),Ref(T(beta)),TD(y),y)
    return y
end

function conv4x(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T}; handle=CUDNN.handle(), alpha=1,
                   o...) where {T} # padding=0, stride=1, dilation=1, mode=0, group=1
    beta = 0
    dx = similar(x)
    (algo,workSpace) = conv4x_algo(w,x,dy,dx; handle=handle, o...)
    algo = CUDNN.CUDA.CUDNN.cudnnConvolutionBwdDataAlgo_t(algo)
    CUDNN.cudnnConvolutionBackwardData(handle,Ref(T(alpha)),FD(w),w,TD(dy),dy,CD(w,x;o...),algo,workSpace,bytes(workSpace),Ref(T(beta)),TD(dx),dx)
    return dx
end

function conv4w(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T}; handle=CUDNN.handle(), alpha=1,
                   o...) where {T} # padding=0, stride=1, dilation=1, mode=0, group=1
    beta = 0
    dw = similar(w)
    (algo,workSpace) = conv4w_algo(w,x,dy,dw;handle=handle,o...)
    algo = CUDA.CUDNN.cudnnConvolutionBwdFilterAlgo_t(algo)
    CUDNN.cudnnConvolutionBackwardFilter(handle,Ref(T(alpha)),TD(x),x,TD(dy),dy,CD(w,x;o...),algo,workSpace,bytes(workSpace),Ref(T(beta)),FD(dw),dw)
    return dw
end

function pool(x::KnetArray{T}; handle=CUDNN.handle(), alpha=1,
                 o...) where {T} # window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=0
    y = similar(x, pdims(x; o...))
    beta = 0
    CUDNN.cudnnPoolingForward(handle,PD(x;o...),Ref(T(alpha)),TD(x),x,    Ref(T(beta)),TD(y),y)
    return y
end

function poolx(x::KnetArray{T},y::KnetArray{T},dy::KnetArray{T}; handle=CUDNN.handle(), alpha=1, mode=0,
                  o...) where {T} # window=2, padding=0, stride=window, maxpoolingNanOpt=0
    dx = similar(x)
    beta = 0
    CUDNN.cudnnPoolingBackward(handle,PD(x;mode=mode,o...),Ref(T(alpha)),TD(y),y,TD(dy),dy,TD(x),x,Ref(T(beta)),TD(dx),dx)
    return dx
end


# cudnn descriptors

mutable struct TD; ptr; end
TD(a::KnetArray{T}) where {T} = TD(T,size(a))
TD(T::Type, dims::Integer...) = TD(T, dims)
function TD(T::Type, dims)
    d = Cptr[0]
    # @cudnn(cudnnCreateTensorDescriptor,(Ptr{Cptr},),d)
    CUDNN.cudnnCreateTensorDescriptor(d)
    n = length(dims)
    sz = [Cint(dims[i]) for i=n:-1:1]
    st = similar(sz); st[n] = 1
    for i=(n-1):-1:1; st[i] = st[i+1] * sz[i+1]; end
    # @cudnn(cudnnSetTensorNdDescriptor,
    #       (Cptr,UInt32,Cint,Ptr{Cint},Ptr{Cint}),
    #       d[1], DT(T), n, sz, st)
    dt = CUDNN.cudnnDataType_t(DT(T))
    CUDNN.cudnnSetTensorNdDescriptor(d[1], dt, n, sz, st)
    td = TD(d[1])
    #finalizer(x->@cudnn(cudnnDestroyTensorDescriptor,(Cptr,),x.ptr), td)
    finalizer(x->CUDNN.cudnnDestroyTensorDescriptor(x.ptr), td)
    return td
end

mutable struct FD; ptr; end
FD(a::KnetArray{T}) where {T}=FD(T,size(a))
FD(T::Type, dims::Integer...) = FD(T,dims)
function FD(T::Type, dims)
    d = Cptr[0]
    # @cudnn(cudnnCreateFilterDescriptor,(Ptr{Cptr},),d)
    CUDNN.cudnnCreateFilterDescriptor(d)
    n = length(dims)
    sz = [Cint(dims[i]) for i=n:-1:1]
    # if cudnnVersion >= 5000
    #     @cudnn(cudnnSetFilterNdDescriptor,
    #           (Cptr,UInt32,UInt32,Cint,Ptr{Cint}),
    #           d[1], DT(T), 0,     n,   sz)
    # elseif cudnnVersion >= 4000
    #     @cudnn(cudnnSetFilterNdDescriptor_v4,
    #           (Cptr,UInt32,UInt32,Cint,Ptr{Cint}),
    #           d[1], DT(T), 0,     n,   sz)
    # else
    #     @cudnn(cudnnSetFilterNdDescriptor,
    #           (Cptr,UInt32,Cint,Ptr{Cint}),
    #           d[1], DT(T),    n,   sz)
    # end
    dt = CUDNN.cudnnDataType_t(DT(T))
    tf = CUDNN.cudnnTensorFormat_t(0)
    CUDNN.cudnnSetFilterNdDescriptor(d[1], dt, tf, n, sz)
    fd = FD(d[1])
    # finalizer(x->@cudnn(cudnnDestroyFilterDescriptor,(Cptr,),x.ptr), fd)
    finalizer(x->CUDNN.cudnnDestroyFilterDescriptor(x.ptr), fd)
    return fd
end

mutable struct CD; ptr
    function CD(w::KnetArray,x::KnetArray; padding=0, stride=1, dilation=1, mode=0, upscale=nothing, group=1)
        upscale !== nothing && error("upscale is deprecated, please use dilation instead.")
        d = Cptr[0]
        # @cudnn(cudnnCreateConvolutionDescriptor,(Ptr{Cptr},),d)
        CUDNN.cudnnCreateConvolutionDescriptor(d)
        cd = new(d[1])
        nd = ndims(x)-2
        # if cudnnVersion >= 4000
        #     @cudnn(cudnnSetConvolutionNdDescriptor,
        #           (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},UInt32,UInt32),
        #           d[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(dilation,nd),mode,DT(x))
        # elseif cudnnVersion > 3000 # does not work when cudnnVersion==3000
        #     @cudnn(cudnnSetConvolutionNdDescriptor_v3,
        #           (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},UInt32,UInt32),
        #           d[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(dilation,nd),mode,DT(x))
        # else
        #     @cudnn(cudnnSetConvolutionNdDescriptor,
        #           (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},UInt32),
        #           d[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(dilation,nd),mode)
        # end
        dt = CUDNN.cudnnDataType_t(DT(x))
        mode = CUDNN.cudnnConvolutionMode_t(mode)
        CUDNN.cudnnSetConvolutionNdDescriptor(cd,nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(dilation,nd),mode,dt)
        CUDNN.cudnnSetConvolutionGroupCount(cd,group)
        # finalizer(x->@cudnn(cudnnDestroyConvolutionDescriptor,(Cptr,),x.ptr),cd)
        finalizer(x->CUDNN.cudnnDestroyConvolutionDescriptor(x.ptr),cd)
        return cd
    end
end

mutable struct PD; ptr
    function PD(x::KnetArray; window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=0)
        d = Cptr[0]
        #@cudnn(cudnnCreatePoolingDescriptor,(Ptr{Cptr},),d)
        CUDNN.cudnnCreatePoolingDescriptor(d)
        nd = ndims(x)-2
        # if cudnnVersion >= 5000
        #     @cudnn(cudnnSetPoolingNdDescriptor,
        #           (Cptr,UInt32,UInt32,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),
        #           d[1],mode,maxpoolingNanOpt,nd,cdsize(window,nd),cdsize(padding,nd),cdsize(stride,nd))
        # elseif cudnnVersion >= 4000
        #     @cudnn(cudnnSetPoolingNdDescriptor_v4,
        #           (Cptr,UInt32,UInt32,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),
        #           d[1],mode,maxpoolingNanOpt,nd,cdsize(window,nd),cdsize(padding,nd),cdsize(stride,nd))
        # else
        #     @cudnn(cudnnSetPoolingNdDescriptor,
        #           (Cptr,UInt32,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),
        #           d[1],mode,nd,cdsize(window,nd),cdsize(padding,nd),cdsize(stride,nd))
        # end
        mode = CUDNN.cudnnPoolingMode_t(mode)
        maxpoolingNanOpt = CUDNN.cudnnNanPropagation_t(maxpoolingNanOpt)
        CUDNN.cudnnSetPoolingNdDescriptor(d[1],mode,maxpoolingNanOpt,nd,cdsize(window,nd),cdsize(padding,nd),cdsize(stride,nd))
        pd = new(d[1])
        #finalizer(x->@cudnn(cudnnDestroyPoolingDescriptor,(Cptr,),x.ptr), pd)
        finalizer(x->CUDNN.cudnnDestroyPoolingDescriptor(x.ptr), pd)
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

DT(::KnetArray{Float32})=Cint(0)
DT(::KnetArray{Float64})=Cint(1)
DT(::KnetArray{Float16})=Cint(2)
DT(::Type{Float32}) = Cint(0)
DT(::Type{Float64}) = Cint(1)
DT(::Type{Float16}) = Cint(2)


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
function conv4_algo(w::KnetArray{T}, x::KnetArray{T}, y::KnetArray{T}; handle=CUDNN.handle(), o...) where {T}
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
        # @cudnn(cudnnFindConvolutionForwardAlgorithmEx,
        #       (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Ptr{Cint},Cptr,Cptr,Csize_t),
        #       handle,xd,x,wd,w,cd,yd,y,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,bytes(workSpace))
        CUDNN.cudnnFindConvolutionForwardAlgorithmEx(handle,xd,x,wd,w,cd,yd,y,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,bytes(workSpace))
        workSpace = nothing; Knet.gc(); GC.gc()
        p = perfChoose(perfResults, returnedAlgoCount[1])
        conv4_algos[key] = p
        return (p.algo, cudnnWorkSpace(p.memory))
    end
end

const conv4w_algos = Dict()
function conv4w_algo(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T},dw::KnetArray{T}; handle=CUDNN.handle(), o...) where {T}
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
        # @cudnn(cudnnFindConvolutionBackwardFilterAlgorithmEx,
        #       (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Ptr{Cint},Cptr,Cptr,Csize_t),
        #       handle,xd,x,yd,dy,cd,wd,dw,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,bytes(workSpace))
        CUDNN.cudnnFindConvolutionBackwardFilterAlgorithmEx(handle,xd,x,yd,dy,cd,wd,dw,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,bytes(workSpace))
        workSpace = nothing; Knet.gc(); GC.gc()
        p = perfChoose(perfResults, returnedAlgoCount[1])
        conv4w_algos[key] = p
        return (p.algo, cudnnWorkSpace(p.memory))
    end
end

const conv4x_algos = Dict()
function conv4x_algo(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T},dx::KnetArray{T}; handle=CUDNN.handle(), o...) where {T}
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
        # @cudnn(cudnnFindConvolutionBackwardDataAlgorithmEx,
        #       (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Ptr{Cint},Cptr,Cptr,Csize_t),
        #       handle,wd,w,yd,dy,cd,xd,dx,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,bytes(workSpace))
        CUDNN.cudnnFindConvolutionBackwardDataAlgorithmEx(handle,wd,w,yd,dy,cd,xd,dx,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,bytes(workSpace))
        workSpace = nothing; Knet.gc(); GC.gc()
        p = perfChoose(perfResults, returnedAlgoCount[1])
        conv4x_algos[key] = p
        return (p.algo, cudnnWorkSpace(p.memory))
    end
end


function perfChoose(ps, n)
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


## Dimension helpers:

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

