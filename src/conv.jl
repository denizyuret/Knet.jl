using LinearAlgebra: lmul!
using LinearAlgebra.BLAS: gemm!

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
* `upscale=1`: upscale factor for each dimension.
* `mode=0`: 0 for convolution and 1 for cross-correlation.
* `alpha=1`: can be used to scale the result.
* `handle`: handle to a previously created cuDNN context. Defaults to a Knet allocated handle.

"""
function conv4(w::KnetArray{T},x::KnetArray{T}; handle=cudnnhandle(), alpha=1,
                  o...) where {T} # padding=0, stride=1, upscale=1, mode=0
    beta=0 # nonzero beta does not make sense when we create y
    y = similar(x, cdims(w,x;o...))
    (algo,workSpace) = conv4_algo(w, x, y; handle=handle, o...)
    @cuda(cudnn, cudnnConvolutionForward,
          (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,   UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
          handle,Ref(T(alpha)),TD(x),x,FD(w),w,CD(w,x;o...),algo,workSpace,bytes(workSpace),Ref(T(beta)),TD(y),y)
    return y
end

function conv4x(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T}; handle=cudnnhandle(), alpha=1,
                   o...) where {T} # padding=0, stride=1, upscale=1, mode=0
    beta = 0
    dx = similar(x)
    (algo,workSpace) = conv4x_algo(w,x,dy,dx; handle=handle, o...)
    if cudnnVersion >= 4000
        @cuda(cudnn,cudnnConvolutionBackwardData,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),FD(w),w,TD(dy),dy,CD(w,x;o...),algo,workSpace,bytes(workSpace),Ref(T(beta)),TD(dx),dx)
    elseif cudnnVersion >= 3000
        @cuda(cudnn,cudnnConvolutionBackwardData_v3,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),FD(w),w,TD(dy),dy,CD(w,x;o...),algo,workSpace,bytes(workSpace),Ref(T(beta)),TD(dx),dx)
    else
        @cuda(cudnn,cudnnConvolutionBackwardData,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,       Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),FD(w),w,TD(dy),dy,CD(w,x;o...),Ref(T(beta)),TD(dx),dx)
    end
    return dx
end

function conv4w(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T}; handle=cudnnhandle(), alpha=1,
                   o...) where {T} # padding=0, stride=1, upscale=1, mode=0
    beta = 0
    dw = similar(w)
    (algo,workSpace) = conv4w_algo(w,x,dy,dw;handle=handle,o...)
    if cudnnVersion >= 4000
        @cuda(cudnn,cudnnConvolutionBackwardFilter,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),TD(x),x,TD(dy),dy,CD(w,x;o...),algo,workSpace,bytes(workSpace),Ref(T(beta)),FD(dw),dw)
    elseif cudnnVersion >= 3000
        @cuda(cudnn,cudnnConvolutionBackwardFilter_v3,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),TD(x),x,TD(dy),dy,CD(w,x;o...),algo,workSpace,bytes(workSpace),Ref(T(beta)),FD(dw),dw)
    else
        @cuda(cudnn,cudnnConvolutionBackwardFilter,
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
    @cuda(cudnn, cudnnPoolingForward,
          (Cptr, Cptr,      Ptr{T},    Cptr,Ptr{T},Ptr{T},   Cptr,Ptr{T}),
          handle,PD(x;o...),Ref(T(alpha)),TD(x),x,    Ref(T(beta)),TD(y),y)
    return y
end

function poolx(x::KnetArray{T},y::KnetArray{T},dy::KnetArray{T}; handle=cudnnhandle(), alpha=1, mode=0,
                  o...) where {T} # window=2, padding=0, stride=window, maxpoolingNanOpt=0
    if alpha!=1 && mode==0; error("Gradient of pool(alpha!=1,mode=0) broken in CUDNN"); end
    dx = similar(x)
    beta = 0
    @cuda(cudnn,cudnnPoolingBackward,
          (Cptr,Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Ptr{T},Cptr,Ptr{T}),
          handle,PD(x;mode=mode,o...),Ref(T(alpha)),TD(y),y,TD(dy),dy,TD(x),x,Ref(T(beta)),TD(dx),dx)
    return dx
end

@primitive pool(x;o...),dy,y  poolx(x,y,dy;o...)
@zerograd  poolx(x,y,dy;o...)

"""

Unpooling; `reverse` of pooling.

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

Simulate 4-D deconvolution by using _transposed convolution_ operation. Its forward pass is equivalent to backward pass of a convolution (gradients with respect to input tensor). Likewise, its backward pass (gradients with respect to input tensor) is equivalent to forward pass of a convolution. Since it swaps forward and backward passes of convolution operation, padding and stride options belong to output tensor. See [this report](https://arxiv.org/abs/1603.07285) for further explanation.

Currently KnetArray{Float32/64,4} and Array{Float32/64,4} are
supported as `w` and `x`.  If `w` has dimensions `(W1,W2,...,O,I)` and
`x` has dimensions `(X1,X2,...,I,N)`, the result `y` will have
dimensions `(Y1,Y2,...,O,N)` where

Yi = Wi+stride[i]*(Xi-1)-2*padding[i]

Here I is the number of input channels, O is the number of output channels, N is the number of instances, and Wi,Xi,Yi are spatial dimensions. padding and stride are keyword arguments that can be specified as a single number (in which case they apply to all dimensions), or an array/tuple with entries for each spatial dimension.

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
    @cuda(cudnn,cudnnCreateTensorDescriptor,(Ptr{Cptr},),d)
    n = length(dims)
    sz = [Cint(dims[i]) for i=n:-1:1]
    st = similar(sz); st[n] = 1
    for i=(n-1):-1:1; st[i] = st[i+1] * sz[i+1]; end
    @cuda(cudnn,cudnnSetTensorNdDescriptor,
          (Cptr,UInt32,Cint,Ptr{Cint},Ptr{Cint}),
          d[1], DT(T), n, sz, st)
    td = TD(d[1])
    finalizer(x->@cuda(cudnn,cudnnDestroyTensorDescriptor,(Cptr,),x.ptr), td)
    return td
end

mutable struct FD; ptr; end
FD(a::KnetArray{T}) where {T}=FD(T,size(a))
FD(T::Type, dims::Integer...) = FD(T,dims)
function FD(T::Type, dims)
    d = Cptr[0]
    @cuda(cudnn,cudnnCreateFilterDescriptor,(Ptr{Cptr},),d)
    n = length(dims)
    sz = [Cint(dims[i]) for i=n:-1:1]
    if cudnnVersion >= 5000
        @cuda(cudnn,cudnnSetFilterNdDescriptor,
              (Cptr,UInt32,UInt32,Cint,Ptr{Cint}),
              d[1], DT(T), 0,     n,   sz)
    elseif cudnnVersion >= 4000
        @cuda(cudnn,cudnnSetFilterNdDescriptor_v4,
              (Cptr,UInt32,UInt32,Cint,Ptr{Cint}),
              d[1], DT(T), 0,     n,   sz)
    else
        @cuda(cudnn,cudnnSetFilterNdDescriptor,
              (Cptr,UInt32,Cint,Ptr{Cint}),
              d[1], DT(T),    n,   sz)
    end
    fd = FD(d[1])
    finalizer(x->@cuda(cudnn,cudnnDestroyFilterDescriptor,(Cptr,),x.ptr), fd)
    return fd
end

mutable struct CD; ptr
    function CD(w::KnetArray,x::KnetArray; padding=0, stride=1, upscale=1, mode=0)
        d = Cptr[0]
        @cuda(cudnn,cudnnCreateConvolutionDescriptor,(Ptr{Cptr},),d)
        nd = ndims(x)-2
        if cudnnVersion >= 4000
            @cuda(cudnn,cudnnSetConvolutionNdDescriptor,
                  (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},UInt32,UInt32),
                  d[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(upscale,nd),mode,DT(x))
        elseif cudnnVersion > 3000 # does not work when cudnnVersion==3000
            @cuda(cudnn,cudnnSetConvolutionNdDescriptor_v3,
                  (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},UInt32,UInt32),
                  d[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(upscale,nd),mode,DT(x))
        else
            @cuda(cudnn,cudnnSetConvolutionNdDescriptor,
                  (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},UInt32),
                  d[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(upscale,nd),mode)
        end
        cd = new(d[1])
        finalizer(x->@cuda(cudnn,cudnnDestroyConvolutionDescriptor,(Cptr,),x.ptr),cd)
        return cd
    end
end

mutable struct PD; ptr
    function PD(x::KnetArray; window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=0)
        d = Cptr[0]
        @cuda(cudnn,cudnnCreatePoolingDescriptor,(Ptr{Cptr},),d)
        nd = ndims(x)-2
        if cudnnVersion >= 5000
            @cuda(cudnn,cudnnSetPoolingNdDescriptor,
                  (Cptr,UInt32,UInt32,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),
                  d[1],mode,maxpoolingNanOpt,nd,cdsize(window,nd),cdsize(padding,nd),cdsize(stride,nd))
        elseif cudnnVersion >= 4000
            @cuda(cudnn,cudnnSetPoolingNdDescriptor_v4,
                  (Cptr,UInt32,UInt32,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),
                  d[1],mode,maxpoolingNanOpt,nd,cdsize(window,nd),cdsize(padding,nd),cdsize(stride,nd))
        else
            @cuda(cudnn,cudnnSetPoolingNdDescriptor,
                  (Cptr,UInt32,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),
                  d[1],mode,nd,cdsize(window,nd),cdsize(padding,nd),cdsize(stride,nd))
        end
        pd = new(d[1])
        finalizer(x->@cuda(cudnn,cudnnDestroyPoolingDescriptor,(Cptr,),x.ptr), pd)
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

function cdims(w,x; padding=0, stride=1, o...)
    N = ndims(x)
    ntuple(N) do i
        if i < N-1
            pi = (if isa(padding,Number); padding; else padding[i]; end)
            si = (if isa(stride,Number); stride; else stride[i]; end)
            1 + div(size(x,i) - size(w,i) + 2*pi, si)
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

function dcdims(w,x; padding=0, stride=1, o...)
    N = ndims(x)
    @assert size(x,N-1) == size(w,N)
    ntuple(N) do i
        if i < N-1
            pi = (if isa(padding,Number); padding; else padding[i]; end)
            si = (if isa(stride,Number); stride; else stride[i]; end)
            si*(size(x,i)-1) + size(w,i) - 2*pi
        elseif i == N-1
            size(w,N-1)
        else
            size(x,N)
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


### CPU convolution using im2col from Mocha.jl

# w=Ww,Hw,Cx,Cy
# x=Wx,Hx,Cx,Nx
# y=Wy,Hy,Cy,Nx
# if we apply im2col to a single image:
# w2=(Ww*Hw*Cx),Cy  ;; simple reshape
# x2=(Wy*Hy),(Ww*Hw*Cx)
# y2=(Wy*Hy),Cy     ;; simple reshape after y2=x2*w2

function conv4(w::Array{T,4}, x::Array{T,4};
                  padding=0, stride=1, upscale=1, mode=0, alpha=1,
                  o...) where {T} # Ignoring handle, algo, workSpace, workSpaceSizeInBytes
    if upscale != 1; throw(ArgumentError("CPU conv4 only supports upscale=1.")); end
    if mode != 0 && mode != 1; throw(ArgumentError("conv4 only supports mode=0 or 1.")); end
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(w)
    if Cx!=C1; throw(DimensionMismatch()); end
    Wy,Hy,Cy,Ny = cdims(w,x;padding=padding,stride=stride)
    # @assert Cy==C2 && Ny==Nx
    y = similar(x, (Wy,Hy,Cy,Ny))
    x2dims = im2col_dims(w,x,y)
    x2 = similar(x, x2dims)
    (p1,p2) = psize(padding,x)
    (s1,s2) = psize(stride,x)
    M,N,K,Y = Wy*Hy,Cy,Ww*Hw*Cx,Wy*Hy*Cy
    alpha,beta,yidx = T(alpha),T(0),1
    @inbounds for n in 1:Nx
        im2col!(w, x, x2, n, p1, p2, s1, s2, mode)
        gemm!('N','N',M,N,K,alpha,pointer(x2),pointer(w),beta,pointer(y,yidx))
        yidx += Y
    end
    return y
end

function conv4w(w::Array{T,4},x::Array{T,4},dy::Array{T,4};
                   padding=0, stride=1, upscale=1, mode=0, alpha=1,
                   o...) where {T} # Ignoring handle, algo, workSpace, workSpaceSizeInBytes
    # dw = x'*dy
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(w)
    Wy,Hy,Cy,Ny = size(dy)
    # if upscale != 1; throw(ArgumentError("CPU conv4 only supports upscale=1.")); end
    # if mode != 0 && mode != 1; throw(ArgumentError("conv4 only supports mode=0 or 1.")); end
    # @assert Cx==C1 && Cy==C2 && Ny==Nx
    dw = zero(w)
    x2dims = im2col_dims(w,x,dy)
    x2 = similar(x, x2dims)
    # op(A) is an m-by-k matrix, op(B) is a k-by-n matrix, C is an m-by-n matrix.
    Y,M,N,K = Wy*Hy*Cy,Ww*Hw*Cx,Cy,Wy*Hy
    alpha,beta = T(alpha),T(1)
    (p1,p2) = psize(padding,x)
    (s1,s2) = psize(stride,x)
    dyi = 1
    @inbounds for n in 1:Nx
        im2col!(w, x, x2, n, p1, p2, s1, s2, mode)
        gemm!('T','N',M,N,K,alpha,pointer(x2),pointer(dy,dyi),beta,pointer(dw))
        dyi += Y
    end
    return dw
end

function conv4x(w::Array{T,4},x::Array{T,4},dy::Array{T,4};
                   padding=0, stride=1, upscale=1, mode=0, alpha=1,
                   o...) where {T} # Ignoring handle, algo, workSpace, workSpaceSizeInBytes
    # dx = dy*w'
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(w)
    Wy,Hy,Cy,Ny = size(dy)
    # if upscale != 1; throw(ArgumentError("CPU conv4 only supports upscale=1.")); end
    # if mode != 0 && mode != 1; throw(ArgumentError("conv4 only supports mode=0 or 1.")); end
    @assert Cx==C1 && Cy==C2 && Ny==Nx
    dx = similar(x)
    x2dims = im2col_dims(w,x,dy)
    x2 = similar(x, x2dims)
    # op(A) is an m-by-k matrix, op(B) is a k-by-n matrix, C is an m-by-n matrix.
    Y,M,N,K = Wy*Hy*Cy,Wy*Hy,Ww*Hw*Cx,Cy
    alpha,beta = T(alpha),T(0)
    (p1,p2) = psize(padding,x)
    (s1,s2) = psize(stride,x)
    dyi = 1
    @inbounds for n in 1:Nx
        gemm!('N','T',M,N,K,alpha,pointer(dy,dyi),pointer(w),beta,pointer(x2))
        col2im!(w,dx,x2,n,p1,p2,s1,s2,mode)
        dyi += Y
    end
    return dx
end

im2col_dims(w,x,y)=(size(y,1)*size(y,2), size(w,1)*size(w,2)*size(w,3))

# Functions from conv.cpp:

for (T,S) in ((Float32,32), (Float64,64)); @eval begin

    function im2col!(w::Array{$T,4}, x::Array{$T,4}, x2::Array{$T,2},
                     n::Int, p1::Int, p2::Int, s1::Int, s2::Int, mode::Int)
        Wx,Hx,Cx,Nx = size(x)
        Ww,Hw,C1,C2 = size(w)
        xn = pointer(x, Wx*Hx*Cx*(n-1)+1)
        @knet8($("im2col$S"),
               (Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
               xn,x2,Wx,Hx,Cx,Ww,Hw,p1,p2,s1,s2,mode)
        return x2
    end

    function col2im!(w::Array{$T,4}, x::Array{$T,4}, x2::Array{$T,2},
                     n::Int, p1::Int, p2::Int, s1::Int, s2::Int, mode::Int)
        Wx,Hx,Cx,Nx = size(x)
        Ww,Hw,C1,C2 = size(w)
        xn = pointer(x, Wx*Hx*Cx*(n-1)+1)
        @knet8($("col2im$S"),
               (Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
               x2,xn,Wx,Hx,Cx,Ww,Hw,p1,p2,s1,s2,mode)
        return x
    end

    ### CPU pooling from Mocha.jl

    function pool(x::Array{$T,4}; window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=0, alpha=1, handle=nothing)
        if maxpoolingNanOpt!=0; throw(ArgumentError("CPU pool only supports maxpoolingNanOpt=0")); end
        Wx,Hx,Cx,Nx = size(x);
        Wy,Hy,Cy,Ny = pdims(x;window=window,padding=padding,stride=stride)
        y = similar(x, (Wy,Hy,Cy,Ny))
        (w1,w2) = psize(window, x)
        (p1,p2) = psize(padding, x)
        (s1,s2) = psize(stride, x)
        if mode == 0
            @knet8($("max_pooling_fwd$S"),
                   (Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                   x,y,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
        elseif mode == 1 || (mode == 2 && p1==p2==0)
            @knet8($("mean_pooling_fwd$S"),
                   (Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                   x,y,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
        else
            throw(ArgumentError("mode $mode not supported by cpu pool"))
        end
        if alpha != 1; lmul!(alpha,y); end
        return y
    end

    function poolx(x::Array{$T,4}, y::Array{$T,4}, dy::Array{$T,4};
                   window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=0, alpha=1, handle=nothing)
        if maxpoolingNanOpt!=0; throw(ArgumentError("CPU pool only supports maxpoolingNanOpt=0")); end
        Wx,Hx,Cx,Nx = size(x);
        Wy,Hy,Cy,Ny = size(y);
        dx = similar(x)
        (w1,w2) = psize(window, x)
        (p1,p2) = psize(padding, x)
        (s1,s2) = psize(stride, x)
        if mode == 0
            if alpha != 1; y = y ./ alpha; end
            @knet8($("max_pooling_bwd$S"),
                   (Ptr{$T},Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                   x,y,dy,dx,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
        elseif mode == 1 || (mode == 2 && p1==p2==0)
            @knet8($("mean_pooling_bwd$S"),
                   (Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                   dx,dy,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
        else
            throw(ArgumentError("mode $mode not supported by cpu pool"))
        end
        if alpha != 1; lmul!(alpha,dx); end
        return dx
    end
end;end


# Utilities to find a fast algorithm

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
        Knet.gc()          # TODO: fix memory management
        @cuda(cudnn, cudnnFindConvolutionForwardAlgorithm,
              (Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Ptr{Cint},Cptr),
              handle,TD(x),FD(w),CD(w,x;o...),TD(y),requestedAlgoCount,returnedAlgoCount,perfResults)
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
        Knet.gc()
        @cuda(cudnn, cudnnFindConvolutionBackwardFilterAlgorithm,
              (Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Ptr{Cint},Cptr),
              handle,TD(x),TD(dy),CD(w,x;o...),FD(dw),requestedAlgoCount,returnedAlgoCount,perfResults)
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
        Knet.gc()
        @cuda(cudnn, cudnnFindConvolutionBackwardDataAlgorithm,
              (Cptr,Cptr,Cptr,Cptr,Cptr,Cint,Ptr{Cint},Cptr),
              handle,FD(w),TD(dy),CD(w,x;o...),TD(dx),requestedAlgoCount,returnedAlgoCount,perfResults)
        p = perfChoose(perfResults, returnedAlgoCount[1])
        conv4x_algos[key] = p
        return (p.algo, cudnnWorkSpace(p.memory))
    end
end

CUDNN_WORKSPACE_MAXSIZE = 0     # Will be set to 20% of available gpu memory
function perfChoose(ps, n)
    global CUDNN_WORKSPACE_MAXSIZE
    if n==ps
        warn("returnedAlgoCount==requestedAlgoCount")
    end
    if CUDNN_WORKSPACE_MAXSIZE == 0
        CUDNN_WORKSPACE_MAXSIZE = div(gpufree(),5)
    end
    (ibest,mbest,tbest) = (0,CUDNN_WORKSPACE_MAXSIZE,Inf)
    for i = 1:n
        if ps[i].status == 0 && ps[i].memory < mbest && ps[i].time < tbest * 1.1
            (ibest,mbest,tbest) = (i,ps[i].memory,ps[i].time)
        end
    end
    if ibest == 0; error("No good algo found."); end
    return ps[ibest]
end

# TODO: this assumes one workspace per gpu. What about streams?
const CUDNN_WORKSPACE = []
function cudnnWorkSpace(len=0;dev=gpu())
    global CUDNN_WORKSPACE
    if dev==-1; error("No cudnnWorkSpace for CPU"); end
    i = dev+2
    if isempty(CUDNN_WORKSPACE); resize!(CUDNN_WORKSPACE,gpuCount()+1); end
    if !isassigned(CUDNN_WORKSPACE,i) || length(CUDNN_WORKSPACE[i]) < len
        CUDNN_WORKSPACE[i]=KnetArray{UInt8}(undef,len);
    end
    return CUDNN_WORKSPACE[i]
end
