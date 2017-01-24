# Define some new primitives: conv4 and pool

"""

`conv4(w,x;kwargs...)` executes convolutions or cross-correlations
using filters specified with `w` over tensor `x`.  Currently 4 or 5
dimensional KnetArrays with Float32 or Float64 entries are supported.

If `w` has dimensions (W1,W2,...,I,O) and `x` has dimensions
(X1,X2,...,I,N), the result y will have dimensions (Y1,Y2,...,O,N)
where

    Yi=1+floor((Xi+2*padding[i]-Wi)/stride[i])

Here I is the number of input channels, O is the number of output
channels, N is the number of instances, and Wi,Xi,Yi are spatial
dimensions.  Padding and stride are keyword arguments that can be
specified as a single number (in which case they apply to all
dimensions), or an array/tuple with entries for each spatial
dimension.

Here is a description of all available keyword arguments:

* padding: the number of extra zeros implicitly concatenated at the start and at the end of each dimension. Default=floor((filterSize-1)/2) which preserves the input size when filterSize is odd and stride=1.
* stride: the number of elements to slide to reach the next filtering window. Default=1.
* upscale: upscale factor for each dimension. Default=1.
* mode: 0 for convolution and 1 for cross-correlation.  Default=0.
* alpha: can be used to scale the result. Default=1.
* algo: specifies which convolution algorithm shoud be used to compute the results. Default=0. See the CUDNN User Guide for details.
* workSpace: data pointer to GPU memory to a workspace needed to able to execute the specified algorithm. Default=C_NULL.
* workSpaceSizeInBytes: the size in bytes of the provided workSpace. Default=0.
* handle: handle to a previously created cuDNN context. Default=Knet allocated context.

"""
function conv4{T}(w::KnetArray{T},x::KnetArray{T};
                  handle=cudnnhandle, alpha=one(T), beta=zero(T),
                  algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0, o...)
    y = similar(x, cdims(w,x;o...))
    @cuda(cudnn, cudnnConvolutionForward,
          (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,   UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
          handle,Ref(alpha),TD(x),x,FD(w),w,CD(w,x;o...),algo,workSpace,workSpaceSizeInBytes,Ref(beta),TD(y),y)
    return y
end

function conv4x{T}(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T};
                   handle=cudnnhandle, alpha=one(T), beta=zero(T),
                   algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0, o...)
    dx = similar(x)
    if cudnnVersion >= 4000
        @cuda(cudnn,cudnnConvolutionBackwardData,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(alpha),FD(w),w,TD(dy),dy,CD(w,x;o...),algo,workSpace,workSpaceSizeInBytes,Ref(beta),TD(dx),dx)
    elseif cudnnVersion >= 3000
        @cuda(cudnn,cudnnConvolutionBackwardData_v3,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(alpha),FD(w),w,TD(dy),dy,CD(w,x;o...),algo,workSpace,workSpaceSizeInBytes,Ref(beta),TD(dx),dx)
    else
        @cuda(cudnn,cudnnConvolutionBackwardData,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,       Ptr{T},Cptr,Ptr{T}),
              handle,Ref(alpha),FD(w),w,TD(dy),dy,CD(w,x;o...),Ref(beta),TD(dx),dx)
    end
    return dx
end

function conv4w{T}(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T};
                   handle=cudnnhandle, alpha=one(T), beta=zero(T),
                   algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0, o...)
    dw = similar(w)
    if cudnnVersion >= 4000
        @cuda(cudnn,cudnnConvolutionBackwardFilter,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(alpha),TD(x),x,TD(dy),dy,CD(w,x;o...),algo,workSpace,workSpaceSizeInBytes,Ref(beta),FD(dw),dw)
    elseif cudnnVersion >= 3000
        @cuda(cudnn,cudnnConvolutionBackwardFilter_v3,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(alpha),TD(x),x,TD(dy),dy,CD(w,x;o...),algo,workSpace,workSpaceSizeInBytes,Ref(beta),FD(dw),dw)
    else
        @cuda(cudnn,cudnnConvolutionBackwardFilter,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,       Ptr{T},Cptr,Ptr{T}),
              handle,Ref(alpha),TD(x),x,TD(dy),dy,CD(w,x;o...),Ref(beta),FD(dw),dw)
    end
    return dw
end


@primitive conv4(w,x; o...),dy  conv4w(w,x,dy;o...)  conv4x(w,x,dy;o...)
@zerograd  conv4x(w,x,dy;o...)
@zerograd  conv4w(w,x,dy;o...)


"""

`pool(x;kwargs...)` computes pooling of input values (i.e., the
maximum or average of several adjacent values) to produce an output
with smaller height and/or width.  Currently 4 or 5 dimensional
KnetArrays with Float32 or Float64 entries are supported.

If `x` has dimensions (X1,X2,...,I,N), the result y will have
dimensions (Y1,Y2,...,I,N) where

   Yi=1+floor((Xi+2*padding[i]-window[i])/stride[i])

Here I is the number of input channels, N is the number of instances,
and Xi,Yi are spatial dimensions.  Window, padding and stride are
keyword arguments that can be specified as a single number (in which
case they apply to all dimensions), or an array/tuple with entries for
each spatial dimension.

Here is a description of all available keyword arguments:

* window: the pooling window size for each dimension. Default=2.
* padding: the number of extra zeros implicitly concatenated at the start and at the end of each dimension. Default=0.
* stride: the number of elements to slide to reach the next pooling window. Default=same as window.
* mode: 0 for max, 1 for average including padded values, 2 for average excluding padded values.  Default=0.
* maxpoolingNanOpt: Nan numbers are not propagated if 0, they are propagated if 1. Default=0.
* alpha: can be used to scale the result. Default=1.
* handle: Handle to a previously created cuDNN context. Default=Knet allocated context.

"""
function pool{T}(x::KnetArray{T}; handle=cudnnhandle, alpha=one(T), beta=zero(T), o...)
    y = similar(x, pdims(x; o...))
    @cuda(cudnn, cudnnPoolingForward,
          (Cptr, Cptr,      Ptr{T},    Cptr,Ptr{T},Ptr{T},   Cptr,Ptr{T}),
          handle,PD(x;o...),Ref(alpha),TD(x),x,    Ref(beta),TD(y),y)
    return y
end

function poolx{T}(x::KnetArray{T},y::KnetArray{T},dy::KnetArray{T};
                  handle=cudnnhandle, alpha=one(T), beta=zero(T), o...)
    dx = similar(x)
    @cuda(cudnn,cudnnPoolingBackward,
          (Cptr,Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Ptr{T},Cptr,Ptr{T}),
          handle,PD(x;o...),Ref(alpha),TD(y),y,TD(dy),dy,TD(x),x,Ref(beta),TD(dx),dx)
    return dx
end

@primitive pool(x;o...),dy,y  poolx(x,y,dy;o...)
@zerograd  poolx(x,y,dy;o...)

# cudnn descriptors

type TD; ptr
    function TD(a::KnetArray)
        d = Cptr[0]
        @cuda(cudnn,cudnnCreateTensorDescriptor,(Ptr{Cptr},),d)
        n = ndims(a)
        sz = [Cint(size(a,n-i+1)) for i=1:n]
        st = [Cint(stride(a,n-i+1)) for i=1:n]
        @cuda(cudnn,cudnnSetTensorNdDescriptor,
              (Cptr,UInt32,Cint,Ptr{Cint},Ptr{Cint}),
              d[1], DT(a), n, sz, st)
        td = new(d[1])
        finalizer(td, x->@cuda(cudnn,cudnnDestroyTensorDescriptor,(Cptr,),x.ptr))
        return td
    end
end

type FD; ptr
    function FD(a::KnetArray)
        d = Cptr[0]
        @cuda(cudnn,cudnnCreateFilterDescriptor,(Ptr{Cptr},),d)
        n = ndims(a)
        sz = [Cint(size(a,n-i+1)) for i=1:n]
        if cudnnVersion >= 5000
            @cuda(cudnn,cudnnSetFilterNdDescriptor,
                  (Cptr,UInt32,UInt32,Cint,Ptr{Cint}),
                  d[1], DT(a), 0,     n,   sz)
        elseif cudnnVersion >= 4000
            @cuda(cudnn,cudnnSetFilterNdDescriptor_v4,
                  (Cptr,UInt32,UInt32,Cint,Ptr{Cint}),
                  d[1], DT(a), 0,     n,   sz)
        else
            @cuda(cudnn,cudnnSetFilterNdDescriptor,
                  (Cptr,UInt32,Cint,Ptr{Cint}),
                  d[1], DT(a),    n,   sz)
        end
        fd = new(d[1])
        finalizer(fd, x->@cuda(cudnn,cudnnDestroyFilterDescriptor,(Cptr,),x.ptr))
        return fd
    end
end

type CD; ptr
    function CD(w::KnetArray,x::KnetArray; padding=0, stride=1, upscale=1, mode=0)
        d = Cptr[0]
        @cuda(cudnn,cudnnCreateConvolutionDescriptor,(Ptr{Cptr},),d)
        nd = ndims(x)-2
        if cudnnVersion >= 4000
            @cuda(cudnn,cudnnSetConvolutionNdDescriptor,
                  (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},UInt32,UInt32),
                  d[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(upscale,nd),mode,DT(x))
        elseif cudnnVersion >= 3000
            @cuda(cudnn,cudnnSetConvolutionNdDescriptor_v3,
                  (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},UInt32,UInt32),
                  d[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(upscale,nd),mode,DT(x))
        else
            @cuda(cudnn,cudnnSetConvolutionNdDescriptor,
                  (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},UInt32),
                  d[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(upscale,nd),mode)
        end
        cd = new(d[1])
        finalizer(cd, x->@cuda(cudnn,cudnnDestroyConvolutionDescriptor,(Cptr,),x.ptr))
        return cd
    end
end

type PD; ptr
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
        finalizer(pd, x->@cuda(cudnn,cudnnDestroyPoolingDescriptor,(Cptr,),x.ptr))
        return pd
    end
end

import Base: unsafe_convert
unsafe_convert(::Type{Cptr}, td::TD)=td.ptr
unsafe_convert(::Type{Cptr}, fd::FD)=fd.ptr
unsafe_convert(::Type{Cptr}, cd::CD)=cd.ptr
unsafe_convert(::Type{Cptr}, pd::PD)=pd.ptr

function cdsize(w, nd)
    if isa(w,Integer)
        fill(Cint(w),nd)
    elseif length(w)==nd 
        [ Cint(w[nd-i+1]) for i=1:nd ]
    else
        throw(DimensionMismatch("$w $nd"))
    end
end

DT(::KnetArray{Float32})=UInt32(0)
DT(::KnetArray{Float64})=UInt32(1)
DT(::KnetArray{Float16})=UInt32(2)

function cdims{T,N}(w::KnetArray{T,N},x::KnetArray{T,N}; padding=0, stride=1, o...)
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

function pdims{T,N}(x::KnetArray{T,N}; window=2, padding=0, stride=window, o...)
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

# Manually call this function inside your model
# convolution padding size that preserves the input size when filter size is odd and stride=1
padsize(w)=ntuple(i->div(size(w,i)-1,2), ndims(w))

"""

mat(x) reshapes x into a two-dimensional matrix.  For 1-D inputs mat
returns `reshape(x, (length(x),1))`.  For inputs with more than two
dimensions of size (X1,X2,...,XD), mat returns

    reshape(x, (X1*X2*...*X[D-1],XD))

"""
function mat(x)
    if ndims(x) > 2
        xn = size(x,ndims(x))
        reshape(x, (div(length(x),xn),xn))
    elseif ndims(x)==2
        x
    elseif ndims(x)==1
        reshape(x, (length(x),1))
    else
        throw(MethodError(mat,x))
    end
end


"""

Implements deconvolution, which can be thought as the 'reverse' of the convolution operation.

`deconv4(w,x; kwargs...)` deconvolves `w` and `x`

Example:

Input `x`:

0   10

20  30

Filter `w`:

1   2   3

4   5   6

7   8   9

Output of `deconv4(w,x)`:

0   10  20  30

20  110 170 150

80  290 350 270

140 370 420 270

How is `deconv4(w,x)` calculated ?

Calculate `conv4(w,x; padding=filterSize-1)`

There is a better way (switch forward and backward passes), but this is the easiest...

Here is a description of all available keyword arguments:

* None at the moment.

"""
function deconv4{T}(w::KnetArray{T},x::KnetArray{T}; o...)
    y = similar(x,dcdims(w,x;o...))
    return conv4x(w,y,x;o...)
end

function deconv4w{T}(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T}; o...)
    return conv4w(w,dy,x;o...)
end

function deconv4x{T}(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T}; o...)
    return conv4(w,dy;o...)
end


@primitive deconv4(w,x; o...),dy,y  deconv4w(w,x,dy; o...)  deconv4x(w,x,dy; o...)
@zerograd deconv4w(w,x,dy; o...)
@zerograd deconv4x(w,x,dy; o...)

function dcdims{T,N}(w::KnetArray{T,N},x::KnetArray{T,N}; padding=0, stride=1, o...)
    ntuple(N) do i
        if i < N-1
            pi = (if isa(padding,Number); padding; else padding[i]; end)
            si = (if isa(stride,Number); stride; else stride[i]; end)
            si*(size(x,i)-1) + size(w,i) - 2*pi
        elseif i == N-1
            size(w,N)
        else # i == N
            size(x,N)
        end
    end
end

"""

Implements simple upsampling, which can be thought as the 'reverse' of the pooling operation.

`unpool(x; kwargs...)` upsamples input `x`

Example:

Input `x`:

6   14

8   16

Output of `unpool(x)`:

6   6   14  14

6   6   14  14

8   8   16  16

8   8   16  16

Here is a description of all available keyword arguments:

* window: integer upsampling factor. Default=2.

"""
function unpool{T}(x::KnetArray{T}; window=2)
    y = similar(x,updims(x; window=window))
    Knet.poolx(y,x,x.*window^2; window=window,mode=1)
end

function unpoolx{T}(dy::KnetArray{T}; window=2)
    -pool(-dy; window=window)
end

#Calculates the output dimensions of unpool of x
function updims{T,N}(x::KnetArray{T,N}; window=2)
    if !isa(window,Number) error("Window size must be a number!") end
    if !isa(window,Integer) error("Window size must be an integer!") end
    ntuple(N) do i
        if i < N-1
            size(x,i)*window
        else
            size(x,i)
        end
    end
end

@primitive unpool(x; o...),dy,y  unpoolx(dy; o...)
@zerograd unpoolx(dy; o...)

import Base: transpose
function transpose{T}(x::KnetArray{T})
    ndims(x) != 2 && error("Transpose is supported only for 2D KnetArrays")
    sz = size(x)
    y = similar(x,(sz[2],sz[1]))
    if T<:Float32
        @cuda(cublas, cublasSgeam, (Cptr,UInt32,UInt32,Cint,Cint,Ptr{T},Ptr{T},Cint,Ptr{T},Ptr{T},Cint,Ptr{T},Cint),
            cublashandle,1,1,size(y,1),size(y,2),Ref(T(1.0)),x,size(x,1),Ref(T(0.0)),x,size(x,1),y,size(y,1))
    elseif T<:Float64
        @cuda(cublas, cublasDgeam, (Cptr,UInt32,UInt32,Cint,Cint,Ptr{T},Ptr{T},Cint,Ptr{T},Ptr{T},Cint,Ptr{T},Cint),
            cublashandle,1,1,size(y,1),size(y,2),Ref(T(1.0)),x,size(x,1),Ref(T(0.0)),x,size(x,1),y,size(y,1))
    else
        error("CUBLAS does not support $T")
    end
    return y
end

import Base: permutedims
function permutedims{T,N}(x::KnetArray{T,N}, dims)
    length(dims) != ndims(x) && error("Dimensions mismatch")
    (N < 2 || N > 5) && error("Unsupported number of dimensions")

    funcName = "permutedims$(N)D_"
    for i=1:N
        funcName = funcName * "$(dims[i])_"
    end
    if T<:Float32
        funcName = funcName * "32"
    elseif T<:Float64
        funcName = funcName * "64"
    else
        error("$T not supported")
    end
    funcName = funcName * "_44"
    
    if N == 2
        if dims == [1 2]
            return x
        elseif dims == [2 1]
            return transpose(x)
        end
    elseif N == 3
        if dims == [1 2 3]
            return x
        else
            y = similar(x, size(x,dims[1]), size(x,dims[2]), size(x,dims[3]))
            @eval ccall(($funcName,libknet8),Void,(Ptr{$T},Cint,Cint,Cint,Ptr{$T},Cint,Cint,Cint),
                                                    $x,size($x,1),size($x,2),size($x,3),$y,size($y,1),size($y,2),size($y,3))
            return y
        end
    elseif N == 4 || N == 5
        error("Not yet implemented")
    end
end