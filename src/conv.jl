# cpuconv todo:
# ok: need to get rid of mask in pool_back
# ok: reimplement conv4 in terms of im2col
# ok: need low level blas call with pointers
# reimplement conv4x conv4w using col2im?
# time doing a single im2col instead of N
# try `#pragma omp parallel for` in im2col
# need separate cpu and gpu libraries: condition makefile on finding nvcc, also cond openmpi like mocha/dep
# replace T<: functions with generated code for each type


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
* `algo=0`: specifies which convolution algorithm shoud be used to compute the results. See the CUDNN User Guide for details.
* `workSpace=C_NULL`: data pointer to GPU memory to a workspace needed to able to execute the specified algorithm.
* `workSpaceSizeInBytes=0`: the size in bytes of the provided workSpace. Default=0.
* `handle`: handle to a previously created cuDNN context. Defaults to a Knet allocated handle.

"""
function conv4{T}(w::KnetArray{T},x::KnetArray{T};
                  handle=cudnnhandle(), algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0, alpha=1,
                  o...) # padding=0, stride=1, upscale=1, mode=0
    y = similar(x, cdims(w,x;o...))
    beta=0 # nonzero beta does not make sense when we create y
    @cuda(cudnn, cudnnConvolutionForward,
          (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,   UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
          handle,Ref(T(alpha)),TD(x),x,FD(w),w,CD(w,x;o...),algo,workSpace,workSpaceSizeInBytes,Ref(T(beta)),TD(y),y)
    return y
end

function conv4x{T}(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T};
                   handle=cudnnhandle(), alpha=1, algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0,
                   o...) # padding=0, stride=1, upscale=1, mode=0
    beta = 0
    dx = similar(x)
    if cudnnVersion >= 4000
        @cuda(cudnn,cudnnConvolutionBackwardData,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),FD(w),w,TD(dy),dy,CD(w,x;o...),algo,workSpace,workSpaceSizeInBytes,Ref(T(beta)),TD(dx),dx)
    elseif cudnnVersion >= 3000
        @cuda(cudnn,cudnnConvolutionBackwardData_v3,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),FD(w),w,TD(dy),dy,CD(w,x;o...),algo,workSpace,workSpaceSizeInBytes,Ref(T(beta)),TD(dx),dx)
    else
        @cuda(cudnn,cudnnConvolutionBackwardData,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,       Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),FD(w),w,TD(dy),dy,CD(w,x;o...),Ref(T(beta)),TD(dx),dx)
    end
    return dx
end

function conv4w{T}(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T};
                   handle=cudnnhandle(), alpha=1, algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0,
                   o...) # padding=0, stride=1, upscale=1, mode=0
    beta = 0
    dw = similar(w)
    if cudnnVersion >= 4000
        @cuda(cudnn,cudnnConvolutionBackwardFilter,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),TD(x),x,TD(dy),dy,CD(w,x;o...),algo,workSpace,workSpaceSizeInBytes,Ref(T(beta)),FD(dw),dw)
    elseif cudnnVersion >= 3000
        @cuda(cudnn,cudnnConvolutionBackwardFilter_v3,
              (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
              handle,Ref(T(alpha)),TD(x),x,TD(dy),dy,CD(w,x;o...),algo,workSpace,workSpaceSizeInBytes,Ref(T(beta)),FD(dw),dw)
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
function pool{T}(x::KnetArray{T}; handle=cudnnhandle(), alpha=1, 
                 o...) # window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=0
    y = similar(x, pdims(x; o...))
    beta = 0
    @cuda(cudnn, cudnnPoolingForward,
          (Cptr, Cptr,      Ptr{T},    Cptr,Ptr{T},Ptr{T},   Cptr,Ptr{T}),
          handle,PD(x;o...),Ref(T(alpha)),TD(x),x,    Ref(T(beta)),TD(y),y)
    return y
end

function poolx{T}(x::KnetArray{T},y::KnetArray{T},dy::KnetArray{T}; handle=cudnnhandle(), alpha=1, mode=0,
                  o...) # window=2, padding=0, stride=window, maxpoolingNanOpt=0
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

Deconvolution; `reverse` of convolution.

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

DT(::KnetArray{Float32})=UInt32(0)
DT(::KnetArray{Float64})=UInt32(1)
DT(::KnetArray{Float16})=UInt32(2)

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
    ntuple(N) do i
        if i < N-1
            pi = (if isa(padding,Number); padding; else padding[i]; end)
            si = (if isa(stride,Number); stride; else stride[i]; end)
            si*(size(x,i)-1) + size(w,i) - 2*pi
        elseif i == N-1
            size(w,N)
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

function conv4{T}(w::Array{T,4}, x::Array{T,4};
                  padding=0, stride=1, upscale=1, mode=0, alpha=1,
                  o...) # Ignoring handle, algo, workSpace, workSpaceSizeInBytes
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

function conv4w{T}(w::Array{T,4},x::Array{T,4},dy::Array{T,4};
                   padding=0, stride=1, upscale=1, mode=0, alpha=1,
                   o...) # Ignoring handle, algo, workSpace, workSpaceSizeInBytes
    # dw = x'*dy
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(w)
    Wy,Hy,Cy,Ny = size(dy)
    # if upscale != 1; throw(ArgumentError("CPU conv4 only supports upscale=1.")); end
    # if mode != 0 && mode != 1; throw(ArgumentError("conv4 only supports mode=0 or 1.")); end
    # @assert Cx==C1 && Cy==C2 && Ny==Nx
    dw = zeros(w)
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

function conv4x{T}(w::Array{T,4},x::Array{T,4},dy::Array{T,4};
                   padding=0, stride=1, upscale=1, mode=0, alpha=1,
                   o...) # Ignoring handle, algo, workSpace, workSpaceSizeInBytes
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

for (T,S) in ((Float32,32), (Float64,64))
    @eval begin
        function im2col!(w::Array{$T,4}, x::Array{$T,4}, x2::Array{$T,2},
                         n::Int, p1::Int, p2::Int, s1::Int, s2::Int, mode::Int)
            Wx,Hx,Cx,Nx = size(x)
            Ww,Hw,C1,C2 = size(w)
            xn = pointer(x, Wx*Hx*Cx*(n-1)+1)
            ccall(($("im2col$S"),libknet8),Void,
                  (Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                  xn,x2,Wx,Hx,Cx,Ww,Hw,p1,p2,s1,s2,mode)
            return x2
        end
        function col2im!(w::Array{$T,4}, x::Array{$T,4}, x2::Array{$T,2},
                         n::Int, p1::Int, p2::Int, s1::Int, s2::Int, mode::Int)
            Wx,Hx,Cx,Nx = size(x)
            Ww,Hw,C1,C2 = size(w)
            xn = pointer(x, Wx*Hx*Cx*(n-1)+1)
            ccall(($("col2im$S"),libknet8),Void,
                  (Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                  x2,xn,Wx,Hx,Cx,Ww,Hw,p1,p2,s1,s2,mode)
            return x
        end
    end
end

#@views
function convy{T}(x0::AbstractArray{T,2}, w::AbstractArray{T,2}, padding::Array{Int,1}, stride::Array{Int,1}, mode)
    x=x0
    if any(padding .> 0) # this could be handled better....
        x=zeros(eltype(x0), 2*padding+collect(size(x0))...)
        x[padding[1]+1:end-padding[1],padding[2]+1:end-padding[2]] = x0
    end
    w1 = vec(w); if mode==0; w1 = reverse(w1); end
    row_extend, col_extend = floor(Int, 1 + (collect(size(x)) - collect(size(w))) ./ stride)
    widx = Int[sub2ind(size(x),i,j) for i in 1:stride[1]:size(x,1)-size(w,1)+1, j in 1:stride[2]:size(x,2)-size(w,2)+1] # linear indexes of filter positions in x
    oidx = Int[sub2ind(size(x),i,j) for i in 1:size(w,1), j in 1:size(w,2)] # linear indexes of elements in a filter window
    destidx = Int[i+(j-1) for i in vec(widx), j in vec(oidx)]
    return reshape(x[destidx]*w1,row_extend,col_extend)
end

# dw = rot180(xcorr(x,dy))
#@views
function conv4w1{T}(w::Array{T,4}, x::Array{T,4}, dy::Array{T,4};
                          padding=0, stride=1, upscale=1, mode=0, alpha=1,
                          o...) # Ignoring handle, algo, workSpace, workSpaceSizeInBytes
    if upscale != 1; throw(ArgumentError("CPU conv4 only supports upscale=1.")); end
    if mode != 0 && mode != 1; throw(ArgumentError("conv4 only supports mode=0 or 1.")); end
    padding = psize(padding,x)
    stride  = psize(stride, x)
    # x:    (Wx,Hx,Cx,N)
    # dy:   (Wy,Hy,K,N) 
    # dw:    (Ww,Hw,Cw,K) 
    dw = fill!(similar(w),0)
    Wx,Hx,C,Nx = size(x)
    Wy,Hy,Cy,Ny = size(dy)
    @inbounds for c in 1:C, k in 1:Cy, n in 1:Ny
        # axpy!(1, convdw(x[:,:,c,n], dy[:,:,k,n], dw[:,:,c,k], padding, stride, mode), view(dw,:,:,c,k))
        dw[:,:,c,k] += convdw(x[:,:,c,n], dy[:,:,k,n], dw[:,:,c,k], padding, stride, mode)
    end
    if alpha != 1; scale!(alpha, dw); end
    return dw
end

# dw = rot180(xcorr(x,dy))
#@views
function convdw{T}(x0::AbstractArray{T,2}, dy::AbstractArray{T,2}, w::AbstractArray{T,2}, padding::Array{Int,1}, stride::Array{Int,1}, mode::Int)
    if any(padding .> 0) # this could be handled better...
        x=zeros(eltype(x0), 2*padding+collect(size(x0))...)
        x[padding[1]+1:end-padding[1],padding[2]+1:end-padding[2]] = x0
    else
        x=x0
    end
    x1l = last(collect(take(countfrom(1,stride[1]),size(dy,1))))
    x2l = last(collect(take(countfrom(1,stride[2]),size(dy,2))))
    widx = Int[sub2ind(size(x),i,j) for i in 1:size(w,1), j in 1:size(w,2)]
    oidx = Int[sub2ind(size(x),i,j) for i in 1:stride[1]:x1l, j in 1:stride[2]:x2l] # linear indexes of elements in a filter window
    destidx = Int[i+(j-1) for i in vec(widx), j in vec(oidx)]
    y = reshape(x[destidx]*vec(dy),size(w))
    if mode == 0; y = rot180(y); end
    return y
end

# dx = xcorr(dy, w, 'full')
#@views
function conv4x1{T}(w::Array{T,4}, x::Array{T,4}, dy::Array{T,4};
                          padding=0, stride=1, upscale=1, mode=0, alpha=1,
                          o...) # Ignoring handle, algo, workSpace, workSpaceSizeInBytes
    if upscale != 1; throw(ArgumentError("CPU conv4 only supports upscale=1.")); end
    if mode != 0 && mode != 1; throw(ArgumentError("conv4 only supports mode=0 or 1.")); end
    Wy,Hy,Ky,Nx = size(dy)
    Ww,Hw,C,Kw = size(w)
    if Ky!=Kw; throw(DimensionMismatch()); end
    padding = psize(padding, x)
    stride  = psize(stride, x)
    dx = fill!(similar(x),0)
    @inbounds for n in 1:Nx, c in 1:C, k in 1:Kw
        dx[:,:,c,n] += convdx(dy[:,:,k,n], w[:,:,c,k], dx[:,:,c,n], padding, stride, mode)
    end
    if alpha != 1; scale!(alpha, dx); end
    return dx
end

# dx = xcorr(dy, w, 'full')
#@views
function convdx{T}(dy::AbstractArray{T,2}, w::AbstractArray{T,2}, dx::AbstractArray{T,2}, padding::Array{Int,1}, stride::Array{Int,1}, mode::Int)
    size_tdy = collect(size(dx)) + collect(size(w)) - 1 + 2padding
    tdy = zeros(T, size_tdy...)
    pad1, pad2 = map(x->x-1,size(w))
    for (i,idy) in zip(countfrom(pad1+1,stride[1]), 1:size(dy,1)), (j,jdy) in zip(countfrom(pad2+1,stride[2]), 1:size(dy,2))
        tdy[i,j] = dy[idy,jdy]
    end
    res = convy(tdy, w, [0,0], [1,1], 1-mode)
    if all(padding .== 0)
        return res
    else
        return res[padding[1]+1:end-padding[1],padding[2]+1:end-padding[2]]
    end
end


#=
function _conv2{T}(x::Array{T,2}, w::Array{T,2}; pad=0, stride=1, xcorr=false)
    max_pad = map(x->x-1-pad,size(w))
    y = conv2(x, xcorr ? rot180(w) : w)
    return y[1+max_pad[1]:stride:end-max_pad[1], 1+max_pad[2]:stride:end-max_pad[2]]
end
=#

#=
function getConvolutionNdForwardOutputDim{T}(x::Array{T,4}, w::Array{T,4}; padding=padding,stride=stride)
    padding = isa(padding, Integer) ? [padding,padding] : collect(padding)
    stride = isa(stride, Integer) ? [stride,stride] : collect(stride)
    Wx,Hx,Cx,N = size(x)
    Ww,Hw,Cw,K = size(w)
    @assert Cx==Cw
    Wy,Hy = floor(Int, 1 + (Int[Wx,Hx] + 2*padding - Int[Ww,Hw]) ./ stride)
    return (Wy,Hy,K,N)
end
=#


### CPU pooling from Mocha.jl


#@views
function pool1{T}(x::Array{T,4}; window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=0, alpha=1, handle=nothing)
    if maxpoolingNanOpt!=0; throw(ArgumentError("CPU pool only supports maxpoolingNanOpt=0")); end
    y = fill!(similar(x, pdims(x;window=window,padding=padding,stride=stride)), 0)
    stride = psize(stride, x)
    window = psize(window, x)
    padding = psize(padding, x)
    Wx,Hx,Cx,Nx = size(x);
    Wy,Hy,Cy,Ny = size(y);
    if any(padding .> 0)
        Wx += 2*padding[1]
        Hx += 2*padding[2]
        x0 = fill!(similar(x, (Wx,Hx,Cx,Nx)), 0)
        x0[padding[1]+1:end-padding[1], padding[2]+1:end-padding[2],:,:] = x
        x = x0
    end
    if mode == 0
        @inbounds for n in 1:Nx, c in 1:Cx, jy in 1:Hy, iy in 1:Wy
            # iy, jy = div(i,stride[1])+1, div(j,stride[2])+1
            i, j = 1+stride[1]*(iy-1), 1+stride[2]*(jy-1)
            wx_end = min(i+window[1]-1,Wx)
            hx_end = min(j+window[2]-1,Hx)
            y[iy,jy,c,n] = maximum(x[i:wx_end,j:hx_end,c,n])
        end
    elseif mode == 1 || (mode == 2 && all(padding .== 0))
        @inbounds for n in 1:Nx, c in 1:Cx, jy in 1:Hy, iy in 1:Wy
            # iy, jy = div(i,stride[1])+1, div(j,stride[2])+1
            i, j = 1+stride[1]*(iy-1), 1+stride[2]*(jy-1)
            wx_end = min(i+window[1]-1, Wx)
            hx_end = min(j+window[2]-1, Hx)
            y[iy,jy,c,n] = mean(x[i:wx_end,j:hx_end,c,n])
        end
    else
        throw(ArgumentError("mode $mode not supported by cpu pool"))
    end
    if alpha != 1; scale!(alpha,y); end
    return y
end

function pool{T}(x::Array{T,4}; window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=0, alpha=1, handle=nothing)
    if maxpoolingNanOpt!=0; throw(ArgumentError("CPU pool only supports maxpoolingNanOpt=0")); end
    ydims = pdims(x;window=window,padding=padding,stride=stride)
    y = similar(x, ydims)
    (w1,w2) = psize(window, x)
    (p1,p2) = psize(padding, x)
    (s1,s2) = psize(stride, x)
    Wx,Hx,Cx,Nx = size(x);
    Wy,Hy,Cy,Ny = size(y);
    if mode == 0
        if T<:Float32
            ccall((:max_pooling_fwd32,libknet8),Void,
                  (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                  x,y,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
        elseif T<:Float64
            ccall((:max_pooling_fwd64,libknet8),Void,
                  (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                  x,y,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
        else
            error("$T not supported")
        end
    elseif mode == 1 || (mode == 2 && p1==p2==0)
        if T<:Float32
            ccall((:mean_pooling_fwd32,libknet8),Void,
                  (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                  x,y,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
        elseif T<:Float64
            ccall((:mean_pooling_fwd64,libknet8),Void,
                  (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                  x,y,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
        else
            error("$T not supported")
        end
    else
        throw(ArgumentError("mode $mode not supported by cpu pool"))
    end
    if alpha != 1; scale!(alpha,y); end
    return y
end

function poolx{T}(x::Array{T,4}, y::Array{T,4}, dy::Array{T,4};
                  window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=0, alpha=1, handle=nothing)
    if maxpoolingNanOpt!=0; throw(ArgumentError("CPU pool only supports maxpoolingNanOpt=0")); end
    if alpha != 1; y = y ./ alpha; end
    dx = similar(x)
    (w1,w2) = psize(window, x)
    (p1,p2) = psize(padding, x)
    (s1,s2) = psize(stride, x)
    Wx,Hx,Cx,Nx = size(x);
    Wy,Hy,Cy,Ny = size(y);
    if mode == 0
        if T<:Float32
            ccall((:max_pooling_bwd32,libknet8),Void,
                  (Ptr{T},Ptr{T},Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                  x,y,dy,dx,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
        elseif T<:Float64
            ccall((:max_pooling_bwd64,libknet8),Void,
                  (Ptr{T},Ptr{T},Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                  x,y,dy,dx,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
        else
            error("$T not supported")
        end
    elseif mode == 1 || (mode == 2 && p1==p2==0)
        if T<:Float32
            ccall((:mean_pooling_bwd32,libknet8),Void,
                  (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                  dx,dy,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
        elseif T<:Float64
            ccall((:mean_pooling_bwd64,libknet8),Void,
                  (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                  dx,dy,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
        else
            error("$T not supported")
        end
    else
        throw(ArgumentError("mode $mode not supported by cpu pool"))
    end
    if alpha != 1; scale!(alpha,dx); end
    return dx
end

#@views
function poolx1{T}(x::Array{T,4}, y::Array{T,4}, dy::Array{T,4};
                         window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=0, alpha=1, handle=nothing)
    if maxpoolingNanOpt!=0; throw(ArgumentError("CPU pool only supports maxpoolingNanOpt=0")); end
    stride = psize(stride, x)
    window = psize(window, x)
    padding = psize(padding, x)
    Wx,Hx,Cx,Nx = size(x);
    Wy,Hy,Cy,Ny = size(y);
    if any(padding .> 0)
        Wx += 2*padding[1]
        Hx += 2*padding[2]
        x0 = fill!(similar(x, (Wx,Hx,Cx,Nx)), 0)
        x0[padding[1]+1:end-padding[1], padding[2]+1:end-padding[2],:,:] = x
        x = x0
    end
    dx = fill!(similar(x), 0)
    if mode == 0
        @inbounds for n in 1:Nx, c in 1:Cx, i in 0:stride[1]:Wx-window[1], j in 0:stride[2]:Hx-window[2]
            iy, jy = div(i,stride[1])+1, div(j,stride[2])+1
            a = x[i+1:i+window[1], j+1:j+window[2], c, n]
            m = (a .== maximum(a))
            for im in find(m)
                (di,dj) = ind2sub(m, im)
                dx[i+di,j+dj,c,n] += dy[iy,jy,c,n]
            end
        end
    elseif mode == 1 || (mode == 2 && all(padding .== 0))
        @inbounds for n in 1:Nx, c in 1:Cx, i in 0:stride[1]:Wx-window[1], j in 0:stride[2]:Hx-window[2]
            iy, jy = div(i,stride[1])+1, div(j,stride[2])+1
            dx[i+1:i+window[1],j+1:j+window[2],c,n] += dy[iy,jy,c,n]
        end
        dx = dx ./ prod(window)
    else
        throw(ArgumentError("mode $mode not supported by cpu pool"))
    end
    if any(padding .> 0)
        dx = Array(dx[1+padding[1]:end-padding[1],1+padding[2]:end-padding[2],:,:])
    end
    if alpha!=1
        dx = alpha * dx
    end
    return dx
end

