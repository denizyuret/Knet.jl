# Define some new primitives: conv4 and pool

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
    @cuda(cudnn,cudnnConvolutionBackwardData,
          (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
          handle,Ref(alpha),FD(w),w,TD(dy),dy,CD(w,x;o...),algo,workSpace,workSpaceSizeInBytes,Ref(beta),TD(dx),dx)
    return dx
end

function conv4w{T}(w::KnetArray{T},x::KnetArray{T},dy::KnetArray{T};
                   handle=cudnnhandle, alpha=one(T), beta=zero(T),
                   algo=0, workSpace=C_NULL, workSpaceSizeInBytes=0, o...)
    dw = similar(w)
    @cuda(cudnn,cudnnConvolutionBackwardFilter,
          (Cptr,Ptr{T},Cptr,Ptr{T},Cptr,Ptr{T},Cptr,     UInt32,Cptr,     Csize_t,             Ptr{T},Cptr,Ptr{T}),
          handle,Ref(alpha),TD(x),x,TD(dy),dy,CD(w,x;o...),algo,workSpace,workSpaceSizeInBytes,Ref(beta),FD(dw),dw)
    return dw
end


@primitive conv4(w,x; o...),dy  conv4w(w,x,dy;o...)  conv4x(w,x,dy;o...)
@zerograd  conv4x(w,x,dy;o...)
@zerograd  conv4w(w,x,dy;o...)

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
        @cuda(cudnn,cudnnSetFilterNdDescriptor,
              (Cptr,UInt32,UInt32,Cint,Ptr{Cint}),
              d[1], DT(a), 0,     n,   sz)
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
        @cuda(cudnn,cudnnSetConvolutionNdDescriptor,
              (Cptr,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},UInt32,UInt32),
              d[1],nd,cdsize(padding,nd),cdsize(stride,nd),cdsize(upscale,nd),mode,DT(x))
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
        @cuda(cudnn,cudnnSetPoolingNdDescriptor,
              (Cptr,UInt32,UInt32,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),
              d[1],mode,maxpoolingNanOpt,nd,cdsize(window,nd),cdsize(padding,nd),cdsize(stride,nd))
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
        throw(DimensionMismatch())
    end
end

DT(::KnetArray{Float32})=UInt32(0)
DT(::KnetArray{Float64})=UInt32(1)
DT(::KnetArray{Float16})=UInt32(2)

function cdims{T,N}(w::KnetArray{T,N},x::KnetArray{T,N}; padding=0, stride=1, o...)
    ntuple(N) do i
        if i < N-1
            pi = if isa(padding,Number); padding; else padding[i]; end
            si = if isa(stride,Number); stride; else stride[i]; end
            1 + div(size(x,i) - size(w,i) + 2*pi, si)
        elseif i == N-1
            size(w,N)
        else # i == N
            size(x,N)
        end
    end
end

function pdims{T,N}(x::KnetArray{T,N}; window=2, padding=0, stride=window, o...)
    isa(window,Integer) && (window=fill(window,N-2))
    isa(padding,Integer) && (padding=fill(padding,N-2))
    isa(stride, Integer) && (stride=fill(stride,N-2))
    ntuple(N) do i
        if i < N-1
            wi = if isa(window,Number); window; else window[i]; end
            pi = if isa(padding,Number); padding; else padding[i]; end
            si = if isa(stride,Number); stride; else stride[i]; end
            1 + ceil(Int, (size(x,i) + 2*pi - wi) / si)
        else
            size(x,i)
        end
    end
end

"Convenience function to convert multidimensional arrays to matrices."
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

