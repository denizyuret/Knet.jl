import Base: tanh

# Activation function layers:

abstract Actf <: Op

ninputs(::Actf)=1
overwrites(::Actf)=true
back_reads_x(::Actf)=false
back_reads_y(::Actf)=true

function infersize(::Actf,xdims,ydims)
    if xdims==nothing
        (ydims, ydims)
    elseif ydims==nothing
        (xdims, xdims)
    else
        @assert length(xdims) == length(ydims)
        dims = map(xdims, ydims) do x,y
            x == y ? x :
            x == 0 ? y :
            y == 0 ? x :
            throw(DimensionMismatch())
        end
        (dims, dims)
    end
end


### Common Definitions

for (ltype,lforw,lback,lname) in 
    ((:Sigm, :sigmforw, :sigmback, :sigm),
     (:Tanh, :tanhforw, :tanhback, :tanh),
     (:Relu, :reluforw, :reluback, :relu),
     (:Soft, :softforw, :softback, :soft),
     (:Logp, :logpforw, :logpback, :logp))
    @eval begin
        type $ltype <: Actf; end
        $lname(x,y;o...)=($ltype(),x,y)
        forw(l::$ltype, x, y; o...)=$lforw(x,y;o...)
        back(l::$ltype, dy, dx; y=nothing, o...)=(dx != nothing && $lback(y,dy,dx;o...))
    end
end

### Implementations

@doc "@knet function sigm(x) computes the sigmoid activation function: 1/(1+exp(-x))" sigm
sigmforw(x::Array,y::Array;o...)=(for i=1:length(y); y[i]=(1/(1+exp(-x[i]))); end; y)
sigmback(y::Array,dy::Array,dx::Array;o...)=(for i=1:length(dx); dx[i]=dy[i]*y[i]*(1-y[i]); end; dx)
@gpu sigmforw(x::CudaArray,y::CudaArray;o...)=(cudnnActivationForward(x,y; mode=CUDNN_ACTIVATION_SIGMOID); gpusync(); y)
@gpu sigmback(y::CudaArray,dy::CudaArray,dx::CudaArray;o...)=(cudnnActivationBackward(y, dy, y, dx; mode=CUDNN_ACTIVATION_SIGMOID); gpusync(); dx)

@doc "@knet function tanh(x) computes the hyperbolic tangent activation function." tanh
tanhforw(x::Array,y::Array;o...)=(for i=1:length(y); y[i]=tanh(x[i]); end; y)
tanhback(y::Array,dy::Array,dx::Array;o...)=(for i=1:length(dx); dx[i]=dy[i]*(1+y[i])*(1-y[i]); end; dx)
@gpu tanhforw(x::CudaArray,y::CudaArray;o...)=(cudnnActivationForward(x,y; mode=CUDNN_ACTIVATION_TANH); gpusync(); y)
@gpu tanhback(y::CudaArray,dy::CudaArray,dx::CudaArray;o...)=(cudnnActivationBackward(y, dy, y, dx; mode=CUDNN_ACTIVATION_TANH); gpusync(); dx)

@doc "@knet function relu(x) computes the rectified linear activation function: (x<0 ? 0 : x)" relu
reluforw(x::Array,y::Array;o...)=(for i=1:length(y); y[i]=(x[i]<0 ? 0 : x[i]) end; y)
reluback(y::Array,dy::Array,dx::Array;o...)=(for i=1:length(dx); dx[i]=(y[i]==0 ? 0 : dy[i]) end; dx)
@gpu reluforw(x::CudaArray,y::CudaArray;o...)=(cudnnActivationForward(x,y; mode=CUDNN_ACTIVATION_RELU); gpusync(); y)
@gpu reluback(y::CudaArray,dy::CudaArray,dx::CudaArray;o...)=(cudnnActivationBackward(y, dy, y, dx; mode=CUDNN_ACTIVATION_RELU); gpusync(); dx)

# dy = wx			;; dy is the input to the soft layer
# yi = (exp zi) / (Σ exp zj)	;; y is the output of the soft layer
# ∂yi/∂zk = [(i=k)(exp zi)(Σ exp zj) - (exp zi)(exp zk)] / (Σ exp zj)^2
#         = (i=k) yi - yi yk
# ∂J/∂zk = Σ (∂J/∂yi)(∂yi/∂zk)	;; derivative wrt the input dy
#        = Σ (1-pi/yi)((i=k) yi - yi yk)
#        = Σ ((i=k) yi - yi yk - (i=k) pi + pi yk)
#        = yk - pk - yk Σ (yi - pi)
#        = yk - pk

@doc "@knet function soft(x) computes the softmax activation function: exp(x[i,j])/sum(exp(x[:,j]))" soft
function softforw(x::Array,y::Array;o...)
    (st,nx) = size2(x)
    for j=1:nx
        i1=(j-1)*st+1
        i2=j*st
        xmax = typemin(eltype(x))
        ysum = zero(Float64)
        for i=i1:i2; x[i] > xmax && (xmax = x[i]); end
        for i=i1:i2; ysum += (y[i]=exp(x[i] - xmax)); end
        for i=i1:i2; y[i] /= ysum; end
    end
    return y
end

@gpu softforw(x::CudaArray,y::CudaArray;o...)=(cudnnSoftmaxForward(x,y); gpusync(); y)

function softback(ypred,ygold,dx; mask=nothing, o...) # dx=(ypred-ygold)/ycols
    ycols = ccount(ypred)
    dx===ygold || copy!(dx,ygold)
    scale!(-1/ycols,dx)
    axpy!(1/ycols,ypred,dx)
    mask!=nothing && domask(mask,dx)
    return dx
end

# function softback(y::Array,dy::Array,dx::Array;o...)
#     (st,nx) = size2(dy)
#     for j=1:nx
#         i1=(j-1)*st+1
#         i2=j*st
#         sumydy = zero(Float64)
#         for i=i1:i2; sumydy += y[i] * dy[i]; end
#         for i=i1:i2; dx[i] = y[i] * (dy[i] - sumydy); end
#     end
#     return dx
# end


# @gpu softback(y::CudaArray,dy::CudaArray,dx::CudaArray;o...)=(cudnnSoftmaxBackward(y, dy, dx); gpusync(); dx)

@doc "@knet function logp(x) computes the log softmax activation function: x[i,j])-log(sum(exp(x[:,j])))" logp
function logpforw(x::Array,y::Array;o...)
    (nd,nx) = size2(x)
    for j=1:nx
        i1=(j-1)*nd+1
        i2=j*nd
        xmax = typemin(eltype(x))
        for i=i1:i2; x[i] > xmax && (xmax = x[i]); end
        expy = zero(Float64)
        for i=i1:i2; y[i]=x[i]-xmax; expy += exp(y[i]); end
        logz = log(expy)
        for i=i1:i2; y[i] -= logz; end
    end
    return y
end

logpback(y,dy,dx;o...)=(dx===dy||copy!(dx,dy);dx)

@gpu (logpforw(x::CudaArray{Float32},y::CudaArray{Float32};o...)=
        ((nd,nx) = size2(y);ccall((:logpforw32,libknet),Void,(Cint,Cint,Ptr{Float32},Ptr{Float32}),nd,nx,x,y); gpusync(); y))
@gpu (logpforw(x::CudaArray{Float64},y::CudaArray{Float64};o...)=
        ((nd,nx) = size2(y);ccall((:logpforw64,libknet),Void,(Cint,Cint,Ptr{Float64},Ptr{Float64}),nd,nx,x,y); gpusync(); y))


"@knet function axpb(x;a=1,p=1,b=0) computes y=ax^b+b elementwise."
axpb(x,y;a=1,p=1,b=0)=(Axpb(a,p,b),x,y)

type Axpb <: Actf; a; p; b; end
back_reads_x(::Axpb)=true
back_reads_y(::Axpb)=false

forw(f::Axpb, x, y; o...)=axpb!(x,y; a=f.a,p=f.p,b=f.b)

function back(f::Axpb, dy, dx; x=nothing, o...)
    dx==nothing && return
    x==nothing && error("Need x for axpb back")
    axpb_back!(x, dy, dx; a=f.a,p=f.p)
    return dx
end

function axpb!{T}(x::Array{T}, y::Array{T}=x; a=1,p=1,b=0)
    length(x)==length(y) || throw(DimensionMismatch())
    a=T(a); b=T(b); p=T(p)
    for i=1:length(y); y[i]=a*x[i]^p+b; end
    return y
end

function axpb_back!{T}(x::Array{T}, dy::Array{T}, dx::Array{T}=dy; a=1,p=1)
    length(x)==length(dy)==length(dx) || throw(DimensionMismatch())
    a=T(a); p=T(p)
    for i=1:length(dx); dx[i]=dy[i]*x[i]^(p-1)*a*p; end
    return dx
end

@gpu function axpb!{T}(x::CudaArray{T}, y::CudaArray{T}=x; a=1,p=1,b=0)
    length(x)==length(y) || throw(DimensionMismatch())
    T <: Float32 ? ccall((:axpb32,libknet),Void,(Cint,Cfloat,Ptr{Cfloat},Cfloat,Cfloat,Ptr{Cfloat}), length(x), convert(Cfloat,a), x, convert(Cfloat,p), convert(Cfloat,b), y) :
    T <: Float64 ? ccall((:axpb64,libknet),Void,(Cint,Cdouble,Ptr{Cdouble},Cdouble,Cdouble,Ptr{Cdouble}), length(x), convert(Cdouble,a), x, convert(Cdouble,p), convert(Cdouble,b), y) :
    error("axpb! not defined for $T")
    return y
end

@gpu function axpb_back!{T}(x::CudaArray{T}, dy::CudaArray{T}, dx::CudaArray{T}=dy; a=1,p=1)
    length(x)==length(dy)==length(dx) || throw(DimensionMismatch())
    T <: Float32 ? ccall((:axpb_back32,libknet),Void,(Cint,Cfloat,Ptr{Cfloat},Cfloat,Ptr{Cfloat},Ptr{Cfloat}), length(x), convert(Cfloat,a), x, convert(Cfloat,p), dy, dx) :
    T <: Float64 ? ccall((:axpb_back64,libknet),Void,(Cint,Cdouble,Ptr{Cdouble},Cdouble,Ptr{Cdouble},Ptr{Cdouble}), length(x), convert(Cdouble,a), x, convert(Cdouble,p), dy, dx) :
    error("axpb! not defined for $T")
    return dx
end


### DEAD CODE

        # $lforw(x::KUdense, y::KUdense=x)=($lforw(x.arr,y.arr);y)
        # $lback(y::KUdense, dy::KUdense, dx::KUdense=dy;o...)=($lback(y.arr, dy.arr, dx.arr);dx)
# params(::Actf)=Any[]
# ysize(::Actf,x)=size(x)
# overwrites(::Actf)=true
