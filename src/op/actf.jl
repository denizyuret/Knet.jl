# Activation function layers:

abstract Actf <: Op

ninputs(::Actf)=1
canoverwrite(::Actf)=true
back_reads_x(::Actf)=false
back_reads_y(::Actf)=true

for (ltype,lforw,lback,lname) in 
    ((:Sigm, :sigmforw, :sigmback, :sigm),
     (:Tanh, :tanhforw, :tanhback, :tanh),
     (:Relu, :reluforw, :reluback, :relu),
     (:Soft, :softforw, :softback, :soft),
     (:Logp, :logpforw, :logpback, :logp),
     (:Copy, :copyforw, :copyback, :copy))
    @eval begin
        type $ltype <: Actf; $ltype(;o...)=new(); end
        forw(l::$ltype, x, y; o...)=($lforw(x,y;o...);gpusync();y)
        back(l::$ltype, dy, dx; y=nothing, o...)=(dx==nothing&&return;$lback(y,dy,dx;o...);gpusync();dx) # need o... for mask etc.
    end
    Kenv.kdef(lname,eval(ltype))
end

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

### Implementations

@doc "@knet function copy(x) copies its input to its output" :copy
back_reads_y(::Copy)=false
copyforw(x,y;o...)=(x===y ? y : copysync!(y,x))
copyback(y,dy,dx;o...)=(dx===dy ? dx : copysync!(dx,dy))

@doc "@knet function sigm(x) computes the sigmoid activation function: 1/(1+exp(-x))" :sigm
sigmforw(x::Array,y::Array;o...)=(for i=1:length(y); y[i]=(1/(1+exp(-x[i]))); end)
sigmback(y::Array,dy::Array,dx::Array;o...)=(for i=1:length(dx); dx[i]=dy[i]*y[i]*(1-y[i]); end)
@gpu sigmforw(x::CudaArray,y::CudaArray;o...)=cudnnActivationForward(x,y; mode=CUDNN_ACTIVATION_SIGMOID)
@gpu sigmback(y::CudaArray,dy::CudaArray,dx::CudaArray;o...)=cudnnActivationBackward(y, dy, y, dx; mode=CUDNN_ACTIVATION_SIGMOID)

@doc "@knet function tanh(x) computes the hyperbolic tangent activation function." :tanh
tanhforw(x::Array,y::Array;o...)=(for i=1:length(y); y[i]=tanh(x[i]); end)
tanhback(y::Array,dy::Array,dx::Array;o...)=(for i=1:length(dx); dx[i]=dy[i]*(1+y[i])*(1-y[i]); end)
@gpu tanhforw(x::CudaArray,y::CudaArray;o...)=cudnnActivationForward(x,y; mode=CUDNN_ACTIVATION_TANH)
@gpu tanhback(y::CudaArray,dy::CudaArray,dx::CudaArray;o...)=cudnnActivationBackward(y, dy, y, dx; mode=CUDNN_ACTIVATION_TANH)

@doc "@knet function relu(x) computes the rectified linear activation function: (x<0 ? 0 : x)" :relu
reluforw(x::Array,y::Array;o...)=(for i=1:length(y); y[i]=(x[i]<0 ? 0 : x[i]) end)
reluback(y::Array,dy::Array,dx::Array;o...)=(for i=1:length(dx); dx[i]=(y[i]==0 ? 0 : dy[i]) end)
@gpu reluforw(x::CudaArray,y::CudaArray;o...)=cudnnActivationForward(x,y; mode=CUDNN_ACTIVATION_RELU)
@gpu reluback(y::CudaArray,dy::CudaArray,dx::CudaArray;o...)=cudnnActivationBackward(y, dy, y, dx; mode=CUDNN_ACTIVATION_RELU)

# z = wx			;; z is the input to the soft layer
# qi = (exp zi) / (Σ exp zj)	;; q is the output of the soft layer
# ∂qi/∂zk = [(i=k)(exp zi)(Σ exp zj) - (exp zi)(exp zk)] / (Σ exp zj)^2
#         = (i=k) qi - qi qk
# ∂J/∂zk = Σ (∂J/∂qi)(∂qi/∂zk)	;; derivative wrt the input z
#        = Σ (1-pi/qi)((i=k) qi - qi qk)
#        = Σ ((i=k) qi - qi qk - (i=k) pi + pi qk)
#        = qk - pk - qk Σ (qi - pi)
#        = qk - pk

@doc "@knet function soft(x) computes the softmax activation function: exp(x[i,j])/sum(exp(x[:,j]))" :soft
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

@gpu softforw(x::CudaArray,y::CudaArray;o...)=cudnnSoftmaxForward(x,y)

# Note that softback expects ygold from softloss, not ygrad!
# See the softloss doc for an explanation.
function softback(ypred,ygold,dx; mask=nothing, o...) # dx=(ypred-ygold)/ycols
    ycols = ccount(ypred)
    dx===ygold || copysync!(dx,ygold)
    scale!(-1/ycols,dx)
    axpy!(1/ycols,ypred,dx)
    mask!=nothing && domask(mask,dx)
    gpusync()
    return dx
end

@doc "@knet function logp(x) computes the log softmax activation function: x[i,j])-log(sum(exp(x[:,j])))" :logp
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
end

logpback(y,dy,dx;o...)=(dx===dy||copysync!(dx,dy))

@gpu logpforw(x::CudaArray{Float32},y::CudaArray{Float32};o...)=((nd,nx) = size2(y);ccall((:logpforw32,libknet),Void,(Cint,Cint,Ptr{Float32},Ptr{Float32}),nd,nx,x,y))
@gpu logpforw(x::CudaArray{Float64},y::CudaArray{Float64};o...)=((nd,nx) = size2(y);ccall((:logpforw64,libknet),Void,(Cint,Cint,Ptr{Float64},Ptr{Float64}),nd,nx,x,y))

@doc "@knet function axpb(x;a=1,p=1,b=0) computes y=ax^b+b elementwise." :axpb

# axpb(x,y;a=1,p=1,b=0)=(Axpb(a,p,b),x,y)

type Axpb <: Actf; a; p; b; Axpb(;a=1,p=1,b=0,o...)=new(a,p,b); end
Kenv.kdef(:axpb,Axpb) # TODO: compiler should recognize arithmetic expr for axpb

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
    gpusync()
    return y
end

@gpu function axpb_back!{T}(x::CudaArray{T}, dy::CudaArray{T}, dx::CudaArray{T}=dy; a=1,p=1)
    length(x)==length(dy)==length(dx) || throw(DimensionMismatch())
    T <: Float32 ? ccall((:axpb_back32,libknet),Void,(Cint,Cfloat,Ptr{Cfloat},Cfloat,Ptr{Cfloat},Ptr{Cfloat}), length(x), convert(Cfloat,a), x, convert(Cfloat,p), dy, dx) :
    T <: Float64 ? ccall((:axpb_back64,libknet),Void,(Cint,Cdouble,Ptr{Cdouble},Cdouble,Ptr{Cdouble},Ptr{Cdouble}), length(x), convert(Cdouble,a), x, convert(Cdouble,p), dy, dx) :
    error("axpb! not defined for $T")
    gpusync()
    return dx
end


### DEAD CODE

        # $lforw(x::KUdense, y::KUdense=x)=($lforw(x.arr,y.arr);y)
        # $lback(y::KUdense, dy::KUdense, dx::KUdense=dy)=($lback(y.arr, dy.arr, dx.arr);dx)
# params(::Actf)=Any[]
# ysize(::Actf,x)=size(x)
# overwrites(::Actf)=true

        # type $ltype <: Actf; $ltype(;o...)=new(); end
        # $lname(x,y;o...)=($ltype(),x,y)
