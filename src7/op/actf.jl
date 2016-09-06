# Activation function layers: have single input, same size output, can overwrite.

abstract Actf <: Op

ninputs(::Actf)=1
canoverwrite(::Actf)=true
back_reads_x(::Actf)=false
back_reads_y(::Actf)=true

for (ltype,lforw,lback) in 
    ((:Sigm, :sigmforw, :sigmback),
     (:Tanh, :tanhforw, :tanhback),
     (:Relu, :reluforw, :reluback),
     (:Soft, :softforw, :softback),
     (:Soft73, :soft73forw, :soft73back),
     (:Copy, :copyforw, :copyback))
    @eval begin
        type $ltype <: Actf; $ltype(;o...)=new(); end
        forw(l::$ltype, x, y; o...)=($lforw(x,y;o...);gpusync();y)
        back(l::$ltype, dy, dx; y=nothing, o...)=(dx==nothing&&return;$lback(y,dy,dx;o...);gpusync();dx) # need o... for mask etc.
    end
end

function infersize(::Actf,xdims,ydims)
    if xdims==ydims==nothing
        nothing
    elseif xdims==nothing
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
sigmforw(x::Array,y::Array;o...)=(@inbounds for i=1:length(y); y[i]=(1/(1+exp(-x[i]))); end)
sigmback(y::Array,dy::Array,dx::Array;o...)=(@inbounds for i=1:length(dx); dx[i]=dy[i]*y[i]*(1-y[i]); end)
@gpu sigmforw(x::CudaArray,y::CudaArray;o...)=cudnnActivationForward(x,y; mode=CUDNN_ACTIVATION_SIGMOID)
@gpu sigmback(y::CudaArray,dy::CudaArray,dx::CudaArray;o...)=cudnnActivationBackward(y, dy, y, dx; mode=CUDNN_ACTIVATION_SIGMOID)

@doc "@knet function tanh(x) computes the hyperbolic tangent activation function." :tanh
tanhforw(x::Array,y::Array;o...)=(@inbounds for i=1:length(y); y[i]=tanh(x[i]); end)
tanhback(y::Array,dy::Array,dx::Array;o...)=(@inbounds for i=1:length(dx); dx[i]=dy[i]*(1+y[i])*(1-y[i]); end)
@gpu tanhforw(x::CudaArray,y::CudaArray;o...)=cudnnActivationForward(x,y; mode=CUDNN_ACTIVATION_TANH)
@gpu tanhback(y::CudaArray,dy::CudaArray,dx::CudaArray;o...)=cudnnActivationBackward(y, dy, y, dx; mode=CUDNN_ACTIVATION_TANH)

@doc "@knet function relu(x) computes the rectified linear activation function: (x<0 ? 0 : x)" :relu
# This is too slow, use broadcast:  TODO: find out why. same is not true for tanh.
# reluforw(x::Array,y::Array;o...)=(@inbounds for i=1:length(y); y[i]=(x[i]<0 ? 0 : x[i]) end)
# reluforw(x::Array,y::Array;o...)=broadcast!(max,y,x,0)
reluforw(x::Array,y::Array;o...)=(@inbounds for i=1:length(y); if x[i]>0; y[i]=x[i]; else; y[i]=0; end; end)
# reluback(y::Array,dy::Array,dx::Array;o...)=(@inbounds for i=1:length(dx); dx[i]=(y[i]==0 ? 0 : dy[i]) end)
# reluback(y::Array,dy::Array,dx::Array;o...)=(copy!(dx,dy); dx[y.==0]=0)
reluback(y::Array,dy::Array,dx::Array;o...)=(@inbounds for i=1:length(dx); if y[i]==0; dx[i]=0; else; dx[i]=dy[i]; end; end)
@gpu reluforw(x::CudaArray,y::CudaArray;o...)=cudnnActivationForward(x,y; mode=CUDNN_ACTIVATION_RELU)
@gpu reluback(y::CudaArray,dy::CudaArray,dx::CudaArray;o...)=cudnnActivationBackward(y, dy, y, dx; mode=CUDNN_ACTIVATION_RELU)

@doc "@knet function soft(x) computes the softmax activation function: exp(x[i,j])/sum(exp(x[:,j]))" :soft
# z = wx			;; z is the input to the soft layer
# qi = (exp zi) / (Σ exp zj)	;; q is the output of the soft layer
# ∂qi/∂zk = [(i=k)(exp zi)(Σ exp zj) - (exp zi)(exp zk)] / (Σ exp zj)^2
#         = (i=k) qi - qi qk
# ∂J/∂zk = Σ (∂J/∂qi)(∂qi/∂zk)	;; derivative wrt the input z
#        = Σ (1-pi/qi)((i=k) qi - qi qk)
#        = Σ ((i=k) qi - qi qk - (i=k) pi + pi qk)
#        = qk - pk - qk Σ (qi - pi)
#        = qk - pk
#
# Note that softback expects xgrad from softloss, not ygrad.
# See the softloss doc for an explanation.
back_reads_y(::Soft)=false
softback(y,dy,dx;o...)=(dx===dy ? dx : copysync!(dx,dy))
@gpu softforw(x::CudaArray,y::CudaArray;o...)=cudnnSoftmaxForward(x,y)
function softforw(x::Array,y::Array;o...)
    (st,nx) = size2(x)
    @inbounds for j=1:nx
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

# The exception of expecting xgrad instead of ygrad in the soft layer
# is ok if it is the last layer and is used with softloss, which is
# aware of this exception.  However, if soft is to be used internally
# in a model this does not work, the layer will receive ygrad.  For
# now we will define a soft73 layer to prevent code break and
# eventually retire the soft layer in favor of models that do not have
# a final actf and that use what we used to call xentloss.

# For the math let us use x for the input to softmax, y for output, J
# for loss.  The softforw function computes:
# ;; yi = exp(xi)/Σj exp(xj)
# Softback should compute:
# ;; ∂yi/∂xk = [(i=k)(exp xi)(Σ exp xj) - (exp xi)(exp xk)] / (Σ exp xj)^2
# ;; = (i=k) yi - yi yk
# ;; ∂J/∂xk = Σi (∂J/∂yi)(∂yi/∂xk)
# ;; = Σi (∂J/∂yi)((i=k) yk - yi yk)
# ;; = yk ((∂J/∂yk) - Σi yi (∂J/∂yi))

soft73forw(x,y;o...)=softforw(x,y;o...) # forward does not change

@gpu soft73back(y::CudaArray,dy::CudaArray,dx::CudaArray;o...)=(cudnnSoftmaxBackward(y, dy, dx); gpusync(); dx)

function soft73back(y::Array,dy::Array,dx::Array;o...)
    (st,nx) = size2(dy)
    for j=1:nx
        i1=(j-1)*st+1
        i2=j*st
        sumydy = zero(Float64)
        for i=i1:i2; sumydy += y[i] * dy[i]; end
        for i=i1:i2; dx[i] = y[i] * (dy[i] - sumydy); end
    end
    return dx
end


@doc "@knet function axpb(x;a=1,p=1,b=0) computes y=ax^p+b elementwise." :axpb
type Axpb <: Actf; a; p; b; Axpb(;a=1,p=1,b=0,o...)=new(a,p,b); end
back_reads_x(f::Axpb)=(f.p!=1)
back_reads_y(f::Axpb)=false

function forw(f::Axpb, x, y; o...)
    length(x)==length(y) || throw(DimensionMismatch())
    axpbforw(f.a,x,f.p,f.b,y)
    return y
end

function axpbforw{T}(a::Number,x::Array{T},p::Number,b::Number,y::Array{T})
    a = convert(T,a); p = convert(T, p); b = convert(T, b)
    @inbounds for i=1:length(y)
        yi = x[i]
        p!=1 && (yi^=p)
        a!=1 && (yi*=a)
        b!=0 && (yi+=b)
        y[i] = yi
    end
end

@gpu function axpbforw{T}(a::Number,x::CudaArray{T},p::Number,b::Number,y::CudaArray{T})
    T <: Float32 ? ccall((:axpbforw32,libknet),Void,(Cint,Cfloat, Ptr{Cfloat}, Cfloat, Cfloat, Ptr{Cfloat}), length(x), convert(Cfloat,a), x, convert(Cfloat,p), convert(Cfloat,b), y) :
    T <: Float64 ? ccall((:axpbforw64,libknet),Void,(Cint,Cdouble,Ptr{Cdouble},Cdouble,Cdouble,Ptr{Cdouble}),length(x), convert(Cdouble,a),x, convert(Cdouble,p),convert(Cdouble,b),y) :
    error("$T not supported")
    gpusync()
end

function back(f::Axpb, dy, dx; x=nothing, o...)
    dx==nothing && return
    issimilar(dx,dy) || throw(DimensionMismatch())
    if f.p != 1
        issimilar(x,dy) || throw(DimensionMismatch())
        axpbback(f.a,x,f.p,dy,dx)
    elseif f.a != 1             # dJ/dx=dJ/dy*dy/dx=dJ/dy*a
        dx===dy || copysync!(dx,dy)
        scale!(f.a,dx)
    else
        dx===dy || copysync!(dx,dy)
    end
end

function axpbback{T}(a::Number, x::Array{T}, p::Number, dy::Array{T}, dx::Array{T})
    ap = convert(T,a*p)
    p1 = convert(T,p-1)
    @inbounds for i=1:length(dx)
        dx[i] = dy[i] * ap * x[i]^p1
    end
end

@gpu function axpbback{T}(a::Number, x::CudaArray{T}, p::Number, dy::CudaArray{T}, dx::CudaArray{T})
    T <: Float32 ? ccall((:axpbback32,libknet),Void,(Cint,Cfloat, Ptr{Cfloat}, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}),  length(x), convert(Cfloat,a),  x, convert(Cfloat,p),  dy, dx) :
    T <: Float64 ? ccall((:axpbback64,libknet),Void,(Cint,Cdouble,Ptr{Cdouble},Cdouble,Ptr{Cdouble},Ptr{Cdouble}), length(x), convert(Cdouble,a), x, convert(Cdouble,p), dy, dx) :
    error("$T not supported")
    gpusync()
end

# Utility fn, used by rgen
axpb!(x; a=1, p=1, b=0)=(axpbforw(a,x,p,b,x); x)

