# Activation function layers:

abstract Actf <: Op

ninputs(::Actf)=1
infersize(::Actf,dims)=(dims==nothing ? nothing : (dims,dims))
back_reads_x(::Actf)=false
back_reads_y(::Actf)=true

### Common Definitions

for (ltype,lforw,lback,lname) in 
    ((:Sigm, :sigmforw, :sigmback, :sigm),
     (:Tanh, :tanhforw, :tanhback, :tanh),
     (:Relu, :reluforw, :reluback, :relu),
     (:Soft, :softforw, :softback, :soft),
     (:Logp, :logpforw, :logpback, :logp))
    @eval begin
        type $ltype <: Actf; end
        $lname()=$ltype()
        forw(l::$ltype, x, y; o...)=(issimilar(x,y)||error("x/y"); $lforw(x,y))
        back(l::$ltype, dy, dx; y=nothing, o...)=(dx != nothing && $lback(y,dy,dx))
    end
end

### Implementations

sigmforw(x::Array,y::Array)=(for i=1:length(y); y[i]=(1/(1+exp(-x[i]))); end; y)
sigmback(y::Array,dy::Array,dx::Array)=(for i=1:length(dx); dx[i]=dy[i]*y[i]*(1-y[i]); end; dx)
@gpu (sigmforw(x::CudaArray,y::CudaArray)=cudnnActivationForward(x,y; mode=CUDNN_ACTIVATION_SIGMOID))
@gpu (sigmback(y::CudaArray,dy::CudaArray,dx::CudaArray)=cudnnActivationBackward(y, dy, y, dx; mode=CUDNN_ACTIVATION_SIGMOID))

tanhforw(x::Array,y::Array)=(for i=1:length(y); y[i]=tanh(x[i]); end; y)
tanhback(y::Array,dy::Array,dx::Array)=(for i=1:length(dx); dx[i]=dy[i]*(1+y[i])*(1-y[i]); end; dx)
@gpu (tanhforw(x::CudaArray,y::CudaArray)=cudnnActivationForward(x,y; mode=CUDNN_ACTIVATION_TANH))
@gpu (tanhback(y::CudaArray,dy::CudaArray,dx::CudaArray)=cudnnActivationBackward(y, dy, y, dx; mode=CUDNN_ACTIVATION_TANH))

reluforw(x::Array,y::Array)=(for i=1:length(y); y[i]=(x[i]<0 ? 0 : x[i]) end; y)
reluback(y::Array,dy::Array,dx::Array)=(for i=1:length(dx); dx[i]=(y[i]==0 ? 0 : dy[i]) end; dx)
@gpu (reluforw(x::CudaArray,y::CudaArray)=cudnnActivationForward(x,y; mode=CUDNN_ACTIVATION_RELU))
@gpu (reluback(y::CudaArray,dy::CudaArray,dx::CudaArray)=cudnnActivationBackward(y, dy, y, dx; mode=CUDNN_ACTIVATION_RELU))

function softforw(x::Array,y::Array)
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

function softback(y::Array,dy::Array,dx::Array)
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


# TODO: what happened to the buggy 0.5 factor?
@gpu (softforw(x::CudaArray,y::CudaArray)=cudnnSoftmaxForward(x,y))
@gpu (softback(y::CudaArray,dy::CudaArray,dx::CudaArray)=cudnnSoftmaxBackward(y, dy, dx))

function logpforw(x::Array,y::Array)
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

logpback(y,dy,dx)=(dx===dy||copy!(dx,dy);dx)

@gpu (logpforw(x::CudaArray{Float32},y::CudaArray{Float32})=
        ((nd,nx) = size2(y);ccall((:logpforw32,libkunet),Void,(Cint,Cint,Ptr{Float32},Ptr{Float32}),nd,nx,x,y); y))
@gpu (logpforw(x::CudaArray{Float64},y::CudaArray{Float64})=
        ((nd,nx) = size2(y);ccall((:logpforw64,libkunet),Void,(Cint,Cint,Ptr{Float64},Ptr{Float64}),nd,nx,x,y); y))


### DEAD CODE

        # $lforw(x::KUdense, y::KUdense=x)=($lforw(x.arr,y.arr);y)
        # $lback(y::KUdense, dy::KUdense, dx::KUdense=dy)=($lback(y.arr, dy.arr, dx.arr);dx)
# params(::Actf)=Any[]
# ysize(::Actf,x)=size(x)
# overwrites(::Actf)=true
