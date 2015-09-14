# Activation function layers:

abstract ActfLayer <: Op

overwrites(l::ActfLayer)=true
back_reads_x(l::ActfLayer)=false
back_reads_y(l::ActfLayer)=true

### Common Definitions

for (ltype,lforw,lback) in ((:Sigm, :sigmforw, :sigmback),
                            (:Tanh, :tanhforw, :tanhback),
                            (:Relu, :reluforw, :reluback),
                            (:Soft, :softforw, :softback),
                            (:Logp, :logpforw, :logpback))
    @eval begin
        type $ltype <: ActfLayer; y; dx; $ltype()=new(); end
        $lforw(x::KUdense, y::KUdense=x)=($lforw(x.arr,y.arr);y)
        $lback(y::KUdense, dy::KUdense, dx::KUdense=dy)=($lback(y.arr, dy.arr, dx.arr);dx)
        forw(l::$ltype, x; y=x, o...)=(issimilar(x,y)||error("x/y"); l.y = $lforw(x,y))
        back(l::$ltype, dy; dx=dy, y=l.y, returndx=true, o...)=
            (returndx||return; (issimilar(dy,y) && issimilar(dx,y))||error("Mismatch"); $lback(y,dy,dx))
    end
end

### Implementations

sigmforw(x::Array,y::Array=x)=(for i=1:length(y); y[i]=(1/(1+exp(-x[i]))); end; y)
sigmback(y::Array,dy::Array,dx::Array=dy)=(for i=1:length(dx); dx[i]=dy[i]*y[i]*(1-y[i]); end; dx)
GPU && (sigmforw(x::CudaArray,y::CudaArray=x)=cudnnActivationForward(x,y; mode=CUDNN_ACTIVATION_SIGMOID))
GPU && (sigmback(y::CudaArray,dy::CudaArray,dx::CudaArray=dy)=cudnnActivationBackward(y, dy, y, dx; mode=CUDNN_ACTIVATION_SIGMOID))

tanhforw(x::Array,y::Array=x)=(for i=1:length(y); y[i]=tanh(x[i]); end; y)
tanhback(y::Array,dy::Array,dx::Array=dy)=(for i=1:length(dx); dx[i]=dy[i]*(1+y[i])*(1-y[i]); end; dx)
GPU && (tanhforw(x::CudaArray,y::CudaArray=x)=cudnnActivationForward(x,y; mode=CUDNN_ACTIVATION_TANH))
GPU && (tanhback(y::CudaArray,dy::CudaArray,dx::CudaArray=dy)=cudnnActivationBackward(y, dy, y, dx; mode=CUDNN_ACTIVATION_TANH))

reluforw(x::Array,y::Array=x)=(for i=1:length(y); y[i]=(x[i]<0 ? 0 : x[i]) end; y)
reluback(y::Array,dy::Array,dx::Array=dy)=(for i=1:length(dx); dx[i]=(y[i]==0 ? 0 : dy[i]) end; dx)
GPU && (reluforw(x::CudaArray,y::CudaArray=x)=cudnnActivationForward(x,y; mode=CUDNN_ACTIVATION_RELU))
GPU && (reluback(y::CudaArray,dy::CudaArray,dx::CudaArray=dy)=cudnnActivationBackward(y, dy, y, dx; mode=CUDNN_ACTIVATION_RELU))

function softforw(x::Array,y::Array=x)
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

function softback(y::Array,dy::Array,dx::Array=dy)
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
GPU && (softforw(x::CudaArray,y::CudaArray=x)=cudnnSoftmaxForward(x,y))
GPU && (softback(y::CudaArray,dy::CudaArray,dx::CudaArray=dy)=cudnnSoftmaxBackward(y, dy, dx))

function logpforw(x::Array,y::Array=x)
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

logpback(y,dy,dx=dy)=(dx===dy||copy!(dx,dy);dx)

GPU && (logpforw(x::CudaArray{Float32},y::CudaArray{Float32}=x)=
        ((nd,nx) = size2(y);ccall((:logpforw32,libkunet),Void,(Cint,Cint,Ptr{Float32},Ptr{Float32}),nd,nx,x,y); y))
GPU && (logpforw(x::CudaArray{Float64},y::CudaArray{Float64}=x)=
        ((nd,nx) = size2(y);ccall((:logpforw64,libkunet),Void,(Cint,Cint,Ptr{Float64},Ptr{Float64}),nd,nx,x,y); y))
