# Activation function layers:

### Common Definitions

for (ltype,lforw,lback) in ((:Sigm, :sigmforw, :sigmback),
                            (:Tanh, :tanhforw, :tanhback),
                            (:Relu, :reluforw, :reluback),
                            (:Soft, :softforw, :softback),
                            (:Logp, :logpforw, :logpback))
    @eval begin
        type $ltype <: Layer; y; $ltype()=new(); end
        forw(l::$ltype, y; o...)=($lforw(y); l.y=y)
        back(l::$ltype, dy; returndx=true, o...)=(@assert issimilar(dy,l.y); returndx && ($lback(l.y, dy); dy))
        $lforw(y::KUdense)=$lforw(y.arr)
        $lback(y::KUdense, dy::KUdense)=$lback(y.arr, dy.arr)
    end
end

### Implementations

sigmforw(y::Array)=(y1=one(eltype(y)); for i=1:length(y); y[i]=(y1/(y1+exp(-y[i]))); end)
sigmback(y::Array,dy::Array)=(y1=one(eltype(y)); for i=1:length(dy); dy[i]*=y[i]*(y1-y[i]); end)
GPU && (sigmforw(y::CudaArray)=cudnnActivationForward(y; mode=CUDNN_ACTIVATION_SIGMOID))
GPU && (sigmback(y::CudaArray,dy::CudaArray)=cudnnActivationBackward(y, dy; mode=CUDNN_ACTIVATION_SIGMOID))

tanhforw(y::Array)=(for i=1:length(y); y[i]=tanh(y[i]); end)
tanhback(y::Array,dy::Array)=(y1=one(eltype(y)); for i=1:length(dy); dy[i]=dy[i]*(y1+y[i])*(y1-y[i]); end)
GPU && (tanhforw(y::CudaArray)=cudnnActivationForward(y; mode=CUDNN_ACTIVATION_TANH))
GPU && (tanhback(y::CudaArray,dy::CudaArray)=cudnnActivationBackward(y, dy; mode=CUDNN_ACTIVATION_TANH))

reluforw(y::Array)=(y0=zero(eltype(y)); for i=1:length(y); (y[i]<y0)&&(y[i]=y0) end)
reluback(y::Array, dy::Array)=(y0=zero(eltype(dy)); for i=1:length(dy); (y[i]==y0)&&(dy[i]=y0) end)
GPU && (reluforw(y::CudaArray)=cudnnActivationForward(y; mode=CUDNN_ACTIVATION_RELU))
GPU && (reluback(y::CudaArray,dy::CudaArray)=cudnnActivationBackward(y,dy; mode=CUDNN_ACTIVATION_RELU))

function softforw(y::Array)
    (st,nx) = size2(y)
    for j=1:nx
        i1=(j-1)*st+1
        i2=j*st
        ymax = typemin(eltype(y))
        ysum = zero(Float64)
        for i=i1:i2; y[i] > ymax && (ymax = y[i]); end
        for i=i1:i2; ysum += (y[i]=exp(y[i] - ymax)); end
        for i=i1:i2; y[i] /= ysum; end
    end
end

function softback(y::Array,dy::Array)
    (st,nx) = size2(dy)
    for j=1:nx
        i1=(j-1)*st+1
        i2=j*st
        sumydy = zero(Float64)
        for i=i1:i2; sumydy += y[i] * dy[i]; end
        for i=i1:i2; dy[i] = y[i] * (dy[i] - sumydy); end
    end
end


# TODO: what happened to the buggy 0.5 factor?
GPU && (softforw(y::CudaArray)=cudnnSoftmaxForward(y))
GPU && (softback(y::CudaArray,dy::CudaArray)=cudnnSoftmaxBackward(y, dy))

function logpforw(y::Array)
    (nd,nx) = size2(y)
    for j=1:nx
        i1=(j-1)*nd+1
        i2=j*nd
        ymax = typemin(eltype(y))
        for i=i1:i2; y[i] > ymax && (ymax = y[i]); end
        z = zero(Float64)
        for i=i1:i2; z += exp(y[i] -= ymax); end
        logz = log(z)
        for i=i1:i2; y[i] -= logz; end
    end
end

logpback(y,dy)=dy

GPU && (logpforw(y::CudaArray{Float32})=((nd,nx) = size2(y);ccall((:logpforw32,libknet),Void,(Cint,Cint,Ptr{Float32}),nd,nx,y)))
GPU && (logpforw(y::CudaArray{Float64})=((nd,nx) = size2(y);ccall((:logpforw64,libknet),Void,(Cint,Cint,Ptr{Float64}),nd,nx,y)))
