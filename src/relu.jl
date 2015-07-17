type Relu <: Layer; y; Relu()=new() end

forw(l::Relu,x; o...)=(l.y = reluforw(x))
back(l::Relu,dy; returndx=true, o...)=(returndx && reluback(l.y, dy)) # reshape(dy, size(l.y))))

reluforw(x::Array)=(x0=zero(eltype(x)); for i=1:length(x); (x[i]<x0)&&(x[i]=x0) end; x)
reluback(y::Array, dy::Array)=(y0=zero(eltype(dy)); for i=1:length(dy); (y[i]==y0)&&(dy[i]=y0) end; dy)

reluforw(x::KUdense{Array})=(reluforw(x.arr); x)
reluback(y::KUdense{Array}, dy::KUdense{Array})=(reluback(y.arr, dy.arr); dy)

if GPU
forw(l::Relu,x::KUdense{CudaArray}; o...)=(cudnnActivationForward(x.arr; mode=CUDNN_ACTIVATION_RELU); l.y=x)
back(l::Relu,dy::KUdense{CudaArray}; returndx=true, o...)=(@assert issimilar(dy, l.y); returndx && cudnnActivationBackward(l.y.arr, dy.arr; mode=CUDNN_ACTIVATION_RELU); dy)
end # if GPU
