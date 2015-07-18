type Relu <: Layer; y; Relu()=new() end

forw(l::Relu,y::KUdense; o...)=(reluforw(y.arr); l.y=y)
back(l::Relu,dy::KUdense; returndx=true, o...)=(@assert issimilar(dy,l.y); returndx && (reluback(l.y.arr, dy.arr); dy))

reluforw(y::Array)=(y0=zero(eltype(y)); for i=1:length(y); (y[i]<y0)&&(y[i]=y0) end)
reluback(y::Array, dy::Array)=(y0=zero(eltype(dy)); for i=1:length(dy); (y[i]==y0)&&(dy[i]=y0) end)

if GPU
reluforw(y::CudaArray)=cudnnActivationForward(y; mode=CUDNN_ACTIVATION_RELU)
reluback(y::CudaArray,dy::CudaArray)=cudnnActivationBackward(y,dy; mode=CUDNN_ACTIVATION_RELU)
end # if GPU
