type Tanh <: Layer; y; Tanh()=new() end

forw(l::Tanh,y::KUdense; o...)=(tanhforw(y.arr); l.y=y)
back(l::Tanh,dy::KUdense; returndx=true, o...)=(@assert issimilar(dy,l.y); returndx && (tanhback(l.y.arr, dy.arr); dy))

tanhforw(y::Array)=(for i=1:length(y); y[i]=tanh(y[i]); end)
tanhback(y::Array,dy::Array)=(y1=one(eltype(y)); for i=1:length(dy); dy[i]=dy[i]*(y1+y[i])*(y1-y[i]); end)

if GPU
tanhforw(y::CudaArray)=cudnnActivationForward(y; mode=CUDNN_ACTIVATION_TANH)
tanhback(y::CudaArray,dy::CudaArray)=cudnnActivationBackward(y, dy; mode=CUDNN_ACTIVATION_TANH)
end # if GPU
