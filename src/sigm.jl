type Sigm <: Layer; y; Sigm()=new() end

forw(l::Sigm,y::KUdense; o...)=(sigmforw(y.arr); l.y=y)
back(l::Sigm,dy::KUdense; returndx=true, o...)=(@assert issimilar(dy,l.y); returndx && (sigmback(l.y.arr, dy.arr); dy))

sigmforw(y::Array)=(y1=one(eltype(y)); for i=1:length(y); y[i]=(y1/(y1+exp(-y[i]))); end)
sigmback(y::Array,dy::Array)=(y1=one(eltype(y)); for i=1:length(dy); dy[i]*=y[i]*(y1-y[i]); end)

if GPU
sigmforw(y::CudaArray)=cudnnActivationForward(y; mode=CUDNN_ACTIVATION_SIGMOID)
sigmback(y::CudaArray,dy::CudaArray)=cudnnActivationBackward(y, dy; mode=CUDNN_ACTIVATION_SIGMOID)
end # if GPU
