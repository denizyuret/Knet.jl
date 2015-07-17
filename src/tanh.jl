type Tanh <: Layer; y; Tanh()=new() end

forw(l::Tanh,x; o...)=(l.y = tanhforw(x))
back(l::Tanh,dy; returndx=true, o...)=(returndx && tanhback(l.y, reshape(dy, size(l.y))))

tanhforw(x)=(for i=1:length(x); x[i]=tanh(x[i]); end; x)
tanhback(y,dy)=(y1=one(eltype(y)); for i=1:length(dy); dy[i]=dy[i]*(y1+y[i])*(y1-y[i]); end; dy)

if GPU
forw(l::Tanh,x::KUdense{CudaArray}; o...)=(cudnnActivationForward(x.arr; mode=CUDNN_ACTIVATION_TANH); l.y=x)
back(l::Tanh,dy::KUdense{CudaArray}; returndx=true, o...)=(@assert issimilar(dy,l.y); returndx && cudnnActivationBackward(l.y.arr, dy.arr; mode=CUDNN_ACTIVATION_TANH); dy)
end # if GPU
