type Tanh <: Layer; y; Tanh()=new() end
# copy(l::Tanh;o...)=Tanh()

forw(l::Tanh,x; o...)=(for i=1:length(x); x[i]=tanh(x[i]); end; l.y=x)
back(l::Tanh,dy; returndx=true, o...)=(@assert issimilar(dy,l.y); returndx||return; y1=one(eltype(l.y)); for i=1:length(dy); dy[i]=dy[i]*(y1+l.y[i])*(y1-l.y[i]); end; dy)

if GPU
forw(l::Tanh,x::CudaArray; o...)=(l.y=cudnnActivationForward(x; mode=CUDNN_ACTIVATION_TANH))
back(l::Tanh,dy::CudaArray; returndx=true, o...)=(@assert issimilar(dy,l.y); returndx && cudnnActivationBackward(l.y, dy; mode=CUDNN_ACTIVATION_TANH); dy)
end # if GPU
