type Tanh <: Layer; y; Tanh()=new() end
# copy(l::Tanh;o...)=Tanh()

forw(l::Tanh,x; o...)=(l.y = tanhforw(x))
back(l::Tanh,dy; returndx=true, o...)=(returndx && tanhback(l.y, reshape(dy, size(l.y))))

tanhforw(x)=(for i=1:length(x); x[i]=tanh(x[i]); end; x)
tanhback(y,dy)=(y1=one(eltype(y)); for i=1:length(dy); dy[i]=dy[i]*(y1+y[i])*(y1-y[i]); end; dy)

if GPU
forw(l::Tanh,x::AbstractCudaArray; o...)=(l.y=cudnnActivationForward(x; mode=CUDNN_ACTIVATION_TANH))
back(l::Tanh,dy::AbstractCudaArray; returndx=true, o...)=(@assert issimilar(dy,l.y); returndx && cudnnActivationBackward(l.y, dy; mode=CUDNN_ACTIVATION_TANH); dy)
end # if GPU
