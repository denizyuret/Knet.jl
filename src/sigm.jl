type Sigm <: Layer; y; Sigm()=new() end

forw(l::Sigm,x; o...)=(x1=one(eltype(x)); for i=1:length(x); x[i]=(x1/(x1+exp(-x[i]))); end; l.y=x)
back(l::Sigm,dy; dx=true, o...)=(dx||return; y1=one(eltype(l.y)); for i=1:length(dy); dy[i]*=l.y[i]*(y1-l.y[i]); end; dy)

if GPU
forw(l::Sigm,x::CudaArray; o...)=(l.y=cudnnActivationForward(x; mode=CUDNN_ACTIVATION_SIGMOID))
back(l::Sigm,dy::CudaArray; dx=true, o...)=(dx && cudnnActivationBackward(l.y, dy; mode=CUDNN_ACTIVATION_SIGMOID))
end # if GPU
