type Sigm <: Layer; y; Sigm()=new() end
copy(l::Sigm; o...)=Sigm()

forw(l::Sigm,x; o...)=(l.y = sigmforw(x))
back(l::Sigm,dy; returndx=true, o...)=(returndx && sigmback(l.y, reshape(dy, size(l.y))))

sigmforw(x)=(x1=one(eltype(x)); for i=1:length(x); x[i]=(x1/(x1+exp(-x[i]))); end; x)
sigmback(y,dy)=(y1=one(eltype(y)); for i=1:length(dy); dy[i]*=y[i]*(y1-y[i]); end; dy)

if GPU
sigmforw(x::CudaArray)=(cudnnActivationForward(x; mode=CUDNN_ACTIVATION_SIGMOID); x)
sigmback(y::CudaArray, dy::CudaArray)=(cudnnActivationBackward(y, dy; mode=CUDNN_ACTIVATION_SIGMOID); dy)
end # if GPU
