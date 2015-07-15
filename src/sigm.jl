type Sigm <: Layer; y; Sigm()=new() end
# copy(l::Sigm; o...)=Sigm()

forw(l::Sigm,x; o...)=(l.y = sigmforw(x))
back(l::Sigm,dy; returndx=true, o...)=(returndx && sigmback(l.y, reshape(dy, size(l.y))))

sigmforw(x)=(x1=one(eltype(x)); for i=1:length(x); x[i]=(x1/(x1+exp(-x[i]))); end; x)
sigmback(y,dy)=(y1=one(eltype(y)); for i=1:length(dy); dy[i]*=y[i]*(y1-y[i]); end; dy)

if GPU
forw(l::Sigm,x::AbstractCudaArray; o...)=(l.y=cudnnActivationForward(x; mode=CUDNN_ACTIVATION_SIGMOID))
back(l::Sigm,dy::AbstractCudaArray; returndx=true, o...)=(@assert issimilar(dy, l.y); returndx||return; cudnnActivationBackward(l.y, dy; mode=CUDNN_ACTIVATION_SIGMOID); dy)
end # if GPU
