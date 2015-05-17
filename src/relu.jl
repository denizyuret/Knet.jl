type Relu <: Layer; y; Relu()=new() end
copy(l::Relu; o...)=Relu()

function forw(l::Relu,x; o...)
    x0=zero(eltype(x)); for i=1:length(x); (x[i]<x0)&&(x[i]=x0) end; l.y=x
end

function back(l::Relu,dy; returndx=true, o...)
    @assert issimilar(dy, l.y)
    returndx||return
    y0=zero(eltype(dy))
    for i=1:length(dy); (l.y[i]==y0)&&(dy[i]=y0) end
    return dy
end

if GPU
forw(l::Relu,x::CudaArray; o...)=(l.y=cudnnActivationForward(x; mode=CUDNN_ACTIVATION_RELU))
back(l::Relu,dy::CudaArray; returndx=true, o...)=(@assert issimilar(dy, l.y); returndx && cudnnActivationBackward(l.y, dy; mode=CUDNN_ACTIVATION_RELU); dy)
end # if GPU
