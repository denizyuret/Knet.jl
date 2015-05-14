type Relu <: Layer; y; Relu()=new() end

forw(l::Relu,x; o...)=(x0=zero(eltype(x)); for i=1:length(x); (x[i]<x0)&&(x[i]=x0) end; l.y=x)
back(l::Relu,dy; dx=true, o...)=(dx||return; y0=zero(eltype(dy)); for i=1:length(dy); (l.y[i]==y0)&&(dy[i]=y0) end; dy)

if GPU

forw(l::Relu,x::CudaArray; o...)=(l.y=cudnnActivationForward(x; mode=CUDNN_ACTIVATION_RELU))
back(l::Relu,dy::CudaArray; dx=true, o...)=(dx && cudnnActivationBackward(l.y, dy; mode=CUDNN_ACTIVATION_RELU))

#forw(l::Relu,x::CudaArray; o...)=(ccall((:reluforw,libkunet),Void,(Cint,Cmat),length(x),x); l.y=x)
#back(l::Relu,dy::CudaArray; o...)=(ccall((:reluback,libkunet),Void,(Cint,Cmat,Cmat),length(dy),l.y,dy); dy)

end # if GPU
