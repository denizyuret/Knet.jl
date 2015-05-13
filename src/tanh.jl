type Tanh <: Layer; y; Tanh()=new() end

# TODO: implement cpu version
# forw(l::Tanh,x; o...)=(for i=1:length(x); (x[i]<zero(x[i]))&&(x[i]=zero(x[i])) end; l.y=x)
# back(l::Tanh,dy; o...)=(for i=1:length(dy); (l.y[i]==zero(l.y[i]))&&(dy[i]=zero(dy[i])) end; dy)

forw(l::Tanh,x::CudaArray; o...)=(l.y=cudnnActivationForward(x; mode=CUDNN_ACTIVATION_TANH))
back(l::Tanh,dy::CudaArray; o...)=cudnnActivationBackward(l.y, dy; mode=CUDNN_ACTIVATION_TANH)
