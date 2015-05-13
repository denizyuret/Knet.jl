type Sigm <: Layer; y; Sigm()=new() end

# TODO: implement cpu version
# forw(l::Sigm,x; o...)=(for i=1:length(x); (x[i]<zero(x[i]))&&(x[i]=zero(x[i])) end; l.y=x)
# back(l::Sigm,dy; o...)=(for i=1:length(dy); (l.y[i]==zero(l.y[i]))&&(dy[i]=zero(dy[i])) end; dy)

forw(l::Sigm,x::CudaArray; o...)=(l.y=cudnnActivationForward(x; mode=CUDNN_ACTIVATION_SIGMOID))
back(l::Sigm,dy::CudaArray; o...)=cudnnActivationBackward(l.y, dy; mode=CUDNN_ACTIVATION_SIGMOID)
