type Relu <: Layer; y; Relu()=new() end

forw(l::Relu,x; o...)=(for i=1:length(x); (x[i]<zero(x[i]))&&(x[i]=zero(x[i])) end; l.y=x)
back(l::Relu,dy; o...)=(for i=1:length(dy); (l.y[i]==zero(l.y[i]))&&(dy[i]=zero(dy[i])) end; dy)

forw(l::Relu,x::CudaArray; o...)=(ccall((:reluforw,libkunet),Void,(Cint,Cmat),length(x),x); l.y=x)
back(l::Relu,dy::CudaArray; o...)=(ccall((:reluback,libkunet),Void,(Cint,Cmat,Cmat),length(dy),l.y,dy); dy)
