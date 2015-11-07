type Arr <: Op; init; initialized; out; Arr(init)=new(init,false); end
arr(y; init=nothing, o...)=(Arr(init),y)
infersize(a::Arr,ysize)=size(a.init)
ninputs(::Arr)=0
overwrites(::Arr)=false
back_reads_x(::Arr)=false
back_reads_y(::Arr)=false
back(::Arr,dy;o...)=nothing
forw(a::Arr,y;o...)=(!a.initialized ? (a.initialized=true; a.out=copy!(y, a.init)) : a.out===y ? y : error("Constant modified"))
