type Const <: Op; init; initialized; out; Const(;init=nothing,o...)=new(init,false); end
# const(y; init=nothing, o...)=(Const(init=init),y)
_KENV[:const]=Const
infersize(a::Const,ysize)=tuple(size(a.init))
ninputs(::Const)=0
overwrites(::Const)=false
back_reads_x(::Const)=false
back_reads_y(::Const)=false
back(::Const,dy;o...)=nothing
forw(a::Const,y;o...)=(!a.initialized ? (a.initialized=true; a.out=copy!(y, a.init)) : a.out===y ? y : error("Constant modified"))
