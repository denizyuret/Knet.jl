type Const <: Op; init; initialized; out; Const(;init=nothing,o...)=new(init,false); end
_KENV[:const]=Const
_KENV[:arr]=Const               # old name, to be deprecated
infersize(a::Const,ysize)=tuple(size(a.init))
ninputs(::Const)=0
canoverwrite(::Const)=false
back_reads_x(::Const)=false
back_reads_y(::Const)=false
back(::Const,dy;o...)=nothing

function forw(a::Const,y;o...)
    if !a.initialized
        a.out=copysync!(y, a.init)
        a.initialized=true
    end
    a.out===y || error("Constant modified")
    return y
end
