type Arr <: Op; init; initialized; out; Arr(;init=nothing,o...)=new(init,false); end
Kenv.kdef(:arr,Arr)
infersize(a::Arr,ysize)=tuple(size(a.init))
ninputs(::Arr)=0
canoverwrite(::Arr)=false
back_reads_x(::Arr)=false
back_reads_y(::Arr)=false
back(::Arr,dy;o...)=nothing

function forw(a::Arr,y;o...)
    if !a.initialized
        a.out=copysync!(y, a.init)
        a.initialized=true
    end
    a.out===y || error("Constant modified")
    return y
end
