type Drop <: Layer; dropout; xdrop; Drop()=new(); end
Drop(d)=(@assert 0 <= d <= 1; l=Drop();l.dropout=d;l)

function forw(l::Drop, x; fx=true, xdrop=nothing, o...)
    if fx && (l.dropout > 0)
        chksize(l, :xdrop, x)
        (xdrop == nothing) ? rand!(l.xdrop) : copy!(l.xdrop, xdrop)
        drop(x, l.xdrop, l.dropout, 1/(1-l.dropout))
    end
    return x
end

function back(l::Drop, dy; o...)
    if l.dropout > 0
        drop(dy, l.xdrop, l.dropout, 1/(1-l.dropout))
    end
    return dy
end

function drop(x, xdrop, dropout, scale)
    for i=1:length(x)
        x[i] = (xdrop[i] < dropout ? zero(x[i]) : scale * x[i]) 
    end
end

if GPU
drop(x::CudaArray, xdrop::CudaArray, dropout, scale)=ccall((:drop,libkunet),Void,(Cint,Cmat,Cmat,Cfloat,Cfloat),length(x),x,xdrop,dropout,scale)
end
