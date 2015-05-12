type Drop <: Layer; dropout; xdrop; 
    Drop(d)=(@assert 0 <= d <= 1; new(d))
end

function forw(l::Drop, x; fx=true, o...)
    if fx && (l.dropout > 0)
        chksize(l, :xdrop, x)
        rand!(l.xdrop)
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

function drop(x::AbstractArray, xdrop::AbstractArray, dropout::Number, scale::Number)
    for i=1:length(x)
        x[i] = (xdrop[i] < dropout ? zero(x[i]) : scale * x[i]) 
    end
end

function drop(x::CudaArray, xdrop::CudaArray, dropout::Number, scale::Number)
    ccall((:drop,libkunet),Void,(Cint,Cmat,Cmat,Cfloat,Cfloat),length(x),x,xdrop,dropout,scale)
end
