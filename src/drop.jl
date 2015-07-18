type Drop <: Layer; dropout; xdrop; Drop(d)=new(d); end

function forw(l::Drop, x::KUdense; predict=false, xdrop=nothing, seed=nothing, o...)
    if !predict && (l.dropout > 0)
        similar!(l, :xdrop, x)
        if xdrop != nothing
            copy!(l.xdrop, xdrop)
        elseif seed != nothing
            # setseed(seed)
            # rand!(l.xdrop)
            # we do this instead of rand! on gpu to eliminate cpu/gpu difference:
            srand(seed)
            copy!(l.xdrop, rand(eltype(l.xdrop), size(l.xdrop)))
        else
            rand!(l.xdrop)
        end
        drop(x.arr, l.xdrop.arr, l.dropout, 1/(1-l.dropout))
    end
    return x
end

function back(l::Drop, dy::KUdense; o...)
    @assert issimilar(dy, l.xdrop) "$(summary(dy)) !~ $(summary(l.xdrop))"
    l.dropout > 0 && drop(dy.arr, l.xdrop.arr, l.dropout, 1/(1-l.dropout))
    return dy
end

function drop(x::Array, xdrop::Array, dropout::Number, scale::Number)
    for i=1:length(x)
        x[i] = (xdrop[i] < dropout ? zero(x[i]) : scale * x[i]) 
    end
end

if GPU
drop(x::CudaArray{Float32}, xdrop::CudaArray{Float32}, dropout, scale)=ccall((:drop32,libkunet),Void,(Cint,Ptr{Float32},Ptr{Float32},Cdouble,Cdouble),length(x),x,xdrop,dropout,scale)
drop(x::CudaArray{Float64}, xdrop::CudaArray{Float64}, dropout, scale)=ccall((:drop64,libkunet),Void,(Cint,Ptr{Float64},Ptr{Float64},Cdouble,Cdouble),length(x),x,xdrop,dropout,scale)
end
