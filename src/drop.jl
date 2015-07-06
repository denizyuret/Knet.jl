type Drop <: Layer; dropout; xdrop; Drop(d)=new(d); end
# copy(l::Drop;o...)=Drop(l.dropout)

function forw(l::Drop, x; predict=false, xdrop=nothing, seed=nothing, o...)
    if !predict && (l.dropout > 0)
        similar!(l, :xdrop, x)
        if xdrop != nothing
            copy!(l.xdrop, xdrop)
        elseif seed != nothing
            srand(seed)
            copy!(l.xdrop, rand(eltype(x), size(x)))
        else
            rand!(l.xdrop)
        end
        drop(x, l.xdrop, l.dropout, 1/(1-l.dropout))
    end
    return x
end

function back(l::Drop, dy; o...)
    @assert issimilar(dy, l.xdrop)
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
drop(x::AbstractCudaArray{Float32}, xdrop::AbstractCudaArray{Float32}, dropout, scale)=ccall((:drop32,libkunet),Void,(Cint,Ptr{Float32},Ptr{Float32},Cfloat,Cfloat),length(x),x,xdrop,dropout,scale)
drop(x::AbstractCudaArray{Float64}, xdrop::AbstractCudaArray{Float64}, dropout, scale)=ccall((:drop64,libkunet),Void,(Cint,Ptr{Float64},Ptr{Float64},Cdouble,Cdouble),length(x),x,xdrop,dropout,scale)
end
