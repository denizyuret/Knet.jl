import Base: drop

type Drop <: Op; dropout; xdrop; y; Drop(d)=new(d); end

# TODO: implement Dropout using Mul2: however forw conditional on predict?
## maybe we can make rand conditional and mul2 treat nothing as identity?
# TODO: be careful about corrupting the xdrop matrix in Net

params(::Drop)=Any[]
ninputs(::Drop)=1
ysize(::Drop,x)=size(x)
overwrites(::Drop)=true
back_reads_x(::Drop)=false
back_reads_y(::Drop)=false

function forw(l::Drop, x; y=x, train=true, xdrop=nothing, seed=nothing, o...)
    issimilar(x,y) || error("Input mismatch")
    if train && (l.dropout > 0)
        similar!(l, :xdrop, y)
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
        drop(x, y, l.xdrop, l.dropout, 1/(1-l.dropout))
    else
        y===x || copy!(y,x)
    end
    return y
end

function back(l::Drop, dy; dx=dy, returndx=true, o...)
    returndx || return
    if l.dropout > 0
        @assert issimilar(dy, l.xdrop) "$(summary(dy)) !~ $(summary(l.xdrop)) $(size(l.xdrop))"
        drop(dy, dx, l.xdrop, l.dropout, 1/(1-l.dropout))
    else
        dx===dy || copy!(dx,dy)
    end
    return dx
end

drop(x::Array, y::Array, xdrop::Array, dropout::Number, scale::Number)=(for i=1:length(y); y[i] = (xdrop[i] < dropout ? zero(x[i]) : scale * x[i]); end; y)
drop(x::KUdense, y::KUdense, xdrop::KUdense, dropout::Number, scale::Number)=(drop(x.arr, y.arr, xdrop.arr, dropout, scale); y)

GPU && (drop(x::CudaArray{Float32}, y::CudaArray{Float32}, xdrop::CudaArray{Float32}, dropout, scale)=ccall((:drop32,libknet),Void,(Cint,Ptr{Float32},Ptr{Float32},Ptr{Float32},Cdouble,Cdouble),length(x),x,y,xdrop,dropout,scale))
GPU && (drop(x::CudaArray{Float64}, y::CudaArray{Float64}, xdrop::CudaArray{Float64}, dropout, scale)=ccall((:drop64,libknet),Void,(Cint,Ptr{Float64},Ptr{Float64},Ptr{Float64},Cdouble,Cdouble),length(x),x,y,xdrop,dropout,scale))
