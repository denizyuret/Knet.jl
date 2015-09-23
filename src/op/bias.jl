type Bias <: Op; b; Bias(b::KUparam)=new(b); end

Bias(d1, d...; o...)=Bias(KUparam(d1, d...; o...))
Bias(; o...)=Bias(KUparam(0; o...))

params(l::Bias)=Any[l.b]
ninputs(::Bias)=1
ysize(::Bias,x)=size(x)
overwrites(::Bias)=true
back_reads_x(::Bias)=false
back_reads_y(::Bias)=false

function forw(l::Bias, x; y=x, o...)
    (b,x,y)=initforw(l,x,y;o...)
    biasforw(b,x,y)
    return y
end

function initforw(l::Bias, x, y; train=true, o...)
    issimilar(x,y) || error("Output mismatch")
    nb = size(x, ndims(x)==1 ? 1 : ndims(x)-1)
    if isempty(l.b) 
        nz(l.b,:init,nothing) || (l.b.init = fill!; l.b.initp = 0)
        init(l.b, eltype(x), (nb,))
    end
    length(l.b) == nb || error("length mismatch")
    eltype(l.b) == eltype(x) || error("eltype mismatch")
    return ((!train && nz(l.b, :average, false)) ? 
            (l.b.avg, x, y) : 
            (l.b.arr, x, y))
end

function back(l::Bias, dy; dx=dy, incr=false, returndx=true, o...)
    # display((:back00,incr,vecnorm0(dy),vecnorm0(params(l))))
    initback(l, dy, incr)
    # display((:back01,incr,vecnorm0(dy),vecnorm0(params(l))))
    if incr
        biasback(l.b.inc, dy)
        axpy!(1, l.b.inc, l.b.diff)
    else
        biasback(l.b.diff, dy)
    end
    if returndx
        issimilar(dx,dy) || error("Gradient mismatch")
        (dx===dy ? dx : copy!(dx,dy))
    end
    # display((:back02,incr,vecnorm0(dy),vecnorm0(params(l))))
end

function initback(l::Bias, dy, incr)
    nb = size(dy, ndims(dy)==1 ? 1 : ndims(dy)-1)
    length(l.b) == nb || error("length mismatch")
    eltype(l.b) == eltype(dy) || error("eltype mismatch")
    similar!(l.b, :diff, l.b.arr; fill=0)
    incr && similar!(l.b, :inc, l.b.arr)
end

# We are implementing the CUDNN_ADD_SAME_C mode of cudnn:
# In this mode if x has dimensions (X1,X2,...,C,N) then
# bias has length=C.

biasforw(b::Vector, x::Vector, y::Vector=x)=(for i=1:length(y); y[i] = x[i] + b[i]; end; y)
biasback(db::Vector, dy::Vector)=(for i=1:length(dy); db[i]=dy[i]; end)

biasforw(b::Vector, x::Array, y::Array=x)=(c=ndims(x)-1; for i=1:length(y); y[i] = x[i] + b[ind2sub(size(x),i)[c]]; end; y)
biasback(db::Vector, dy::Array)=(c=ndims(dy)-1; fill!(db, zero(eltype(db))); for i=1:length(dy); db[ind2sub(size(dy),i)[c]] += dy[i]; end)

biasforw(b, x::KUdense, y::KUdense=x)=(biasforw(b,x.arr,y.arr); y)
biasback(db, dy::KUdense)=biasback(db, dy.arr)

GPU && (biasforw(b::CudaArray, x::CudaArray, y::CudaArray=x)=(y===x||copy!(y,x);cudnnAddTensor(b, y; mode=CUDNN_ADD_SAME_C)))
GPU && (biasback(db::CudaArray, dy::CudaArray)=cudnnConvolutionBackwardBias(dy, db))
