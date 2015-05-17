type Bias <: Layer; b::Param; Bias(b::Param)=new(b); end

Bias(b; o...)=Bias(Param(b;o...))
Bias(d::Integer...; o...)=Bias(Param(zeros(d); o...))

copy(l::Bias; o...)=Bias(copy(l.b; o...))
update(l::Bias; o...)=update(l.b; o...)
setparam!(l::Bias; o...)=setparam!(l.b; o...)

# We are implementing the CUDNN_ADD_SAME_C mode of cudnn:
# In this mode if x has dimensions (X1,X2,...,C,N) then
# bias has length=C.

function forw(l::Bias, x; o...)
    b = l.b.data
    if ndims(x) == 1
        @assert length(b) == length(x)
        for i=1:length(x); x[i] += b[i]; end
    else
        c = ndims(x)-1
        @assert length(b) == size(x, c)
        for i=1:length(x); x[i] += b[ind2sub(size(x),i)[c]]; end
    end
    return x
end

function back(l::Bias, dy; o...)
    chksize(l.b, :diff, l.b.data)
    db = l.b.diff
    if ndims(dy) == 1
        @assert length(db) == length(dy)
        for i=1:length(dy); db[i]=dy[i]; end
    else
        c = ndims(dy)-1
        @assert length(db) == size(dy, c)
        fill!(db, zero(eltype(db)))
        for i=1:length(dy); db[ind2sub(size(dy),i)[c]] += dy[i]; end
    end
    # sum!(l.b.diff, dy)
    return dy
end

if GPU
forw(l::Bias, x::CudaArray; o...)=(cudnnAddTensor(l.b.data, x; mode=CUDNN_ADD_SAME_C); x)
back(l::Bias, dy::CudaArray; o...)=(chksize(l.b, :diff, l.b.data); cudnnConvolutionBackwardBias(dy, l.b.diff); dy)
end

