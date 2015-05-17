type Bias <: Layer; b::Param; Bias(b::Param)=new(b); end

Bias(b; o...)=Bias(Param(b;o...))
Bias(d::Integer...; o...)=Bias(Param(zeros(d); o...))

copy(l::Bias; o...)=Bias(copy(l.b; o...))
update(l::Bias; o...)=update(l.b; o...)
setparam!(l::Bias; o...)=setparam!(l.b; o...)

# We are implementing the CUDNN_ADD_SAME_C mode of cudnn:
# In this mode if x has dimensions (X1,X2,...,C,N) then
# bias has length=C.

function forw(l::Bias, x, y=x; o...)
    @assert issimilar(x,y)
    b = l.b.data
    if ndims(x) == 1
        @assert length(b) == length(x)
        for i=1:length(x); y[i] = x[i] + b[i]; end
    else
        c = ndims(x)-1
        @assert length(b) == size(x, c)
        for i=1:length(x); y[i] = x[i] + b[ind2sub(size(x),i)[c]]; end
    end
    return y
end

function back(l::Bias, dy; o...)
    similar!(l.b, :diff, l.b.data)
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
function forw(l::Bias, x::CudaArray, y=x; o...)
    @assert issimilar(x,y)
    y===x || copy!(y,x)
    cudnnAddTensor(l.b.data, y; mode=CUDNN_ADD_SAME_C)
    return y
end

back(l::Bias, dy::CudaArray; o...)=(similar!(l.b, :diff, l.b.data); cudnnConvolutionBackwardBias(dy, l.b.diff); dy)
end

