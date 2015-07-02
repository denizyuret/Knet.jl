type Bias <: Layer; w; 
    Bias(d...; init=initzero, o...)=new(Param(d...; init=init, o...))
    Bias()=new(Param(0))
end

# copy(l::Bias; o...)=Bias(copy(l.w; o...))
update(l::Bias; o...)=update(l.w; o...)
setparam!(l::Bias; o...)=setparam!(l.w; o...)

# We are implementing the CUDNN_ADD_SAME_C mode of cudnn:
# In this mode if x has dimensions (X1,X2,...,C,N) then
# bias has length=C.

function forw(l::Bias, x, y=x; o...)
    (w,c) = initforw(l,x,y)
    if ndims(x) == 1
        for i=1:length(x); y[i] = x[i] + w[i]; end
    else
        for i=1:length(x); y[i] = x[i] + w[ind2sub(size(x),i)[c]]; end
    end
    return y
end

function initforw(l::Bias, x, y)
    @assert issimilar(x,y)
    y===x || copy!(y,x)
    c = ndims(x)-1
    nb = size(x, c==0 ? 1 : c)
    w = l.w.data
    isempty(w) && (w = l.w.data = initzero((gpu()?CudaArray:Array)(eltype(x), nb)))
    @assert length(w) == nb
    @assert eltype(w) == eltype(x)
    return (w,c)
end

function back(l::Bias, dy; o...)
    (db, c) = initback(l, dy)
    # sum!(l.w.diff, dy)
    if ndims(dy) == 1
        for i=1:length(dy); db[i]=dy[i]; end
    else
        fill!(db, zero(eltype(db)))
        for i=1:length(dy); db[ind2sub(size(dy),i)[c]] += dy[i]; end
    end
    return dy
end

function initback(l::Bias, dy)
    similar!(l.w, :diff, l.w.data)
    db = l.w.diff
    c = ndims(dy)-1
    nb = size(dy, c==0 ? 1 : c)
    @assert length(db) == nb
    @assert eltype(db) == eltype(dy)
    return (db, c)
end

if GPU
forw(l::Bias, x::CudaArray, y=x; o...)=(initforw(l,x,y); cudnnAddTensor(l.w.data, y; mode=CUDNN_ADD_SAME_C); y)
back(l::Bias, dy::CudaArray; o...)=(initback(l,dy); cudnnConvolutionBackwardBias(dy, l.w.diff); dy)
end

