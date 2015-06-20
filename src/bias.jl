type Bias <: Layer; b; 
    Bias(d...; o...)=new(Param(d...; o...))
    Bias()=new()
end

# copy(l::Bias; o...)=Bias(copy(l.b; o...))
update(l::Bias; o...)=update(l.b; o...)
setparam!(l::Bias; o...)=setparam!(l.b; o...)

# We are implementing the CUDNN_ADD_SAME_C mode of cudnn:
# In this mode if x has dimensions (X1,X2,...,C,N) then
# bias has length=C.

function forw(l::Bias, x, y=x; o...)
    (b,c) = initforw(l,x,y)
    if ndims(x) == 1
        for i=1:length(x); y[i] = x[i] + b[i]; end
    else
        for i=1:length(x); y[i] = x[i] + b[ind2sub(size(x),i)[c]]; end
    end
    return y
end

function initforw(l::Bias, x, y)
    @assert issimilar(x,y)
    y===x || copy!(y,x)
    c = ndims(x)-1
    nb = size(x, c==0 ? 1 : c)
    isdefined(l,:b) || (l.b = Param(eltype(x), nb; init=initzero))
    b = l.b.data
    @assert length(b) == nb
    @assert eltype(b) == eltype(x)
    return (b,c)
end

initzero(a)=fill!(a,0)

function back(l::Bias, dy; o...)
    (db, c) = initback(l, dy)
    # sum!(l.b.diff, dy)
    if ndims(dy) == 1
        for i=1:length(dy); db[i]=dy[i]; end
    else
        fill!(db, zero(eltype(db)))
        for i=1:length(dy); db[ind2sub(size(dy),i)[c]] += dy[i]; end
    end
    return dy
end

function initback(l::Bias, dy)
    similar!(l.b, :diff, l.b.data)
    db = l.b.diff
    c = ndims(dy)-1
    nb = size(dy, c==0 ? 1 : c)
    @assert length(db) == nb
    @assert eltype(db) == eltype(dy)
    return (db, c)
end

if GPU
forw(l::Bias, x::CudaArray, y=x; o...)=(initforw(l,x,y); cudnnAddTensor(l.b.data, y; mode=CUDNN_ADD_SAME_C); y)
back(l::Bias, dy::CudaArray; o...)=(initback(l,dy); cudnnConvolutionBackwardBias(dy, l.b.diff); dy)
end

