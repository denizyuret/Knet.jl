type Bias <: Layer; b; Bias(p::KUparam)=new(p); end

param(l::Bias)=l.b
default_init(::Type{Bias})=initzero

Bias(d...; init=default_init(Bias), o...)=Bias(KUparam(d...; init=init, o...))
Bias()=Bias(KUparam(0))

# We are implementing the CUDNN_ADD_SAME_C mode of cudnn:
# In this mode if x has dimensions (X1,X2,...,C,N) then
# bias has length=C.

function forw(l::Bias, x, y=x; o...)
    initforw(l,x,y)
    (ya, xa, ba) = (y.arr, x.arr, l.b.arr)
    if ndims(x) == 1
        for i=1:length(x); ya[i] = xa[i] + ba[i]; end
    else
        c = ndims(x)-1
        for i=1:length(x); ya[i] = xa[i] + ba[ind2sub(size(x),i)[c]]; end
    end
    return y
end

function back(l::Bias, dy; o...)
    initback(l, dy)
    (ya, ba) = (dy.arr, l.b.diff)
    if ndims(dy) == 1
        for i=1:length(dy); ba[i]=ya[i]; end
    else
        c = ndims(dy)-1
        fill!(ba, zero(eltype(ba)))
        for i=1:length(dy); ba[ind2sub(size(dy),i)[c]] += ya[i]; end
    end
    return dy
end

function initforw(l::Bias, x, y)
    @assert issimilar(x,y)
    y===x || copy!(y,x)
    c = ndims(x)-1
    nb = size(x, c==0 ? 1 : c)
    isempty(l.b) && (l.b=KUparam(eltype(x), (nb,); init=default_init(Bias)))
    @assert length(l.b) == nb
    @assert eltype(l.b) == eltype(x)
end

function initback(l::Bias, dy)
    initdiff(l.b)
    c = ndims(dy)-1
    nb = size(dy, c==0 ? 1 : c)
    @assert length(l.b) == nb
    @assert eltype(l.b) == eltype(dy)
end

if GPU
forw(l::Bias, x::KUdense{CudaArray}, y=x; o...)=(initforw(l,x,y); cudnnAddTensor(l.b.arr, y.arr; mode=CUDNN_ADD_SAME_C); y)
back(l::Bias, dy::KUdense{CudaArray}; o...)=(initback(l,dy); cudnnConvolutionBackwardBias(dy.arr, l.b.diff); dy)
end
