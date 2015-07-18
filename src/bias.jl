type Bias <: Layer; b; Bias(p::KUparam)=new(p); end

param(l::Bias)=l.b
default_init(::Type{Bias})=initzero

Bias(d...; init=default_init(Bias), o...)=Bias(KUparam(d...; init=init, o...))
Bias()=Bias(KUparam(0))

forw(l::Bias, x::KUdense; o...)=(initforw(l,x); biasforw(l.b.arr, x.arr); x)
back(l::Bias, dy::KUdense; returndx=true, o...)=(initback(l,dy); biasback(l.b.diff, dy.arr); returndx && dy)

function initforw(l::Bias, x::KUdense)
    nb = size(x, ndims(x)==1 ? 1 : ndims(x)-1)
    isempty(l.b) && (l.b=KUparam(eltype(x), (nb,); init=default_init(Bias)))
    @assert length(l.b) == nb
    @assert eltype(l.b) == eltype(x)
end

function initback(l::Bias, dy::KUdense)
    initdiff(l.b)
    nb = size(dy, ndims(dy)==1 ? 1 : ndims(dy)-1)
    @assert length(l.b) == nb
    @assert eltype(l.b) == eltype(dy)
end

# We are implementing the CUDNN_ADD_SAME_C mode of cudnn:
# In this mode if x has dimensions (X1,X2,...,C,N) then
# bias has length=C.

biasforw(b::Vector, x::Vector)=(for i=1:length(x); x[i] = x[i] + b[i]; end)
biasforw(b::Vector, x::Array)=(c=ndims(x)-1; for i=1:length(x); x[i] = x[i] + b[ind2sub(size(x),i)[c]]; end)
biasback(db::Vector, dy::Vector)=(for i=1:length(dy); db[i]=dy[i]; end)
biasback(db::Vector, dy::Array)=(c=ndims(dy)-1; fill!(db, zero(eltype(db))); for i=1:length(dy); db[ind2sub(size(dy),i)[c]] += dy[i]; end)

if GPU
biasforw(b::CudaArray, x::CudaArray)=cudnnAddTensor(b, x; mode=CUDNN_ADD_SAME_C)
biasback(db::CudaArray, dy::CudaArray)=cudnnConvolutionBackwardBias(dy, db)
end
