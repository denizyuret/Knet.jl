"""
Rgen is an abstract type whose subtypes represent random
distributions and other array fillers.  Currently implemented subtypes
are listed below.

    * Gaussian(mean, std)
    * Uniform(min, max)
    * Constant(val)
    * Identity(scale)
    * Xavier()
    * Bernoulli(p,scale)
"""
abstract Rgen
type Gaussian <: Rgen; mean; std; Gaussian(mean=0,std=1)=new(mean,std); end
type Uniform  <: Rgen; min; max; Uniform(min=0,max=1)=new(min,max); end
type Constant <: Rgen; val; Constant(val=0)=new(val); end
type Identity <: Rgen; scale; Identity(x=1)=new(x); end
type Xavier <: Rgen; end        # lasagne calls this GlorotUniform?
type Bernoulli <: Rgen; p; scale; Bernoulli(p=0.5, s=1)=new(p,s); end
type GlorotUniform <: Rgen; gain; GlorotUniform(gain=1)=new(gain); end
type GlorotNormal <: Rgen; gain; GlorotNormal(gain=1)=new(gain); end

function rgen!(r::Rgen, y)
    (isa(r, Constant)  ? fillsync!(y, r.val) :
     isa(r, Uniform)   ? axpb!(rand!(y); a=r.max - r.min, b=r.min) :
     isa(r, Gaussian)  ? axpb!(randn!(y); a=r.std, b=r.mean) :
     isa(r, Identity)  ? scale!(r.scale, copysync!(y, eye(eltype(y), size(y)...))) :
     isa(r, Xavier)    ? (fanin = length(y) / (size(y)[end]); scale = sqrt(3 / fanin); axpb!(rand!(y); a=2*scale, b=-scale)) : # This is wrong for fully connected
     isa(r, Bernoulli) ? bernoulli!(r.p, r.scale, rand!(y)) :
     isa(r, GlorotUniform) ? (s = r.gain*glorot(y); axpb!(rand!(y); a=2s, b=-s)) :
     isa(r, GlorotNormal)  ? (s = r.gain*glorot(y); scale!(s, randn!(y))) :
     error("Unknown Rgen=$r"))
    return y
end

bernoulli!{T}(p,s,x::Array{T})=(p=T(p);s=T(s);@inbounds for i=1:length(x); x[i] = (rand(T) < p ? s : 0); end; x)
@gpu bernoulli!(p,s,x::CudaArray{Float32})=(ccall((:bernoulli32,libknet), Void, (Cint,Cfloat,Cfloat,Ptr{Cfloat}), length(x), p, s, x); gpusync(); x)
@gpu bernoulli!(p,s,x::CudaArray{Float64})=(ccall((:bernoulli64,libknet), Void, (Cint,Cdouble,Cdouble,Ptr{Cdouble}), length(x), p, s, x); gpusync(); x)

function glorot(w)
     # The old implementation was not right for fully connected layers:
     # (fanin = length(y) / (size(y)[end]); scale = sqrt(3 / fanin); axpb!(rand!(y); a=2*scale, b=-scale)) :
    if ndims(w) < 2
        error("ndims=$(ndims(w)) in xavier")
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        fanout = size(w, ndims(w)) # Caffe disagrees: http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1XavierFiller.html#details
        fanin = div(length(w), fanout)
    end
    sqrt(2 / (fanin + fanout))
end

function Base.isequal(a::Rgen,b::Rgen)
    typeof(a)==typeof(b) || return false
    for n in fieldnames(a)
        if isdefined(a,n) && isdefined(b,n)
            isequal(a.(n), b.(n)) || return false
        elseif isdefined(a,n) || isdefined(b,n)
            return false
        end
    end
    return true
end
