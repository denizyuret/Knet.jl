using Base.LinAlg.BLAS: axpy!, scal!

type KUparam{A,T,N}; arr; diff; lr; l1reg; l2reg; adagrad; ada; momentum; mom; nesterov; nes; KUparam()=new(); end

setparam!(p::KUparam; o...)=(for (n,v) in o; p.(n)=v; end; p)
KUparam(w; init=nothing, o...)=setparam!(KUparam{atype(w),eltype(w),ndims(w)}(); arr=(init==nothing?w:init(w)), o...)
KUparam(A::Type, T::Type, d::Dims; o...)=KUparam(A(T,d); o...)
KUparam(T::Type, d::Dims; o...)=KUparam((gpu()?CudaArray:Array)(T,d); o...)
KUparam(d1::Int,d::Int...; o...)=KUparam(Float64,tuple(d1,d...); o...)

# We probably don't need this copy, just implement cpucopy and gpucopy.
# copy(p::KUparam; o...)=(q=KUparam(); for n in names(p); isdefined(p,n) && q.(n)=copy(p.(n)); end; q)

function update(p::KUparam; o...)
    initupdate(p)
    nz(p,:l1reg) && l1reg!(p.l1reg, p.arr, p.diff)
    nz(p,:l2reg) && l2reg!(p.l2reg, p.arr, p.diff)
    nz(p,:adagrad) && adagrad!(p.adagrad, p.ada, p.diff)
    nz(p,:momentum) && momentum!(p.momentum, p.mom, p.diff)
    nz(p,:nesterov) && nesterov!(p.nesterov, p.nes, p.diff)
    nz(p,:lr,1) && scal!(length(p.diff), convert(eltype(p.diff),p.lr), p.diff, 1)
    axpy!(length(p.arr), -one(eltype(p.arr)), p.diff, 1, p.arr, 1)
    # nz(p,:maxnorm) && maxnorm!(p.maxnorm, p.arr)
end

nz(p,n,v=0)=(isdefined(p,n) && (p.(n) != v))

function initupdate(p::KUparam)
    isdefined(p,:adagrad)  && (p.adagrad  > 0) && similar!(p, :ada, p.diff; fill=0)
    isdefined(p,:momentum) && (p.momentum > 0) && similar!(p, :mom, p.diff; fill=0)
    isdefined(p,:nesterov) && (p.nesterov > 0) && similar!(p, :nes, p.diff; fill=0)
end

l1reg!(l1, w, dw)=for i=1:length(dw); (w[i]>zero(w[i])) ? (dw[i]+=l1) : (w[i]<zero(w[i])) ? (dw[i]-=l1) : 0; end
l2reg!(l2, w, dw)=axpy!(length(dw), convert(eltype(w),l2), w, 1, dw, 1)
adagrad!(eps, dw2, dw)=for i=1:length(dw); dw2[i] += dw[i] * dw[i]; dw[i] /= (eps + sqrt(dw2[i])); end
momentum!(m, dw2, dw)=(m=convert(eltype(dw2),m); axpy!(length(dw), m, dw2, 1, dw, 1); copy!(dw2,dw))
nesterov!(m, dw2, dw)=(nw=length(dw); m=convert(eltype(dw2),m); scal!(nw, m, dw2, 1); axpy!(nw, one(eltype(dw)), dw, 1, dw2, 1); axpy!(nw, m, dw2, 1, dw, 1))

initzero(a)=(fill!(a,zero(eltype(a))); a)
initgaussian(a, std=0.01, mean=0.0)=(randn!(a,std,mean); a)
initxavier(a)=(fanin = length(a) / (size(a)[end]); scale = sqrt(3 / fanin); rand!(a, -scale, scale); a)

if GPU
adagrad!(eps, dw2::CudaArray{Float32}, dw::CudaArray{Float32})=ccall((:adagrad32,libkunet),Void,(Cint,Cfloat,Ptr{Float32},Ptr{Float32}),length(dw),eps,dw2,dw)
adagrad!(eps, dw2::CudaArray{Float64}, dw::CudaArray{Float64})=ccall((:adagrad64,libkunet),Void,(Cint,Cdouble,Ptr{Float64},Ptr{Float64}),length(dw),eps,dw2,dw)
l1reg!(l1, w::CudaArray{Float32}, dw::CudaArray{Float32})=ccall((:l1reg32,libkunet),Void,(Cint,Cfloat,Ptr{Float32},Ptr{Float32}),length(dw),l1,w,dw)
l1reg!(l1, w::CudaArray{Float64}, dw::CudaArray{Float64})=ccall((:l1reg64,libkunet),Void,(Cint,Cdouble,Ptr{Float64},Ptr{Float64}),length(dw),l1,w,dw)
end #if GPU

# BASIC ARRAY OPS:

for fname in (:eltype, :length, :ndims, :size, :strides, :pointer, :isempty)
    @eval (Base.$fname)(a::KUparam)=$fname(a.arr)
end

for fname in (:size, :stride)
    @eval (Base.$fname)(a::KUparam,n)=$fname(a.arr,n)
end

atype{A}(::KUparam{A})=A
diff(a::KUparam)=a.diff

update(::Nothing;o...)=nothing
setparam!(::Nothing;o...)=nothing
initdiff(w::KUparam;o...)=similar!(w, :diff, w.arr)

# We need to fix cpu/gpu copy so the type changes appropriately:
function cpucopy_internal{T,N}(x::KUparam{CudaArray,T,N},d::ObjectIdDict)
    haskey(d,x) && return d[x]
    y = KUparam{Array,T,N}()
    for n in names(x)
        isdefined(x,n) || continue
        y.(n) = cpucopy_internal(x.(n),d)
    end
    d[x] = y
end

function gpucopy_internal{T,N}(x::KUparam{Array,T,N},d::ObjectIdDict)
    haskey(d,x) && return d[x]
    y = KUparam{CudaArray,T,N}()
    for n in names(x)
        isdefined(x,n) || continue
        y.(n) = gpucopy_internal(x.(n),d)
    end
    d[x] = y
end

# function maxnorm!(maxnorm, w)
#     error("Did not debug maxnorm yet.")
#     norms = sqrt(sum(w.^2, 2))
#     if any(norms > p.maxnorm)
#         scale = min(p.maxnorm ./ norms, 1)
#         w *= scale
#     end
# end

