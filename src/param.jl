using Base.LinAlg.BLAS: axpy!, scal!

type Param; data; diff; lr; l1reg; l2reg; adagrad; ada; momentum; mom; nesterov; nes; 
    Param(w;o...)=setparam!(new(convert(Atype{Ftype},w));o...)
end

setparam!(p::Param; o...)=(for (n,v) in o; p.(n)=v; end; p)

function copy(p::Param; o...)
    q = Param(p.data)
    for n in names(p)
        isdefined(p,n) || continue
        if ((isa(p.(n), Array) || isa(p.(n), CudaArray)) && !isa(p.(n), Atype{Ftype}))
            q.(n) = convert(Atype{Ftype}, p.(n))
        else
            q.(n) = copy(p.(n))
        end
    end
    return q
end

function update(p::Param; o...)
    initupdate(p)
    nz(p,:l1reg) && l1reg!(p.l1reg, p.data, p.diff)
    nz(p,:l2reg) && l2reg!(p.l2reg, p.data, p.diff)
    nz(p,:adagrad) && adagrad!(p.adagrad, p.ada, p.diff)
    nz(p,:lr,1) && scal!(length(p.diff), convert(eltype(p.diff),p.lr), p.diff, 1)
    nz(p,:momentum) && momentum!(p.momentum, p.mom, p.diff)
    nz(p,:nesterov) && nesterov!(p.nesterov, p.nes, p.diff)
    axpy!(length(p.data), -one(eltype(p.data)), p.diff, 1, p.data, 1)
    # nz(p,:maxnorm) && maxnorm!(p.maxnorm, p.data)
end

nz(p,n,v=zero(Ftype))=(isdefined(p,n) && (p.(n) != v))

function initupdate(p::Param)
    isdefined(p,:adagrad)  && (p.adagrad  > zero(p.adagrad))  && similar!(p, :ada, p.diff; fill=zero(Ftype))
    isdefined(p,:momentum) && (p.momentum > zero(p.momentum)) && similar!(p, :mom, p.diff; fill=zero(Ftype))
    isdefined(p,:nesterov) && (p.nesterov > zero(p.nesterov)) && similar!(p, :nes, p.diff; fill=zero(Ftype))
end

l1reg!(l1, w, dw)=for i=1:length(dw); (w[i]>zero(w[i])) ? (dw[i]+=l1) : (w[i]<zero(w[i])) ? (dw[i]-=l1) : 0; end
l2reg!(l2, w, dw)=axpy!(length(dw), convert(eltype(w),l2), w, 1, dw, 1)
adagrad!(eps, dw2, dw)=for i=1:length(dw); dw2[i] += dw[i] * dw[i]; dw[i] /= (eps + sqrt(dw2[i])); end
momentum!(m, dw2, dw)=(m=convert(eltype(dw2),m); axpy!(length(dw), m, dw2, 1, dw, 1); copy!(dw2,dw))
nesterov!(m, dw2, dw)=(nw=length(dw); m=convert(eltype(dw2),m); scal!(nw, m, dw2, 1); axpy!(nw, one(eltype(dw)), dw, 1, dw2, 1); axpy!(nw, m, dw2, 1, dw, 1))

if GPU
adagrad!(eps, dw2::CudaArray{Float32}, dw::CudaArray{Float32})=ccall((:adagrad32,libkunet),Void,(Cint,Cfloat,Ptr{Float32},Ptr{Float32}),length(dw),eps,dw2,dw)
l1reg!(l1, w::CudaArray{Float32}, dw::CudaArray{Float32})=ccall((:l1reg32,libkunet),Void,(Cint,Cfloat,Ptr{Float32},Ptr{Float32}),length(dw),l1,w,dw)
adagrad!(eps, dw2::CudaArray{Float64}, dw::CudaArray{Float64})=ccall((:adagrad64,libkunet),Void,(Cint,Cdouble,Ptr{Float64},Ptr{Float64}),length(dw),eps,dw2,dw)
l1reg!(l1, w::CudaArray{Float64}, dw::CudaArray{Float64})=ccall((:l1reg64,libkunet),Void,(Cint,Cdouble,Ptr{Float64},Ptr{Float64}),length(dw),l1,w,dw)
end

# function maxnorm!(maxnorm, w)
#     error("Did not debug maxnorm yet.")
#     norms = sqrt(sum(w.^2, 2))
#     if any(norms > p.maxnorm)
#         scale = min(p.maxnorm ./ norms, 1)
#         w *= scale
#     end
# end

