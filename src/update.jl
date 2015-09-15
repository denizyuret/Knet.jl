using Base.LinAlg: axpy!, scale!

function update!(p::KUparam; gclip=0, o...)
    initupdate(p)
    gclip > 0 && scale!(gclip, p.diff) # this is not a per-parameter deal, we need the gnorm for the whole model
    nz(p,:l1reg,0) && l1reg!(p.l1reg, p.arr, p.diff)
    nz(p,:l2reg,0) && l2reg!(p.l2reg, p.arr, p.diff)
    nz(p,:adagrad,false) && adagrad!(1e-8, p.ada, p.diff)
    nz(p,:momentum,0) && momentum!(p.momentum, p.mom, p.diff)
    nz(p,:nesterov,0) && nesterov!(p.nesterov, p.nes, p.diff)
    nz(p,:lr,1) && scale!(p.lr, p.diff)
    axpy!(-1, p.diff, p.arr)
    nz(p,:average,false) && axpy!(1,p.arr,p.avg)
    # nz(p,:maxnorm,0) && maxnorm!(p.maxnorm, p.arr)
end

nz(p,n,v)=(isdefined(p,n) && (p.(n) != v))

function initupdate(p::KUparam)
    nz(p,:average,false) && similar!(p, :avg, p.arr; fill=0)
    nz(p,:adagrad,false) && similar!(p, :ada, p.diff; fill=0)
    nz(p,:momentum,0) && similar!(p, :mom, p.diff; fill=0)
    nz(p,:nesterov,0) && similar!(p, :nes, p.diff; fill=0)
end

l1reg!(l1, w, dw)=for i=1:length(dw); (w[i]>zero(w[i])) ? (dw[i]+=l1) : (w[i]<zero(w[i])) ? (dw[i]-=l1) : 0; end
l2reg!(l2, w, dw)=axpy!(l2, w, dw)
adagrad!(eps, dw2, dw)=for i=1:length(dw); dw2[i] += dw[i] * dw[i]; dw[i] /= (eps + sqrt(dw2[i])); end
momentum!(m, dw2, dw)=(axpy!(m, dw2, dw); copy!(dw2,dw))
nesterov!(m, dw2, dw)=(scale!(m, dw2); axpy!(1, dw, dw2); axpy!(m, dw2, dw))

if GPU
adagrad!(eps, dw2::CudaArray{Float32}, dw::CudaArray{Float32})=ccall((:adagrad32,libkunet),Void,(Cint,Cdouble,Ptr{Float32},Ptr{Float32}),length(dw),eps,dw2,dw)
adagrad!(eps, dw2::CudaArray{Float64}, dw::CudaArray{Float64})=ccall((:adagrad64,libkunet),Void,(Cint,Cdouble,Ptr{Float64},Ptr{Float64}),length(dw),eps,dw2,dw)
l1reg!(l1, w::CudaArray{Float32}, dw::CudaArray{Float32})=ccall((:l1reg32,libkunet),Void,(Cint,Cdouble,Ptr{Float32},Ptr{Float32}),length(dw),l1,w,dw)
l1reg!(l1, w::CudaArray{Float64}, dw::CudaArray{Float64})=ccall((:l1reg64,libkunet),Void,(Cint,Cdouble,Ptr{Float64},Ptr{Float64}),length(dw),l1,w,dw)
end #if GPU

# function maxnorm!(maxnorm, w)
#     error("Did not debug maxnorm yet.")
#     norms = sqrt(sum(w.^2, 2))
#     if any(norms > p.maxnorm)
#         scale = min(p.maxnorm ./ norms, 1)
#         w *= scale
#     end
# end

