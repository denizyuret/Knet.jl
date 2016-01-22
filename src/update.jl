using Base.LinAlg: axpy!, scale!

function update!(r::Reg; gscale=1, o...)
    initupdate(r)
    getp(r,:l1reg,0)!=0 && l1reg!(getp(r,:l1reg), r.out, r.dif)
    getp(r,:l2reg,0)!=0 && l2reg!(getp(r,:l2reg), r.out, r.dif)
    getp(r,:adagrad) && adagrad!(1e-8, getp(r,:ada), r.dif) # TODO: make 1e-8 a parameter
    getp(r,:rmsprop) && rmsprop!(1e-8, 0.9, getp(r,:rms), r.dif) # TODO: make 1e-8 a parameter
    getp(r,:momentum,0)!=0 && momentum!(getp(r,:momentum), getp(r,:mom), r.dif)
    getp(r,:nesterov,0)!=0 && nesterov!(getp(r,:nesterov), getp(r,:nes), r.dif)
    if getp(r, :adam)
        incp(r, :t)
        adam!(getp(r, :t), getp(r, :eps), getp(r, :b1), getp(r, :b2), getp(r, :fstm), getp(r, :scndm), r.dif)
    end
    scale = -1 * getp(r,:lr,1) * gscale # TODO: make scale a parameter for callbacks
    axpy!(scale, r.dif, r.out)
    getp(r,:average) && axpy!(1,r.out,getp(r,:avg))
end

function initupdate(r::Reg)
    isa(r.op, Par) || error("Update called on $(typeof(r.op))")
    if r.dif != nothing
        getp(r,:adagrad) && similarp(r, :ada, r.dif)
        getp(r,:rmsprop) && similarp(r, :rms, r.dif)
        getp(r,:momentum,0)!=0 && similarp(r, :mom, r.dif)
        getp(r,:nesterov,0)!=0 && similarp(r, :nes, r.dif)
        if getp(r,:adam)
            similarp(r, :fstm, r.dif)
            similarp(r, :scndm, r.dif)
            getp(r, :eps) == 0 && setp(r, :eps, 1e-8)
            getp(r, :b1) == 0 && setp(r, :b1, 0.9)
            getp(r, :b2) == 0 && setp(r, :b2, 0.999)
        end
    else
        error("r.dif==nothing in $r")
    end
    if r.out != nothing
        getp(r,:average) && similarp(r, :avg, r.out)
    else
        error("r.out==nothing in $r")
    end
end

function similarp(r,n,a)
    b = getp(r,n)
    issimilar(a,b) && return b
    b != false && warn("Changing $n")
    setp(r,n,fillsync!(similar(a),0))
end

l1reg!(l1, w, dw)=for i=1:length(dw); (w[i]>zero(w[i])) ? (dw[i]+=l1) : (w[i]<zero(w[i])) ? (dw[i]-=l1) : 0; end
l2reg!(l2, w, dw)=axpy!(l2, w, dw)
adagrad!(eps, dw2, dw)=for i=1:length(dw); dw2[i] += dw[i] * dw[i]; dw[i] /= sqrt(dw2[i] + eps); end
momentum!(m, dw2, dw)=(axpy!(m, dw2, dw); copysync!(dw2,dw))
nesterov!(m, dw2, dw)=(scale!(m, dw2); axpy!(1, dw, dw2); axpy!(m, dw2, dw))
rmsprop!(eps, rho, dw2, dw)=for i=1:length(dw); dw2[i] = dw2[i] * rho + (1 - rho) * dw[i] * dw[i]; dw[i] /= sqrt(dw2[i] + eps); end

function adam!(t, eps, b1, b2, fstm, scndm, dw)
    for i=1:length(dw)
        fstm[i] = b1*fstm[i] + (1-b1)*dw[i]
        scndm[i] = b2*scndm[i] + (1-b2)*(dw[i] *dw[i])
        dw[i] = (fstm[i] / (1 - b1 ^ t)) / (sqrt(scndm[i] / (1 - b2 ^ t)) + eps)
    end
end

@gpu adagrad!(eps, dw2::CudaArray{Float32}, dw::CudaArray{Float32})=ccall((:adagrad32,libknet),Void,(Cint,Cdouble,Ptr{Float32},Ptr{Float32}),length(dw),eps,dw2,dw)
@gpu adagrad!(eps, dw2::CudaArray{Float64}, dw::CudaArray{Float64})=ccall((:adagrad64,libknet),Void,(Cint,Cdouble,Ptr{Float64},Ptr{Float64}),length(dw),eps,dw2,dw)
@gpu l1reg!(l1, w::CudaArray{Float32}, dw::CudaArray{Float32})=ccall((:l1reg32,libknet),Void,(Cint,Cdouble,Ptr{Float32},Ptr{Float32}),length(dw),l1,w,dw)
@gpu l1reg!(l1, w::CudaArray{Float64}, dw::CudaArray{Float64})=ccall((:l1reg64,libknet),Void,(Cint,Cdouble,Ptr{Float64},Ptr{Float64}),length(dw),l1,w,dw)
@gpu rmsprop!(eps, rho, dw2::CudaArray{Float32}, dw::CudaArray{Float32})=ccall((:rmsprop32,libknet),Void,(Cint,Cdouble, Cdouble,Ptr{Float32},Ptr{Float32}),length(dw),eps, rho, dw2,dw)
@gpu rmsprop!(eps, rho, dw2::CudaArray{Float64}, dw::CudaArray{Float64})=ccall((:rmsprop64,libknet),Void,(Cint,Cdouble,Cdouble,Ptr{Float64},Ptr{Float64}),length(dw),eps, rho, dw2,dw)
@gpu adam!(t, eps, b1, b2, fstm::CudaArray{Float32}, scndm::CudaArray{Float32}, dw::CudaArray{Float32})=ccall((:adam32,libknet),Void,(Cint,Cint,Cdouble,Cdouble, Cdouble,Ptr{Float32},Ptr{Float32},Ptr{Float32}),length(dw),t,eps,b1,b2,fstm,scndm,dw)
@gpu adam!(t, eps, b1, b2, fstm::CudaArray{Float64}, scndm::CudaArray{Float64}, dw::CudaArray{Float64})=ccall((:adam64,libknet),Void,(Cint,Cint,Cdouble,Cdouble, Cdouble,Ptr{Float64},Ptr{Float64},Ptr{Float64}),length(dw),t,eps,b1,b2,fstm,scndm,dw)

# function maxnorm!(maxnorm, w)
#     error("Did not debug maxnorm yet.")
#     norms = sqrt(sum(w.^2, 2))
#     if any(norms > p.maxnorm)
#         scale = min(p.maxnorm ./ norms, 1)
#         w *= scale
#     end
# end

# nz(r,n,v)=(isdefined(r,n) && (r.(n) != v))

