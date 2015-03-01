function update(w, dw, o::UpdateParam)
    initupdate(w, dw, o)
    nz(o,:l1reg) && l1reg!(o.l1reg, w, dw)
    nz(o,:l2reg) && l2reg!(o.l2reg, w, dw)
    nz(o,:adagrad) && adagrad!(o.adagrad, o.ada, dw)
    nz(o,:learningRate,1f0) && (@in1! dw .* o.learningRate)
    nz(o,:momentum) && momentum!(o.momentum, o.mom, dw)
    nz(o,:nesterov) && nesterov!(o.nesterov, o.nes, dw)
    @in1! w .- dw
    nz(o,:maxnorm) && maxnorm!(o.maxnorm, w)
end

nz(o,n,v=0f0)=(isdefined(o,n) && (o.(n) != v))

l1reg!(l1, w, dw)=for i=1:length(dw); (w[i]>zero(w[i])) ? (dw[i]+=l1) : (w[i]<zero(w[i])) ? (dw[i]-=l1) : 0; end
l2reg!(l2, w, dw)=axpy!(length(dw), l2, w, 1, dw, 1)
adagrad!(eps, dw2, dw)=for i=1:length(dw); dw2[i] += dw[i] * dw[i]; dw[i] /= (eps + sqrt(dw2[i])); end
momentum!(m, dw2, dw)=(axpy!(length(dw), m, dw2, 1, dw, 1);copy!(dw2,dw))
nesterov!(m, dw2, dw)=(nw=length(dw); scal!(nw, m, dw2, 1); axpy!(nw, one(eltype(dw)), dw, 1, dw2, 1); axpy!(nw, m, dw2, 1, dw, 1))

if usegpu
    adagrad!(eps, dw2::CudaArray, dw::CudaArray)=ccall((:adagrad,libkunet),Void,(Cint,Cfloat,Cmat,Cmat),length(dw),eps,dw2,dw)
    l1reg!(l1, w::CudaArray, dw::CudaArray)=ccall((:l1reg,libkunet),Void,(Cint,Cfloat,Cmat,Cmat),length(dw),l1,w,dw)
end

function maxnorm!(maxnorm, w)
    error("Did not debug maxnorm yet.")
    norms = sqrt(sum(w.^2, 2))
    if any(norms > o.maxnorm)
        scale = min(o.maxnorm ./ norms, 1)
        w *= scale
    end
end

function initupdate(w, dw, o::UpdateParam)
    isdefined(o,:adagrad)  && (o.adagrad > zero(o.adagrad))   && chksize(o, :ada, dw; fill=0f0)
    isdefined(o,:momentum) && (o.momentum > zero(o.momentum)) && chksize(o, :mom, dw; fill=0f0)
    isdefined(o,:nesterov) && (o.nesterov > zero(o.nesterov)) && chksize(o, :nes, dw; fill=0f0)
end

