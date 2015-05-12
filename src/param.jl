type Param; data; diff; learningRate; l1reg; l2reg; maxnorm; adagrad; ada; momentum; mom; nesterov; nes; 
    Param(data;args...)=(p=new(paramcheck(data)); for (k,v) in args; p.(k)=v; end; p)
end

paramcheck(a)=(isa(a, CudaArray) ? (usegpu ? a : to_host(a)) : (usegpu ? CudaArray(a) : a))

function update(p::Param)
    initupdate(p)
    nz(p,:l1reg) && l1reg!(p.l1reg, p.data, p.diff)
    nz(p,:l2reg) && l2reg!(p.l2reg, p.data, p.diff)
    nz(p,:adagrad) && adagrad!(p.adagrad, p.ada, p.diff)
    nz(p,:learningRate,1f0) && (@in1! p.diff .* p.learningRate)
    nz(p,:momentum) && momentum!(p.momentum, p.mom, p.diff)
    nz(p,:nesterov) && nesterov!(p.nesterov, p.nes, p.diff)
    @in1! p.data .- p.diff
    nz(p,:maxnorm) && maxnorm!(p.maxnorm, p.data)
end

nz(p,n,v=0f0)=(isdefined(p,n) && (p.(n) != v))

function initupdate(p::Param)
    isdefined(p,:adagrad)  && (p.adagrad  > zero(p.adagrad))  && chksize(p, :ada, p.diff; fill=0f0)
    isdefined(p,:momentum) && (p.momentum > zero(p.momentum)) && chksize(p, :mom, p.diff; fill=0f0)
    isdefined(p,:nesterov) && (p.nesterov > zero(p.nesterov)) && chksize(p, :nes, p.diff; fill=0f0)
end

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
    if any(norms > p.maxnorm)
        scale = min(p.maxnorm ./ norms, 1)
        w *= scale
    end
end

