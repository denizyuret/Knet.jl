type LogpLoss <: LossLayer; y; LogpLoss()=new(); end
# copy(l::LogpLoss;o...)=LogpLoss()

# Cross entropy loss to use after the Logp layer.
# l.y should be normalized log probabilities output by the model.
# p has normalized probabilities from the answer key.
# Normalization is across the last dimension, i.e. sum(p[:,...,:,i])==1
# Overwrites p with the gradient of the loss wrt y, i.e. exp(y)-p:
#
# z = sum(exp(y))   ;; normalization constant (should be 1 here)
# q = exp(y)/z      ;; model probabilities
# logq = y - logz   ;; model (normalized) log prob
# dlogz/dy = q      
#
# J = (1/N) Σ[nc] -p[nc]*logq[nc]  ;; n=1..N: instance, c=1..C: class
#   = (1/N) Σ[nc] -p[nc]*(y[nc]-logz[n])
#   = (1/N) ((Σ[n] logz[n]) - (Σ[nc] p[nc]*y[nc]))
#   = (1/N) (Σ[nc] -p[nc]*y[nc])   ;; all logz are 0
#
# dJ/dy[md] = (1/N) (q[md] - p[md])

forw(l::LogpLoss, x; o...)=(l.y=x)

function back(l::LogpLoss, p; returndx=true, o...)
    @assert issimilar(p, l.y)
    returndx || return
    (st,nx) = size2(p)
    for i=1:length(p)
        p[i] = (exp(l.y[i])-p[i])/nx
    end
    return p
end

function loss(l::LogpLoss, p, y=l.y)
    @assert issimilar(p, y)
    (st,nx) = size2(p)
    cost = zero(Float64)
    for i=1:length(p)
        cost -= (p[i]*y[i])
    end
    return cost/nx
end


if GPU

loss(l::LogpLoss, p::AbstractCudaArray)=loss(l, to_host(p), to_host(l.y))

function back(l::LogpLoss, p::AbstractCudaArray{Float32}; returndx=true, o...)
    @assert issimilar(p, l.y)
    returndx || return
    (st,nx) = size2(p)
    ccall((:logploss32,libkunet),Void,(Cint,Cfloat,Ptr{Float32},Ptr{Float32}),length(p),1/nx,l.y,p)
    return p
end

function back(l::LogpLoss, p::AbstractCudaArray{Float64}; returndx=true, o...)
    @assert issimilar(p, l.y)
    returndx || return
    (st,nx) = size2(p)
    ccall((:logploss64,libkunet),Void,(Cint,Cdouble,Ptr{Float64},Ptr{Float64}),length(p),1/nx,l.y,p)
    return p
end
end # if GPU

