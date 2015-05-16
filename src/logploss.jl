type LogpLoss <: LossLayer; y; LogpLoss()=new(); end

# Cross entropy loss to use after the Logp layer.
# l.y should be normalized log probabilities output by the model.
# p has normalized probabilities from the answer key.
# Normalization is across the last dimension, i.e. sum(p[:,...,:,i])==1
# Overwrites p with the gradient of the loss wrt y, i.e. exp(y)-p:
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

function back(l::LogpLoss, p; dx=true, o...)
    @assert issimilar(p, l.y)
    dx || return
    (st,nx) = size2(p)
    for i=1:length(p)
        p[i] = (exp(l.y[i])-p[i])/nx
    end
    return p
end

function loss(l::LogpLoss, p)
    @assert issimilar(p, l.y)
    p = to_host(p)
    y = to_host(l.y)
    (st,nx) = size2(p)
    cost = zero(Float64)
    for i=1:length(p)
        cost -= (p[i]*y[i])
    end
    return cost/nx
end

if GPU
function back(l::LogpLoss, p::CudaArray{Float32}; dx=true, o...)
    @assert issimilar(p, l.y)
    dx || return
    (st,nx) = size2(p)
    ccall((:slogploss,libkunet),Void,(Cint,Cfloat,Ptr{Float32},Ptr{Float32}),length(p),1/nx,l.y,p)
    return p
end

function back(l::LogpLoss, p::CudaArray{Float64}; dx=true, o...)
    @assert issimilar(p, l.y)
    dx || return
    (st,nx) = size2(p)
    ccall((:dlogploss,libkunet),Void,(Cint,Cdouble,Ptr{Float64},Ptr{Float64}),length(p),1/nx,l.y,p)
    return p
end
end # if GPU

