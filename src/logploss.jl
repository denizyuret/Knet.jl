type LogpLoss <: LossLayer; y; LogpLoss()=new(); end

# Cross entropy loss to use after the Logp layer.
# l.y should be normalized log probabilities output by the model.
# p has normalized probabilities from the answer key.
# Normalization is across the last dimension, i.e. sum(p[:,...,:,i])==1
# Overwrites p with the gradient of the loss wrt y, i.e. exp(y)-p:
# z = sum(exp(y))   ;; normalization constant (should be 1 here)
# q = exp(y)/z      ;; model probabilities
# logq = y - logz   ;; model (normalized) log prob
# Loss = J = -sum[p logq] = -sum[p (y-logz)] = logz - sum[py]
# dlogz/dy = q      
# dJ/dy = q - p

forw(l::LogpLoss, x; o...)=(l.y=x)

function back(l::LogpLoss, p; dx=true, o...)
    @assert size(p) == size(l.y)
    dx || return
    (st,nx) = size2(p)
    for i=1:length(p)
        p[i] = (exp(l.y[i])-p[i])/nx
    end
    return p
end

function loss(l::LogpLoss, p)
    @assert size(p) == size(l.y)
    (st,nx) = size2(p)
    cost = zero(eltype(p))
    for i=1:length(p)
        cost -= (p[i]*l.y[i])
    end
    return cost/nx
end

if GPU
# TODO: float64 support, N-D arrays, return loss, probabilistic dy, check formula
# logploss(y::CudaArray,dy::CudaArray)=ccall((:logploss,libkunet),Cfloat,(Cint,Cint,Cmat,Cmat),size(dy,1),size(dy,2),y,dy)
back(l::LogpLoss, x::CudaArray; o...)=error("Not implemented")
end # if GPU

