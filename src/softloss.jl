type SoftLoss <: LossLayer; y; SoftLoss()=new(); end

# Cross entropy loss to use after the Soft layer.
# l.y should have normalized probabilities output by the model.
# p has normalized probabilities from the answer key.
# Normalization is across the last dimension, i.e. sum(p[:,...,:,i])==1
# Overwrites p with the gradient of the loss wrt y, i.e. 1-p/y
# Loss = -sum[p log(y)]

forw(l::SoftLoss, x; o...)=(l.y=x)

function back(l::SoftLoss, p; dx=true, o...)
    @assert size(p) == size(l.y)
    dx || return
    (st,nx) = size2(p)
    for i=1:length(p)
        p[i] = ((l.y[i]-p[i])/l.y[i])/nx
    end
    return p
end

function loss(l::SoftLoss, p)
    @assert size(p) == size(l.y)
    (st,nx) = size2(p)
    cost = zero(eltype(p))
    for i=1:length(p)
        cost -= (p[i]*log(l.y[i]))
    end
    return cost/nx
end

if GPU
# TODO: float64 support, N-D arrays, return loss, probabilistic dy, check formula
# softloss(y::CudaArray,dy::CudaArray)=ccall((:logploss,libkunet),Cfloat,(Cint,Cint,Cmat,Cmat),size(dy,1),size(dy,2),y,dy)
back(l::SoftLoss, p::CudaArray; o...)=error("softloss for gpu not implemented yet.")
end # if GPU
