type SoftLoss <: LossLayer; y; SoftLoss()=new(); end
# copy(l::SoftLoss;o...)=SoftLoss()

# Cross entropy loss to use after the Soft layer.
# l.y should have normalized probabilities output by the model.
# p has normalized probabilities from the answer key.
# Normalization is across the last dimension, i.e. sum(p[:,...,:,i])==1
# Overwrites p with the gradient of the loss wrt y, i.e. 1-p/y

# Math:
#
# J = -Σ pi log yi		;; loss function
#   = -Σ pi log (yi/Σyj)	;; should make normalization explicit
#   = (-Σ pi log yi) + Σ pi log Σ yj
#   = (-Σ pi log yi) + log Σ yj
#
# ∂J/∂yk = -pk/yk + (1/Σ yj)
#        = -pk/yk + 1
#
# z = wx			;; z is the input to the soft layer
# yi = (exp zi) / (Σ exp zj)	;; y is the output of the soft layer
# ∂yi/∂zk = [(i=k)(exp zi)(Σ exp zj) - (exp zi)(exp zk)] / (Σ exp zj)^2
#         = (i=k) yi - yi yk
# ∂J/∂zk = Σ (∂J/∂yi)(∂yi/∂zk)	;; derivative wrt the input z
#        = Σ (1-pi/yi)((i=k) yi - yi yk)
#        = Σ ((i=k) yi - yi yk - (i=k) pi + pi yk)
#        = yk - pk - yk Σ (yi - pi)
#        = yk - pk


forw(l::SoftLoss, x; o...)=(l.y=x)

function back(l::SoftLoss, p; returndx=true, o...)
    @assert issimilar(p,l.y)
    returndx || return
    (nd,nx) = size2(p)
    for i=1:length(p)
        p[i] = ((l.y[i]-p[i])/l.y[i])/nx
    end
    return p
end

function loss(l::SoftLoss, p, y=l.y)
    @assert issimilar(p,y)
    (nd,nx) = size2(p)
    cost = zero(Float64)
    for i=1:length(p)
        cost -= (p[i]*log(y[i]))
    end
    return cost/nx
end

if GPU

loss(l::SoftLoss, p::AbstractCudaArray)=loss(l, to_host(p), to_host(l.y))

function back(l::SoftLoss, p::AbstractCudaArray{Float32}; returndx=true, o...)
    @assert issimilar(p, l.y)
    returndx || return
    (st,nx) = size2(p)
    ccall((:softloss32,libkunet),Void,(Cint,Cfloat,Ptr{Float32},Ptr{Float32}),length(p),1/nx,l.y,p)
    return p
end

function back(l::SoftLoss, p::AbstractCudaArray{Float64}; returndx=true, o...)
    @assert issimilar(p, l.y)
    returndx || return
    (st,nx) = size2(p)
    ccall((:softloss64,libkunet),Void,(Cint,Cdouble,Ptr{Float64},Ptr{Float64}),length(p),1/nx,l.y,p)
    return p
end
end # if GPU
