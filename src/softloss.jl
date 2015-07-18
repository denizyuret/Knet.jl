type SoftLoss <: LossLayer; y; SoftLoss()=new(); end

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
back(l::SoftLoss, p::KUdense; returndx=true, o...)=(@assert issimilar(p,l.y); returndx && (softlossback(l.y.arr,p.arr); p))
loss(l::SoftLoss, p::KUdense)=(@assert issimilar(p,l.y); softlossloss(l.y.arr,p.arr))

softlossback(y::Array,p::Array)=(nx=ccount(p); for i=1:length(p); p[i] = ((y[i]-p[i])/y[i])/nx; end)
softlossloss(y::Array,p::Array)=(cost=zero(Float64); for i=1:length(p); cost -= (p[i]*log(y[i])); end; cost/ccount(p))

if GPU

softlossloss(y::CudaArray, p::CudaArray)=softlossloss(to_host(y), to_host(p))
softlossback(y::CudaArray{Float32}, p::CudaArray{Float32})=ccall((:softloss32,libkunet),Void,(Cint,Cdouble,Ptr{Cfloat},Ptr{Cfloat}),length(p),1/ccount(p),y,p)
softlossback(y::CudaArray{Float64}, p::CudaArray{Float64})=ccall((:softloss64,libkunet),Void,(Cint,Cdouble,Ptr{Cdouble},Ptr{Cdouble}),length(p),1/ccount(p),y,p)

end # if GPU
