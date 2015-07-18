type LogpLoss <: LossLayer; y; LogpLoss()=new(); end

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
back(l::LogpLoss, p::KUdense; returndx=true, o...)=(@assert issimilar(p,l.y); returndx && (logplossback(l.y.arr,p.arr); p))
loss(l::LogpLoss, p::KUdense)=(@assert issimilar(p,l.y); logplossloss(l.y.arr,p.arr))

logplossback(y::Array, p::Array)=(nx = ccount(p); for i=1:length(p); p[i] = (exp(y[i])-p[i])/nx; end)
logplossloss(y::Array, p::Array)=(nx = ccount(p); cost = zero(Float64); for i=1:length(p); cost -= (p[i]*y[i]); end; cost/nx)

if GPU

logplossloss(y::CudaArray, p::CudaArray)=logplossloss(to_host(y), to_host(p))
logplossback(y::CudaArray{Float32}, p::CudaArray{Float32})=ccall((:logploss32,libkunet),Void,(Cint,Cdouble,Ptr{Cfloat},Ptr{Cfloat}),length(p),1/ccount(p),y,p)
logplossback(y::CudaArray{Float64}, p::CudaArray{Float64})=ccall((:logploss64,libkunet),Void,(Cint,Cdouble,Ptr{Cdouble},Ptr{Cdouble}),length(p),1/ccount(p),y,p)

end # if GPU

