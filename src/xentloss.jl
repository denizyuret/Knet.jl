type XentLoss <: LossLayer; y; XentLoss()=new(); end

# Cross entropy loss to use after an unnormalized layer.
# l.y is treated as unnormalized log probabilities output by the model.
# p has normalized probabilities from the answer key.
# Normalization is across the last dimension, i.e. sum(p[:,...,:,i])==1
# Overwrites p with the gradient of the loss wrt y, i.e. q-p:
#
# z = sum(exp(y))   ;; normalization constant
# q = exp(y)/z      ;; model probabilities
# logq = y - logz   ;; model (normalized) log prob
# dlogz/dy = q      
#
# J = (1/N) Σ[nc] -p[nc]*logq[nc]  ;; n=1..N: instance, c=1..C: class
#   = (1/N) Σ[nc] -p[nc]*(y[nc]-logz[n])
#   = (1/N) ((Σ[n] logz[n]) - (Σ[nc] p[nc]*y[nc]))
#
# dJ/dy[md] = (1/N) (q[md] - p[md])


forw(l::XentLoss, x; o...)=(l.y=x)
back(l::XentLoss, p::KUdense; returndx=true, o...)=(@assert issimilar(p,l.y); returndx && (xentlossback(l.y.arr,p.arr); p))
loss(l::XentLoss, p::KUdense)=(@assert issimilar(p,l.y); xentlossloss(l.y.arr,p.arr))

function xentlossback(y::Array, p::Array)
    (nd,nx) = size2(p)
    # cuda cannot handle allocation, we will overwrite y for compatibility
    # qz = similar(p, nd)
    for j=1:nx
        i1=(j-1)*nd+1; i2=j*nd
        z = zero(Float64)
        ymax = typemin(eltype(y)) # subtract ymax for numerical stability
        for i=i1:i2; y[i] > ymax && (ymax = y[i]); end
        for i=i1:i2; y[i] = exp(y[i]-ymax); z+=y[i]; end
        for i=i1:i2; y[i]/=z; p[i] = (y[i] - p[i])/nx; end
        #for i=i1:i2; z += (qz[i-i1+1] = exp(y[i]-ymax)); end
        #for i=i1:i2; p[i] = (qz[i-i1+1]/z - p[i])/nx; end
    end
    return p
end

function xentlossloss(y::Array, p::Array)
    cost = zero(Float64)
    (nd,nx) = size2(p)
    for j=1:nx
        i1=(j-1)*nd+1; i2=j*nd
        z = sumpy = zero(Float64)
        ymax = typemin(eltype(y))
        for i=i1:i2; y[i] > ymax && (ymax = y[i]); end
        for i=i1:i2; yi=y[i]-ymax; z += exp(yi); sumpy += p[i]*yi; end
        cost += (log(z) - sumpy)
    end
    return cost/nx
end

if GPU

xentlossloss(y::CudaArray, p::CudaArray)=xentlossloss(to_host(y), to_host(p))
xentlossback(y::CudaArray{Float32}, p::CudaArray{Float32})=((nd,nx)=size2(p);ccall((:xentloss32,libkunet),Void,(Cint,Cint,Ptr{Cfloat},Ptr{Cfloat}),nd,nx,y,p))
xentlossback(y::CudaArray{Float64}, p::CudaArray{Float64})=((nd,nx)=size2(p);ccall((:xentloss64,libkunet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble}),nd,nx,y,p))

end # if GPU
