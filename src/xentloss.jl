type XentLoss <: LossLayer; y; XentLoss()=new(); end
# copy(l::XentLoss;o...)=XentLoss()

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

function back(l::XentLoss, p; returndx=true, o...)
    @assert issimilar(p, l.y)
    returndx || return
    (nd,nx) = size2(p)
    # cuda cannot handle allocation, we will overwrite l.y for compatibility
    # qz = similar(p, nd)
    for j=1:nx
        i1=(j-1)*nd+1; i2=j*nd
        z = zero(Float64)
        ymax = typemin(eltype(l.y)) # subtract ymax for numerical stability
        for i=i1:i2; l.y[i] > ymax && (ymax = l.y[i]); end
        for i=i1:i2; l.y[i] = exp(l.y[i]-ymax); z+=l.y[i]; end
        for i=i1:i2; l.y[i]/=z; p[i] = (l.y[i] - p[i])/nx; end
        #for i=i1:i2; z += (qz[i-i1+1] = exp(l.y[i]-ymax)); end
        #for i=i1:i2; p[i] = (qz[i-i1+1]/z - p[i])/nx; end
    end
    return p
end

function loss(l::XentLoss, p, y=l.y)
    @assert issimilar(p,y)
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

loss(l::XentLoss, p::CudaArray)=loss(l, to_host(p), to_host(l.y))

function back(l::XentLoss, p::CudaArray{Float32}; returndx=true, o...)
    @assert issimilar(p, l.y)
    returndx || return
    (nd,nx) = size2(p)
    ccall((:xentloss32,libkunet),Void,(Cint,Cint,Ptr{Float32},Ptr{Float32}),nd,nx,l.y,p)
    return p;
end

function back(l::XentLoss, p::CudaArray{Float64}; returndx=true, o...)
    @assert issimilar(p, l.y)
    returndx || return
    (nd,nx) = size2(p)
    ccall((:xentloss64,libkunet),Void,(Cint,Cint,Ptr{Float64},Ptr{Float64}),nd,nx,l.y,p)
    return p;
end
end # if GPU
