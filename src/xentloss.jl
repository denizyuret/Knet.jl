type XentLoss <: LossLayer; y; XentLoss()=new(); end

# Cross entropy loss to use after an unnormalized layer.
# l.y is treated as unnormalized log probabilities output by the model.
# p has normalized probabilities from the answer key.
# Normalization is across the last dimension, i.e. sum(p[:,...,:,i])==1
# Overwrites p with the gradient of the loss wrt y, i.e. q-p:
# z = sum(exp(y))   ;; normalization constant
# q = exp(y)/z      ;; model probabilities
# logq = y - logz   ;; model (normalized) log prob
# Loss = J = -sum[p logq] = -sum[p (y-logz)] = logz - sum[py]
# dlogz/dy = q      
# dJ/dy = q - p

forw(l::XentLoss, x; o...)=(l.y=x)

function back(l::XentLoss, p; dx=true, o...)
    @assert size(p) == size(l.y)
    dx || return
    inst = size(p, ndims(p))
    # loss = zero(eltype(p))
    nd = ndims(p)
    sz = size(p, nd)
    st = stride(p, nd)
    qz = similar(p, st)
    for j=1:sz
        i1=(j-1)*st+1; i2=j*st
        z = sumpy = zero(eltype(p))
        ymax = typemin(eltype(l.y))
        for i=i1:i2; l.y[i] > ymax && (ymax = l.y[i]); end
        for i=i1:i2; yi=l.y[i]-ymax; sumpy+=p[i]*yi; z += (qz[i-i1+1] = exp(yi)); end
        for i=i1:i2; p[i] = (qz[i-i1+1]/z - p[i])/inst; end
        # loss += (log(z) - sumpy)/inst
    end
    return p
end

function loss(l::XentLoss, p)
    @assert size(p) == size(l.y)
    inst = size(p, ndims(p))
    loss = zero(eltype(p))
    nd = ndims(p)
    sz = size(p, nd)
    st = stride(p, nd)
    for j=1:sz
        i1=(j-1)*st+1; i2=j*st
        z = sumpy = zero(eltype(p))
        ymax = typemin(eltype(l.y))
        for i=i1:i2; l.y[i] > ymax && (ymax = l.y[i]); end
        for i=i1:i2; yi=l.y[i]-ymax; sumpy += p[i]*yi; z += exp(yi); end
        loss += (log(z) - sumpy)
    end
    return loss/inst
end

if GPU
back(l::XentLoss, p::CudaArray; o...)=error("xentloss for gpu not implemented yet.")
loss(l::XentLoss, p::CudaArray)=error("xentloss for gpu not implemented yet.")
end # if GPU
