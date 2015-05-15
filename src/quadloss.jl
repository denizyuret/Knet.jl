type QuadLoss <: LossLayer; y; QuadLoss()=new(); end

# Quadratic loss:
# l.y stores the model output.
# z is the desired output.
# Overwrites z with the gradient of quadratic loss wrt y, i.e. y-z
# J = 0.5*sum((yi-zi)^2)
# dJ/dy = y-z

forw(l::QuadLoss, x; o...)=(l.y=x)

function back(l::QuadLoss, z; dx=true, o...)
    @assert size(z) == size(l.y)
    dx || return
    inst = size(z, ndims(z))
    for i=1:length(z)
        z[i] = (l.y[i]-z[i])/inst
    end
    return z
end

function loss(l::QuadLoss, z)
    @assert size(z) == size(l.y)
    inst = size(z, ndims(z))
    loss = zero(eltype(z))
    for i=1:length(z)
        loss += (l.y[i]-z[i])^2
    end
    return 0.5*loss/inst
end

if GPU
# TODO: float64 support, N-D arrays, return loss, probabilistic dy, check formula
# quadloss(y::CudaArray,dy::CudaArray)=ccall((:logploss,libkunet),Cfloat,(Cint,Cint,Cmat,Cmat),size(dy,1),size(dy,2),y,dy)
# quadloss(y::CudaArray,dy::CudaArray)=error("quadloss for gpu not implemented yet.")
back(l::QuadLoss, z::CudaArray; dx=true, o...)=error("quadloss for gpu not implemented yet.")
end # if GPU
