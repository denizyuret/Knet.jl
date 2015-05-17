type QuadLoss <: LossLayer; y; QuadLoss()=new(); end
copy(l::QuadLoss; o...)=QuadLoss()

# Quadratic loss:
# l.y stores the model output.
# z is the desired output.
# Overwrites z with the gradient of quadratic loss wrt y, i.e. y-z
# J = 0.5*sum((yi-zi)^2)
# dJ/dy = y-z

forw(l::QuadLoss, x; o...)=(l.y=x)

function back(l::QuadLoss, z; returndx=true, o...)
    @assert issimilar(z,l.y)
    returndx || return
    (st,nx) = size2(z)
    for i=1:length(z)
        z[i] = (l.y[i]-z[i])/nx
    end
    return z
end

function loss(l::QuadLoss, z)
    @assert issimilar(z,l.y)
    y = to_host(l.y)
    z = to_host(z)
    (st,nx) = size2(z)
    cost = zero(Float64)
    for i=1:length(z)
        cost += (y[i]-z[i])^2
    end
    return 0.5*cost/nx
end

if GPU

function back(l::QuadLoss, z::CudaArray; returndx=true, o...)
    @assert issimilar(z,l.y)
    returndx || return
    (st,nx) = size2(z)
    cudnnTransformTensor(1/nx, l.y, -1/nx, z)
end

end # if GPU
