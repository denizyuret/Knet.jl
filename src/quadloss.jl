type QuadLoss <: LossLayer; y; QuadLoss()=new(); end

# Quadratic loss:
# l.y stores the model output.
# z is the desired output.
# Overwrites z with the gradient of quadratic loss wrt y, i.e. y-z
# J = 0.5*sum((yi-zi)^2)
# dJ/dy = y-z

forw(l::QuadLoss, x; o...)=(l.y=x)
back(l::QuadLoss, z::KUdense; returndx=true, o...)=(@assert issimilar(z,l.y); returndx && (quadlossback(l.y.arr,z.arr); z))
loss(l::QuadLoss, z::KUdense)=(@assert issimilar(z,l.y); quadlossloss(l.y.arr,z.arr))

quadlossback(y::Array, z::Array)=(nx=ccount(z); for i=1:length(z); z[i] = (y[i]-z[i])/nx; end)
quadlossloss(y::Array, z::Array)=(cost=zero(Float64); for i=1:length(z); cost += (y[i]-z[i])^2; end; 0.5*cost/ccount(z))

if GPU

quadlossloss(y::CudaArray, z::CudaArray)=quadlossloss(to_host(y), to_host(z))
quadlossback(y::CudaArray, z::CudaArray)=cudnnTransformTensor(1/ccount(y), y, -1/ccount(y), z)

end # if GPU
