using Knet, CUDArt, CUSPARSE
using Knet: loss, back

m = 10
n = 1000
dy = sparse(rand(1:m,n), 1:n, 1.0, m, n)
y = rand(m,n)
dx = similar(y)

for (yT,dyT) in ((Array,Array),
                 (Array,SparseMatrixCSC),
                 (CudaArray,CudaArray),
                 (CudaArray,CudaSparseMatrixCSC))
    display((:softloss, loss(SoftLoss(), convert(dyT, dy); y=convert(yT, y)), yT, dyT))
end

for (yT,dyT) in ((Array,Array),
                 (Array,SparseMatrixCSC),
                 (CudaArray,CudaArray),
                 (CudaArray,CudaSparseMatrixCSC))
    display((:softlossback, vecnorm(back(SoftLoss(), convert(dyT, dy); y=convert(yT, y), dx=convert(yT, dx))), yT, dyT))
end

for (yT,dyT) in ((Array,Array), (CudaArray,CudaArray))
    display((:quadloss, loss(QuadLoss(), convert(dyT, dy); y=convert(yT, y)), yT, dyT))
end

for (yT,dyT) in ((Array,Array), (CudaArray,CudaArray))
    display((:quadlossback, vecnorm(back(QuadLoss(), convert(dyT, dy); y=convert(yT, y), dx=convert(yT, dx))), yT, dyT))
end
