# Handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.
# Testing sparse arrays.

using Base.Test
using KUnet
using KUnet: params
setseed(42)
nbatch=100

### fixes

using CUDArt
using CUSPARSE

Base.isempty(a::CudaSparseMatrix) = (length(a) == 0)
Base.issparse(a::CudaSparseMatrix) = true
Base.vecnorm(a::CudaSparseMatrix) = vecnorm(a.nzVal)
Base.scale!(c,a::CudaSparseMatrix) = (scale!(c,a.nzVal); a)
Base.convert{T<:Array}(::Type{T},a::CudaSparseMatrix)=full(to_host(a))

function resizecopy!{T}(a::CudaVector{T}, b::Vector{T})
    resize!(a, length(b))
    copy!(a, b)
end

function Base.copy!{T}(a::CudaSparseMatrixCSC{T}, b::SparseMatrixCSC{T})
    a.dims = (b.m,b.n)
    a.nnz = convert(Cint, length(b.nzval))
    resizecopy!(a.colPtr, convert(Vector{Cint},b.colptr))
    resizecopy!(a.rowVal, convert(Vector{Cint},b.rowval))
    resizecopy!(a.nzVal, b.nzval)
    return a
end

using Base: dims2string
Base.summary(a::Union(KUdense,KUparam,CudaArray,CudaSparseMatrixCSC)) = string(dims2string(size(a)), " ", typeof(a))

### 

isdefined(:MNIST) || include("mnist.jl")
dtrn = ItemTensor(sparse(MNIST.xtrn), MNIST.ytrn; batch=nbatch)
dtst = ItemTensor(sparse(MNIST.xtst), MNIST.ytst; batch=nbatch)
# dtrn1 = ItemTensor(MNIST.xtrn, MNIST.ytrn; batch=nbatch, epoch=nbatch)
# dtst1 = ItemTensor(MNIST.xtst, MNIST.ytst; batch=nbatch, epoch=nbatch)
# dtrn = ItemTensor(sparse(MNIST.xtrn), sparse(MNIST.ytrn); batch=nbatch)
# dtst = ItemTensor(sparse(MNIST.xtst), sparse(MNIST.ytst); batch=nbatch)
# dtrn = ItemTensor(sprand(784,60000,0.2),sprand(10,60000,0.1); batch=nbatch)
# dtst = ItemTensor(sprand(784,10000,0.2),sprand(10,10000,0.1); batch=nbatch)
# dtrn = ItemTensor(sprand(784,6000,0.2),full(sprand(10,6000,0.1)); batch=nbatch)
# dtst = ItemTensor(sprand(784,1000,0.2),full(sprand(10,1000,0.1)); batch=nbatch)

x0 = copy(dtrn.data[1])
y0 = copy(dtrn.data[2])

info("Testing simple mlp with sparse arrays.")
net = Net(Mmul(64), Bias(), Relu(), Mmul(10), Bias(), XentLoss())
setparam!(net, lr=0.5)

# test(net, dtst)                 # to init weights
# net.out0[end]=Any[]
# net1 = deepcopy(net)

@time for i=1:3
    # TODO: gcheck does not work 
    @show (l,w,g) = train(net, dtrn; gclip=0, gcheck=0, getloss=true, getnorm=true)
    @show (test(net, dtrn), accuracy(net, dtrn))
    @show (test(net, dtst), accuracy(net, dtst))
    # @show (l,w,g) = train(net1, dtrn1; gclip=0, gcheck=0, getloss=true, getnorm=true)
    # train(mlp, dtrn.data[1], dtrn.data[2]; batch=nbatch)
    # @test all(map(isequal, params(net), params(mlp)))
    # println((i, accuracy(dtst.data[2], predict(mlp, dtst.data[1])),
    #             accuracy(dtrn.data[2], predict(mlp, dtrn.data[1]))))
end

@test isequal(x0,dtrn.data[1])
@test isequal(y0,dtrn.data[2])

