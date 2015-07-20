using CUDArt, KUnet
import KUnet: ytype
@time require(Pkg.dir("KUnet/test/mnist.jl"))

ytype(X::DataType)=
    (X <: KUsparse{Array} ? KUdense{Array} :
     X <: KUsparse{CudaArray} ? KUdense{CudaArray} :
     X <: Sparse{Array} ? Array :
     X <: Sparse{CudaArray} ? CudaArray :
     X <: SparseMatrixCSC ? Array : X)

for X in (
          Array, 
          CudaArray,
          SparseMatrixCSC,
          KUdense{Array},
          KUdense{CudaArray},
          KUsparse{Array},
          KUsparse{CudaArray},
          Sparse{Array},
          Sparse{CudaArray},
          )
    Y = ytype(X)
    @show (X,Y)
    # These two nets should behave the same:
    net0 = [Perceptron(10; bias=true, average=false), PercLoss()]
    net1 = [Mmul(10; init=initzero), Bias(), PercLoss()]
    xtrn = convert(X, copy(MNIST.xtrn))
    xtst = convert(X, copy(MNIST.xtst))
    ytrn = convert(Y, copy(MNIST.ytrn))
    ytst = convert(Y, copy(MNIST.ytst))
    @show map(summary, (xtrn, ytrn, xtst, ytst))
    for i=1:10
        train(net0, xtrn, ytrn)
        train(net1, xtrn, ytrn)
        @show accuracy(ytst, predict(net0, xtst))
        @show accuracy(ytst, predict(net1, xtst))
        @show accuracy(ytrn, predict(net0, xtrn))
        @show accuracy(ytrn, predict(net1, xtrn))
    end
end # for X


# DEAD CODE:

# using KUnet
# using KUnet: accuracy
# @time require(Pkg.dir("KUnet/test/mnist.jl"))

# KUnet.gpu(false)
# xtrn = float32(255*MNIST.xtrn)
# xtst = float32(255*MNIST.xtst)
# ytrn = float32(MNIST.ytrn)
# ytst = float32(MNIST.ytst)
# strn = sparse(xtrn)
# stst = sparse(xtst)

# if true
# info("Dense input")
# xnet = Layer[Perceptron(10)]
# @time for i=1:5
#     train(xnet, xtrn, ytrn)
#     println((i,
#              accuracy(ytst, predict(xnet, xtst)),
#              accuracy(ytrn, predict(xnet, xtrn))))
# end
# end # if false

# if true
# info("Sparse input")
# snet = Layer[Perceptron(10)]
# @time for i=1:5
#     train(snet, strn, ytrn)
#     println((i,
#              accuracy(ytst, predict(snet, stst)),
#              accuracy(ytrn, predict(snet, strn))))
# end
# end # if false

# # This is the stuff that needs to be defined for CudaArrays:
# # We probably need gpu kernels for forw, update
# using CUDArt
# import Base: .+, getindex, ./, -

# if false
# .+(a::CudaArray{Float32,2}, b::CudaArray{Float32,2})=a
# getindex(::CudaArray{Float32,2}, ::Int64, ::Int64)=0
# ./(a::CudaArray{Float32,2}, ::Int64)=a
# -(a::CudaArray{Float32,2}, b::CudaArray{Float32,2})=a

# # KUnet.atype(CudaArray)
# KUnet.gpu(true)
# cnet = Layer[Perceptron(10)]
# @time for i=1:5
#     train(cnet, xtrn, ytrn; iters=100)
#     println((i,
#              accuracy(ytst, predict(cnet, xtst)),
#              accuracy(ytrn, predict(cnet, xtrn))))
# end
# # KUnet.atype(Array)
# end # if false

# # Similar stuff needs to be defined for sparse cuda arrays.