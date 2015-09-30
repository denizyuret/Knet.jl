using CUDArt, Knet, Base.Test
include("isapprox.jl")
@time include(Pkg.dir("Knet/test/mnist.jl"))

ytype(X::DataType)=
    (X <: KUsparse{Array} ? KUdense{Array} :
     X <: KUsparse{CudaArray} ? KUdense{CudaArray} :
     X <: SparseMatrixCSC ? Array : X)

net=nothing
xtrn=xtst=nothing
ytrn=ytst=nothing
w0 = w1 = b0 = b1 = nothing

for X in (
          SparseMatrixCSC{Float32,Int32},
          KUsparse{Array},
          KUsparse{CudaArray},
          Array, 
          CudaArray,
          KUdense{Array},
          KUdense{CudaArray},
          )
    Y = ytype(X)
    @show (X,Y)
    net = [Mmul(10), Bias(), PercLoss()]
    setparam!(net, average=true, init=initzero)
    # xtrn = convert(X, copy(MNIST.xtrn))
    xtst = convert(X, copy(MNIST.xtst))
    # ytrn = convert(Y, copy(MNIST.ytrn))
    ytst = convert(Y, copy(MNIST.ytst))
    # @show map(summary, (xtrn, ytrn, xtst, ytst))
    for i=1:3
        # train(net, xtrn, ytrn)
        train(net, xtst, ytst)
        println((i, accuracy(ytst, predict(net, xtst)), 
                 )) # accuracy(ytrn, predict(net, xtrn))))
    end
    if w0 == nothing
        w0 = convert(Array, net[1].w.arr)
        w1 = convert(Array, net[1].w.avg)
        b0 = convert(Array, net[2].b.arr)
        b1 = convert(Array, net[2].b.avg)
    else
        @test isapprox(w0, net[1].w.arr)
        @test isapprox(w1, net[1].w.avg)
        @test isapprox(b0, net[2].b.arr)
        @test isapprox(b1, net[2].b.avg)
    end
end # for X


# DEAD CODE:

# using Knet
# using Knet: accuracy
# @time require(Pkg.dir("Knet/test/mnist.jl"))

# Knet.gpu(false)
# xtrn = float32(255*MNIST.xtrn)
# xtst = float32(255*MNIST.xtst)
# ytrn = float32(MNIST.ytrn)
# ytst = float32(MNIST.ytst)
# strn = sparse(xtrn)
# stst = sparse(xtst)

# if true
# info("Dense input")
# xnet = Op[Perceptron(10)]
# @time for i=1:5
#     train(xnet, xtrn, ytrn)
#     println((i,
#              accuracy(ytst, predict(xnet, xtst)),
#              accuracy(ytrn, predict(xnet, xtrn))))
# end
# end # if false

# if true
# info("Sparse input")
# snet = Op[Perceptron(10)]
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

# # Knet.atype(CudaArray)
# Knet.gpu(true)
# cnet = Op[Perceptron(10)]
# @time for i=1:5
#     train(cnet, xtrn, ytrn; iters=100)
#     println((i,
#              accuracy(ytst, predict(cnet, xtst)),
#              accuracy(ytrn, predict(cnet, xtrn))))
# end
# # Knet.atype(Array)
# end # if false

# # Similar stuff needs to be defined for sparse cuda arrays.