using Base.Test
using CUDArt
using Knet
include(Pkg.dir("Knet/test/mnist.jl"))
sparse32{T}(a::Array{T})=convert(SparseMatrixCSC{T,Int32}, a)

xtrn = MNIST.xtrn
xtst = MNIST.xtst
ytrn = MNIST.ytrn
ytst = MNIST.ytst
w0 = similar(ytst, size(ytst,1), 0)
d0 = 6.0
c0 = 1.0
g0 = 0.1
niter = 100
nbatch = 128
net = nothing
nc = size(ytrn,1)


for ker in (
            (:kpoly, [c0,d0]),
            (:kgauss, [g0]),
            (:klinear, nothing),
            (:perceptron, nothing),
            )
    aa = ss = nothing
    for prc in (
                :double,
                :single,
                )
        for fmt in (
                    :sparse,
                    :dense, 
                    )
            for loc in (
                        :cpu, 
                        :gpu,
                        )
                # loc == :gpu && ker[1] == :perceptron && continue
                println("\n$ker, $prc, $fmt, $loc")
                Knet.gpu(loc == :gpu)
                for p in (:xtrn, :xtst, :ytrn, :ytst); @eval $p=copy(MNIST.$p); end
                prc == :double && (for p in (:xtrn, :xtst, :ytrn, :ytst); @eval $p=convert(Array{Float64},$p); end)
                fmt == :sparse && (for p in (:xtrn, :xtst); @eval $p=sparse32($p); end)

                xtrn,ytrn=xtst,ytst # DBG: For quick results

                net = (ker[1] == :perceptron ? 
                       Op[Mmul(nc;average=true,init=initzero), PercLoss()] :
                       Op[KPerceptron(nc, Knet.(ker[1]), ker[2])])
                gc(); @date train(net, xtrn, ytrn; iters=niter,batch=nbatch)
                @date a = accuracy(ytst, predict(net, xtst))
                s = isdefined(net[1],:s) ? size(net[1].s) : 0
                @show (s,a)
                aa == nothing && (aa=a)
                ss == nothing && (ss=s)
                @test aa == a
                @test ss == s
            end
        end
    end
end


# with naive hcat!:

# (:klinear0,nothing), cpu, dense
# 2015-06-23T03:19:20 train(net,xtrn,ytrn; iters=niter,batch=nbatch)
# elapsed time: 0.780812858 seconds (789688232 bytes allocated, 42.90% gc time)
# ((:klinear0,nothing),:cpu,:dense,(784,2926),0.877)
# elapsed time: 0.366831263 seconds (120308196 bytes allocated, 11.45% gc time)

# (:klinear0,nothing), cpu, sparse
# 2015-06-23T03:19:22 train(net,xtrn,ytrn; iters=niter,batch=nbatch)
# elapsed time: 4.665061582 seconds (1148766840 bytes allocated, 10.10% gc time)
# ((:klinear0,nothing),:cpu,:sparse,(784,2926),0.877)
# elapsed time: 5.582400282 seconds (864266820 bytes allocated, 6.32% gc time)

# with sparse hcat!:

# (:klinear0,nothing), cpu, sparse
# 2015-06-23T03:22:54 train(net,xtrn,ytrn; iters=niter,batch=nbatch)
# elapsed time: 4.136324843 seconds (834224992 bytes allocated, 8.03% gc time)
# ((:klinear0,nothing),:cpu,:sparse,(784,2926),0.877)
# elapsed time: 5.153346267 seconds (864266820 bytes allocated, 6.83% gc time)



# if false; info("KPerceptron+klinear2")
# for i=1:2
# ftrn = sparse(MNIST.xtrn)
# ftst = sparse(MNIST.xtst)
# @show fnet = Op[KPerceptron(10, Knet.klinear0, [0f0])]
#     gc()
#     @date train(fnet, ftrn, ytrn; iters=niter,batch=nbatch)
#     gc()
#     @time println((i, size(fnet[1].s), 
#                    accuracy(ytst, predict(fnet, ftst)),
#                    # accuracy(ytrn, predict(lnet, ltrn)),
#                    )) 
# end; end

# if false; info("KPerceptron+klinear")
# for i=1:2
# ltrn = sparse(MNIST.xtrn)
# ltst = sparse(MNIST.xtst)
# @show lnet = Op[KPerceptron(10, Knet.klinear, [0f0])]
# gc()
#     @date train(lnet, ltrn, ytrn; iters=niter,batch=nbatch)
# gc()
#     @time println((i, size(lnet[1].s), 
#                    accuracy(ytst, predict(lnet, ltst)),
#                    # accuracy(ytrn, predict(lnet, ltrn)),
#                    )) 
# end; end

# if false; info("KPerceptron+kgauss")
# for i=1:2
# qtrn = sparse(MNIST.xtrn)
# qtst = sparse(MNIST.xtst)
# @show qnet = Op[KPerceptron(10, Knet.kgauss, [g0])]
#     @date train(qnet, qtrn, ytrn; iters=niter,batch=nbatch)
#     @time println((i, size(qnet[1].s), 
#                    accuracy(ytst, predict(qnet, qtst)),
#                    # accuracy(ytrn, predict(qnet, qtrn)),
#              ))
# end; end

# if false; info("KPerceptron+kgauss0")
# for i=1:2
# qtrn = sparse(MNIST.xtrn)
# qtst = sparse(MNIST.xtst)
# @show qnet = Op[KPerceptron(10, Knet.kgauss0, [g0])]
#     @date train(qnet, qtrn, ytrn; iters=niter,batch=nbatch)
#     @time println((i, size(qnet[1].s), 
#                    accuracy(ytst, predict(qnet, qtst)),
#                    # accuracy(ytrn, predict(qnet, qtrn)),
#              ))
# end; end

# if false; info("KPerceptron+kpoly")
# for i=1:2
# ktrn = sparse(MNIST.xtrn)
# ktst = sparse(MNIST.xtst)
# @show knet = Op[KPerceptron(10, Knet.kpoly, [c0,d0])]
#     @date train(knet, ktrn, ytrn; iters=niter,batch=nbatch)
#     @time println((i, size(knet[1].s), 
#                    accuracy(ytst, predict(knet, ktst)),
#                    # accuracy(ytrn, predict(knet, ktrn)),
#                    )) 
# end; end

# if false; info("KPerceptron+kpoly0")
# for i=1:2
# ktrn = sparse(MNIST.xtrn)
# ktst = sparse(MNIST.xtst)
# @show knet = Op[KPerceptron(10, Knet.kpoly, [c0,d0])]
#     @date train(knet, ktrn, ytrn; iters=niter,batch=nbatch)
#     @time println((i, size(knet[1].s), 
#                    accuracy(ytst, predict(knet, ktst)),
#                    # accuracy(ytrn, predict(knet, ktrn)),
#                    )) 
# end; end

# if false; info("Perceptron (sparse)")
# ptrn = sparse(MNIST.xtrn)
# ptst = sparse(MNIST.xtst)
# @show pnet = Op[Perceptron(10)]
# for i=1:1
#     @date train(pnet, ptrn, ytrn; iters=niter,batch=nbatch)
#     @time println((i,
#                    accuracy(ytst, predict(pnet, ptst)),
#                    # accuracy(ytrn, predict(pnet, ptrn)),
#                    )) 
# end; end

# if false; info("Perceptron (dense)")
# dtrn = copy(MNIST.xtrn)
# dtst = copy(MNIST.xtst)
# @show dnet = Op[Perceptron(10)]
# for i=1:1
#     @date train(dnet, dtrn, ytrn; iters=niter,batch=nbatch)
#     @time println((i,
#                    accuracy(ytst, predict(dnet, dtst)),
#                    # accuracy(ytrn, predict(dnet, dtrn)),
#                    )) 
# end; end

# if false; info("Poly+PercLoss")
# strn = sparse(MNIST.xtrn)
# stst = sparse(MNIST.xtst)
# @show snet = Op[Poly(10;c=c0,d=d0,w=w0), PercLoss()]
# for i=1:1
#     @date train(snet, strn, ytrn; iters=niter,batch=nbatch)
#     @time println((i, size(snet[1].s), 
#                    accuracy(ytst, predict(snet, stst)),
#                    # accuracy(ytrn, predict(snet, strn)),
#                    ))
# end; end

# if false
# @show knet[1].s == snet[1].s
# @show knet[1].w0 == snet[1].v
# @show knet[1].w2 == snet[1].w
# end # if false

# if false; info("Rbfk+PercLoss")
# rtrn = sparse(MNIST.xtrn)
# rtst = sparse(MNIST.xtst)
# @show rnet = Op[Rbfk(10;gamma=0.1), PercLoss()]
# for i=1:1
#     @date train(rnet, rtrn, ytrn; iters=niter,batch=nbatch)
#     println((i, size(rnet[1].s), 
#              accuracy(ytst, predict(rnet, rtst)),
#              # accuracy(ytrn, predict(rnet, rtrn)),
#              ))
# end; end

# if false; info("KPerceptron+kgauss")
# qtrn = sparse(MNIST.xtrn)
# qtst = sparse(MNIST.xtst)
# @show qnet = Op[KPerceptron(10, Knet.kgauss, [g0])]
# for i=1:1
#     @date train(qnet, qtrn, ytrn; iters=niter,batch=nbatch)
#     @time println((i, size(qnet[1].s), 
#                    accuracy(ytst, predict(qnet, qtst)),
#                    # accuracy(ytrn, predict(qnet, qtrn)),
#              ))
# end; end

# if false
# @show qnet[1].s == rnet[1].s
# @show qnet[1].w0 == rnet[1].v
# @show qnet[1].w2 == rnet[1].w
# end # if false

# if false; info("Poly+PercLoss+dense")
# xnet = Op[Poly(c=c0,d=d0,w=w0), PercLoss()]
# for i=1:1
#     @time train(xnet, xtrn, ytrn; iters=niter,batch=nbatch)
#     println((i, size(xnet[1].s), 
#              accuracy(ytst, predict(xnet, xtst)),
#              )) # accuracy(ytrn, predict(xnet, xtrn))))
# end; end

# if false; info("Poly+PercLoss+GPU+dense") # mmul, hcat, ctranspose do not work
# Knet.gpu(true)
# cnet = Op[Poly(c=c0,d=d0,w=CudaArray(w0)), PercLoss()]
# for i=1:1
#     @time train(cnet, xtrn, ytrn; iters=niter,batch=nbatch)
#     println((i, size(cnet[1].s), 
#              accuracy(ytst, predict(cnet, xtst)),
#              )) # accuracy(ytrn, predict(cnet, xtrn))))
# end; end
