using CUDArt
using KUnet
require(Pkg.dir("KUnet/test/mnist.jl"))

KUnet.gpu(false)
xtrn = MNIST.xtrn
xtst = MNIST.xtst
ytrn = MNIST.ytrn
ytst = MNIST.ytst
w0 = similar(ytst, size(ytst,1), 0)
d0 = 6
c0 = 1
g0 = 0.1
niter = 100
nbatch = 128

if true
info("Poly+PercLoss")
strn = sparse(MNIST.xtrn)
stst = sparse(MNIST.xtst)
@show snet = Layer[Poly(10;c=c0,d=d0,w=w0), PercLoss()]
for i=1:1
    @date train(snet, strn, ytrn; iters=niter,batch=nbatch)
    @time println((i, size(snet[1].s), 
                   accuracy(ytst, predict(snet, stst)),
                   # accuracy(ytrn, predict(snet, strn)),
                   ))
end

info("KPerceptron+kpoly")
ktrn = sparse(MNIST.xtrn)
ktst = sparse(MNIST.xtst)
@show knet = Layer[KPerceptron(10, KUnet.kpoly, [c0,d0])]
for i=1:1
    @date train(knet, ktrn, ytrn; iters=niter,batch=nbatch)
    @time println((i, size(knet[1].s), 
                   accuracy(ytst, predict(knet, ktst)),
                   # accuracy(ytrn, predict(knet, ktrn)),
                   )) 
end

@show knet[1].s == snet[1].s
@show knet[1].w0 == snet[1].v
@show knet[1].w2 == snet[1].w

end # if false

if false

rtrn = sparse(MNIST.xtrn)
rtst = sparse(MNIST.xtst)
@show rnet = Layer[Rbfk(10;gamma=0.1), PercLoss()]
for i=1:1
    @date train(rnet, rtrn, ytrn; iters=niter,batch=nbatch)
    println((i, size(rnet[1].s), 
             accuracy(ytst, predict(rnet, rtst)),
             # accuracy(ytrn, predict(rnet, rtrn)),
             ))
end

qtrn = sparse(MNIST.xtrn)
qtst = sparse(MNIST.xtst)
@show qnet = Layer[KPerceptron(10, KUnet.krbf, [g0])]
for i=1:1
    @date train(qnet, qtrn, ytrn; iters=niter,batch=nbatch)
    println((i, size(qnet[1].s), 
             accuracy(ytst, predict(qnet, qtst)),
             # accuracy(ytrn, predict(qnet, qtrn)),
             ))

end

@show qnet[1].s == rnet[1].s
@show qnet[1].w0 == rnet[1].v
@show qnet[1].w2 == rnet[1].w

end # if false

if false # do sparse for now
xnet = Layer[Poly(c=c0,d=d0,w=w0), PercLoss()]
for i=1:1
    @time train(xnet, xtrn, ytrn; iters=niter,batch=nbatch)
    println((i, size(xnet[1].s), 
             accuracy(ytst, predict(xnet, xtst)),
             )) # accuracy(ytrn, predict(xnet, xtrn))))
end
end # if false

if false # mmul, hcat, ctranspose do not work
KUnet.gpu(true)
cnet = Layer[Poly(c=c0,d=d0,w=CudaArray(w0)), PercLoss()]
for i=1:1
    @time train(cnet, xtrn, ytrn; iters=niter,batch=nbatch)
    println((i, size(cnet[1].s), 
             accuracy(ytst, predict(cnet, xtst)),
             )) # accuracy(ytrn, predict(cnet, xtrn))))
end
end # if false
