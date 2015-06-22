using KUnet
require(Pkg.dir("KUnet/test/mnist.jl"))

KUnet.gpu(false)
xtrn = MNIST.xtrn
xtst = MNIST.xtst
ytrn = MNIST.ytrn
ytst = MNIST.ytst
w0 = similar(ytst, size(ytst,1), 0)
d0 = 6f0
c0 = 1f0
g0 = 0.1f0
niter = 100
nbatch = 128
dense = copy
net = nothing
nc = size(ytrn,1)

for kernel in ((:klinear0, []),
               (:klinear, []),
               (:kpoly0, [c0,d0]),
               (:kpoly, [c0,d0]),
               (:kgauss0, [g0]),
               (:kgauss, [g0]))
    for data in (dense, sparse)
        xtrn = data(MNIST.xtrn)
        xtst = data(MNIST.xtst)
        net = Layer[KPerceptron(nc, KUnet.(kernel[1]), kernel[2])]
        gc(); @date train(net, xtrn, ytrn; iters=niter,batch=nbatch)
        gc(); @time println((kernel, data, size(net[1].s), 
                             accuracy(ytst, predict(net, xtst)),
                             # accuracy(ytrn, predict(net, xtrn)),
                             ))
    end
end

if false; info("KPerceptron+klinear2")
for i=1:2
ftrn = sparse(MNIST.xtrn)
ftst = sparse(MNIST.xtst)
@show fnet = Layer[KPerceptron(10, KUnet.klinear0, [0f0])]
    gc()
    @date train(fnet, ftrn, ytrn; iters=niter,batch=nbatch)
    gc()
    @time println((i, size(fnet[1].s), 
                   accuracy(ytst, predict(fnet, ftst)),
                   # accuracy(ytrn, predict(lnet, ltrn)),
                   )) 
end; end

if false; info("KPerceptron+klinear")
for i=1:2
ltrn = sparse(MNIST.xtrn)
ltst = sparse(MNIST.xtst)
@show lnet = Layer[KPerceptron(10, KUnet.klinear, [0f0])]
gc()
    @date train(lnet, ltrn, ytrn; iters=niter,batch=nbatch)
gc()
    @time println((i, size(lnet[1].s), 
                   accuracy(ytst, predict(lnet, ltst)),
                   # accuracy(ytrn, predict(lnet, ltrn)),
                   )) 
end; end

if false; info("KPerceptron+kgauss")
for i=1:2
qtrn = sparse(MNIST.xtrn)
qtst = sparse(MNIST.xtst)
@show qnet = Layer[KPerceptron(10, KUnet.kgauss, [g0])]
    @date train(qnet, qtrn, ytrn; iters=niter,batch=nbatch)
    @time println((i, size(qnet[1].s), 
                   accuracy(ytst, predict(qnet, qtst)),
                   # accuracy(ytrn, predict(qnet, qtrn)),
             ))
end; end

if false; info("KPerceptron+kgauss0")
for i=1:2
qtrn = sparse(MNIST.xtrn)
qtst = sparse(MNIST.xtst)
@show qnet = Layer[KPerceptron(10, KUnet.kgauss0, [g0])]
    @date train(qnet, qtrn, ytrn; iters=niter,batch=nbatch)
    @time println((i, size(qnet[1].s), 
                   accuracy(ytst, predict(qnet, qtst)),
                   # accuracy(ytrn, predict(qnet, qtrn)),
             ))
end; end

if false; info("KPerceptron+kpoly")
for i=1:2
ktrn = sparse(MNIST.xtrn)
ktst = sparse(MNIST.xtst)
@show knet = Layer[KPerceptron(10, KUnet.kpoly, [c0,d0])]
    @date train(knet, ktrn, ytrn; iters=niter,batch=nbatch)
    @time println((i, size(knet[1].s), 
                   accuracy(ytst, predict(knet, ktst)),
                   # accuracy(ytrn, predict(knet, ktrn)),
                   )) 
end; end

if false; info("KPerceptron+kpoly0")
for i=1:2
ktrn = sparse(MNIST.xtrn)
ktst = sparse(MNIST.xtst)
@show knet = Layer[KPerceptron(10, KUnet.kpoly, [c0,d0])]
    @date train(knet, ktrn, ytrn; iters=niter,batch=nbatch)
    @time println((i, size(knet[1].s), 
                   accuracy(ytst, predict(knet, ktst)),
                   # accuracy(ytrn, predict(knet, ktrn)),
                   )) 
end; end

if false; info("Perceptron (sparse)")
ptrn = sparse(MNIST.xtrn)
ptst = sparse(MNIST.xtst)
@show pnet = Layer[Perceptron(10)]
for i=1:1
    @date train(pnet, ptrn, ytrn; iters=niter,batch=nbatch)
    @time println((i,
                   accuracy(ytst, predict(pnet, ptst)),
                   # accuracy(ytrn, predict(pnet, ptrn)),
                   )) 
end; end

if false; info("Perceptron (dense)")
dtrn = copy(MNIST.xtrn)
dtst = copy(MNIST.xtst)
@show dnet = Layer[Perceptron(10)]
for i=1:1
    @date train(dnet, dtrn, ytrn; iters=niter,batch=nbatch)
    @time println((i,
                   accuracy(ytst, predict(dnet, dtst)),
                   # accuracy(ytrn, predict(dnet, dtrn)),
                   )) 
end; end

if false; info("Poly+PercLoss")
strn = sparse(MNIST.xtrn)
stst = sparse(MNIST.xtst)
@show snet = Layer[Poly(10;c=c0,d=d0,w=w0), PercLoss()]
for i=1:1
    @date train(snet, strn, ytrn; iters=niter,batch=nbatch)
    @time println((i, size(snet[1].s), 
                   accuracy(ytst, predict(snet, stst)),
                   # accuracy(ytrn, predict(snet, strn)),
                   ))
end; end

if false
@show knet[1].s == snet[1].s
@show knet[1].w0 == snet[1].v
@show knet[1].w2 == snet[1].w
end # if false

if false; info("Rbfk+PercLoss")
rtrn = sparse(MNIST.xtrn)
rtst = sparse(MNIST.xtst)
@show rnet = Layer[Rbfk(10;gamma=0.1), PercLoss()]
for i=1:1
    @date train(rnet, rtrn, ytrn; iters=niter,batch=nbatch)
    println((i, size(rnet[1].s), 
             accuracy(ytst, predict(rnet, rtst)),
             # accuracy(ytrn, predict(rnet, rtrn)),
             ))
end; end

if false; info("KPerceptron+kgauss")
qtrn = sparse(MNIST.xtrn)
qtst = sparse(MNIST.xtst)
@show qnet = Layer[KPerceptron(10, KUnet.kgauss, [g0])]
for i=1:1
    @date train(qnet, qtrn, ytrn; iters=niter,batch=nbatch)
    @time println((i, size(qnet[1].s), 
                   accuracy(ytst, predict(qnet, qtst)),
                   # accuracy(ytrn, predict(qnet, qtrn)),
             ))
end; end

if false
@show qnet[1].s == rnet[1].s
@show qnet[1].w0 == rnet[1].v
@show qnet[1].w2 == rnet[1].w
end # if false

if false; info("Poly+PercLoss+dense")
xnet = Layer[Poly(c=c0,d=d0,w=w0), PercLoss()]
for i=1:1
    @time train(xnet, xtrn, ytrn; iters=niter,batch=nbatch)
    println((i, size(xnet[1].s), 
             accuracy(ytst, predict(xnet, xtst)),
             )) # accuracy(ytrn, predict(xnet, xtrn))))
end; end

if false; info("Poly+PercLoss+GPU+dense") # mmul, hcat, ctranspose do not work
KUnet.gpu(true)
cnet = Layer[Poly(c=c0,d=d0,w=CudaArray(w0)), PercLoss()]
for i=1:1
    @time train(cnet, xtrn, ytrn; iters=niter,batch=nbatch)
    println((i, size(cnet[1].s), 
             accuracy(ytst, predict(cnet, xtst)),
             )) # accuracy(ytrn, predict(cnet, xtrn))))
end; end
