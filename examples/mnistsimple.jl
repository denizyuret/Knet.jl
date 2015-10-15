using Knet

# Load MNIST

if !isdefined(:MNIST)
    include(Pkg.dir("Knet/examples/mnist.jl"))
end

# Prepare data

batchsize = 100
dtrn = Any[]
for i=1:batchsize:size(MNIST.xtrn,2)
    j=i+batchsize-1
    push!(dtrn, (MNIST.xtrn[:,i:j],MNIST.ytrn[:,i:j]))
end
dtst = Any[]
for i=1:batchsize:size(MNIST.xtst,2)
    j=i+batchsize-1
    push!(dtst, (MNIST.xtst[:,i:j],MNIST.ytst[:,i:j]))
end

# Prepare model

@knet function mnistsimple(x1)
    w1 = par(;dims=(64,0))
    z1 = dot(w1,x1)
    b1 = par(;dims=(0,))
    y1 = add(b1,z1)
    x2 = relu(y1)
    w2 = par(;dims=(10,0))
    z2 = dot(w2,x2)
    b2 = par(;dims=(0,))
    y2 = add(b2,z2)
    ou = soft(y2)
end

model = FNN(mnistsimple)

# Train and evaluate

setopt!(model, lr=0.5)
for epoch=1:10
    (l,w,g) = train(model, dtrn, softloss)
    (l1,a1) = (test(model, dtrn, softloss), 1-test(model, dtrn, zeroone))
    (l2,a2) = (test(model, dtst, softloss), 1-test(model, dtst, zeroone))
    println("epoch=$epoch tstacc=$a2 trnacc=$a1 tstloss=$l2 trnloss=$l1 wnorm=$w gnorm=$g")
end


### SAMPLE RUN

# epoch=1 tstacc=0.9362 trnacc=0.9387500000000001 tstloss=0.19834305 trnloss=0.19569032 wnorm=18.56881 gnorm=3.0631502
# epoch=2 tstacc=0.9589000000000001 trnacc=0.96365 tstloss=0.1326303 trnloss=0.12078557 wnorm=22.30527 gnorm=4.1069307
# epoch=3 tstacc=0.9663 trnacc=0.9712666666666667 tstloss=0.11089823 trnloss=0.09421845 wnorm=24.958153 gnorm=3.7882617
# epoch=4 tstacc=0.9686 trnacc=0.9760666666666669 tstloss=0.100210935 trnloss=0.078416765 wnorm=27.135939 gnorm=3.2001014
# epoch=5 tstacc=0.972 trnacc=0.9791333333333335 tstloss=0.092702836 trnloss=0.06718973 wnorm=28.992907 gnorm=2.4996655
# epoch=6 tstacc=0.9716 trnacc=0.9807333333333335 tstloss=0.09260971 trnloss=0.06175735 wnorm=30.643934 gnorm=2.2204309
# epoch=7 tstacc=0.9716 trnacc=0.9817333333333335 tstloss=0.09355682 trnloss=0.056200992 wnorm=32.127914 gnorm=2.1083603
# epoch=8 tstacc=0.9729 trnacc=0.9836000000000001 tstloss=0.09248012 trnloss=0.049878325 wnorm=33.49334 gnorm=2.1731966
# epoch=9 tstacc=0.9719 trnacc=0.9832333333333334 tstloss=0.09715143 trnloss=0.04935321 wnorm=34.74469 gnorm=1.9201015
# epoch=10 tstacc=0.9732000000000001 trnacc=0.9856166666666668 tstloss=0.09411778 trnloss=0.04231144 wnorm=35.89667 gnorm=1.8556117
