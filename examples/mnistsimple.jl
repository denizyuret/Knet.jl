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

prog = quote
    x1 = input()
    w1 = par(64,0)
    z1 = dot(w1,x1)
    b1 = par(0)
    y1 = add(b1,z1)
    x2 = relu(y1)
    w2 = par(10,0)
    z2 = dot(w2,x2)
    b2 = par(0)
    y2 = add(b2,z2)
    ou = softmax(y2)
end

model = Net(prog)

# Train and evaluate

setopt!(model, lr=0.5)
for epoch=1:10
    (l,w,g) = train(model, dtrn)
    (l1,a1) = (test(model, dtrn), accuracy(model, dtrn))
    (l2,a2) = (test(model, dtst), accuracy(model, dtst))
    println("epoch=$epoch tstacc=$a2 trnacc=$a1 tstloss=$l2 trnloss=$l1 wnorm=$w gnorm=$g")
end
