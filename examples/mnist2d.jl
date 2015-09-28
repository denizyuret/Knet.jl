# Handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.

using Base.Test
using KUnet
isdefined(:MNIST) || include("mnist.jl")
setseed(42)
nbatch=100

dtrn = ItemTensor(MNIST.xtrn, MNIST.ytrn; batch=nbatch)
dtst = ItemTensor(MNIST.xtst, MNIST.ytst; batch=nbatch)

x0 = copy(dtrn.data[1])
y0 = copy(dtrn.data[2])

info("Testing simple mlp")

softmax() = quote
    x1 = input()
    x2 = soft(x1)
    x3 = softloss(x2)
end

layer(;n=1,f=nothing) = quote
    x1 = input()
    w1 = par($n,0)
    x2 = dot(w1,x1)
    b2 = par(0)
    x3 = add(b2,x2)
    y3 = $(symbol(f))(x3)
end

function mlp(loss, actf, hidden...)
    prog = quote
        x0 = input()
    end
    N = length(hidden)
    for n=1:N
        x1 = symbol("x$(n-1)")
        x2 = symbol("x$n")
        push!(prog.args, :($x2 = layer($x1; n=$(hidden[n]), f=$(n<N ? actf : loss))))
    end
    return prog
end

prog = mlp(softmax, relu, 64, 10)

net = Net(prog)

setopt!(net, lr=0.5)
@time for i=1:3
    @show (l,w,g) = train(net, dtrn; gclip=0, gcheck=100, getloss=true, getnorm=true, atol=0.01, rtol=0.001)
    @show (test(net, dtrn), accuracy(net, dtrn))
    @show (test(net, dtst), accuracy(net, dtst))
end

@test isequal(x0,dtrn.data[1])
@test isequal(y0,dtrn.data[2])
