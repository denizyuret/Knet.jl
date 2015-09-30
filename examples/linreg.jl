using Knet

# Simple linear regression.

# Data generator:
import Base: start, next, done

type LinReg; w; batchsize; epochsize; noise; end

function LinReg(outputs,inputs,batchsize,epochsize,noise)
    LinReg(rand(outputs,inputs),batchsize,epochsize,noise)
end

function next(l::LinReg, n)
    (outputs, inputs) = size(l.w)
    x = rand(inputs, batchsize)
    y = l.w * x + scale(l.noise, randn(outputs, l.batchsize))
    return ((x,y), n+batchsize)
end

start(::LinReg)=0
done(l::LinReg,n)=(n >= l.epochsize)

# main()

# function main()
    inputs = 100
    outputs = 10
    batchsize = 20
    epochsize = 1000
    epochs = 10
    noise = 0.01
    data = LinReg(outputs, inputs, batchsize, epochsize, noise)
    prog = quote
        x = input()
        w = par($outputs,0)
        y = dot(w,x)
        z = qloss(y)
    end
    net = Net(prog)
     net.dbg = true
    for epoch = 1:epochs
        @show train(net, data)
    end
# end

