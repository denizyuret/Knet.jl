using CUDArt,Knet

@knet function addtest(x; alpha=1, beta=1, init=nothing)
    b = par(;init=init)
    z = add(b,x; alpha=alpha, beta=beta)
end

@knet function multest(x; alpha=1, beta=1, init=nothing)
    w = par(;init=init)
    z = mul(w,x; alpha=alpha, beta=beta)
end

function rtest(;s1=nothing, s2=nothing, a=nothing, b=nothing, batchsize=8)
    global prog = (rand()<=0 ? addtest : multest)
    global dims1, dims2, ndims1, ndims2
    if s2 != nothing
        dims2 = s2
        ndims2 = length(s2)
    else
        ndims2 = 1
        while rand()<0.5 && ndims2<7
            ndims2 += 1
        end
        dims2 = ntuple(i->rand(1:10), ndims2)
    end
    if s1 != nothing
        dims1 = s1
        ndims1 = length(s1)
    else
        ndims1 = rand()<0.5 ? ndims2 : rand()<0.5 ? 1 : rand(1:ndims2)
        dims1 = ntuple(i->(rand()<0.5 ? dims2[i] : 1), ndims1)
        if prog==addtest && length(dims1)==1 # CUDNN_ADD_SAME_C exception
            dims1 = (length(dims2) == 1 ? dims2 : dims2[end:end])
        end
    end
    @show (dims1,dims2)
    global x = randn(dims2...,batchsize)
    global y = randn(dims2...,batchsize)
    global w = randn(dims1...)
    data = ItemTensor(x,y; batch=batchsize)

    global alpha = (a!=nothing ? a : rand()<0.5 ? 1 : rand()<0.5 ? -1 : rand()<0.5 ? randn() : 0)
    global beta  = (b!=nothing ? b : rand()<0.5 ? 1 : rand()<0.5 ? -1 : rand()<0.5 ? randn() : 0)
    @show (alpha,beta)
    global net = FNN(prog; alpha=alpha, beta=beta, init=w)
    @show net.net.op[end]

    gradcheck(net, data, quadloss)
end
