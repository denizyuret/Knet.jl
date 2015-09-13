# This is the adding problem from: Le, Q. V., Jaitly, N., & Hinton,
# G. E. (2015). A Simple Way to Initialize Recurrent Networks of
# Rectified Linear Units. arXiv preprint arXiv:1504.00941.

# TODO:
# + minibatching (16)
# + dif = nothing, dif0 = array
# + profiling
# x separate into two networks
# + xfer train/predict to kunet (tforw? figure out right interface)
# - gradient clipping (1/10/100)
# - adadelta (not this paper)

using CUDArt
using KUnet
using ArgParse
# using Base.Test
# using KUnet: forwback1, forwback2, param, train2
include("../src/train.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--train"
        help = "number of training examples"
        arg_type = Int
        default = 100000
        "--test"
        help = "number of testing examples"
        arg_type = Int
        default = 10000
        "--length"
        help = "length of the input sequence"
        arg_type = Int
        default = 100
        "--hidden"
        help = "number of hidden units"
        arg_type = Int
        default = 100
        "--batch"
        help = "minibatch size"
        arg_type = Int
        default = 16
        "--type"
        help = "type of network"
        default = "irnn" # "lstm"
        "--lr"
        help = "learning rate"
        arg_type = Float64
        default =  0.01
        "--gc"
        help = "gradient clip"
        arg_type = Float64
        default =  100.0
        "--fb"
        help = "forget gate bias"
        arg_type = Float64
        default =  1.0
        "--epochs"
        help = "Number of epochs to train"
        arg_type = Int
        default = 100
        "--seed"
        help = "Random seed"
        arg_type = Int
        default = 1001
    end
    parse_args(s)
end

function gendata(ni, nt)
    x = cell(ni)
    y = cell(ni)
    for i=1:ni
        x[i] = cell(nt)
        y[i] = cell(nt)
        for t=1:nt
            x[i][t] = Float64[rand(), 0]
            y[i][t] = nothing
        end
        t1 = rand(1:nt)
        t2 = rand(1:nt)
        while t1==t2; t2 = rand(1:nt); end
        x[i][t1][2] = 1
        x[i][t2][2] = 1
        y[i][nt] = Float64[x[i][t1][1]+x[i][t2][1]]
    end
    return (x,y)
end

function maxnorm(r::RNN, mw=0, mg=0)
    for o in r.op
        p = param(o)
        p == nothing && continue
        w = vecnorm(p.arr)
        w > mw && (mw = w)
        g = vecnorm(p.diff)
        g > mg && (mg = g)
    end
    return (mw, mg)
end

args = parse_commandline()
args["seed"] > 0 && setseed(args["seed"])

nx = 2
ny = 1
nh = args["hidden"]
# net0 = (args["type"] == "irnn" ? RNN(irnn(nh),quadlosslayer(ny)) :
#         args["type"] == "lstm" ? RNN(lstm(nh),quadlosslayer(ny)) : 
#         error("Unknown network type "*args["type"]))
net1 = (args["type"] == "irnn" ? irnn(nh) :
        args["type"] == "lstm" ? lstm(nh) : 
        error("Unknown network type "*args["type"]))
net2 = quadlosslayer(ny)

# setparam!(net0; lr=args["lr"], gc=args["gc"])
setparam!(net1; lr=args["lr"], gc=args["gc"])
setparam!(net2; lr=args["lr"], gc=args["gc"])
setparam!(net2.op[1]; init=randn!, initp=(0,0.001))
# args["type"] == "lstm" && setparam!(net0.op[9]; init=fill!, initp=args["fb"])
args["type"] == "lstm" && setparam!(net1.op[9]; init=fill!, initp=args["fb"])

ntrn = args["train"]
ntst = args["test"]
nt = args["length"]
(xtst1,ytst1) = gendata(ntst, nt)
(xtst,ytst) = batch(xtst1, ytst1, args["batch"])
(xtrn1,ytrn1) = gendata(ntrn, nt)
(xtrn,ytrn) = batch(xtrn1, ytrn1, args["batch"])

@time for epoch=1:args["epochs"]
    trnloss = tstloss = maxg = maxw = 0
    for i=1:length(xtrn)
        (xi,yi) = (xtrn[i], ytrn[i][end])
        forw2(net1, net2, xi, yi)
        trnloss += loss(net2, yi) # loss = sqerr/2
        back2(net1, net2, xi, yi)
        (maxw,maxg) = maxnorm(net1, maxw, maxg)
        (maxw,maxg) = maxnorm(net2, maxw, maxg)
        update(net1)
        update(net2)
    end
    trnmse = 2*trnloss/length(xtrn)
    for i=1:length(xtst)
        (xi,yi) = (xtst[i], ytst[i][end])
        forw2(net1, net2, xi, yi; train=false)
        tstloss += loss(net2, yi)
    end
    tstmse = 2*tstloss/length(xtst)
    println(tuple(epoch,trnmse,tstmse,maxw,maxg))
    flush(STDOUT)
end


### DEAD CODE:


# nb = args["batch"]
# xb = Array(Float32, nx, nb, nt)
# xx = [ KUdense(CudaArray, Float32, nx, nb) for t=1:nt ]
# for b=1:nb:ntrn
#     e=min(ntrn, b + nb - 1)
#     permutedims!(xb, sub(xtrn,:,:,b:e), [1,3,2])
#     for t=1:nt
#         copy!(xx, 1, xb, 1+(t-1)*stride(xb,3), length(xx))
#     end
# end

# for b=1:nb:args["test"]

# end



    # x = zeros(Float32, nx, nt, ni)
    # y = zeros(Float32, ny, ni)
    # for i=1:ni
    #     t1 = rand(1:nt)
    #     t2 = rand(1:nt)
    #     while t1==t2; t2 = rand(1:nt); end
    #     x[2,t1,i] = x[2,t2,i] = 1
    #     y[i] = 0
    #     for t=1:nt
    #         x[1,t,i] = rand()
    #         x[2,t,i] == 1 && (y[i] += x[1,t,i])
    #     end
    # end
    # return (x,y)

        # if true
        #     nt = length(xi)
        #     y = convert(Array,r.out[nops(r)])
        #     a = convert(Array,yi[nt])
        #     e = sum((y-a).^2)/ccount(y)
        #     @show (e,2*err)
        # end

# Organization of the training set and training batches:
# D:dimensionality, I:instance, B:batch-instance, T:time
# In FFNN size(xtrain)=(D,I), size(xbatch)=(D,B) where D fastest
# In RNN size(xtrain)=(D,T,I) or xtrain=[(D,T1),(D,T2),...]
# i.e. we want each instance to be contiguous.
# In RNN size(xbatch)=(D,B,T) or xbatch=[(D,B1),(D,B2),...]
# i.e. we want each time-step to be contiguous.
# train->batch will need to do some shuffling

# CANCEL: splitting net into two parts may be more efficient
# this way forw calculation by the last three layers are wasted
# but this way we are testing the general input/output

# import KUnet: forw, back, ninputs, param, similar!, gpu, initforw, initback, setparam!, update, loss, axpy! # push, pop, get1
# import KUnet: backprop, train, predict, nz
# include("../src/rnn.jl")

# setparam!(net1; nesterov=0.01)
# net1.dbg = true

# net0.dbg = net1.dbg = net2.dbg = true
# setseed(42); e1 = forwback1(net0, xtst[1], ytst[1][end])
# setseed(42); e2 = forwback2(net1, net2, xtst[1], ytst[1][end])
# @test @show e1 == e2
# n0 = nops(net0); n1 = nops(net1); n2 = nops(net2)
# for n=n0:-1:1
#     @show n
#     p0 = param(net0.op[n])
#     p1 = (n<=n1 ? param(net1.op[n]) : param(net2.op[n-n1]))
#     p0 == nothing && p1 == nothing && continue
#     @test convert(Array, p1.arr)==convert(Array, p0.arr)
#     @test convert(Array, p1.diff)==convert(Array, p0.diff)
# end

# trnerr = trnerr0 = 0
# maxg = maxg0 = 0

    # c = gradcheck(net0, xtrn[1], ytrn[1]; ncheck=10)
    # c == nothing || println(tuple(:gradcheck, c...))
    # (trnerr0,maxg0) = train2(net0, xtrn, ytrn)
    # @test isapprox(trnerr, trnerr0; atol=1e-3)
    # @test isapprox(maxg, maxg0; atol=1e-2)
    # tsterr = test(net1, xtst, ytst)
    # println(tuple(epoch,sqrt(2trnerr),sqrt(2tsterr),maxgnorm))

