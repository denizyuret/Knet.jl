# This is the adding problem from: Le, Q. V., Jaitly, N., & Hinton,
# G. E. (2015). A Simple Way to Initialize Recurrent Networks of
# Rectified Linear Units. arXiv preprint arXiv:1504.00941.

using ArgParse
using KUnet

function parse_commandline(a=ARGS)
    s = ArgParseSettings()
    @add_arg_table s begin
        "--epochs"
        help = "Number of epochs to train"
        arg_type = Int
        default = 20 # 100
        "--epochsize"
        help = "number of training examples per epoch"
        arg_type = Int
        default = 2000 # 10000
        "--batchsize"
        help = "minibatch size"
        arg_type = Int
        default = 16
        "--length"
        help = "length of the input sequence"
        arg_type = Int
        default = 10 # 100
        "--hidden"
        help = "number of hidden units"
        arg_type = Int
        default = 5 # 100
        "--lr"
        help = "learning rate"
        arg_type = Float64
        default =  0.05 # 0.01
        "--gclip"
        help = "gradient clip"
        arg_type = Float64
        default = 1.0 # 1.0
        "--gcheck"
        help = "gradient check"
        arg_type = Int
        default = 10 # 1.0
        "--type"
        help = "type of network"
        default = "irnn" # "lstm"
        "--fb"
        help = "forget gate bias"
        arg_type = Float64
        default =  1.0
        "--seed"
        help = "Random seed"
        arg_type = Int
        default = 1003
    end
    parse_args(a,s)
end

args = parse_commandline()
# args = parse_commandline(split("--epochsize 2000 --length 10 --hidden 5 --lr 0.05 --gc 0 --epochs 20 --seed 1003"))
# args = parse_commandline(split("--epochsize 10000 --test 2000 --length 100 --hidden 100 --lr 0.01 --gc 1.0 --epochs 100"))
println(args)
args["seed"] > 0 && setseed(args["seed"])

data = Adding(args["length"], args["batchsize"], args["epochsize"])

nx = 2
ny = 1
nh = args["hidden"]
net1 = (args["type"] == "irnn" ? IRNN(nh) :
        args["type"] == "lstm" ? LSTM(nh) : 
        error("Unknown network type "*args["type"]))
args["type"] == "lstm" && setparam!(net1.op[9]; init=fill!, initp=args["fb"])

net2 = Net(Mmul(ny), Bias(), QuadLoss())
# setparam!(net2.op[1]; init=randn!, initp=(0,0.001)) # TODO: paper uses 0.001, does it make a difference?

net = S2C(net1, net2)
setparam!(net; lr=args["lr"])

@time for epoch=1:args["epochs"]
    (l,maxw,maxg) = train(net, data; gclip=args["gclip"], gcheck=args["gcheck"])
    mse = 2*l
    println(tuple(epoch*data.epochsize,mse,maxw,maxg))
    flush(STDOUT)
end

:ok

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
# In Net size(xtrain)=(D,T,I) or xtrain=[(D,T1),(D,T2),...]
# i.e. we want each instance to be contiguous.
# In Net size(xbatch)=(D,B,T) or xbatch=[(D,B1),(D,B2),...]
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

# TODO:
# + minibatching (16)
# + dif = nothing, dif0 = array
# + profiling
# x separate into two networks
# + xfer train/predict to kunet (tforw? figure out right interface)
# + gradient clipping (1/10/100)
# x adadelta (not this paper)

# setparam!(net0; lr=args["lr"], gc=args["gc"])
# setparam!(net1; lr=args["lr"], gc=args["gc"])
# setparam!(net2; lr=args["lr"], gc=args["gc"])
# args["type"] == "lstm" && setparam!(net0.op[9]; init=fill!, initp=args["fb"])

# function gendata(ni, nt)
#     x = cell(ni)
#     y = cell(ni)
#     for i=1:ni
#         x[i] = cell(nt)
#         y[i] = cell(nt)
#         for t=1:nt
#             x[i][t] = Float32[rand(), 0]
#             y[i][t] = nothing
#         end
#         t1 = rand(1:nt)
#         t2 = rand(1:nt)
#         while t1==t2; t2 = rand(1:nt); end
#         x[i][t1][2] = 1
#         x[i][t2][2] = 1
#         y[i][nt] = Float32[x[i][t1][1]+x[i][t2][1]]
#     end
#     return (x,y)
# end

# # TODO: fix this so y[i] is not a sequence
# function batch(x, y, nb)
#     isempty(x) && return (x,y)
#     xx = Any[]
#     yy = Any[]
#     ni = length(x)    # assume x and y are same length
#     nt = length(x[1]) # assume all x[i] are same length
#     for i1=1:nb:ni
#         i2=min(ni,i1+nb-1)
#         xi = Any[]
#         yi = Any[]
#         for t=1:nt
#             xit = x[i1][t]
#             xt = similar(xit, tuple(size(xit)..., i2-i1+1))
#             for i=i1:i2; xt[:,i-i1+1] = x[i][t]; end
#             push!(xi, xt)
#             yit = y[i1][t]
#             if yit == nothing
#                 yt = nothing # assumes yit=nothing for all i
#             else
#                 yt = similar(yit, tuple(size(yit)..., i2-i1+1))
#                 for i=i1:i2; yt[:,i-i1+1] = y[i][t]; end
#             end
#             push!(yi, yt)
#         end
#         push!(xx, xi)
#         push!(yy, yi)
#     end
#     return(xx, yy)
# end
# net0 = (args["type"] == "irnn" ? Net(irnn(nh),quadlosslayer(ny)) :
#         args["type"] == "lstm" ? Net(lstm(nh),quadlosslayer(ny)) : 
#         error("Unknown network type "*args["type"]))
# setparam!(net; lr=args["lr"], gc=args["gc"])  # do a global gclip instead of per parameter
        # "--test"
        # help = "number of testing examples"
        # arg_type = Int
        # default = 2000
    # trnmse = tstmse = maxg = maxw = 0
    # for i=1:length(xtrn)
    #     (l,w,g) = train(net, xtrn[i], ytrn[i][end]; getloss=true, getnorm=true, gclip=args["gc"])
    #     trnmse += l
    #     w > maxw && (maxw = w)
    #     g > maxg && (maxg = g)
    # end
    # for i=1:length(xtst)
    #     tstmse += test(net, xtst[i], ytst[i][end])
    # end
    # trnmse = 2*trnmse/length(xtrn)
    # tstmse = 2*tstmse/length(xtst)
# ntrn = args["train"]
# ntst = args["test"]
# nt = args["length"]
# (xtst1,ytst1) = gendata(ntst, nt)
# (xtst,ytst) = batch(xtst1, ytst1, args["batch"])
# (xtrn1,ytrn1) = gendata(ntrn, nt)
# (xtrn,ytrn) = batch(xtrn1, ytrn1, args["batch"])

# gradcheck(net, xtrn[1], ytrn[1][end]; ncheck=100, rtol=.01, atol=.01)
    # gradcheck(deepcopy(net), xtrn[1], ytrn[1][end]; ncheck=10, rtol=.01, atol=.01)
    # gradcheck(deepcopy(net), xtrn[1], ytrn[1][end]; ncheck=typemax(Int), rtol=.01, atol=0.001)
# DONE: move batch somewhere else

# Sample run for debugging:
# include("adding.jl")
# Dict{AbstractString,Any}("epochs"=>20,"length"=>10,"hidden"=>5,"lr"=>0.05,"gc"=>0.0,"seed"=>1003,"epochsize"=>2000,"type"=>"irnn","batchsize"=>16,"fb"=>1.0)
# (2000,0.2367687518065747,3.4275942f0,3.3859315f0)
# (4000,0.15307395973077917,3.464923f0,4.7202573f0)
# (6000,0.14563310024940243,3.6585462f0,5.872257f0)
# (8000,0.13914305934002114,3.8016677f0,3.1696885f0)
# (10000,0.14323858796367964,3.8592012f0,3.2897224f0)
# (12000,0.1367174871237112,3.985711f0,3.6923754f0)
# (14000,0.13409161991421098,4.122253f0,3.546906f0)
# (16000,0.14308425651714993,4.29228f0,4.408668f0)
# (18000,0.13665285526663074,4.4640274f0,4.3833494f0)
# (20000,0.13405299006340385,4.6561775f0,5.563099f0)
# (:gc,1,1,-0.014068735f0,-0.012999516911804303)
# (:gc,3,2,1.0189252f0,1.0168674634770678)
# (22000,0.11635054438228343,4.6980543f0,4.930481f0)
# (:gc,3,1,-0.017460648f0,-0.014846227713860662)
# (24000,0.13276363812079117,4.8182077f0,6.6221495f0)
# (26000,0.11608889580161318,5.0058174f0,4.4876924f0)
# (28000,0.09768098581235139,5.2528496f0,3.7506382f0)
# (30000,0.08683158848625311,5.476772f0,5.6964407f0)
# (32000,0.07696116060071267,5.66774f0,4.3827868f0)
# (34000,0.06580346676474488,5.834197f0,3.7996128f0)
# (36000,0.0653655101321814,5.957285f0,4.3073344f0)
# (38000,0.059040538369345454,6.0668435f0,4.8617268f0)
# (:gc,1,1,-0.04241219f0,-0.03291861048638019)
# (:gc,3,1,-0.04151296f0,-0.040597419683763535)
# (40000,0.05559959733567962,6.150661f0,7.046003f0)
#  12.254361 seconds (24.96 M allocations: 1.034 GB, 2.07% gc time)
# :ok

# len	hidden	lr	gc	mse<0.1
# 2	1	0.3		4000
# 3	2	0.2		4000
# 5	2	0.1		8000
# 10	5	0.05		26000
# 20	10	0.03		80000
# 40	30	0.01		220000
# 60	40	0.01	10	440000
# 100	100	0.01	1	870000
# 100	100	0.01	10	960000
# 150	100	0.01	1	1390000
# 200	100	0.01	1	2080000
# 300	100	0.01	1	3860000
# 400	100	0.01	1	7400000 speed=400k/h, unstable

# TODO: check out unstability of 400.
# TODO: share results with authors.

# DONE: lstm does not work with 10x5 find out why: much larger --lr=1.0, also large --fb >= 1 helps.
# The following settings solve 10x5 in 12000 iterations:
# "hidden"=>5,"lr"=>0.7,"batchsize"=>16,"length"=>10,"gclip"=>1.0,"fb"=>100.0,"gcheck"=>10,"epochs"=>10,"seed"=>1003,"epochsize"=>2000,"type"=>"lstm")

