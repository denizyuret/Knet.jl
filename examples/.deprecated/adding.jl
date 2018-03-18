# This is the adding problem from: Le, Q. V., Jaitly, N., & Hinton,
# G. E. (2015). A Simple Way to Initialize Recurrent Networks of
# Rectified Linear Units. arXiv preprint arXiv:1504.00941.
# Usage: julia adding.jl [opts], use --help for a full list of opts.

module Adding
using Knet, AutoGrad, ArgParse, JLD

function main(args=ARGS)
    info("Adding problem from Le et al. 2015.")
    isa(args, AbstractString) && (args=split(args))
    opts = parse_commandline(args) # TODO: get option processing back in here
    println(opts)
    opts["seed"] > 0 && setseed(opts["seed"])
    global data = Data(opts["length"], opts["batchsize"], opts["epochsize"])
    if opts["load"] != nothing
        global net = load(opts["load"],"net")
    else
        # global net = S2C(p1, p2; rnn=eval(parse(opts["nettype"])), hidden=opts["hidden"], winit=Gaussian(0,opts["winit"]), fbias=opts["fbias"])
        global net = compile(:adding; rnn=symbol(opts["nettype"]), hidden=opts["hidden"], winit=Gaussian(0,opts["winit"]), fbias=opts["fbias"])
    end
    # TODO: can pass opts in directly to compile as hash if keys were symbols: parse_args has an as_symbols option.
    # TODO: winit should probably be Xavier, at least give the user the option
    setp(net; lr=opts["lrate"])
    if opts["nosharing"]
        setp(net, :forwoverwrite, false)
        setp(net, :backoverwrite, false)
    end
    mse = 0; l=[0f0,0f0]; m=[0f0,0f0]
    for epoch=1:opts["epochs"]
        train(net, data, quadloss; gclip=opts["gclip"], losscnt=fill!(l,0), maxnorm=fill!(m,0))
        mse = 2*l[1]/l[2]
        println(tuple(epoch*data.epochsize,mse,m...))
        opts["gcheck"] > 0 && gradcheck(net,f->gradloss(f,data,quadloss;grad=true),f->gradloss(f,data,quadloss);gcheck=opts["gcheck"])
        flush(STDOUT)
    end
    return (mse, m...)
end

@knet function adding(x; rnn=nothing, hidden=0, o...)
    h = rnn(x; o..., out=hidden)
    if predict
        return wb(h; o..., out=1)
    end
end

function train(f, data, loss; gclip=0, losscnt=nothing, maxnorm=nothing)
    reset!(f)
    for (x,ygold) in data
        if ygold == nothing
            sforw(f, x; predict=false)
        else
            ypred = sforw(f, x; predict=true)
            losscnt[1] += loss(ypred, ygold); losscnt[2] += 1
            sback(f, ygold, loss)
            while !stack_isempty(f); sback(f); end
            g = gnorm(f); g > maxnorm[2] && (maxnorm[2]=g)
            gscale = (g > gclip > 0 ? gclip/g : 1)
            update!(f; gscale=gscale)
            w = wnorm(f); w > maxnorm[1] && (maxnorm[1]=w) # TODO: take this maxnorm calculation elsewhere, have a statistics optional function? just be smart about scaling?  batch-normalization, msra etc.
            reset!(f)
        end
    end
end

function gradloss(f, data, loss; grad=false, seed=42)
    data_rng = data.rng
    data.rng = MersenneTwister(seed)
    reset!(f)
    myforw = grad ? sforw : forw
    loss1 = 0
    for (x,ygold) in data
        if ygold == nothing
            myforw(f, x; predict=false)
        else
            ypred = myforw(f, x; predict=true)
            loss1 = loss(ypred, ygold)
            if grad
                sback(f, ygold, loss)
                while !stack_isempty(f); sback(f); end
            end
            break
        end
    end
    data.rng = data_rng
    return loss1
end

# Data generator:

import Base: start, next, done

type Data; len; batchsize; epochsize; batch; sum; cnt; rng;
    Data(len, batchsize, epochsize; rng=Base.GLOBAL_RNG) =
    new(len, batchsize, epochsize, zeros(Float32,2,batchsize),
        zeros(Float32,1,batchsize), zeros(Int,1,batchsize), rng)
end

# state keeps track of instance number, time step
start(a::Data)=(0,0)

done(a::Data,s)=(s[1] >= a.epochsize)

function next(a::Data, s)
    (n,t) = s
    t == 0 && (fill!(a.sum, 0); fill!(a.cnt, 0))
    rand!(a.rng, sub(a.batch,1,:))
    fill!(sub(a.batch,2,:),0)
    togo = a.len - t
    for b=1:a.batchsize
        if (a.cnt[b]==0 ? rand(a.rng) <= 2/togo :
            a.cnt[b]==1 ? rand(a.rng) <= 1/togo : false)
            a.batch[2,b] = 1
            a.cnt[b] += 1
            a.sum[b] += a.batch[1,b]
        end
    end
    if t+1 < a.len
        return ((a.batch,nothing), (n,t+1))
    else
        return ((a.batch,a.sum), (n+a.batchsize,0))
    end
end

function parse_commandline(args)
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
        "--lrate"
        help = "learning rate"
        arg_type = Float64
        default = 0.05 # 0.01
        "--gclip"
        help = "gradient clip"
        arg_type = Float64
        default = 0.0 # 1.0
        "--gcheck"
        help = "gradient check"
        arg_type = Int
        default = 0
        "--nettype"
        help = "type of network"
        default = "irnn" # "lstm"
        "--load"
        help = "load network from file"
        "--fbias"
        help = "forget gate bias (for lstm)"
        arg_type = Float64
        default = 1.0
        "--winit"
        help = "stdev for weight initialization (for irnn)"
        arg_type = Float64
        default =  0.01 # 0.001
        "--nosharing"
        help = "Do not share register arrays (this is for debugging, should not change results)"
        action = :store_true
        "--seed"
        help = "Random seed"
        arg_type = Int
        default = 1003
    end
    parse_args(args,s)
end

!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)

end #module

### SAMPLE RUNS: Mon Sep 28 21:22:01 PDT 2015

# [dy_052@hpc3013 examples]$ julia adding.jl
# Dict{AbstractString,Any}("hidden"=>5,"batchsize"=>16,"lrate"=>0.05,"length"=>10,"gclip"=>0.0,"std"=>0.01,"gcheck"=>10,"fbias"=>1.0,"epochs"=>20,"seed"=>1003,"epochsize"=>2000,"type"=>"irnn")
# (:gc,3,5,0.006502123f0,0.007450580596923828)
# (2000,0.22126968f0,3.401929f0,5.9513392f0)
# (4000,0.15322119f0,3.428556f0,4.774887f0)
# (6000,0.14666896f0,3.5303187f0,5.8686156f0)
# (8000,0.14103648f0,3.6803625f0,3.3791587f0)
# (10000,0.14484541f0,3.730668f0,3.6132479f0)
# (:gc,3,1,-0.008531229f0,-0.007376074790954634)
# (12000,0.13800602f0,3.8211298f0,3.4911914f0)
# (14000,0.13632888f0,3.8948088f0,3.0174415f0)
# (:gc,3,5,0.08060618f0,0.07353723049163863)
# (16000,0.14532445f0,4.0158386f0,4.1214657f0)
# (:gc,3,1,-0.07339329f0,-0.07551163434982083)
# (18000,0.13509625f0,4.1566772f0,3.2232072f0)
# (20000,0.12674719f0,4.329744f0,3.4179578f0)
# (:gc,1,5,0.072262116f0,0.07338821887969761)
# (22000,0.10952723f0,4.54435f0,6.0535564f0)
# (24000,0.106625326f0,4.769523f0,3.0753422f0)
# (26000,0.09111189f0,4.863088f0,3.6007364f0)
# (28000,0.08520666f0,4.994016f0,5.3230906f0)
# (30000,0.07867393f0,5.157688f0,6.0482206f0)
# (:gc,3,5,-0.09625465f0,-0.10021030902863652)
# (32000,0.07547985f0,5.239566f0,4.534141f0)
# (:gc,3,2,0.12882976f0,0.12787058949471927)
# (34000,0.06994344f0,5.2877007f0,5.1183004f0)
# (:gc,3,2,0.4618018f0,0.46066939830785103)
# (:gc,3,5,0.10286887f0,0.09141862392426543)
# (36000,0.05994114f0,5.384079f0,4.1663117f0)
# (38000,0.06648293f0,5.5462112f0,5.588353f0)
# (40000,0.048857126f0,5.6036315f0,3.805253f0)
#  16.701477 seconds (22.66 M allocations: 1.070 GB, 2.03% gc time)


# [dy_052@hpc3013 examples]$ julia adding.jl --type lstm
# Dict{AbstractString,Any}("hidden"=>5,"batchsize"=>16,"lrate"=>0.05,"length"=>10,"gclip"=>0.0,"std"=>0.01,"gcheck"=>10,"fbias"=>1.0,"epochs"=>20,"seed"=>1003,"epochsize"=>2000,"type"=>"lstm")
# (:gc,10,4,-0.0032901722f0,-0.002384185791015619)
# (:gc,10,6,-0.0020495693f0,-0.0011920928955078197)
# (:gc,10,7,0.0009976054f0,-0.0)
# (:gc,10,10,0.002146461f0,0.0011920928955078095)
# (:gc,12,5,0.008184996f0,0.0095367431640625)
# (:gc,13,4,-0.0014661846f0,-0.0005960464477539098)
# (2000,0.25688365f0,3.5912254f0,1.1217515f0)
# (4000,0.16381074f0,3.5942483f0,0.2719825f0)
# (6000,0.16333067f0,3.6215296f0,0.4068982f0)
# (8000,0.1633528f0,3.6199408f0,0.30277357f0)
# (10000,0.16167283f0,3.6377945f0,0.3591256f0)
# (12000,0.16095969f0,3.6270704f0,0.26166877f0)
# (14000,0.15849201f0,3.6007977f0,0.3542457f0)
# (16000,0.17288947f0,3.6203938f0,0.28349283f0)
# (18000,0.16657862f0,3.6240203f0,0.3288658f0)
# (20000,0.16593297f0,3.6399243f0,0.31581247f0)
# (22000,0.15971881f0,3.6533332f0,0.2592598f0)
# (24000,0.17853853f0,3.6475596f0,0.36688146f0)
# (26000,0.16586855f0,3.6920733f0,0.2798261f0)
# (28000,0.16403076f0,3.6743417f0,0.37097052f0)
# (30000,0.16771112f0,3.7095623f0,0.39885145f0)
# (32000,0.16923295f0,3.7148843f0,0.25710136f0)
# (34000,0.16505243f0,3.7452762f0,0.30730715f0)
# (36000,0.17146099f0,3.7212203f0,0.33892983f0)
# (38000,0.1655073f0,3.7557068f0,0.32211268f0)
# (40000,0.16552721f0,3.7773042f0,0.34974277f0)
#  55.710653 seconds (87.87 M allocations: 4.044 GB, 2.16% gc time)



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

# import Knet: forw, back, ninputs, param, similar!, gpu, initforw, initback, setparam!, update, loss, axpy! # push, pop, get1
# import Knet: backprop, train, predict, nz
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

# DONE:
# + minibatching (16)
# + dif = nothing, dif0 = array
# + profiling
# x separate into two networks
# + xfer train/predict to knet (tforw? figure out right interface)
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

# # DONE: fix this so y[i] is not a sequence
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
# julia> include("adding.jl")
# Dict{AbstractString,Any}("hidden"=>5,"batchsize"=>16,"lrate"=>0.05,"length"=>10,"gclip"=>0.0,"std"=>0.01,"gcheck"=>10,"fbias"=>1.0,"epochs"=>20,"seed"=>1003,"epochsize"=>2000,"type"=>"irnn")
# (2000,0.22126971078821514,3.401929f0,5.9513392f0)
# (4000,0.15322119775290655,3.428556f0,4.774887f0)
# (6000,0.14666901898846646,3.5303187f0,5.8686156f0)
# (8000,0.1410364930989558,3.6803625f0,3.3791587f0)
# (10000,0.14484541128223494,3.730668f0,3.6132479f0)
# (:gc,3,1,-0.008531229f0,-0.0073469345807098295)
# (12000,0.13800603687082819,3.8211298f0,3.4911914f0)
# (14000,0.13632891371942893,3.8948088f0,3.0174415f0)
# (:gc,3,5,0.08060618f0,0.07355656634899832)
# (16000,0.1453244475960048,4.0158386f0,4.1214657f0)
# (:gc,3,1,-0.07339329f0,-0.07550548616563685)
# (18000,0.13509626036176464,4.1566772f0,3.2232072f0)
# (20000,0.1267472031369284,4.329744f0,3.4179578f0)
# (:gc,1,5,0.072262116f0,0.07338319164773356)
# (22000,0.10952723081752598,4.54435f0,6.0535564f0)
# (24000,0.10662533053077042,4.769523f0,3.0753422f0)
# (26000,0.09111188449872992,4.863088f0,3.6007364f0)
# (28000,0.08520665419544599,4.994016f0,5.3230906f0)
# (30000,0.07867394052687623,5.157688f0,6.0482206f0)
# (:gc,3,5,-0.09625465f0,-0.10007152013714596)
# (32000,0.07547986407922849,5.239566f0,4.534141f0)
# (:gc,3,2,0.12882976f0,0.1279206117033168)
# (34000,0.06994342621574566,5.2877007f0,5.1183004f0)
# (:gc,3,2,0.4618018f0,0.460615519841718)
# (:gc,3,5,0.10286887f0,0.09136574590230435)
# (36000,0.059941143911885625,5.384079f0,4.1663117f0)
# (38000,0.06648293199506522,5.5462112f0,5.588353f0)
# (40000,0.048857121442293316,5.6036315f0,3.805253f0)
#  13.332866 seconds (23.97 M allocations: 1017.019 MB, 2.08% gc time)
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

# DONE: lstm does not work with 10x5 find out why: much larger --lr=1.0, also large --fb >= 1 helps.
# The following settings solve 10x5 in 12000 iterations:
# "hidden"=>5,"lr"=>0.7,"batchsize"=>16,"length"=>10,"gclip"=>1.0,"fb"=>100.0,"gcheck"=>10,"epochs"=>10,"seed"=>1003,"epochsize"=>2000,"type"=>"lstm")

    # args = parse_commandline()
    # args = parse_commandline(split("--epochsize 2000 --length 10 --hidden 5 --lrate 0.05 --gc 0 --epochs 20 --seed 1003"))
    # args = parse_commandline(split("--epochsize 10000 --test 2000 --length 100 --hidden 100 --lrate 0.01 --gc 1.0 --epochs 100"))

# Sample run for LSTM:
# julia> main(split("--type lstm"))
# Dict{AbstractString,Any}("hidden"=>5,"batchsize"=>16,"lrate"=>0.05,"length"=>10,"gclip"=>0.0,"std"=>0.01,"gcheck"=>10,"fbias"=>1.0,"epochs"=>20,"seed"=>1003,"epochsize"=>2000,"type"=>"lstm")
# (:gc,10,4,-0.0032901722f0,-0.002384185791015619)
# (:gc,10,6,-0.0020495693f0,-0.0011920928955078197)
# (:gc,10,7,0.0009976054f0,-0.0)
# (:gc,10,10,0.002146461f0,0.0011920928955078095)
# (:gc,12,5,0.008184996f0,0.0095367431640625)
# (:gc,13,4,-0.0014661846f0,-0.0005960464477539098)
# (2000,0.25688365f0,3.5912254f0,1.1217515f0)
# (4000,0.16381074f0,3.5942483f0,0.2719825f0)
# (6000,0.16333067f0,3.6215296f0,0.4068982f0)
# (8000,0.1633528f0,3.6199408f0,0.30277357f0)
# (10000,0.16167283f0,3.6377945f0,0.3591256f0)
# (12000,0.16095969f0,3.6270704f0,0.26166877f0)
# (14000,0.15849201f0,3.6007977f0,0.3542457f0)
# (16000,0.17288947f0,3.6203938f0,0.28349283f0)
# (18000,0.16657862f0,3.6240203f0,0.3288658f0)
# (20000,0.16593297f0,3.6399243f0,0.31581247f0)
# (22000,0.15971881f0,3.6533332f0,0.2592598f0)
# (24000,0.17853853f0,3.6475596f0,0.36688146f0)
# (26000,0.16586855f0,3.6920733f0,0.2798261f0)
# (28000,0.16403076f0,3.6743417f0,0.37097052f0)
# (30000,0.16771112f0,3.7095623f0,0.39885145f0)
# (32000,0.16923295f0,3.7148843f0,0.25710136f0)
# (34000,0.16505243f0,3.7452762f0,0.30730715f0)
# (36000,0.17146099f0,3.7212203f0,0.33892983f0)
# (38000,0.1655073f0,3.7557068f0,0.32211268f0)
# (40000,0.16552721f0,3.7773042f0,0.34974277f0)
#  55.892459 seconds (93.09 M allocations: 4.137 GB, 2.54% gc time)

# OLD GENERATOR FOR COMPARISON:

# type Adding1; len; batchsize; epochsize; b; x; y;
#     Adding1(len, batchsize, epochsize; o...)=new(len,batchsize,epochsize,Adding0(len,batchsize,epochsize; o...))
# end

# start(a::Adding1)=(0,0)
# done(a::Adding1,s)=(s[1] >= a.b.epochsize)

# function next(a::Adding1, s)
#     (n,t) = s
#     t == 0 && ((xy,n1) = next(a.b,n); (a.x,a.y)=xy)
#     t += 1
#     if t < a.b.len
#         return ((a.x[t], nothing), (n,t))
#     else
#         return ((a.x[t], a.y), (n+a.b.batchsize, 0))
#     end
# end

# type Adding0; len; batchsize; epochsize; rng;
#     Adding0(len, batchsize, epochsize; rng=MersenneTwister(0))=new(len, batchsize, epochsize, rng)
# end

# start(a::Adding0)=0

# done(a::Adding0,n)=(n >= a.epochsize)

# function next(a::Adding0, n)
#     nb = min(a.batchsize, a.epochsize-n)
#     x = [ vcat(rand(a.rng,Float32,1,nb),zeros(Float32,1,nb)) for t=1:a.len ]
#     y = Array(Float32,1,nb)
#     t1 = rand(a.rng,1:a.len,nb)
#     t2 = rand(a.rng,1:a.len,nb)
#     for b=1:nb
#         while t2[b]==t1[b]
#             t2[b]=rand(a.rng,1:a.len)
#         end
#         x[t1[b]][2,b]=1
#         x[t2[b]][2,b]=1
#         y[b] = x[t1[b]][1,b] + x[t2[b]][1,b]
#     end
#     return ((x,y), n+nb)
# end


    # p1 = (opts["nettype"] == "irnn" ? Net(irnn; out=opts["hidden"], winit=Gaussian(0,opts["winit"])) :
    #       opts["nettype"] == "lstm" ? Net(lstm; out=opts["hidden"], fbias=opts["fbias"]) :
    #       error("Unknown network type "*opts["nettype"]))
    # p2 = Net(wb; out=1, winit=Gaussian(0,opts["winit"]))

# @knet function p1(x; rnn=nothing, hidden=0, o...)
#     y = rnn(x; o..., out=hidden)
# end

# @knet function p2(x; o...)
#     y = wb(x; o..., out=1)
# end
