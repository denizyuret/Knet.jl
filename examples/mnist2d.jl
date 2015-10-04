# Handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.

using Knet,ArgParse,Base.Test
isdefined(:MNIST) || include("mnist.jl")
include("mlp.jl")

function mnist2d(args=ARGS)
    info("Testing simple mlp on MNIST")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=42)
        ("--nbatch"; arg_type=Int; default=100)
        ("--epochs"; arg_type=Int; default=3)
        ("--xsparse"; action=:store_true)
        ("--ysparse"; action=:store_true)
    end
    isa(args, AbstractString) && (args=split(args))
    opts = parse_args(args,s)
    println(opts)
    for (k,v) in opts; @eval ($(symbol(k))=$v); end
    seed > 0 && setseed(seed)

    fx = (xsparse ? sparse : identity)
    fy = (ysparse ? sparse : identity)
    dtrn = ItemTensor(fx(MNIST.xtrn), fy(MNIST.ytrn); batch=nbatch)
    dtst = ItemTensor(fx(MNIST.xtst), fy(MNIST.ytst); batch=nbatch)

    prog = mlp(layers=(64,10), loss=softmax, actf=relu, winit=Gaussian(0,.01), binit=Constant(0))
    net = Net(prog)
    setopt!(net, lr=0.5)

    l=w=g=0
    for epoch=1:epochs
        (l,w,g) = train(net, dtrn; gclip=0, gcheck=100, getloss=true, getnorm=true, atol=0.01, rtol=0.001) # t:3053
        ltrn = test(net, dtrn)  # t:815
        atrn = accuracy(net, dtrn) # t:877
        ltst = 0 # test(net, dtst)
        atst = 0 # accuracy(net, dtst)
        @show (epoch,l,w,g,ltrn,atrn,ltst,atst)
    end
    return (l,w,g)
end

!isinteractive() && !isdefined(:load_only) && mnist2d(ARGS)

### SAMPLE RUN

# INFO: Loading MNIST...
#   5.736248 seconds (362.24 k allocations: 502.003 MB, 1.35% gc time)
# INFO: Testing simple mlp
# (l,w,g) = train(net,dtrn; gclip=0,gcheck=100,getloss=true,getnorm=true,atol=0.01,rtol=0.001) = (0.37387532f0,18.511799f0,2.8433793f0)
# (test(net,dtrn),accuracy(net,dtrn)) = (0.21288027f0,0.9327666666666666)
# (test(net,dtst),accuracy(net,dtst)) = (0.2148458f0,0.9289)
# (l,w,g) = train(net,dtrn; gclip=0,gcheck=100,getloss=true,getnorm=true,atol=0.01,rtol=0.001) = (0.14995994f0,22.26936f0,3.9932733f0)
# (test(net,dtrn),accuracy(net,dtrn)) = (0.13567321f0,0.9574)
# (test(net,dtst),accuracy(net,dtst)) = (0.14322147f0,0.9546)
# (l,w,g) = train(net,dtrn; gclip=0,gcheck=100,getloss=true,getnorm=true,atol=0.01,rtol=0.001) = (0.10628127f0,24.865437f0,3.5134742f0)
# (test(net,dtrn),accuracy(net,dtrn)) = (0.100041345f0,0.9681833333333333)
# (test(net,dtst),accuracy(net,dtst)) = (0.114785746f0,0.9641)
#  10.373502 seconds (11.14 M allocations: 557.680 MB, 1.25% gc time)
