using CUDArt, CUSPARSE, CUDNN, CUBLAS
#using Knet, 
#import Knet: forw, back, loss, ninputs, overwrites, back_reads_x, back_reads_y, gpu, axpb!, @gpu, issimilar, mul2!
#using Base.LinAlg: axpy!, scale!

include("util/gpu.jl")
include("util/cudart.jl")
include("util/curand.jl")
include("util/cusparse.jl")
include("util/linalg.jl")
include("model.jl")
include("op.jl")
include("op/actf.jl");
include("op/add.jl");
include("op/dot.jl");
include("op/input.jl");
include("op/loss.jl");
include("op/mul.jl");
include("op/par.jl");
include("net.jl")
include("net/initforw.jl")
include("net/initback.jl")
include("net/forw.jl")
include("net/back.jl")
include("net/util.jl")
include("update.jl")

using Base.Test
include("data.jl")
# include("util/dense.jl")
include("util/colops.jl")
isdefined(:MNIST) || include("../examples/mnist.jl")
setseed(42)
nbatch=100
f64(x)=map(Float64,x)

dtrn = ItemTensor(f64(MNIST.xtrn), f64(MNIST.ytrn); batch=nbatch)
dtst = ItemTensor(f64(MNIST.xtst), f64(MNIST.ytst); batch=nbatch)

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
    net = quote
        x0 = input()
    end
    N = length(hidden)
    for n=1:N
        x1 = symbol("x$(n-1)")
        x2 = symbol("x$n")
        push!(net.args, :($x2 = layer($x1; n=$(hidden[n]), f=$(n<N ? actf : loss))))
    end
    return net
end

net = Net(mlp(softmax, relu, 64, 10))
DBG = false

setopt!(net, lr=0.5)
@time for i=1:3
    @show (l,w,g) = train(net, dtrn; gclip=0, gcheck=1000, getloss=true, getnorm=true, atol=0.0001, rtol=0.0001)
    @show (test(net, dtrn), accuracy(net, dtrn))
    @show (test(net, dtst), accuracy(net, dtst))
end

@test isequal(x0,dtrn.data[1])
@test isequal(y0,dtrn.data[2])



# DBG = true
# nout = 2
# winit = rand(4,3)
# prog = quote
#     x = input()
#     w = par($nout, 0; init=Gaussian(1,3))
#     # w = par($winit)
#     y = dot(w,x)
#     r = relu(y)
#     z = qloss(r)
# end
# net1 = Net(prog)
# x = rand(3,5)
# forw(net1, x)

# function lstm2(; n=1)
#     quote
#         x  = input()
#         i  = lin2(x,h;n=$n,f=sigm)
#         f  = lin2(x,h;n=$n,f=sigm)
#         o  = lin2(x,h;n=$n,f=sigm)
#         g  = lin2(x,h;n=$n,f=tanh)
#         ig = mul(i,g)
#         fc = mul(f,c)
#         c  = add(ig,fc)
#         tc = tanh(c)
#         h  = mul(tc,o)
#     end
# end

# function lin2(; n=1,f=sigm)
#     quote
#         w1 = par($n,0)
#         x1 = input()
#         y1 = dot(w1,x1)
#         w2 = par($n,0)
#         x2 = input()
#         y2 = dot(w2,x2)
#         w3 = par(0)
#         x3 = add(y1,y2)
#         y3 = add(w3,x3)
#         # ou = symbol($f)(y3)
#         ou = relu(y3)
#     end
# end

# net = Net(lstm2(n=10))
# forw(net, x)
# forw(net, x)
# dy = convert(Array, net.out0[end])
# back(net, dy)

# s = sprand(3,5,.5)
# snet = Net(lstm2(n=10))
# forw(snet, s)
# forw(snet, s)
# sy = convert(SparseMatrixCSC, net.out0[end])
# # TODO: this is failing
# # back(snet, dy)

# # Op			out0	dif	size	flags

# # (1,Input)		7e00 	0	(3,5)	tosave,toincr,!toback
# # (2,Par)               8000 	2400	(10,3)	
# # (3,Dot,2,1)           8200+ 	1c00+	(10,5)	
# # (4,Par)               8400 	2000	(10,10)	
# # (5,Dot,4,38)          8800*	1c00+	(10,5)	
# # (6,Par)               8a00    1e00	(10,)	
# # (7,Add,3,5)           8800*	1c00+	(10,5)	
# # (8,Add,6,7)           8800*	1c00+	(10,5)	
# # (9,Knet.Relu,8)	8c00    1c00+	(10,5)	tosave
# # (10,Par)              8e00    1a00	(10,3)	
# # (11,Dot,10,1)         8800*	1200*	(10,5)	
# # (12,Par)              9000    1600	(10,10)	
# # (13,Dot,12,38)        8200+	1200*	(10,5)	
# # (14,Par)              9400    1400	(10,)	
# # (15,Add,11,13)        8200+	1200*	(10,5)	
# # (16,Add,14,15)        8200+	1200*	(10,5)	
# # (17,Knet.Relu,16)	9600    1200*	(10,5)	tosave
# # (18,Par)              9800    1000	(10,3)	
# # (19,Dot,18,1)         8200+	0800!	(10,5)	
# # (20,Par)              9a00    0c00	(10,10)	
# # (21,Dot,20,38)        8800*	0800!	(10,5)	
# # (22,Par)              9e00    0a00	(10,)	
# # (23,Add,19,21)        8800*	0800!	(10,5)	
# # (24,Add,22,23)        8800*	0800!	(10,5)	
# # (25,Knet.Relu,24)	a000    0800!	(10,5)	tosave
# # (26,Par)	        a200    0600	(10,3)	
# # (27,Dot,26,1)         8800*	fe00@	(10,5)	
# # (28,Par)              a400    0200	(10,10)	
# # (29,Dot,28,38)        8200+	fe00@	(10,5)	
# # (30,Par)              a800    0000	(10,)	
# # (31,Add,27,29)        8200+	fe00@	(10,5)	
# # (32,Add,30,31)        8200+	fe00@	(10,5)	
# # (33,Knet.Relu,32)	aa00    fe00@	(10,5)	tosave
# # (34,Mul,9,33)         8200+	fc00	(10,5)	
# # (35,Mul,17,36)        8800*	fa00	(10,5)	
# # (36,Add,34,35)	ac00    f800	(10,5)	tosave,toincr,tozero,tmp=f400
# # (37,Knet.Tanh,36)	ae00    f600	(10,5)	tosave
# # (38,Mul,37,25)	b000	f200	(10,5)	tosave,toincr,tozero,tmp=f400

# # ops: 38

# # out0: 22 unique registers, 12 par, 8 tosave, 2 temp

# # dif: 20 unique registers, 12 par, 2 toincr

# # tmp: where is tmp for par? only needed if sequence.
