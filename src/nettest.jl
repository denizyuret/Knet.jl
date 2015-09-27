using KUnet, CUDArt, CUSPARSE, CUDNN
import KUnet: forw, back, loss, ninputs, overwrites, back_reads_x, back_reads_y, gpu, axpb!, @gpu, issimilar, mul2!
using Base.LinAlg: axpy!, scale!

include("op/actf.jl");
include("op/add.jl");
include("op/dot.jl");
include("op/input.jl");
include("op/loss.jl");
include("op/mul.jl");
include("op/par.jl");
include("netcomp.jl")
include("netinit.jl")
include("netutil.jl")
include("netforw.jl")

DBG = true
outputs = 2
winit = rand(4,3)
prog = quote
    x = input()
    w = par($outputs, 0; init=Gaussian(1,3))
    # w = par($winit)
    y = dot(w,x)
    r = relu(y)
    z = qloss(r)
end
# net1 = Net(prog)
x = rand(3,5)
# forw(net1, x)

function lstm2(; n=1)
    quote
        x  = input()
        i  = lin2(x,h;n=$n,f=sigm)
        f  = lin2(x,h;n=$n,f=sigm)
        o  = lin2(x,h;n=$n,f=sigm)
        g  = lin2(x,h;n=$n,f=tanh)
        ig = mul(i,g)
        fc = mul(f,c)
        c  = add(ig,fc)
        tc = tanh(c)
        h  = mul(tc,o)
    end
end

function lin2(; n=1,f=sigm)
    quote
        w1 = par($n,0)
        x1 = input()
        y1 = dot(w1,x1)
        w2 = par($n,0)
        x2 = input()
        y2 = dot(w2,x2)
        w3 = par(0)
        x3 = add(y1,y2)
        y3 = add(w3,x3)
        # ou = symbol($f)(y3)
        ou = relu(y3)
    end
end

net = Net(lstm2(n=10))
forw(net, x)
forw(net, x)

# (1,Input)		7e00 	(3,5)	tosave,toincr,!toback
# (2,Par)               8000 	(10,3)	
# (3,Dot,2,1)           8200+ 	(10,5)	
# (4,Par)               8400 	(10,10)	
# (5,Dot,4,38)          8800*	(10,5)	
# (6,Par)               8a00    (10,)	
# (7,Add,3,5)           8800*	(10,5)	
# (8,Add,6,7)           8800*	(10,5)	
# (9,KUnet.Relu,8)	8c00    (10,5)	tosave
# (10,Par)              8e00    (10,3)	
# (11,Dot,10,1)         8800*	(10,5)	
# (12,Par)              9000    (10,10)	
# (13,Dot,12,38)        8200+	(10,5)	
# (14,Par)              9400    (10,)	
# (15,Add,11,13)        8200+	(10,5)	
# (16,Add,14,15)        8200+	(10,5)	
# (17,KUnet.Relu,16)	9600    (10,5)	tosave
# (18,Par)              9800    (10,3)	
# (19,Dot,18,1)         8200+	(10,5)	
# (20,Par)              9a00    (10,10)	
# (21,Dot,20,38)        8800*	(10,5)	
# (22,Par)              9e00    (10,)	
# (23,Add,19,21)        8800*	(10,5)	
# (24,Add,22,23)        8800*	(10,5)	
# (25,KUnet.Relu,24)	a000    (10,5)	tosave
# (26,Par)	        a200    (10,3)	
# (27,Dot,26,1)         8800*	(10,5)	
# (28,Par)              a400    (10,10)	
# (29,Dot,28,38)        8200+	(10,5)	
# (30,Par)              a800    (10,)	
# (31,Add,27,29)        8200+	(10,5)	
# (32,Add,30,31)        8200+	(10,5)	
# (33,KUnet.Relu,32)	aa00    (10,5)	tosave
# (34,Mul,9,33)         8200+	(10,5)	
# (35,Mul,17,36)        8800*	(10,5)	
# (36,Add,34,35)	ac00    (10,5)	tosave,toincr,tozero
# (37,KUnet.Tanh,36)	ae00    (10,5)	tosave
# (38,Mul,37,25)	b000	(10,5)	tosave,toincr,tozero

# 22 unique registers
# 12 par
# 8  tosave
# 2  temp
