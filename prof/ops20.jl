using BenchmarkTools, Knet, CUDA, Printf, Random
CUDA.allowscalar(true)
CUDA.rand(10)[1] # get the warning out of the way
pt = BenchmarkTools.prettytime

function kbench(f,x...; o...)
    pr(c,b) = (@printf("%s:%-10s ", c, pt(minimum(b.times))); flush(stdout); push!(ns,length(b.times)))
    seconds = 1.0
    global ff = f
    global os = o
    global ns = []
    global ds = Base.dims2string.(size.(x)); fn = @sprintf("%s(%s)", ff, join(ds,",")); @printf("%-34s", fn); flush(stdout); ds = nothing; GC.gc(true);
    global xs = Param.(x); ff(xs...;os...); pr("a1",@benchmark(ff(xs...;os...); seconds=seconds)); xs = nothing; GC.gc(true);
    global ks = Param.(KnetArray.(x)); ff(ks...;os...); pr("k1",@benchmark((ff(ks...;os...); synchronize()), seconds=seconds)); ks = nothing; GC.gc(true);
    global cs = Param.(CuArray.(x)); ff(cs...;os...); pr("c1",@benchmark((ff(cs...;os...); synchronize()), seconds=seconds)); cs = nothing; GC.gc(true);
    global xs = Param.(x); @diff ff(xs...;os...)[1]; pr("a2",@benchmark((@diff (ff(xs...;os...)[1])), seconds=seconds)); xs = nothing; GC.gc(true);
    global ks = Param.(KnetArray.(x)); @diff ff(ks...;os...)[1]; pr("k2",@benchmark((@diff (ff(ks...;os...)[1])), seconds=seconds)); ks = nothing; GC.gc(true);
    global cs = Param.(CuArray.(x)); @diff ff(cs...;os...)[1]; pr("c2",@benchmark((@diff (ff(cs...;os...)[1])), seconds=seconds)); cs = nothing; GC.gc(true);
    @printf("n=%d-%d\n",minimum(ns),maximum(ns)); flush(stdout); ns = nothing; GC.gc(true);
end

B = 32

getindex1(x)=x[1]
drop(x)=dropout(x,0.5,drop=true)
y1 = randn(Float32,1000,B)
kbench(identity,y1)
kbench(getindex1,y1)
kbench(sum,y1)
kbench(drop,y1)

kbench(logp, y1; dims=1)
kbench(softmax, y1; dims=1)
kbench(logsumexp, y1; dims=1)

nll1(x;a)=nll(x,a)
accuracy1(x;a)=accuracy(x,a)
a1 = rand(1:size(y1,1),size(y1,2))
kbench(nll1,y1;a=a1)
kbench(accuracy1,y1;a=a1)

bce1(t;b)=bce(t,b)
logistic1(t;b)=logistic(t,b)
t1 = randn(Float32,1000)
t01 = rand((0,1),1000)
t11 = rand((1,1),1000)
kbench(bce1, t1; b=t01)
kbench(logistic1, t1; b=t11)

adddot(x,y)=(x .+ y)
x1 = rand(Float32,2048,B)
w1 = rand(Float32,1000,size(x1,1))
b1 = rand(Float32,size(w1,1))
kbench(*,w1,x1)
kbench(adddot,y1,b1)

x2 = rand(Float32,14,14,256,B)
w2 = rand(Float32,3,3,256,256)
y2 = conv4(w2,x2;padding=1)
z2 = pool(y2)
kbench(conv4,w2,x2)
kbench(deconv4,w2,y2)
kbench(pool,y2)
kbench(unpool,z2)
kbench(mat,x2)

eludot(x)=elu.(x)
reludot(x)=relu.(x)
seludot(x)=selu.(x)
sigmdot(x)=sigm.(x)
kbench(eludot,x2)
kbench(reludot,x2)
kbench(seludot,x2)
kbench(sigmdot,x2)

m2 = bnmoments()
p2 = bnparams(eltype(x2),size(x2,3))
bn(x,p;m)=(m=(typeof(m.mean)===typeof(x) ? m : bnmoments()); batchnorm(x,m,p))
kbench(bn, x2, p2; m=m2)

k3 = rand(Float32,64,256,256,B)
q3 = rand(Float32,256,64,256,B)
kbench(bmm,k3,q3)

r3 = RNN(256,256; atype=Array{Float32})
x3 = rand(Float32,256,B,256)
w3 = r3.w.value
rnntest(w,x;r=r3)=(r.w = w; r(x))
kbench(rnntest,w3,x3)

embed(e;s)=e[:,s]
e4 = rand(Float32,256,10000)
s4 = rand(1:10000,256,256)
kbench(embed,e4;s=s4)

# Using resnet dims as reference for conv.
# 224×224×3×1  = 150528 (1)
# 112×112×64×1 = 802816 (1)
# 55×55×64×1   = 193600 (3)
# 28×28×128×1  = 100352 (4)
# 14×14×256×1  = 50176  (23)
# 7×7×512×1    = 25088  (3)
# 1×1×2048×1   = 2048   (1)
# 1000×1       = 1000   (1)

# Using bert dims ÷ 4 as reference for bmm.
# 512x1024x1024x256

nothing
