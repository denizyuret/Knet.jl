using BenchmarkTools, Knet, CUDA, Printf, Random
CUDA.allowscalar(true)
CUDA.rand(10)[1] # get the warning out of the way
pt = BenchmarkTools.prettytime

function kbench(f,x...; o...)
    seconds = 1.0
    global ff = f
    global ds = Base.dims2string.(size.(x))
    global xs = Param.(x)
    global ks = Param.(KnetArray.(x))
    global cs = Param.(CuArray.(x))
    global os = o
    global ns = []
    pr(c,b) = (@printf("%s:%-10s ", c, pt(minimum(b.times))); flush(stdout); push!(ns,length(b.times)))
    fn = @sprintf("%s(%s)", ff, join(ds,","))
    @printf("%-34s", fn); flush(stdout)
    ff(xs...;os...); pr("a1",@benchmark(ff(xs...;os...); seconds=seconds))
    ff(ks...;os...); pr("k1",@benchmark((ff(ks...;os...); synchronize()), seconds=seconds))
    ff(cs...;os...); pr("c1",@benchmark((ff(cs...;os...); synchronize()), seconds=seconds))
    @diff ff(xs...;os...)[1]; pr("a2",@benchmark((@diff (ff(xs...;os...)[1])), seconds=seconds))
    @diff ff(ks...;os...)[1]; pr("k2",@benchmark((@diff (ff(ks...;os...)[1])), seconds=seconds))
    @diff ff(cs...;os...)[1]; pr("c2",@benchmark((@diff (ff(cs...;os...)[1])), seconds=seconds))
    @printf("n=%d-%d\n",minimum(ns),maximum(ns)); flush(stdout)
end

B = 32

x1(x)=x[1]
drop(x)=dropout(x,0.5,drop=true)
y = randn(Float32,1000,B)
# kbench(identity,y)
# kbench(x1,y)
# kbench(sum,y)
# kbench(drop,y)

# kbench(logp, y; dims=1)
# kbench(softmax, y; dims=1)
# kbench(logsumexp, y; dims=1)

nll1(x;a)=nll(x,a)
accuracy1(x;a)=accuracy(x,a)
a = rand(1:size(y,1),size(y,2))
# kbench(nll1,y;a=a)
# kbench(accuracy1,y;a=a)

bce1(t;b)=bce(t,b)
logistic1(t;b)=logistic(t,b)
t = randn(Float32,1000)
t01 = rand((0,1),1000)
t11 = rand((1,1),1000)
# kbench(bce1, t; b=t01)
# kbench(logistic1, t; b=t11)

eludot(x)=elu.(x)
reludot(x)=relu.(x)
seludot(x)=selu.(x)
sigmdot(x)=sigm.(x)
x = rand(Float32,2048,B)
# kbench(eludot,x)
# kbench(reludot,x)
# kbench(seludot,x)
# kbench(sigmdot,x)

w = rand(Float32,1000,2048)
# kbench(*,w,x)

adddot(x,y)=(x .+ y)
b = rand(Float32,1000)
# kbench(adddot,y,b)

x = rand(Float32,14,14,256,B)
w = rand(Float32,3,3,256,256)
y = conv4(w,x;padding=1)
z = pool(y)
# kbench(conv4,w,x)
# kbench(deconv4,w,y)
# kbench(pool,y)
# kbench(unpool,z)
# kbench(mat,x)

m = bnmoments()
p = bnparams(eltype(x),size(x,3))
bn(x,p;m)=(m=(typeof(m.mean)===typeof(x) ? m : bnmoments()); batchnorm(x,m,p))
# kbench(bn, x, p; m=m) ## ERROR

k = rand(Float32,64,256,256,B)
q = rand(Float32,256,64,256,B)
# kbench(bmm,k,q)

rnntest(w,x;r)=(r.w=w; r(x))
r = RNN(256,256; atype=Array{Float32})
x = rand(Float32,256,B,256)
w = r.w.value
# kbench(rnntest,w,x;r=r)

embed(e;s)=e[:,s]
e = rand(Float32,256,10000)
s = rand(1:10000,256,256)
kbench(embed,e;s=s)

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

