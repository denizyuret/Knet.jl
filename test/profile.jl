# fun		cpu	af	kn	kn+gc1	kn+gc2
# 1 mul		0.94	0.56	0.56	0.56	0.56
# 2 bias	1.05	0.56	0.60	0.59	0.59
# 3 max		1.34	0.56	0.63	0.62	0.62
# 4 mul		1.44	0.74	0.76	0.75	0.75
# 5 bias	1.48	0.75	0.90	0.78	0.78
# 6 sub		1.49	0.81	1.05	0.82	0.81
# 7 sq		1.62	0.93	1.19	0.85	0.84
# 8 sum		1.62	1.22	-	-	1.08
# 9 forw	2.47	2.60	-	-	1.46,1.75?
# 10 grad	5.52	6.53
# 
# (*) timeall(weights(), weights(64), data(), 10)
# (*) af results with gc_enable=false and sync()
# (*) kn uses `similar`, +gc1 runs tmpfree every epoch, +gc2 runs tmpfree every iteration (minibatch)
# AF: The forw records arrays preventing their reuse?
# AF: They are merging consecutive ops in one kernel, which breaks down with forw?

using AutoGrad,GZip
using AutoGrad: forward_pass

fun = []

# push!(fun,(w,x,y)->w[1]*x)
# push!(fun,(w,x,y)->w[1]*x.+w[2])
# push!(fun,(w,x,y)->max(0,w[1]*x.+w[2]))
# push!(fun,(w,x,y)->w[3]*max(0,w[1]*x.+w[2]))
# push!(fun,(w,x,y)->w[3]*max(0,w[1]*x.+w[2]).+w[4])
# push!(fun,(w,x,y)->((w[3]*max(0,w[1]*x.+w[2]).+w[4])-y))
push!(fun,(w,x,y)->(((w[3]*max(0,w[1]*x.+w[2]).+w[4])-y).^2))
push!(fun,(w,x,y)->sum(((w[3]*max(0,w[1]*x.+w[2]).+w[4])-y).^2))
fun1 = fun[end]
push!(fun,(w,x,y)->forward_pass(fun1,(w,x,y),(),1))
# push!(fun,grad(fun1))

function timeall(w=w2,d=d0,t=10)
    for i=1:length(fun)
        printfun(fun[i])
        for j=1:2
            sleep(2)
            @time loop(fun[i],w,d,t)
        end
    end
end

function loop(f,w,d,t)
    for i in 1:t
        for (x,y) in d
            f(w,x,y)
        end
    end
end

function weights(h...; seed=nothing)
    seed==nothing || srand(seed)
    w = Array{Float32}[]
    x = 28*28
    for y in [h..., 10]
        push!(w, convert(Array{Float32}, 0.1*randn(y,x)))
        push!(w, zeros(Float32,y))
        x = y
    end
    return w
end

function data()
    info("Loading data...")
    xshape(a)=reshape(a./255f0,784,div(length(a),784))
    yshape(a)=(a[a.==0]=10; full(sparse(convert(Vector{Int},a),1:length(a),1f0)))
    xtrn = xshape(gzload("train-images-idx3-ubyte.gz")[17:end])
    ytrn = yshape(gzload("train-labels-idx1-ubyte.gz")[9:end])
    #xtst = xshape(gzload("t10k-images-idx3-ubyte.gz")[17:end])
    #ytst = yshape(gzload("t10k-labels-idx1-ubyte.gz")[9:end])
    batch(xtrn,ytrn,100)
end

function gzload(file; dir=Pkg.dir("AutoGrad/data/"), url="http://yann.lecun.com/exdb/mnist/")
    path = dir*file
    isfile(path) || download(url*file, path)
    f = gzopen(path)
    a = readbytes(f)
    close(f)
    return(a)
end

function batch(x, y, batchsize)
    data = Any[]
    nx = size(x,2)
    for i=1:batchsize:nx
        j=min(i+batchsize-1,nx)
        push!(data, (x[:,i:j], y[:,i:j]))
    end
    return data
end

function printfun(x)
    if isdefined(x,:code)
        println(Base.uncompressed_ast(x.code).args[3].args[2].args[1])
    else
        println(x)
    end
end

if !isdefined(:d0)
    d0 = data()
    w1 = weights(seed=1)
    w2 = weights(64;seed=1)
end

:ok
