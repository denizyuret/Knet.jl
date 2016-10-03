# Adapted from mnist.jl

using AutoGrad
using ArrayFire
using GZip
using Compat

function timeit(f=train0, w = weights(64; seed=1); epochs=10)
    isdefined(:dtrn) || loaddata()
    for i=1:3
        sleep(2)
        #gc_enable(false)
        @time f(w; epochs=epochs)
        #gc_enable(true)
    end
end

function profit(f=train0, w = weights(64; seed=1); epochs=10)
    timeit(f, w; epochs=epochs)
    sleep(2)
    #gc_enable(false)
    @profile f(w; epochs=epochs)
    #gc_enable(true)
end

function predict(w, x)
    i = 1
    while i+2 < length(w)
        x = max(0, w[i]*x .+ w[i+1])
        i += 2
    end
    return w[i]*x .+ w[i+1]
end

function loss(w,x,ygold)
    ypred = predict(w, x)
    sum((ypred-ygold).^2)
end

function loss1(w,x,ygold)
    ypred = predict(w, x)
    ynorm = ypred .- log(sum(exp(ypred),1))
    -sum(ygold .* ynorm) / size(ygold,2)
end

function train0(w=weights(64); lr=.1, epochs=1)
    gradfun = grad(loss)
    for epoch=1:epochs
        for (x,y) in dtrn
            g = gradfun(w, x, y)
            for i in 1:length(w)
                w[i] -= lr*g[i]
            end
        end
    end
    return w
end

function train1(w=weights(64); lr=.1, epochs=1)
    gradfun = grad(loss)
    for epoch=1:epochs
        for (x,y) in dtrn
            g = gradfun(w, x, y)
        end
    end
end

function train2(w=weights(64); lr=.1, epochs=1)
    for epoch=1:epochs
        for (x,y) in dtrn
            z = loss(w, x, y)
        end
    end
end

function train3(w=weights(64); lr=.1, epochs=1)
    for epoch=1:epochs
        for (x,ygold) = dtrn
            y = predict(w,x) 
        end
    end
end

function weights(h...; seed=nothing)
    seed==nothing || srand(seed)
    w = Any[]
    x = 28*28
    for y in [h..., 10]
        push!(w, convert(Array{Float32}, 0.1*randn(y,x)))
        push!(w, zeros(Float32,y))
        x = y
    end
    return map(AFArray,w)
end

function loaddata()
    info("Loading data...")
    global xtrn, xtst, ytrn, ytst, dtrn
    xshape(a)=reshape(a./255f0,784,div(length(a),784))
    yshape(a)=(a[a.==0]=10; full(sparse(convert(Vector{Int},a),1:length(a),1f0)))
    xtrn = xshape(gzload("train-images-idx3-ubyte.gz")[17:end])
    xtst = xshape(gzload("t10k-images-idx3-ubyte.gz")[17:end])
    ytrn = yshape(gzload("train-labels-idx1-ubyte.gz")[9:end])
    ytst = yshape(gzload("t10k-labels-idx1-ubyte.gz")[9:end])
    dtrn = map(a->map(AFArray,a), minibatch(xtrn, ytrn, 100))
    # dtrn = [(CudaArray(xtst),CudaArray(ytst))]
    info("Loading done...")
end

function gzload(file; path=Knet.dir("data",file), url="http://yann.lecun.com/exdb/mnist/$file")
    isfile(path) || download(url, path)
    f = gzopen(path)
    a = @compat read(f)
    close(f)
    return(a)
end

function minibatch(x, y, batchsize)
    data = Any[]
    nx = size(x,2)
    for i=1:batchsize:nx
        j=min(i+batchsize-1,nx)
        push!(data, (x[:,i:j], y[:,i:j]))
    end
    return data
end

isdefined(:dtrn) || loaddata()

using AutoGrad: id2
import AutoGrad: _dbg, sum_helper
_dbg(x::AFArray)=Symbol("AF$(join([id2(x),size(x)...,8*sizeof(eltype(x))],'_'))")
sum_helper{T}(a::AFArray{T},b::AFArray{T},c::AFArray{T}...) = +(a,b,c...)

w1 = weights();
w2 = weights(64);
(x,y) = first(dtrn);
g = grad(loss)

# timeit(train0,w1); # => 6.713386 seconds (7.34 M allocations: 276.245 MB, 4.68% gc time) # compare to 1.977053 seconds (2.91 M allocations: 173.705 MB)  # compare to 1.8276 in Knet7
# timeit(train1,w1); # => 5.613988 seconds (6.75 M allocations: 258.575 MB, 4.26% gc time) # compare to 1.898212 seconds (2.88 M allocations: 173.156 MB)
# timeit(train2,w1); # => 1.213765 seconds (668.72 k allocations: 25.752 MB) # compare to 1.025943 seconds (631.80 k allocations: 74.277 MB)
# timeit(train3,w1); # => 0.464109 seconds (259.80 k allocations: 11.838 MB) # compare to 0.608363 seconds (601.80 k allocations: 27.127 MB)

# timeit(train0,w2); # => 11.899407 seconds (11.37 M allocations: 421.264 MB, 4.53% gc time) # compare to 3.659655 seconds (6.97 M allocations: 344.909 MB)  # compare to 2.9201 in Knet7
# timeit(train1,w2); # => 9.333509 seconds (10.21 M allocations: 386.199 MB, 4.29% gc time) # compare to 3.505348 seconds (6.90 M allocations: 343.810 MB)
# timeit(train2,w2); # => 1.874707 seconds (907.80 k allocations: 33.353 MB) # compare to 1.391519 seconds (1.36 M allocations: 107.053 MB)
# timeit(train3,w2); # => 0.856982 seconds (511.80 k allocations: 20.078 MB) # compare to 0.985798 seconds (1.33 M allocations: 59.903 MB)

# Plain forward is faster.  relu and softmax are much slower.  Also probably too much alloc.

# Replacing softloss with quadloss:
# timeit(train0,w1); # => 4.318921 seconds (4.92 M allocations: 181.396 MB, 5.20% gc time)
# timeit(train1,w1); # => 3.246138 seconds (4.33 M allocations: 163.726 MB, 5.24% gc time)
# timeit(train2,w1); # => 0.847536 seconds (373.80 k allocations: 12.113 MB)
# timeit(train3,w1); # => 0.464719 seconds (109.80 k allocations: 3.690 MB)

# timeit(train0,w2); # => 8.889076 seconds (8.94 M allocations: 324.584 MB, 4.87% gc time)
# timeit(train1,w2); # => 6.513208 seconds (7.78 M allocations: 289.520 MB, 4.31% gc time)
# timeit(train2,w2); # => 1.375004 seconds (625.80 k allocations: 20.352 MB, 3.89% gc time)
# timeit(train3,w2); # => 0.749930 seconds (361.80 k allocations: 11.929 MB)

# There is still a big slowdown in the backward_pass.  What operation is responsible?
# The number of allocations is smaller in forward, larger in backward compared to old impl.
# Is it related to not being able to fuse loops?
# Is there some ungetindex applied to a AFArray?
