# Adapted from mnist.jl


if !isdefined(:sforw)
using Knet
end
using AutoGrad
using CUDArt
using GZip
using Compat

function timeit(f=train0, w = weights(64; seed=1); epochs=10)
    isdefined(:dtrn) || loaddata()
    for i=1:3
        sleep(2)
        gc_enable(false)
        @time (f(w; epochs=epochs); gpusync())
        gc_enable(true)
    end
end

function predict(w, x)
    i = 1
    while i+2 < length(w)
        x = relu(w[i]*x .+ w[i+1])
        i += 2
    end
    return w[i]*x .+ w[i+1]
end

function loss(w,x,ygold)
    ypred = predict(w,x)
    return xentloss(ypred,ygold)
end

function train0(w=weights(64); lr=.1, epochs=1)
    gradfun = grad(loss)
    for epoch=1:epochs
        for (x,y) in dtrn
            tmpfree()
            g = gradfun(w, x, y)
            for i in 1:length(w)
                Base.axpy!(-lr, g[i], w[i])
            end
        end
    end
    return w
end

function train1(w=weights(64); lr=.1, epochs=1)
    gradfun = grad(loss)
    for epoch=1:epochs
        for (x,y) in dtrn
            tmpfree()
            g = gradfun(w, x, y)
        end
    end
end

function train2(w=weights(64); lr=.1, epochs=1)
    for epoch=1:epochs
        for (x,y) in dtrn
            tmpfree()
            z = loss(w, x, y)
        end
    end
end

function train3(w=weights(64); lr=.1, epochs=1)
    for epoch=1:epochs
        for (x,ygold) = dtrn
            tmpfree()
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
    return map(CudaArray,w)
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
    dtrn = map(a->map(CudaArray,a), minibatch(xtrn, ytrn, 100))
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

isdefined(:dtrn) || loaddata()

using AutoGrad: recorder,Node,Grad,getval

tmplike(a::Node,i...)=tmplike(a.value,i...)
zimilar(a...)=tmplike(a...)

#import Knet: reluback
relu(x)=(y=zimilar(x);reluforw(x,y);y)
@primitive relu(x)::y (dy->(dx=zimilar(x);reluback(y,dy,dx);dx))
@zerograd reluback(y,dy,dx)

importall Base
(*)(a::CudaArray,b::CudaArray)=A_mul_B!(zimilar(a,(size(a,1),size(b,2))),a,b)
Ac_mul_B(a::CudaArray,b::CudaArray)=At_mul_B!(zimilar(a,(size(a,2),size(b,2))),a,b)
A_mul_Bc(a::CudaArray,b::CudaArray)=A_mul_Bt!(zimilar(a,(size(a,1),size(b,1))),a,b)
(*)(a::Number,b::CudaArray)=a.*b
(*)(a::CudaArray,b::Number)=a.*b
(-)(a::CudaArray,b::CudaArray)=a.-b

(.+)(a::CudaArray,b::CudaArray)=broadcast!(+,zimilar(larger(a,b)),a,b)
(.*)(a::CudaArray,b::CudaArray)=broadcast!(*,zimilar(larger(a,b)),a,b)
(.-)(a::CudaArray,b::CudaArray)=broadcast!(+,zimilar(larger(a,b)),a,scale!(copy!(zimilar(b),b),-1))
(.*){T}(x::Number,a::CudaArray{T})=scale!(T(x),copy!(zimilar(a),a))
(.*){T}(a::CudaArray{T},x::Number)=scale!(T(x),copy!(zimilar(a),a))
larger(a,b)=(length(a)>length(b) ? a : b)

# log(sum(exp(ypred),1))
# for sum(x) we could try CUBLAS.asum, or Knet.vecnorm1 (all entries are positive).
# but we need sum(x,i).  we have log/exp defined in Knet.

function sum(a::CudaMatrix,i::Integer)
    b = (i==1 ? zimilar(a,(1,size(a,2))) :
         i==2 ? zimilar(a,(size(a,1),1)) :
         error("$i out of range"))
    # sum!(b,a) # works for 2 but not 1!
    baddback(a,b)
    return b
end

# # broadcast does not work for 1xn arrays!
# # we need a good broadcast implementation for CudaArrays.
# # in the short run we need warnings.  right now we get wrong answers for (.-)

importall AutoGrad
using AutoGrad: matmul2arg, recorder, Node, Grad, broadcast2arg, unbroadcast, math1arg, getval, isfloat, tofloat, _dbg, id2
import AutoGrad: matmul2arg, recorder, Node, Grad, broadcast2arg, unbroadcast, math1arg, getval, isfloat, tofloat, _dbg, id2
_dbg(x::CudaArray)=Symbol("C$(join([id2(x),size(x)...],'_'))")
isfloat{T<:AbstractFloat}(x::CudaArray{T})=true
isfloat(x::CudaArray)=false
tofloat{T<:AbstractFloat}(x::CudaArray{T})=x

for (f,d) in matmul2arg
    @eval @primitive $f(x1::CudaArray,x2::CudaArray)::y $(d[1]) $(d[2])
end

for (f,g) in broadcast2arg
    @eval @primitive $f(x1::CudaArray,x2::CudaArray)::y  unbroadcast(y,x1,dy->dy.*$(g[1]))  unbroadcast(y,x2,dy->dy.*$(g[2]))
end

for (f,g) in math1arg
    @eval @primitive  $f(x::CudaArray)::y  (dy->dy.*$g)
end

@primitive  sum(x::CudaArray,i...)  (dy->dy.+zeros(x))

# import Knet: xentloss

@primitive xentloss(y,p)  (dy->dy.*xentloss(y,p,zimilar(y)))
@zerograd xentloss(y,p,dx)

Base.zeros(a::CudaArray)=fill!(zimilar(a),0)


# Experiments:
# timeit() with similar: 7.128271 seconds (9.63 M allocations: 407.365 MB)
# timeit() with tmplike: 4.993464 seconds (8.07 M allocations: 369.399 MB)
# why still so many allocations?
# timeit(train3): 0.918988 seconds (1.17 M allocations: 50.839 MB)
# @time y=w[1]*x =>  0.000070 seconds (26 allocations: 1.078 KB) (comes from cublas.gemm!)
# @time z=y.+w[2] =>  0.000043 seconds (53 allocations: 2.297 KB) (comes from cudnnAddTensor3)
# @time train3(w1) => 0.019915 seconds (44.58 k allocations: 1.852 MB) with tmplike, w1=weights()
# @time train3(w1) => 0.050419 seconds (66.92 k allocations: 2.468 MB) with similar
# timeit(train3,w1) => 0.563168 seconds (445.80 k allocations: 18.521 MB) with tmplike
# timeit(train3,w1) => 0.609260 seconds (681.78 k allocations: 24.869 MB) with similar
# CUDArt.cuda_ptrs gives the number of CudaArrays allocated.
# After loading profgpu.jl we have 1200 arrays (dtrn)
# with w=weights() we have 1202
# after train3() we have 1208, each epoch adds 4 more?  The dict has only 2 entries!? who allocs 4 arrays every iter?
# cannot replicate with single minibatch... disappeared by itself :(

# @show length(CUDArt.cuda_ptrs)
# @show w=weights()
# @show length(CUDArt.cuda_ptrs)
# @show train1(w)
# @show length(CUDArt.cuda_ptrs)
# @show train1(w)
# @show length(CUDArt.cuda_ptrs)
# @show train1(w; epochs=10)
# @show length(CUDArt.cuda_ptrs)
# @show sum(map(s->length(s.arr), values(tmpdict(w[1]))))

# ok, train2 and train3 have no extra gpu allocation with tmplike.
# train0 and train1 still alloc like crazy: 3000 arrays per epoch.  5
# arrays per iteration.

# fixed some copy problem in profgpu.

# next is cudadims used in broadcast.jl!  how does cudnn pass dims to gpu?  can we avoid copy as well as alloc?
# is this why my badd is slow?
# I created some temp arrays and started copy, now no more cuda allocs during training.

# OK, no more CudaArray allocations:

numarrays()=(length(CUDArt.cuda_ptrs), sum(map(s->length(s.arr), values(tmpdict(CudaArray{Float32})))))
@show numarrays()
@show w1 = weights()
@show w2 = weights(64)
@show numarrays()
@show timeit(train0,w1) # => 1.977053 seconds (2.91 M allocations: 173.705 MB)  # compare to 1.8276 in Knet7
@show timeit(train1,w1) # => 1.898212 seconds (2.88 M allocations: 173.156 MB)
@show timeit(train2,w1) # => 1.025943 seconds (631.80 k allocations: 74.277 MB)
@show timeit(train3,w1) # => 0.608363 seconds (601.80 k allocations: 27.127 MB)
@show numarrays()
@show timeit(train0,w2) # => 3.659655 seconds (6.97 M allocations: 344.909 MB)  # compare to 2.9201 in Knet7
@show timeit(train1,w2) # => 3.505348 seconds (6.90 M allocations: 343.810 MB)
@show timeit(train2,w2) # => 1.391519 seconds (1.36 M allocations: 107.053 MB)
@show timeit(train3,w2) # => 0.985798 seconds (1.33 M allocations: 59.903 MB)
@show numarrays()
