using Knet
include(Pkg.dir("Knet", "examples", "resnet", "resnetlib.jl"))
using ResNetLib: resnet50init, resnet50


const BATCH_SIZE = 32
const RESNET_FEATURES = 2048
const BATCHES_GPU = 40
const BATCHES_CPU = 8

function fakedata(batches)
    srand(0)
    x = rand(Float32, 224, 224, 3, BATCH_SIZE * batches)
    #y = Int.(ones(size(x,4))) #dummy
    return x
end

function predictfn(w, m, x, bsize; atype=KnetArray)
    dataset = minibatch(x,
                        Array(1:size(x,4)), bsize;
                        xtype=atype)
    out = zeros(Float32, RESNET_FEATURES, size(x,4))
    idx = 1
    for (x, _) in dataset
        pred = Array(mat(resnet50(w, m, x; stage=5)))
        out[:, (idx-1)*bsize+1:idx*bsize] = pred
        idx += 1
    end
    return out
    
end

function benchmark_gpu()
    w, m = resnet50init(;stage=5, trained=true)
    x = fakedata(BATCHES_GPU)
    cold_start = predictfn(w, m, x, BATCH_SIZE)
    info("Benchmarking")
    @time begin
       y = predictfn(w, m, x, BATCH_SIZE)
    end
    nothing
end

function benchmark_cpu()
    temp = gpu()
    gpu(-1)
    w, m = resnet50init(;stage=5, trained=true, atype=Array)
    x = fakedata(BATCHES_CPU)
    info("running cold start")
    cold_start = predictfn(w, m, x, BATCH_SIZE; atype=Array)
    info("running benchmark")
    @time begin
       y = predictfn(w, m, x, BATCH_SIZE; atype=Array)
    end
    gpu(temp)
end
