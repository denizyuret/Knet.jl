module Deprecate
export gpu, invx, ka, knetgc, gc, setseed, seed!, zeroone, dir, training, atype
import Knet, CUDA, Random

function gpu(x...)
    @warn "gpu() is deprecated, please use CUDA.device instead" maxlog=1
    CUDA.functional() ? CUDA.device().handle : -1
end

function invx(x)
    @warn "invx() is deprecated, please use 1/x instead" maxlog=1
    1/x
end

function ka(x...)
    @warn "ka() is deprecated, please use KnetArray instead" maxlog=1
    KnetArray(x...)
end

function knetgc()
    @warn "knetgc() is deprecated, please use GC.gc() instead" maxlog=1
    Knet.KnetArrays.gc()
end

function gc()
    @warn "Knet.gc() is deprecated, please use GC.gc() instead" maxlog=1
    Knet.KnetArrays.gc()
end

function setseed(x)
    @warn "setseed() is deprecated, please use Random.seed!() and/or CUDA.seed!() instead" maxlog=1
    Random.seed!(x); CUDA.seed!(x)
end

function seed!(x)
    @warn "Knet.seed!() is deprecated, please use Random.seed!() and/or CUDA.seed!() instead" maxlog=1
    Random.seed!(x); CUDA.seed!(x)
end

function zeroone(x...; o...)
    @warn "zeroone() is deprecated, please use 1-accuracy() instead" maxlog=1
    1-accuracy(x...; o...)
end

function dir(path...)
    joinpath(dirname(@__DIR__),path...)
end

function training()
    AutoGrad.recording()
end

function atype()
    @warn "atype() is deprecated, please use Knet.array_type[] instead" maxlog=1
    Knet.Train20.array_type[]
end

function atype(x)
    @warn "atype() is deprecated, please use Knet.array_type[] instead" maxlog=1
    convert(atype(),x)
end

end # module
