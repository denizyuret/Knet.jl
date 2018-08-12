# curand functions:

using Random
import Random: rand!, randn!
rand!(a::KnetArray{Float32})=(@cuda(curand,curandGenerateUniform,(Cptr,Ptr{Cfloat},Csize_t),rng(),a,length(a)); a)
rand!(a::KnetArray{Float64})=(@cuda(curand,curandGenerateUniformDouble,(Cptr,Ptr{Cdouble},Csize_t),rng(),a,length(a)); a)

function randn!(a::KnetArray{Float32}, mean = 0, stdev = 1)
    @cuda(curand,curandGenerateNormal,(Cptr,Ptr{Cfloat},Csize_t, Cfloat, Cfloat),
        rng(), a, length(a), Cfloat(mean), Cfloat(stdev))
    a
end

function randn!(a::KnetArray{Float64}, mean = 0, stdev = 1)
    @cuda(curand,curandGenerateNormalDouble,(Cptr,Ptr{Cdouble},Csize_t, Cdouble, Cdouble),
        rng(), a, length(a), Cdouble(mean), Cdouble(stdev))
    a
end

let RNG=0
global rng
function rng(init=false)
    if RNG==0 || init
        ptr = Cptr[0]
        # CURAND_RNG_PSEUDO_DEFAULT = 100, ///< Default pseudorandom generator
        @cuda(curand,curandCreateGenerator,(Cptr,Cint),ptr,100)
        RNG = ptr[1]
    end
    return RNG
end
end

"""
    Knet.seed!(n::Integer)

Run seed!(n) on both cpu and gpu.
"""
function seed!(n::Integer)
    # need to regenerate RNG for the seed to take effect for some reason
    if gpu() >= 0
        @cuda1(curand,curandSetPseudoRandomGeneratorSeed,(Cptr,Culonglong),rng(true),n)
    end
    Random.seed!(n)
end

@deprecate setseed Knet.seed!
