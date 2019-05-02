# curand functions:

using Random
import Random: rand!, randn!
rand!(a::KnetArray{Float32})=(@curand(curandGenerateUniform,(Cptr,Ptr{Cfloat},Csize_t),rng(),a,length(a)); a)
rand!(a::KnetArray{Float64})=(@curand(curandGenerateUniformDouble,(Cptr,Ptr{Cdouble},Csize_t),rng(),a,length(a)); a)

# curandNormal functions only work with even length arrays
_randn!(a::KnetArray{Float32},n,mean,stdev)=@curand(curandGenerateNormal,(Cptr,Ptr{Cfloat},Csize_t, Cfloat, Cfloat), rng(), a, Csize_t(n), Cfloat(mean), Cfloat(stdev))
_randn!(a::KnetArray{Float64},n,mean,stdev)=@curand(curandGenerateNormalDouble,(Cptr,Ptr{Cdouble},Csize_t, Cdouble, Cdouble), rng(), a, Csize_t(n), Cdouble(mean), Cdouble(stdev))

function randn!(a::KnetArray{T}, mean = 0, stdev = 1)
    n = length(a)
    if isodd(n)
        a[end] = randn()*stdev+mean
        n = n-1
    end
    if n > 0
        _randn!(a,n,mean,stdev)
    end
    return a
end

let RNG=0
global rng
function rng(init=false)
    if RNG==0 || init
        ptr = Cptr[0]
        # CURAND_RNG_PSEUDO_DEFAULT = 100, ///< Default pseudorandom generator
        @curand(curandCreateGenerator,(Cptr,Cint),ptr,100)
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
        @curand(curandSetPseudoRandomGeneratorSeed,(Cptr,Culonglong),rng(true),n)
    end
    Random.seed!(n)
end

function setseed(n)
    @warn "setseed is deprecated, use Knet.seed! instead." maxlog=1
    seed!(n)
end
