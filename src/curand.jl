import Base: rand!, randn!

const libcurand = find_library(["libcurand"], [])
RNG=0

function rng(init=false)
    global RNG
    if RNG==0 || init
        ptr = Ptr{Void}[0]
        # CURAND_RNG_PSEUDO_DEFAULT = 100, ///< Default pseudorandom generator
        @assert 0==ccall((:curandCreateGenerator,libcurand),Cint,(Ptr{Void},Cint),ptr,100)
        RNG = ptr[1]
    end
    return RNG
end

# our version of srand sets both gpu and cpu
function gpuseed(n::Integer)
    # need to regenerate RNG for the seed to take effect for some reason
    @assert 0==ccall((:curandSetPseudoRandomGeneratorSeed,libcurand),Cint,(Ptr{Void},Culonglong),rng(true),n)
    srand(n)
end

rand!(a::AbstractCudaArray{Float32})=(@assert 0==ccall((:curandGenerateUniform,libcurand),Cint,(Ptr{Void},Ptr{Cfloat},Cint),rng(),a,length(a)); a)
rand!(a::AbstractCudaArray{Float64})=(@assert 0==ccall((:curandGenerateUniformDouble,libcurand),Cint,(Ptr{Void},Ptr{Cdouble},Cint),rng(),a,length(a)); a)

# These are a pain, curand insists that array length should be even!
function randn!(a::AbstractCudaArray{Float32},stddev=1f0,mean=0f0)
    if length(a) % 2 == 0
        @assert 0==ccall((:curandGenerateNormal,libcurand),Cint,(Ptr{Void},Ptr{Cfloat},Cint,Cfloat,Cfloat),rng(),a,length(a),mean,stddev)
    else
        @assert 0==ccall((:curandGenerateNormal,libcurand),Cint,(Ptr{Void},Ptr{Cfloat},Cint,Cfloat,Cfloat),rng(),a,length(a)-1,mean,stddev)
        copy!(a,length(a),Float32[randn()*stddev+mean],1,1)
    end
    return a
end

function randn!(a::AbstractCudaArray{Float64},stddev=1f0,mean=0f0)
    if length(a) % 2 == 0
        @assert 0==ccall((:curandGenerateNormalDouble,libcurand),Cint,(Ptr{Void},Ptr{Cdouble},Cint,Cdouble,Cdouble),rng(),a,length(a),mean,stddev)
    else
        @assert 0==ccall((:curandGenerateNormalDouble,libcurand),Cint,(Ptr{Void},Ptr{Cdouble},Cint,Cdouble,Cdouble),rng(),a,length(a)-1,mean,stddev)
        copy!(a,length(a),Float64[randn()*stddev+mean],1,1)
    end
    return a
end
