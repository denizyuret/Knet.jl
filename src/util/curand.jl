import Base: rand!, randn!

const libcurand = Libdl.find_library(["libcurand"], [])
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
function setseed(n::Integer)
    # need to regenerate RNG for the seed to take effect for some reason
    @assert 0==ccall((:curandSetPseudoRandomGeneratorSeed,libcurand),Cint,(Ptr{Void},Culonglong),rng(true),n)
    srand(n)
end

rand!(a::CudaArray{Float32})=(@assert 0==ccall((:curandGenerateUniform,libcurand),Cint,(Ptr{Void},Ptr{Cfloat},Csize_t),rng(),a,length(a)); a)
rand!(a::CudaArray{Float64})=(@assert 0==ccall((:curandGenerateUniformDouble,libcurand),Cint,(Ptr{Void},Ptr{Cdouble},Csize_t),rng(),a,length(a)); a)
rand!(a::Union{CudaArray{UInt32},CudaArray{Int32}})=(@assert 0==ccall((:curandGenerate,libcurand),Cint,(Ptr{Void},Ptr{Cuint},Csize_t),rng(),a,length(a)); a)
rand!(a::Union{CudaArray{UInt64},CudaArray{Int64}})=(@assert 0==ccall((:curandGenerate,libcurand),Cint,(Ptr{Void},Ptr{Cuint},Csize_t),rng(),a,2*length(a)); a)

# These are a pain, curand insists that array length should be even!
function randn!(a::CudaArray{Float32},mean=0f0,stddev=1f0)
    if length(a) % 2 == 0
        @assert 0==ccall((:curandGenerateNormal,libcurand),Cint,(Ptr{Void},Ptr{Cfloat},Csize_t,Cfloat,Cfloat),rng(),a,length(a),mean,stddev)
    else
        @assert 0==ccall((:curandGenerateNormal,libcurand),Cint,(Ptr{Void},Ptr{Cfloat},Csize_t,Cfloat,Cfloat),rng(),a,length(a)-1,mean,stddev)
        copysync!(a,length(a),Float32[randn()*stddev+mean],1,1)
    end
    return a
end

function randn!(a::CudaArray{Float64},mean=0e0,stddev=1e0)
    if length(a) % 2 == 0
        @assert 0==ccall((:curandGenerateNormalDouble,libcurand),Cint,(Ptr{Void},Ptr{Cdouble},Csize_t,Cdouble,Cdouble),rng(),a,length(a),mean,stddev)
    else
        @assert 0==ccall((:curandGenerateNormalDouble,libcurand),Cint,(Ptr{Void},Ptr{Cdouble},Csize_t,Cdouble,Cdouble),rng(),a,length(a)-1,mean,stddev)
        copysync!(a,length(a),Float64[randn()*stddev+mean],1,1)
    end
    return a
end

# randn!{T}(a::Array{T}, mean=zero(T), std=one(T))=(for i=1:length(a); a[i] = mean + std * randn(); end; a)
# rand!(a::BaseArray, x0, x1)=(rand!(a); axpb!(x1-x0, x0, a); a)

