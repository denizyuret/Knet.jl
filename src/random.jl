# # curand functions:
#
# import Base: rand!
# rand!(a::KnetArray{Float32})=(@cuda(curand,curandGenerateUniform,(Cptr,Ptr{Cfloat},Csize_t),rng(),a,length(a)); a)
# rand!(a::KnetArray{Float64})=(@cuda(curand,curandGenerateUniformDouble,(Cptr,Ptr{Cdouble},Csize_t),rng(),a,length(a)); a)
#
# let RNG=0
# global rng
# function rng(init=false)
#     if RNG==0 || init
#         ptr = Cptr[0]
#         # CURAND_RNG_PSEUDO_DEFAULT = 100, ///< Default pseudorandom generator
#         @cuda(curand,curandCreateGenerator,(Cptr,Cint),ptr,100)
#         RNG = ptr[1]
#     end
#     return RNG
# end
# end
#
# """
#     setseed(n::Integer)
#
# Run srand(n) on both cpu and gpu.
# """
function setseed(n::Integer)
    # need to regenerate RNG for the seed to take effect for some reason
    # @cuda1(curand,curandSetPseudoRandomGeneratorSeed,(Cptr,Culonglong),rng(true),n)
    srand(n)
end
#
