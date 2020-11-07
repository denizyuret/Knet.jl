using Test
using Base: unsafe_convert
using CUDA: @retry_reclaim
using CUDA.CUDNN: cudnnTensorDescriptor, cudnnCreateTensorDescriptor, cudnnFilterDescriptor, cudnnDataType, CUDNN_TENSOR_NCHW, CUDNN_STATUS_SUCCESS, cudnnDataType_t

if CUDA.functional(); @testset "cudnn/common" begin

    x = CUDA.rand(1,1,1,2)

    TD = cudnnTensorDescriptor
    FD = cudnnFilterDescriptor
    DT = cudnnDataType

    @test TD(C_NULL) isa TD
    @test unsafe_convert(Ptr, TD(C_NULL)) isa Ptr
    @test TD(x) isa TD
    @test TD(CUDNN_TENSOR_NCHW, DT(eltype(x)), Cint(ndims(x)), Cint[reverse(size(x))...]) isa TD

    @test FD(C_NULL) isa FD
    @test unsafe_convert(Ptr, FD(C_NULL)) isa Ptr
    @test FD(x) isa FD
    @test FD(DT(eltype(x)),CUDNN_TENSOR_NCHW,Cint(ndims(x)),Cint[reverse(size(x))...]) isa FD

    @test DT(Float32) isa cudnnDataType_t

    @test (@retry_reclaim(x->(x!==CUDNN_STATUS_SUCCESS),cudnnCreateTensorDescriptor(Ptr[C_NULL]))) isa Nothing

end; end
