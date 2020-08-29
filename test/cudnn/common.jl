using Test
#using Knet.CUDNN: cudnnTensorDescriptor, TD, cudnnFilterDescriptor, FD, cudnnDataType, DT, @retry, CUDNN_TENSOR_NCHW, cudnnDataType_t

if CUDA.functional(); @testset "cudnn/common" begin

    x = CUDA.rand(1,1,1,2)

    @test TD === cudnnTensorDescriptor
    @test TD(C_NULL) isa TD
    @test unsafe_convert(Ptr, TD(C_NULL)) isa Ptr
    @test TD(x) isa TD
    @test TD(eltype(x),size(x),CUDNN_TENSOR_NCHW) isa TD

    @test FD === cudnnFilterDescriptor
    @test FD(C_NULL) isa FD
    @test unsafe_convert(Ptr, FD(C_NULL)) isa Ptr
    @test FD(x) isa FD
    @test FD(eltype(x),size(x),CUDNN_TENSOR_NCHW) isa FD

    @test DT === cudnnDataType
    @test DT(Float32) isa cudnnDataType_t

    @test (@retry cudnnCreateTensorDescriptor(Ptr[C_NULL])) isa Nothing

end; end
