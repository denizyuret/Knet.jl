using Test, AutoGrad
using Base: unsafe_convert
using CUDA.CUDNN: cudnnActivationDescriptor, cudnnActivationForward, CUDNN_NOT_PROPAGATE_NAN, CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_RELU, CUDNN_ACTIVATION_TANH, CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_ACTIVATION_ELU

if CUDA.functional(); @testset "cudnn/activation" begin

    @test cudnnActivationDescriptor(C_NULL) isa cudnnActivationDescriptor
    @test unsafe_convert(Ptr, cudnnActivationDescriptor(C_NULL)) isa Ptr
    @test cudnnActivationDescriptor(CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN,0) isa cudnnActivationDescriptor

    x = Param(CUDA.randn(Float64,10,10))
    @test @gcheck cudnnActivationForward(x, mode=CUDNN_ACTIVATION_SIGMOID)
    @test @gcheck cudnnActivationForward(x, mode=CUDNN_ACTIVATION_RELU)
    @test @gcheck cudnnActivationForward(x, mode=CUDNN_ACTIVATION_TANH)
    @test @gcheck cudnnActivationForward(x, mode=CUDNN_ACTIVATION_CLIPPED_RELU)
    @test @gcheck cudnnActivationForward(x, mode=CUDNN_ACTIVATION_ELU)

end; end
