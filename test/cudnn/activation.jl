using CUDA, Test, AutoGrad
using CUDA.CUDNN: 
    cudnnActivationForward,
    cudnnActivationForward!,
    cudnnActivationBackward,
    cudnnActivationDescriptor,
        cudnnActivationDescriptor_t,
        cudnnCreateActivationDescriptor,
        cudnnSetActivationDescriptor,
        cudnnGetActivationDescriptor,
        cudnnDestroyActivationDescriptor,
    cudnnActivationMode_t,
        CUDNN_ACTIVATION_SIGMOID,
        CUDNN_ACTIVATION_RELU,
        CUDNN_ACTIVATION_TANH,
        CUDNN_ACTIVATION_CLIPPED_RELU,
        CUDNN_ACTIVATION_ELU,
        CUDNN_ACTIVATION_IDENTITY,
    cudnnNanPropagation_t,
        CUDNN_NOT_PROPAGATE_NAN,
        CUDNN_PROPAGATE_NAN,
    handle


@testset "cudnn/activation" begin

    x = Param(CUDA.randn(Float64,10))
    @test @gcheck cudnnActivationForward(x, alpha=1, coef=1, nanOpt=CUDNN_NOT_PROPAGATE_NAN, mode=CUDNN_ACTIVATION_SIGMOID)
    @test @gcheck cudnnActivationForward(x, alpha=1, coef=1, nanOpt=CUDNN_NOT_PROPAGATE_NAN, mode=CUDNN_ACTIVATION_RELU)
    @test @gcheck cudnnActivationForward(x, alpha=1, coef=1, nanOpt=CUDNN_NOT_PROPAGATE_NAN, mode=CUDNN_ACTIVATION_TANH)
    @test @gcheck cudnnActivationForward(x, alpha=1, coef=1, nanOpt=CUDNN_NOT_PROPAGATE_NAN, mode=CUDNN_ACTIVATION_CLIPPED_RELU)
    @test @gcheck cudnnActivationForward(x, alpha=1, coef=1, nanOpt=CUDNN_NOT_PROPAGATE_NAN, mode=CUDNN_ACTIVATION_ELU)
    @test @gcheck cudnnActivationForward(x, alpha=2, coef=1, nanOpt=CUDNN_NOT_PROPAGATE_NAN, mode=CUDNN_ACTIVATION_SIGMOID)
    @test @gcheck cudnnActivationForward(x, alpha=1, coef=2, nanOpt=CUDNN_NOT_PROPAGATE_NAN, mode=CUDNN_ACTIVATION_CLIPPED_RELU)
    @test @gcheck cudnnActivationForward(x, alpha=1, coef=2, nanOpt=CUDNN_NOT_PROPAGATE_NAN, mode=CUDNN_ACTIVATION_ELU)
    @test @gcheck cudnnActivationForward(x, alpha=1, coef=1, nanOpt=CUDNN_PROPAGATE_NAN,     mode=CUDNN_ACTIVATION_SIGMOID)

end
