using Test, Random, CUDA, Knet, AutoGrad, Statistics

using CUDA.CUDNN:
    cudnnNormalizationForward,
    cudnnNormalizationForward!,
    cudnnNormalizationForwardInference,
    cudnnNormalizationForwardTraining,
    cudnnNormalizationBackward,
    cudnnActivationDescriptor,
    cudnnNormMode_t,
        CUDNN_NORM_PER_ACTIVATION, # 0, bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice, normalized over N slice)
        CUDNN_NORM_PER_CHANNEL,    # 1, bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors)
    cudnnNormOps_t,
        CUDNN_NORM_OPS_NORM,                # 0, /* do normalization only */
        CUDNN_NORM_OPS_NORM_ACTIVATION,     # 1, /* do Norm, then activation */
        CUDNN_NORM_OPS_NORM_ADD_ACTIVATION, # 2, /* do Norm, then elemWiseAdd, then activation */
    cudnnNormAlgo_t,
        CUDNN_NORM_ALGO_STANDARD, # 0
        CUDNN_NORM_ALGO_PERSIST,  # 1
    cudnnActivationMode_t,
        CUDNN_ACTIVATION_SIGMOID,      # 0
        CUDNN_ACTIVATION_RELU,         # 1
        CUDNN_ACTIVATION_TANH,         # 2
        CUDNN_ACTIVATION_CLIPPED_RELU, # 3
        CUDNN_ACTIVATION_ELU,          # 4
        CUDNN_ACTIVATION_IDENTITY,     # 5
    cudnnNanPropagation_t,
        CUDNN_NOT_PROPAGATE_NAN, # 0
        CUDNN_PROPAGATE_NAN,     # 1
    cudnnTensorFormat_t,
        CUDNN_TENSOR_NCHW,        # 0, /* row major (wStride = 1, hStride = w) */
        CUDNN_TENSOR_NHWC,        # 1, /* feature maps interleaved ( cStride = 1 )*/
        CUDNN_TENSOR_NCHW_VECT_C, # 2, /* each image point is vector of element of C, vector length in data type */
    handle


@testset "cudnn/normalization" begin

    x, s, b, s2, b2, s3, b3 = (Param(CUDA.randn(Float64,x...)) for x in ((5,4,3,2),(1,1,3,1),(1,1,3,1),(5,4,3,1),(5,4,3,1),(5,1,1,1),(5,1,1,1)))

    # @gcheck needs Inference and Training to compute the same function
    # For this they need to use the same mean/var
    # @gcheck sums the output of non-scalar functions by default
    # This has the effect of multiplying scale with a 0 mean array
    # So we'll use getindex(y,1) instead
    function normtest(x,s,b; dims=(1,2,4), o...)
        m = mean(value(x);dims)
        v = var(value(x);mean=m,corrected=false,dims)
        y = cudnnNormalizationForward(x,m,v,b,s; training=Knet.training(), o...)
        y[1]
    end

    @test normtest(x,s,b) ≈ value(@diff(normtest(x,s,b)))
    @test @gcheck normtest(x,s,b)
    @test @gcheck normtest(x,s2,b2; mode = CUDNN_NORM_PER_ACTIVATION, dims=4)
    @test @gcheck normtest(x,s,b; algo = CUDNN_NORM_ALGO_PERSIST)
    @test @gcheck normtest(x,s3,b3; algo = CUDNN_NORM_ALGO_PERSIST, format = CUDNN_TENSOR_NHWC, dims=(2,3,4))
    @test @gcheck normtest(x,s,b; alpha = 2)
    @test @gcheck normtest(x,s,b; epsilon = 0)
    @test @gcheck normtest(x,s3,b3; format = CUDNN_TENSOR_NHWC, dims=(2,3,4))
    @test @gcheck normtest(x,s,b; exponentialAverageFactor = 0.01)
    @test @gcheck normtest(x,s,b; savedMean = similar(s), savedInvVariance = similar(s))
    # cudnn-8.0.5: Currently, CUDNN_NORM_OPS_NORM_ACTIVATION and CUDNN_NORM_OPS_NORM_ADD_ACTIVATION are not supported.
    #@test_skip @gcheck normtest(x,s,b; normOps = CUDNN_NORM_OPS_NORM_ACTIVATION, activationMode = CUDNN_ACTIVATION_RELU)
    #@test_skip @gcheck normtest(x,s,b; normOps = CUDNN_NORM_OPS_NORM_ADD_ACTIVATION, activationMode = CUDNN_ACTIVATION_RELU, z)
    #@test_skip @gcheck normtest(x,s,b; groupCnt = 2) # Currently only groupCnt=1 is supported
    #@test_skip @gcheck normtest(x,s,b; beta = 2) # we don't use beta in functional code
end
