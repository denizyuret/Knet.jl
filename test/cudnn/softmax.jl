using Test, CUDA, Knet, AutoGrad
using CUDA.CUDNN:
    cudnnSoftmaxForward,
    cudnnSoftmaxForward!,
    cudnnSoftmaxBackward,
    cudnnSoftmaxAlgorithm_t,
        CUDNN_SOFTMAX_FAST,     # 0, /* straightforward implementation */
        CUDNN_SOFTMAX_ACCURATE, # 1, /* subtract max from every point to avoid overflow */
        CUDNN_SOFTMAX_LOG,      # 2
    cudnnSoftmaxMode_t,
        CUDNN_SOFTMAX_MODE_INSTANCE, # 0, /* compute the softmax over all C, H, W for each N */
        CUDNN_SOFTMAX_MODE_CHANNEL,  # 1  /* compute the softmax over all C for each H, W, N */
    handle


if CUDA.functional(); @testset "cudnn/softmax" begin

    x = CUDA.randn(Float64,10,10)
    x2 = Param(x)
    x3 = Param(reshape(x, 10, 10, 1))
    @test @gcheck cudnnSoftmaxForward(x2, alpha=1, algo=CUDNN_SOFTMAX_FAST,     mode=CUDNN_SOFTMAX_MODE_INSTANCE)
    @test @gcheck cudnnSoftmaxForward(x2, alpha=2, algo=CUDNN_SOFTMAX_FAST,     mode=CUDNN_SOFTMAX_MODE_INSTANCE)
    @test @gcheck cudnnSoftmaxForward(x2, alpha=1, algo=CUDNN_SOFTMAX_ACCURATE, mode=CUDNN_SOFTMAX_MODE_INSTANCE)
    @test @gcheck cudnnSoftmaxForward(x2, alpha=2, algo=CUDNN_SOFTMAX_ACCURATE, mode=CUDNN_SOFTMAX_MODE_INSTANCE)
    @test @gcheck cudnnSoftmaxForward(x2, alpha=1, algo=CUDNN_SOFTMAX_LOG,      mode=CUDNN_SOFTMAX_MODE_INSTANCE)
    @test @gcheck cudnnSoftmaxForward(x2, alpha=2, algo=CUDNN_SOFTMAX_LOG,      mode=CUDNN_SOFTMAX_MODE_INSTANCE)
    @test @gcheck cudnnSoftmaxForward(x3, alpha=1, algo=CUDNN_SOFTMAX_FAST,     mode=CUDNN_SOFTMAX_MODE_CHANNEL)
    @test @gcheck cudnnSoftmaxForward(x3, alpha=2, algo=CUDNN_SOFTMAX_FAST,     mode=CUDNN_SOFTMAX_MODE_CHANNEL)
    @test @gcheck cudnnSoftmaxForward(x3, alpha=1, algo=CUDNN_SOFTMAX_ACCURATE, mode=CUDNN_SOFTMAX_MODE_CHANNEL)
    @test @gcheck cudnnSoftmaxForward(x3, alpha=2, algo=CUDNN_SOFTMAX_ACCURATE, mode=CUDNN_SOFTMAX_MODE_CHANNEL)
    @test @gcheck cudnnSoftmaxForward(x3, alpha=1, algo=CUDNN_SOFTMAX_LOG,      mode=CUDNN_SOFTMAX_MODE_CHANNEL)
    @test @gcheck cudnnSoftmaxForward(x3, alpha=2, algo=CUDNN_SOFTMAX_LOG,      mode=CUDNN_SOFTMAX_MODE_CHANNEL)

end; end
