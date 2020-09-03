using Test, AutoGrad
using CUDA.CUDNN: CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE, CUDNN_SOFTMAX_MODE_CHANNEL
#using Knet.CUDNN: cudnnSoftmaxForward


if CUDA.functional(); @testset "cudnn/softmax" begin

    x = Param(CUDA.randn(Float64,10,10))
    @test @gcheck cudnnSoftmaxForward(x, algo=CUDNN_SOFTMAX_FAST, mode=CUDNN_SOFTMAX_MODE_INSTANCE)
    @test @gcheck cudnnSoftmaxForward(x, algo=CUDNN_SOFTMAX_ACCURATE, mode=CUDNN_SOFTMAX_MODE_INSTANCE)
    @test @gcheck cudnnSoftmaxForward(x, algo=CUDNN_SOFTMAX_LOG, mode=CUDNN_SOFTMAX_MODE_INSTANCE)
    @test @gcheck cudnnSoftmaxForward(x, algo=CUDNN_SOFTMAX_FAST, mode=CUDNN_SOFTMAX_MODE_CHANNEL)
    @test @gcheck cudnnSoftmaxForward(x, algo=CUDNN_SOFTMAX_ACCURATE, mode=CUDNN_SOFTMAX_MODE_CHANNEL)
    @test @gcheck cudnnSoftmaxForward(x, algo=CUDNN_SOFTMAX_LOG, mode=CUDNN_SOFTMAX_MODE_CHANNEL)

end; end
