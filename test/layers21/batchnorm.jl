using Test, Random, CUDA, Knet, AutoGrad, Statistics, Knet.Layers21

@testset "layers21/batchnorm" begin

    x, s, b, s2, b2, s3, b3 = (Param(CUDA.randn(Float64,x...)) for x in ((5,4,3,2),(1,1,3,1),(1,1,3,1),(5,4,3,1),(5,4,3,1),(5,1,1,1),(5,1,1,1)))

    # @gcheck needs Inference and Training to compute the same function
    # For this they need to use the same mean/var
    # @gcheck sums the output of non-scalar functions by default
    # This has the effect of multiplying scale with a 0 mean array
    # So we'll use getindex(y,1) instead
    function bnormtest(x,s,b; dims=(1,2,4), o...)
        global m,v,bn,y,z
        m = mean(value(x); dims)
        v = var(value(x); mean=m,corrected=false,dims)
        bn = BatchNorm(; mean=m, variance=v, bias=b, scale=s, o...)
        y = bn(x)
        z = CUDA.CUDNN.cudnnNormalizationForward(x, m, v, b, s; o...)
        y[1]
    end

    @test bnormtest(x,s,b) ≈ value(@diff(bnormtest(x,s,b)))
    @test @gcheck bnormtest(x,s,b)
    @test @gcheck bnormtest(x,s2,b2; mode = CUDNN_NORM_PER_ACTIVATION, dims=4)
    @test @gcheck bnormtest(x,s,b; algo = CUDNN_NORM_ALGO_PERSIST)
    @test @gcheck bnormtest(x,s3,b3; algo = CUDNN_NORM_ALGO_PERSIST, format = CUDNN_TENSOR_NHWC, dims=(2,3,4))
    @test @gcheck bnormtest(x,s,b; alpha = 2)
    @test @gcheck bnormtest(x,s,b; epsilon = 0)
    @test @gcheck bnormtest(x,s3,b3; format = CUDNN_TENSOR_NHWC, dims=(2,3,4))
    @test @gcheck bnormtest(x,s,b; exponentialAverageFactor = 0.01)
    @test @gcheck bnormtest(x,s,b; savedMean = similar(s))
    @test @gcheck bnormtest(x,s,b; savedInvVariance = similar(s))
    # cudnn-8.0.5: Currently, CUDNN_NORM_OPS_NORM_ACTIVATION and CUDNN_NORM_OPS_NORM_ADD_ACTIVATION are not supported.
    #@test_skip @gcheck bnormtest(x,s,b; normOps = CUDNN_NORM_OPS_NORM_ACTIVATION, activationMode = CUDNN_ACTIVATION_RELU)
    #@test_skip @gcheck bnormtest(x,s,b; normOps = CUDNN_NORM_OPS_NORM_ADD_ACTIVATION, activationMode = CUDNN_ACTIVATION_RELU, z)
    #@test_skip @gcheck bnormtest(x,s,b; groupCnt = 2) # Currently only groupCnt=1 is supported
    #@test_skip @gcheck bnormtest(x,s,b; beta = 2) # we don't use beta in functional code
end
