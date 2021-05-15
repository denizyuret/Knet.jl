using Test, CUDA, Knet, AutoGrad
using CUDA.CUDNN: 
    cudnnDropoutForward,
    cudnnDropoutForward!,
    cudnnDropoutBackward,
    cudnnDropoutSeed,
    cudnnDropoutDescriptor,
        cudnnDropoutDescriptor_t,
        cudnnCreateDropoutDescriptor,
        cudnnSetDropoutDescriptor,
        cudnnGetDropoutDescriptor,
        cudnnRestoreDropoutDescriptor,
        cudnnDestroyDropoutDescriptor,
    cudnnDropoutGetStatesSize,
    cudnnDropoutGetReserveSpaceSize,
    handle


@testset "cudnn/dropout" begin

    N,P = 1000, 0.7
    x = Param(CUDA.rand(N))
    d = cudnnDropoutDescriptor(P)
    cudnnDropoutSeed[] = 1
    @test @gcheck cudnnDropoutForward(x; dropout = P)
    @test @gcheck cudnnDropoutForward(x, d)
    @test @gcheck cudnnDropoutForward!(similar(x), x; dropout = P)
    @test @gcheck cudnnDropoutForward!(similar(x), x, d)
    cudnnDropoutSeed[] = -1

end
