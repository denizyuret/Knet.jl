using Test, AutoGrad
#using Knet.CUDNN: cudnnDropoutDescriptor, cudnnDropoutSeed, cudnnDropoutForward

if CUDA.functional(); @testset "cudnn/dropout" begin

    @test cudnnDropoutDescriptor(C_NULL) isa cudnnDropoutDescriptor
    @test unsafe_convert(Ptr, cudnnDropoutDescriptor(C_NULL)) isa Ptr
    @test cudnnDropoutDescriptor(0.5) isa cudnnDropoutDescriptor

    x = Param(CUDA.randn(Float64,10,10))
    cudnnDropoutSeed[] = 1
    @test @gcheck cudnnDropoutForward(x)
    cudnnDropoutSeed[] = -1

end; end
