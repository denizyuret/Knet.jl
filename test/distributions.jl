if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

using Knet

@testset "distributions" begin
    @test isa(gaussian(10),Array)
    @test isa(xavier(10,10),Array)
    @test isa(bilinear(Float32,2,2,128,128),Array)
end

nothing
