include("header.jl")

@testset "distributions" begin
    @test isa(gaussian(10),Array)
    #deprecated @test isa(xavier(10,10),Array)
    @test isa(xavier_uniform(10,10),Array)
    @test isa(xavier_normal(10,10),Array)
    @test isa(bilinear(Float32,2,2,128,128),Array)
end

nothing
