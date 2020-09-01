include("header.jl")

@testset "nonlinear" begin
     x = randn(10,2)
     s = Sigm()
     @test s(x) == sigm.(x)
     r = ReLU()
     @test r(x) == relu.(x)
     lgp = LogSoftMax()
     @test lgp(x) == logp(x)
     sft = SoftMax()
     @test sft(x) == softmax(x)
     elu = ELU()
     elu(x)
     @test true
end
