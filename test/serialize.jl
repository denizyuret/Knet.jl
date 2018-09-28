include("header.jl")

@testset "serialize" begin
    M = RNN(1,1)
    M2 = M |> cpu
    @test typeof(M2.w.value) <: Array
    if gpu() >= 0 
        M3 = M2 |> gpu
        @test typeof(M3.w.value) <: KnetArray
    end
end
       

