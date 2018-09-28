include("header.jl")

@testset "serialize" begin
    M1 = RNN(2,3)
    M2 = M1 |> cpucopy
    @test typeof(M2.w.value) <: Array
    @test M2.w.value == M1.w.value
    if gpu() >= 0 
        M3 = M2 |> gpucopy
        @test typeof(M3.w.value) <: KnetArray
        @test M3.w.value == M2.w.value
    end
end
       

