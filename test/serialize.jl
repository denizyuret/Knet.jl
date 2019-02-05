include("header.jl")
import Knet: serialize
struct M370; layer; end;

@testset "serialize" begin
    M1 = RNN(2,3)
    M2 = M1 |> cpucopy
    @test typeof(M2.w.value) <: Array
    @test M2.w.value == M1.w.value
    @test Array{Float32} == serialize(Array{Float32})
    if gpu() >= 0 
        M3 = M2 |> gpucopy
        @test typeof(M3.w.value) <: KnetArray
        @test M3.w.value == M2.w.value
        array_of_ka = map(value,params(M3))
        array_of_ca = array_of_ka |> cpucopy
        @test first(array_of_ca) isa Array
        
        # 370-1
        m = M370(param(5,5,1,1))
        mcpu = m |> cpucopy
        @test Knet.save("foo.jld2","mcpu",mcpu) === nothing

        # 370-2
        @test conv4(mcpu.layer,randn(Float32,20,20,1,1)) isa Array
        
    end
end


