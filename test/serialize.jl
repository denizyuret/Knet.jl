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

        # 370-1
        struct Model; layer; end;
        m = Model(param(5,5,1,1))
        mcpu = m |> cpucopy
        @test Knet.save("foo.jld2","mcpu",mcpu) === nothing

        # 370-2
        @test conv4(mcpu.layer,randn(Float32,20,20,1,1)) isa Array
        
    end
end


