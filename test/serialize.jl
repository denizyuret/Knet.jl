include("header.jl")
using Knet: serialize
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
        
        # 506
        function m1test(M1,xgpu,xcpu)
            Knet.save("/tmp/foo.jld2", "model", M1)
            Ms = Knet.load("/tmp/foo.jld2", "model")
            Mg = gpucopy(M1)
            Mc = cpucopy(M1)
            y,h,c = M1(xgpu),M1.h,M1.c
            @test (y,h,c) == (Ms(xgpu),Ms.h,Ms.c)
            @test (y,h,c) == (Mg(xgpu),Mg.h,Mg.c)
            @test_broken (y,h,c) == (Mc(xcpu),Mc.h,Mc.c)
        end
        xcpu = randn(Float32,2,4,8)
        xgpu = KnetArray(xcpu)
        M1 = RNN(2,3)
        M1.h = M1.c = 0
        M1.dx = M1.dhx = M1.dcx = nothing
        m1test(M1,xgpu,xcpu)  # sets M1.h,c
        @test M1.h isa KnetArray
        @test M1.c isa KnetArray
        m1test(M1,xgpu,xcpu)
        @diff sum(M1(xgpu)) # sets M1.dx,dhx,dcx
        @test M1.dx isa KnetArray
        @test M1.dhx isa KnetArray
        @test M1.dcx isa KnetArray
        m1test(M1,xgpu,xcpu)
    end
end


