using Test, Random, FileIO, JLD2
using Knet.KnetArrays: KnetArray, Cptr, jld2serialize, cpucopy, gpucopy
using Knet.Ops20: RNN, conv4
using CUDA: CUDA, functional
using AutoGrad: Param, params, @diff, value

struct M370; layer; end;

@testset "jld2serialize" begin
    M1 = RNN(2,3)
    M2 = M1 |> cpucopy
    @test typeof(M2.w.value) <: Array
    @test M2.w.value == M1.w.value
    @test Array{Float32} == jld2serialize(Array{Float32})
    if CUDA.functional()
        M3 = M2 |> gpucopy
        @test typeof(M3.w.value) <: KnetArray
        @test M3.w.value == M2.w.value
        array_of_ka = map(value,params(M3))
        array_of_ca = array_of_ka |> cpucopy
        @test first(array_of_ca) isa Array
        
        # 370-1
        m = M370(Param(KnetArray(randn(Float32,5,5,1,1))))
        mcpu = m |> cpucopy
        path = tempname()*".jld2"
        @test save(path,"mcpu",mcpu) === nothing

        # 370-2
        @test conv4(mcpu.layer,randn(Float32,20,20,1,1)) isa Array
        
        # 506
        function m1test(M1,xgpu,xcpu)
            path = tempname()*".jld2"
            save(path, "model", M1)
            Ms = load(path, "model")
            Mg = gpucopy(deepcopy(M1))
            Mc = cpucopy(M1)
            y,h,c = M1(xgpu),M1.h,M1.c
            @test (y,h,c) == (Ms(xgpu),Ms.h,Ms.c)
            @test (y,h,c) == (Mg(xgpu),Mg.h,Mg.c)
            @test all(isapprox(a,b) for (a,b) in ((y, Mc(xcpu)), (h, Mc.h), (c, Mc.c)))
        end
        xcpu = randn(Float32,2,4,8)
        xgpu = KnetArray(xcpu)
        M1 = RNN(2,3; atype=KnetArray{Float32})
        M1.h = M1.c = 0
        M1.dx = M1.dhx = M1.dcx = nothing
        m1test(M1,xgpu,xcpu)  # sets M1.h,c
        @test all(p isa KnetArray for p in (M1.h, M1.c))
        m1test(M1,xgpu,xcpu)
        @diff sum(M1(xgpu)) # sets M1.dx,dhx,dcx
        @test all(p isa KnetArray for p in (M1.dx, M1.dhx, M1.dcx))
        m1test(M1,xgpu,xcpu)
    end
end


