include("header.jl")

if gpu() >= 0; @testset "gcnode" begin

    # 506: knetgcnode garbage collects rnn fields
    save_gcnode = AutoGrad.gcnode
    AutoGrad.set_gc_function(Knet.knetgcnode)
    M1 = RNN(2,3)
    M1.h = M1.c = 0
    M1.dx = M1.dhx = M1.dcx = nothing
    M1(xgpu) # sets M1.h,c
    @diff sum(M1(xgpu)) # sets M1.h,c,dx,dhx,dcx
    @test_broken pointer(M1.h) != C_NULL
    @test_broken pointer(M1.c) != C_NULL
    @test pointer(M1.dx) != C_NULL
    @test pointer(M1.dhx) != C_NULL
    @test pointer(M1.dcx) != C_NULL
    AutoGrad.set_gc_function(save_gcnode)
end; end
