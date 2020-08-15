using Test
using CUDA: CUDA, functional
using AutoGrad: AutoGrad, gcnode, set_gc_function, @diff
using Knet.AutoGrad_gpu: knetgcnode
using Knet.KnetArrays: KnetArray
using Knet.Ops20: RNN

if CUDA.functional(); @testset "gcnode" begin

    # 506: knetgcnode garbage collects rnn fields
    save_gcnode = AutoGrad.gcnode
    set_gc_function(knetgcnode)
    M1 = RNN(2,3)
    M1.h = M1.c = 0
    M1.dx = M1.dhx = M1.dcx = nothing
    xcpu = randn(Float32,2,4,8)
    xgpu = KnetArray(xcpu)
    M1(xgpu) # sets M1.h,c
    @diff sum(M1(xgpu)) # sets M1.h,c,dx,dhx,dcx
    @test pointer(M1.h) != C_NULL
    @test pointer(M1.c) != C_NULL
    @test pointer(M1.dx) != C_NULL
    @test pointer(M1.dhx) != C_NULL
    @test pointer(M1.dcx) != C_NULL
    set_gc_function(save_gcnode)

end; end
