using Test, Random
using Knet.Ops20: conv4, deconv4, pool, unpool
using CUDA: CUDA, functional, seed!
using Knet.KnetArrays: KnetArray
using AutoGrad: gradcheck, Param

CUDA.functional() && CUDA.seed!(42); Random.seed!(42);
struct M370; layer; end;

@testset "conv" begin

    conv41(a;o...)=conv4(a[1],a[2];o...)
    deconv41(a;o...)=deconv4(a[1],a[2];o...)
    rand41(d...)=reshape(0.01*collect(Float64,1:prod(d)),d)

    TOL=0.2
    ax = rand41(5,4,3,2)
    aw = rand41(3,3,3,4)
    ad = permutedims(aw, (1,2,4,3))
    ax32 = convert(Array{Float32}, ax)
    aw32 = convert(Array{Float32}, aw)
    ad32 = convert(Array{Float32}, ad)
    ax5 = rand41(6,5,4,3,2)
    aw5 = rand41(3,3,3,3,3)
    if CUDA.functional()
        kx = KnetArray(ax)
        kw = KnetArray(aw)
        kd = KnetArray(ad)
        kx32 = KnetArray(ax32)
        kw32 = KnetArray(aw32)
        kd32 = KnetArray(ad32)
        kx5 = KnetArray(ax5)
        kw5 = KnetArray(aw5)
    end

    # @warn "cpuconv tests commented out until we figure out why they are failing on gitlab-ci"

    @testset "cpuconv" begin
        ### Default
        @test gradcheck(pool, ax)
        @test gradcheck(unpool, ax)
        @test isapprox(pool(unpool(ax)),ax)
        @test gradcheck(conv41, (aw,ax); rtol=TOL)
        @test gradcheck(deconv41, (ad,ax); rtol=TOL)

        ### Float32
        @test gradcheck(pool, ax32; rtol=TOL) # TODO: sensitive to seed
        @test gradcheck(unpool, ax32; rtol=TOL) # TODO: sensitive to seed
        @test isapprox(pool(unpool(ax32)),ax32)
        @test gradcheck(conv41, (aw32,ax32); rtol=0.5) # TODO: sensitive to seed
        @test gradcheck(deconv41, (ad32,ax32); rtol=TOL) # TODO: sensitive to seed
        
        ### 5D
        @test gradcheck(pool, ax5)
        @test gradcheck(unpool, ax5)
        @test isapprox(pool(unpool(ax5)),ax5)
        @test gradcheck(conv41, (aw5,ax5); rtol=TOL)
        @test gradcheck(deconv41, (aw5,ax5); rtol=TOL)

        ### window=3 (default=2) only for pool
        @test gradcheck(pool, ax; kw=[(:window,3)])
        @test gradcheck(unpool, ax; kw=[(:window,3)])
        @test isapprox(pool(unpool(ax;window=3);window=3),ax)
        @test gradcheck(pool, ax; kw=[(:window,(3,3))])
        @test gradcheck(unpool, ax; kw=[(:window,(3,3))])
        @test isapprox(pool(unpool(ax;window=(3,3));window=(3,3)),ax)

        ### padding=1 (default=0)
        @test gradcheck(pool, ax; kw=[(:padding,1)])
        @test gradcheck(unpool, ax; kw=[(:padding,1)])
        @test isapprox(pool(unpool(ax;padding=1);padding=1),ax)
        @test gradcheck(conv41, (aw,ax); rtol=TOL, kw=[(:padding,1)])
        @test gradcheck(deconv41, (ad,ax); rtol=TOL, kw=[(:padding,1)])
        @test gradcheck(pool, ax; kw=[(:padding,(1,1))])
        @test gradcheck(unpool, ax; kw=[(:padding,(1,1))])
        @test isapprox(pool(unpool(ax;padding=(1,1));padding=(1,1)),ax)
        @test gradcheck(conv41, (aw,ax); rtol=TOL, kw=[(:padding,(1,1))])
        @test gradcheck(deconv41, (ad,ax); rtol=TOL, kw=[(:padding,(1,1))])

        ### stride=3 (default=1 for conv, window=2 for pool)
        @test gradcheck(pool, ax; kw=[(:stride,3)])
        @test gradcheck(unpool, ax; kw=[(:stride,3)])
        @test isapprox(pool(unpool(ax;stride=3);stride=3),ax)
        @test gradcheck(conv41, (aw,ax); rtol=TOL, kw=[(:stride,3)])
        @test gradcheck(deconv41, (ad,ax); rtol=TOL, kw=[(:stride,3)])
        @test gradcheck(pool, ax; kw=[(:stride,(3,3))])
        @test gradcheck(unpool, ax; kw=[(:stride,(3,3))])
        @test isapprox(pool(unpool(ax;stride=(3,3));stride=(3,3)),ax)
        @test gradcheck(conv41, (aw,ax); rtol=TOL, kw=[(:stride,(3,3))])
        @test gradcheck(deconv41, (ad,ax); rtol=TOL, kw=[(:stride,(3,3))])

        ### mode=1 (default=0)
        @test gradcheck(pool, ax; kw=[(:mode,1),(:padding,1)])
        @test gradcheck(unpool, ax; kw=[(:mode,1),(:padding,1)])
        @test isapprox(pool(unpool(ax;mode=1);mode=1),ax)
        @test_broken isapprox(pool(unpool(ax;mode=1,padding=1);mode=1,padding=1),ax)
        @test gradcheck(conv41, (aw,ax); rtol=TOL, kw=[(:mode,1),(:padding,1)])
        @test gradcheck(deconv41, (ad,ax); rtol=TOL, kw=[(:mode,1),(:padding,1)])

        ### mode=2 (only for pool) -- is not supported in NNlib #218
        # @test gradcheck(pool, ax; kw=[(:mode,2),(:padding,1)])
        # @test gradcheck(unpool, ax; kw=[(:mode,2),(:padding,1)])
        # @test isapprox(pool(unpool(ax;mode=2);mode=2),ax)
        # @test isapprox(pool(unpool(ax;mode=2,padding=1);mode=2,padding=1),ax)

        ### alpha=2 (default=1)
        @test gradcheck(pool, ax; kw=[(:alpha,2)])
        @test gradcheck(unpool, ax; kw=[(:alpha,2)])
        @test isapprox(pool(unpool(ax;alpha=2);alpha=2),ax)
        @test gradcheck(pool, ax; kw=[(:alpha,2),(:mode,1),(:padding,1)])
        @test gradcheck(unpool, ax; kw=[(:alpha,2),(:mode,1),(:padding,1)])
        @test isapprox(pool(unpool(ax;alpha=2,mode=1);alpha=2,mode=1),ax)
        @test_broken isapprox(pool(unpool(ax;alpha=2,mode=1,padding=1);alpha=2,mode=1,padding=1),ax)
        @test gradcheck(conv41, (aw,ax); rtol=TOL, kw=[(:alpha,2)])
        @test gradcheck(deconv41, (ad,ax); rtol=TOL, kw=[(:alpha,2)])
    end

    if CUDA.functional(); @testset "gpuconv" begin
        ### Default
        @test isapprox(pool(kx), pool(ax))
        @test gradcheck(pool, kx)
        @test isapprox(unpool(kx), unpool(ax))
        @test gradcheck(unpool, kx)
        @test isapprox(conv4(kw,kx), conv4(aw,ax))
        @test gradcheck(conv41, (kw,kx); rtol=TOL)
        @test isapprox(deconv4(kd,kx), deconv4(ad,ax))
        @test gradcheck(deconv41, (kd,kx); rtol=TOL)

        ### Float32
        @test isapprox(pool(kx32), pool(ax32))
        @test gradcheck(pool, kx32)
        @test isapprox(unpool(kx32), unpool(ax32))
        @test gradcheck(unpool, kx32)  # TODO: sensitive to seed
        @test isapprox(conv4(kw32,kx32), conv4(aw32,ax32))
        @test gradcheck(conv41, (kw32,kx32); rtol=TOL)
        @test isapprox(deconv4(kd32,kx32), deconv4(ad32,ax32))
        @test gradcheck(deconv41, (kd32,kx32); rtol=TOL)

        ### 5D
        @test isapprox(pool(kx5), pool(ax5))
        @test gradcheck(pool, kx5)
        @test isapprox(unpool(kx5), unpool(ax5))
        @test gradcheck(unpool, kx5)
        @test isapprox(conv4(kw5,kx5), conv4(aw5,ax5))
        @test gradcheck(conv41, (kw5,kx5); rtol=TOL)
        @test isapprox(deconv4(kw5,kx5), deconv4(aw5,ax5))
        @test gradcheck(deconv41, (kw5,kx5); rtol=TOL)

        ### window=3 (default=2) only for pool
        @test isapprox(pool(kx;window=3), pool(ax;window=3))
        @test gradcheck(pool, kx; kw=[(:window,3)])
        @test isapprox(unpool(kx;window=3), unpool(ax;window=3))
        @test gradcheck(unpool, kx; kw=[(:window,3)])
        @test isapprox(pool(kx;window=(3,3)), pool(ax;window=(3,3)))
        @test gradcheck(pool, kx; kw=[(:window,(3,3))])
        @test isapprox(unpool(kx;window=(3,3)), unpool(ax;window=(3,3)))
        @test gradcheck(unpool, kx; kw=[(:window,(3,3))])

        ### padding=1 (default=0)
        @test isapprox(pool(kx;padding=1), pool(ax;padding=1))
        @test gradcheck(pool, kx; kw=[(:padding,1)])
        @test isapprox(unpool(kx;padding=1), unpool(ax;padding=1))
        @test gradcheck(unpool, kx; kw=[(:padding,1)])
        @test isapprox(conv4(kw,kx;padding=1), conv4(aw,ax;padding=1))
        @test gradcheck(conv41, (kw,kx); rtol=TOL, kw=[(:padding,1)])
        @test isapprox(deconv4(kd,kx;padding=1), deconv4(ad,ax;padding=1))
        @test gradcheck(deconv41, (kd,kx); rtol=TOL, kw=[(:padding,1)])

        @test isapprox(pool(kx;padding=(1,1)), pool(ax;padding=(1,1)))
        @test gradcheck(pool, kx; kw=[(:padding,(1,1))])
        @test isapprox(unpool(kx;padding=(1,1)), unpool(ax;padding=(1,1)))
        @test gradcheck(unpool, kx; kw=[(:padding,(1,1))])
        @test isapprox(conv4(kw,kx;padding=(1,1)), conv4(aw,ax;padding=(1,1)))
        @test gradcheck(conv41, (kw,kx); rtol=TOL, kw=[(:padding,(1,1))])
        @test isapprox(deconv4(kd,kx;padding=(1,1)), deconv4(ad,ax;padding=(1,1)))
        @test gradcheck(deconv41, (kd,kx); rtol=TOL, kw=[(:padding,(1,1))])

        ### stride=3 (default=1 for conv, window=2 for pool)
        @test isapprox(pool(kx;stride=3), pool(ax;stride=3))
        @test gradcheck(pool, kx; kw=[(:stride,3)])
        @test isapprox(unpool(kx;stride=3), unpool(ax;stride=3))
        @test gradcheck(unpool, kx; kw=[(:stride,3)])
        @test isapprox(conv4(kw,kx;stride=3), conv4(aw,ax;stride=3))
        @test gradcheck(conv41, (kw,kx); rtol=TOL, kw=[(:stride,3)])
        @test isapprox(deconv4(kd,kx;stride=3), deconv4(ad,ax;stride=3); rtol=1e-6)
        @test gradcheck(deconv41, (kd,kx); rtol=TOL, kw=[(:stride,3)])

        @test isapprox(pool(kx;stride=(3,3)), pool(ax;stride=(3,3)))
        @test gradcheck(pool, kx; kw=[(:stride,(3,3))])
        @test isapprox(unpool(kx;stride=(3,3)), unpool(ax;stride=(3,3)))
        @test gradcheck(unpool, kx; kw=[(:stride,(3,3))])
        @test isapprox(conv4(kw,kx;stride=(3,3)), conv4(aw,ax;stride=(3,3)))
        @test gradcheck(conv41, (kw,kx); rtol=TOL, kw=[(:stride,(3,3))])
        @test isapprox(deconv4(kd,kx;stride=(3,3)), deconv4(ad,ax;stride=(3,3)); rtol=1e-6)
        @test gradcheck(deconv41, (kd,kx); rtol=TOL, kw=[(:stride,(3,3))])

        ### mode=1 (default=0)
        @test isapprox(pool(kx;mode=1,padding=1), pool(ax;mode=1,padding=1))
        @test gradcheck(pool, kx; kw=[(:mode,1),(:padding,1)])
        @test isapprox(unpool(kx;mode=1,padding=1), unpool(ax;mode=1,padding=1))
        @test gradcheck(unpool, kx; kw=[(:mode,1),(:padding,1)])
        @test isapprox(conv4(kw,kx;mode=1,padding=1), conv4(aw,ax;mode=1,padding=1))
        @test gradcheck(conv41, (kw,kx); rtol=TOL, kw=[(:mode,1),(:padding,1)])
        @test isapprox(deconv4(kd,kx;mode=1,padding=1), deconv4(ad,ax;mode=1,padding=1))
        @test gradcheck(deconv41, (kd,kx); rtol=TOL, kw=[(:mode,1),(:padding,1)])

        ### mode=2 (only for pool)
        # @test isapprox(pool(kx;mode=2,padding=1), pool(ax;mode=2,padding=1)) ## mode=2 is not supported in NNlib #218.
        @test gradcheck(pool, kx; kw=[(:mode,2),(:padding,1)])
        # @test isapprox(unpool(kx;mode=2,padding=1), unpool(ax;mode=2,padding=1))  ## mode=2 is not supported in NNlib #218.
        @test gradcheck(unpool, kx; kw=[(:mode,2),(:padding,1)])

        ### alpha=2 (default=1)
        @test isapprox(pool(kx;alpha=2), pool(ax;alpha=2))
        @test gradcheck(pool, kx; kw=[(:alpha,2)])
        @test isapprox(unpool(kx;alpha=2), unpool(ax;alpha=2))
        @test gradcheck(unpool, kx; kw=[(:alpha,2)])
        @test isapprox(pool(kx;alpha=2,mode=1,padding=1), pool(ax;alpha=2,mode=1,padding=1))
        @test gradcheck(pool, kx; kw=[(:alpha,2),(:mode,1),(:padding,1)])
        @test isapprox(unpool(kx;alpha=2,mode=1,padding=1), unpool(ax;alpha=2,mode=1,padding=1))
        @test gradcheck(unpool, kx; kw=[(:alpha,2),(:mode,1),(:padding,1)])

        # @test isapprox(pool(kx;alpha=2,mode=2,padding=1), pool(ax;alpha=2,mode=2,padding=1)) ## mode=2 is not supported in NNlib #218.
        @test gradcheck(pool, kx; kw=[(:alpha,2),(:mode,2),(:padding,1)]) 
        # @test isapprox(unpool(kx;alpha=2,mode=2,padding=1), unpool(ax;alpha=2,mode=2,padding=1)) ## mode=2 is not supported in NNlib #218.
        @test gradcheck(unpool, kx; kw=[(:alpha,2),(:mode,2),(:padding,1)])

        @test isapprox(conv4(kw,kx;alpha=2), conv4(aw,ax;alpha=2))
        @test gradcheck(conv41, (kw,kx); rtol=TOL, kw=[(:alpha,2)])
        @test isapprox(deconv4(kd,kx;alpha=2), deconv4(ad,ax;alpha=2))
        @test gradcheck(deconv41, (kd,kx); rtol=TOL, kw=[(:alpha,2)])

        # 370-3: This test may not be possible with CUDA.jl, once CUDA.functional() there is no way to turn it off
        # Tim Besard: there's has_cuda and has_cuda_gpu for higher level functionality
        # you could run julia with CUDA_VISIBLE_DEVICES=
        # or using docker, blocking access to the GPUs
        #= 
        m = M370(Param(KnetArray(randn(Float32,5,5,1,1))))
        path = tempname()*".jld2"
        Knet.save(path,"m",m)
        gpusave = gpu()
        gpu(-1)
        mcpu = Knet.load(path,"m")
        @test conv4(mcpu.layer,randn(Float32,20,20,1,1)) isa Array
        gpu(gpusave)
        rm(path)
        =#
    end
    end
end

nothing
