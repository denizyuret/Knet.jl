include("header.jl")

conv41(a;o...)=conv4(a[1],a[2];o...)
deconv41(a;o...)=deconv4(a[1],a[2];o...)
rand41(d...)=reshape(0.01*collect(Float64,1:prod(d)),d)

ax = rand41(5,4,3,2)
aw = rand41(3,3,3,3)
ax32 = convert(Array{Float32}, ax)
aw32 = convert(Array{Float32}, aw)
ax5 = rand41(6,5,4,3,2)
aw5 = rand41(3,3,3,3,3)
if gpu() >= 0
    kx = KnetArray(ax)
    kw = KnetArray(aw)
    kx32 = KnetArray(ax32)
    kw32 = KnetArray(aw32)
    kx5 = KnetArray(ax5)
    kw5 = KnetArray(aw5)
end

@testset "conv" begin
@testset "cpuconv" begin
    ### Default
    @test gradcheck(pool, ax)
    @test gradcheck(unpool, ax)
    @test gradcheck(conv41, (aw,ax); rtol=0.05)
    @test gradcheck(deconv41, (aw,ax); rtol=0.05)

    ### Float32
    @test gradcheck(pool, ax32)
    @test gradcheck(unpool, ax32)
    @test gradcheck(conv41, (aw32,ax32); rtol=0.05)
    @test gradcheck(deconv41, (aw32,ax32); rtol=0.05)

    ### 5D
    #FAIL @test gradcheck(pool, ax5)
    #FAIL @test gradcheck(unpool, ax5)
    #FAIL @test gradcheck(conv41, (aw5,ax5); rtol=0.05)
    #FAIL @test gradcheck(deconv41, (aw5,ax5); rtol=0.05)

    ### window=3 (default=2) only for pool
    @test gradcheck(pool, ax; kwargs=[(:window,3)])
    @test gradcheck(unpool, ax; kwargs=[(:window,3)])
    @test gradcheck(pool, ax; kwargs=[(:window,(3,3))])
    @test gradcheck(unpool, ax; kwargs=[(:window,(3,3))])

    ### padding=1 (default=0)
    @test gradcheck(pool, ax; kwargs=[(:padding,1)])
    @test gradcheck(unpool, ax; kwargs=[(:padding,1)])
    @test gradcheck(conv41, (aw,ax); rtol=0.05, kwargs=[(:padding,1)])
    @test gradcheck(deconv41, (aw,ax); rtol=0.05, kwargs=[(:padding,1)])
    @test gradcheck(pool, ax; kwargs=[(:padding,(1,1))])
    @test gradcheck(unpool, ax; kwargs=[(:padding,(1,1))])
    @test gradcheck(conv41, (aw,ax); rtol=0.05, kwargs=[(:padding,(1,1))])
    @test gradcheck(deconv41, (aw,ax); rtol=0.05, kwargs=[(:padding,(1,1))])

    ### stride=3 (default=1 for conv, window=2 for pool)
    @test gradcheck(pool, ax; kwargs=[(:stride,3)])
    @test gradcheck(unpool, ax; kwargs=[(:stride,3)])
    @test gradcheck(conv41, (aw,ax); rtol=0.05, kwargs=[(:stride,3)])
    @test gradcheck(deconv41, (aw,ax); rtol=0.05, kwargs=[(:stride,3)])
    @test gradcheck(pool, ax; kwargs=[(:stride,(3,3))])
    @test gradcheck(unpool, ax; kwargs=[(:stride,(3,3))])
    @test gradcheck(conv41, (aw,ax); rtol=0.05, kwargs=[(:stride,(3,3))])
    @test gradcheck(deconv41, (aw,ax); rtol=0.05, kwargs=[(:stride,(3,3))])

    ### mode=1 (default=0)
    @test gradcheck(pool, ax; kwargs=[(:mode,1)])
    @test gradcheck(unpool, ax; kwargs=[(:mode,1)])
    @test gradcheck(conv41, (aw,ax); rtol=0.05, kwargs=[(:mode,1)])
    @test gradcheck(deconv41, (aw,ax); rtol=0.05, kwargs=[(:mode,1)])

    ### mode=2 (only for pool)
    @test gradcheck(pool, ax; kwargs=[(:mode,2)])
    @test gradcheck(unpool, ax; kwargs=[(:mode,2)])

    ### alpha=2 (default=1)
    @test gradcheck(pool, ax; kwargs=[(:alpha,2)])
    @test gradcheck(unpool, ax; kwargs=[(:alpha,2)])
    @test gradcheck(pool, ax; kwargs=[(:alpha,2),(:mode,1)])
    @test gradcheck(unpool, ax; kwargs=[(:alpha,2),(:mode,1)])
    @test gradcheck(conv41, (aw,ax); rtol=0.05, kwargs=[(:alpha,2)])
    @test gradcheck(deconv41, (aw,ax); rtol=0.05, kwargs=[(:alpha,2)])
end
if gpu() >= 0; @testset "gpuconv" begin
    ### Default
    @test isapprox(pool(kx), pool(ax))
    @test gradcheck(pool, kx)
    @test isapprox(unpool(kx), unpool(ax))
    @test gradcheck(unpool, kx)
    @test isapprox(conv4(kw,kx), conv4(aw,ax))
    @test gradcheck(conv41, (kw,kx); rtol=0.05)
    @test isapprox(deconv4(kw,kx), deconv4(aw,ax))
    @test gradcheck(deconv41, (kw,kx); rtol=0.05)

    ### Float32
    @test isapprox(pool(kx32), pool(ax32))
    @test gradcheck(pool, kx32)
    @test isapprox(unpool(kx32), unpool(ax32))
    @test gradcheck(unpool, kx32)
    @test isapprox(conv4(kw32,kx32), conv4(aw32,ax32))
    @test gradcheck(conv41, (kw32,kx32); rtol=0.05)
    @test isapprox(deconv4(kw32,kx32), deconv4(aw32,ax32))
    @test gradcheck(deconv41, (kw32,kx32); rtol=0.05)

    ### 5D
    #FAIL @test isapprox(pool(kx5), pool(ax5))
    @test gradcheck(pool, kx5)
    #FAIL @test isapprox(unpool(kx5), unpool(ax5))
    @test gradcheck(unpool, kx5)
    #FAIL @test isapprox(conv4(kw5,kx5), conv4(aw5,ax5))
    @test gradcheck(conv41, (kw5,kx5); rtol=0.05)
    #FAIL @test isapprox(deconv4(kw5,kx5), deconv4(aw5,ax5))
    @test gradcheck(deconv41, (kw5,kx5); rtol=0.05)

    ### window=3 (default=2) only for pool
    @test isapprox(pool(kx;window=3), pool(ax;window=3))
    @test gradcheck(pool, kx; kwargs=[(:window,3)])
    @test isapprox(unpool(kx;window=3), unpool(ax;window=3))
    @test gradcheck(unpool, kx; kwargs=[(:window,3)])
    @test isapprox(pool(kx;window=(3,3)), pool(ax;window=(3,3)))
    @test gradcheck(pool, kx; kwargs=[(:window,(3,3))])
    @test isapprox(unpool(kx;window=(3,3)), unpool(ax;window=(3,3)))
    @test gradcheck(unpool, kx; kwargs=[(:window,(3,3))])

    ### padding=1 (default=0)
    @test isapprox(pool(kx;padding=1), pool(ax;padding=1))
    @test gradcheck(pool, kx; kwargs=[(:padding,1)])
    @test isapprox(unpool(kx;padding=1), unpool(ax;padding=1))
    @test gradcheck(unpool, kx; kwargs=[(:padding,1)])
    @test isapprox(conv4(kw,kx;padding=1), conv4(aw,ax;padding=1))
    @test gradcheck(conv41, (kw,kx); rtol=0.05, kwargs=[(:padding,1)])
    @test isapprox(deconv4(kw,kx;padding=1), deconv4(aw,ax;padding=1))
    @test gradcheck(deconv41, (kw,kx); rtol=0.05, kwargs=[(:padding,1)])

    @test isapprox(pool(kx;padding=(1,1)), pool(ax;padding=(1,1)))
    @test gradcheck(pool, kx; kwargs=[(:padding,(1,1))])
    @test isapprox(unpool(kx;padding=(1,1)), unpool(ax;padding=(1,1)))
    @test gradcheck(unpool, kx; kwargs=[(:padding,(1,1))])
    @test isapprox(conv4(kw,kx;padding=(1,1)), conv4(aw,ax;padding=(1,1)))
    @test gradcheck(conv41, (kw,kx); rtol=0.05, kwargs=[(:padding,(1,1))])
    @test isapprox(deconv4(kw,kx;padding=(1,1)), deconv4(aw,ax;padding=(1,1)))
    @test gradcheck(deconv41, (kw,kx); rtol=0.05, kwargs=[(:padding,(1,1))])

    ### stride=3 (default=1 for conv, window=2 for pool)
    @test isapprox(pool(kx;stride=3), pool(ax;stride=3))
    @test gradcheck(pool, kx; kwargs=[(:stride,3)])
    @test isapprox(unpool(kx;stride=3), unpool(ax;stride=3))
    @test gradcheck(unpool, kx; kwargs=[(:stride,3)])
    @test isapprox(conv4(kw,kx;stride=3), conv4(aw,ax;stride=3))
    @test gradcheck(conv41, (kw,kx); rtol=0.05, kwargs=[(:stride,3)])
    @test isapprox(deconv4(kw,kx;stride=3), deconv4(aw,ax;stride=3); rtol=1e-6)
    @test gradcheck(deconv41, (kw,kx); rtol=0.05, kwargs=[(:stride,3)])

    @test isapprox(pool(kx;stride=(3,3)), pool(ax;stride=(3,3)))
    @test gradcheck(pool, kx; kwargs=[(:stride,(3,3))])
    @test isapprox(unpool(kx;stride=(3,3)), unpool(ax;stride=(3,3)))
    @test gradcheck(unpool, kx; kwargs=[(:stride,(3,3))])
    @test isapprox(conv4(kw,kx;stride=(3,3)), conv4(aw,ax;stride=(3,3)))
    @test gradcheck(conv41, (kw,kx); rtol=0.05, kwargs=[(:stride,(3,3))])
    @test isapprox(deconv4(kw,kx;stride=(3,3)), deconv4(aw,ax;stride=(3,3)); rtol=1e-6)
    @test gradcheck(deconv41, (kw,kx); rtol=0.05, kwargs=[(:stride,(3,3))])

    ### mode=1 (default=0)
    @test isapprox(pool(kx;mode=1), pool(ax;mode=1))
    @test gradcheck(pool, kx; kwargs=[(:mode,1)])
    @test isapprox(unpool(kx;mode=1), unpool(ax;mode=1))
    @test gradcheck(unpool, kx; kwargs=[(:mode,1)])
    @test isapprox(conv4(kw,kx;mode=1), conv4(aw,ax;mode=1))
    @test gradcheck(conv41, (kw,kx); rtol=0.05, kwargs=[(:mode,1)])
    @test isapprox(deconv4(kw,kx;mode=1), deconv4(aw,ax;mode=1))
    @test gradcheck(deconv41, (kw,kx); rtol=0.05, kwargs=[(:mode,1)])

    ### mode=2 (only for pool)
    @test isapprox(pool(kx;mode=2), pool(ax;mode=2))
    @test gradcheck(pool, kx; kwargs=[(:mode,2)])
    @test isapprox(unpool(kx;mode=2), unpool(ax;mode=2))
    @test gradcheck(unpool, kx; kwargs=[(:mode,2)])

    ### alpha=2 (default=1)
    @test isapprox(pool(kx;alpha=2), pool(ax;alpha=2))
    #FAIL @test gradcheck(pool, kx; kwargs=[(:alpha,2)]) # CUDNN bug
    @test isapprox(unpool(kx;alpha=2), unpool(ax;alpha=2))
    @test gradcheck(unpool, kx; kwargs=[(:alpha,2)])
    @test isapprox(pool(kx;alpha=2,mode=1), pool(ax;alpha=2,mode=1))
    @test gradcheck(pool, kx; kwargs=[(:alpha,2),(:mode,1)])
    @test isapprox(unpool(kx;alpha=2,mode=1), unpool(ax;alpha=2,mode=1))
    @test gradcheck(unpool, kx; kwargs=[(:alpha,2),(:mode,1)])
    @test isapprox(pool(kx;alpha=2,mode=2), pool(ax;alpha=2,mode=2))
    @test gradcheck(pool, kx; kwargs=[(:alpha,2),(:mode,2)])
    @test isapprox(unpool(kx;alpha=2,mode=2), unpool(ax;alpha=2,mode=2))
    @test gradcheck(unpool, kx; kwargs=[(:alpha,2),(:mode,2)])
    @test isapprox(conv4(kw,kx;alpha=2), conv4(aw,ax;alpha=2))
    @test gradcheck(conv41, (kw,kx); rtol=0.05, kwargs=[(:alpha,2)])
    @test isapprox(deconv4(kw,kx;alpha=2), deconv4(aw,ax;alpha=2))
    @test gradcheck(deconv41, (kw,kx); rtol=0.05, kwargs=[(:alpha,2)])
end
end
end

nothing
