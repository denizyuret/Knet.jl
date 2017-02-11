# TODO: implement cpu versions and compare
# TODO: any way to compare conv4 vs deconv4 or pool vs unpool?
# TODO: unpool gradient broken
# TODO: pool with alpha=2 broken

using Base.Test, Knet

conv41(a;o...)=conv4(a[1],a[2];o...)
deconv41(a;o...)=deconv4(a[1],a[2];o...)

@testset "conv" begin
    # Default
    x = KnetArray(rand(5,4,3,2))
    w = KnetArray(rand(3,3,3,3))
    @test gradcheck(pool, x)
    @test_broken gradcheck(unpool, x)
    @test gradcheck(conv41, (w,x); rtol=0.05)
    @test gradcheck(deconv41, (w,x); rtol=0.05)
    # Float32
    x32 = convert(KnetArray{Float32}, x)
    w32 = convert(KnetArray{Float32}, w)
    @test gradcheck(pool, x32)
    @test_broken gradcheck(unpool, x32)
    @test gradcheck(conv41, (w32,x32); rtol=0.05)
    @test gradcheck(deconv41, (w32,x32); rtol=0.05)
    # 5D
    x5 = KnetArray(rand(6,5,4,3,2))
    w5 = KnetArray(rand(3,3,3,3,3))
    @test gradcheck(pool, x5)
    @test_broken gradcheck(unpool, x5)
    @test gradcheck(conv41, (w5,x5); rtol=0.05)
    @test gradcheck(deconv41, (w5,x5); rtol=0.05)
    # window=3 (default=2) only for pool
    @test gradcheck(pool, x; kwargs=[:window=>3])
    @test_broken gradcheck(unpool, x; kwargs=[:window=>3])
    # padding=1 (default=0)
    @test gradcheck(pool, x; kwargs=[:padding=>1])
    @test_broken gradcheck(unpool, x; kwargs=[:padding=>1])
    @test gradcheck(conv41, (w,x); rtol=0.05, kwargs=[:padding=>1])
    @test gradcheck(deconv41, (w,x); rtol=0.05, kwargs=[:padding=>1])
    # stride=3 (default=1 for conv, window=2 for pool)
    @test gradcheck(pool, x; kwargs=[:stride=>3])
    @test_broken gradcheck(unpool, x; kwargs=[:stride=>3])
    @test gradcheck(conv41, (w,x); rtol=0.05, kwargs=[:stride=>3])
    @test gradcheck(deconv41, (w,x); rtol=0.05, kwargs=[:stride=>3])
    # mode=1 (default=0)
    @test gradcheck(pool, x; kwargs=[:mode=>1])
    @test_broken gradcheck(unpool, x; kwargs=[:mode=>1])
    @test gradcheck(conv41, (w,x); rtol=0.05, kwargs=[:mode=>1])
    @test gradcheck(deconv41, (w,x); rtol=0.05, kwargs=[:mode=>1])
    # mode=2 (only for pool)
    @test gradcheck(pool, x; kwargs=[:mode=>2])
    @test_broken gradcheck(unpool, x; kwargs=[:mode=>2])
    # alpha=2 (default=1)
    @test_broken gradcheck(pool, x; kwargs=[:alpha=>2])
    @test_broken gradcheck(unpool, x; kwargs=[:alpha=>2])
    @test gradcheck(conv41, (w,x); rtol=0.05, kwargs=[:alpha=>2])
    @test gradcheck(deconv41, (w,x); rtol=0.05, kwargs=[:alpha=>2])
end

nothing
