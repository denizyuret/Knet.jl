using Base.Test, Knet

conv41(a;o...)=conv4(a[1],a[2];o...)
deconv41(a;o...)=deconv4(a[1],a[2];o...)

@testset "conv" begin
    for T in (Float32,Float64)
        for D in (4,5)
            xdims = ((D+1):-1:2...)
            x = rand!(KnetArray(T,xdims))
            @test gradcheck(pool, x)
            @test_broken gradcheck(unpool, x)
            wdims = ntuple(i->3,D)
            w = rand!(KnetArray(T,wdims))
            @test gradcheck(conv41, (w,x))
            # @test gradcheck(deconv41, (w,x))
        end
    end
    # TODO: other keyword args
    # TODO: implement cpu versions and compare
    # TODO: any way to compare conv4 vs deconv4 or pool vs unpool?
end

nothing
