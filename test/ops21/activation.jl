using Test, CUDA, AutoGrad
using Knet.Ops21

@testset "ops21/activation" begin

    function acttest(act; atype=Array)
        ax = Param(randn(5,4,3,2))
        if atype === Array
            @gcheck act.(ax)
        else
            cx = Param(convert(atype, ax.value))
            ((@gcheck act.(ax)) &&
             (@gcheck act.(cx)) &&
             (act.(ax) ≈ act.(cx)))
        end
    end

    @test acttest(elu)
    @test acttest(gelu)
    @test acttest(relu)
    @test acttest(selu)
    @test acttest(sigm)
    @test acttest(swish)
    @test acttest(tanh)

    if CUDA.functional()
        @test acttest(elu; atype=CuArray)
        @test acttest(gelu; atype=CuArray)
        @test acttest(relu; atype=CuArray)
        @test acttest(selu; atype=CuArray)
        @test acttest(sigm; atype=CuArray)
        @test acttest(swish; atype=CuArray)
        @test acttest(tanh; atype=CuArray)

        @test acttest(elu; atype=KnetArray)
        @test acttest(gelu; atype=KnetArray)
        @test acttest(relu; atype=KnetArray)
        @test acttest(selu; atype=KnetArray)
        @test acttest(sigm; atype=KnetArray)
        @test acttest(swish; atype=KnetArray)
        @test acttest(tanh; atype=KnetArray)
    end
end
