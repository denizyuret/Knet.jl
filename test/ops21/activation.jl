using Test, CUDA, AutoGrad
using Knet.KnetArrays: KnetArray
using Knet.Ops21

@testset "ops21/activation" begin

    function acttest(act; atype=Array, o...)
        ax = Param(randn(3,4))
        if atype === Array
            ((act(ax; o...) ≈ act.(ax; o...)) &&
             (@gcheck act(ax[1]; o...)) &&
             (@gcheck act(ax; o...)))
            # TODO check back functions' derivatives
        else
            cx = Param(convert(atype, ax.value))
            ((act.(ax; o...) ≈ act(ax; o...)) &&
             (act(cx; o...) ≈ act(ax; o...)) &&
             (@gcheck act(ax[1]; o...)) &&
             (@gcheck act(ax; o...)) &&
             (@gcheck act(cx; o...)))
        end
    end

    @test acttest(elu)
    @test acttest(gelu)
    @test acttest(hardsigmoid)
    @test acttest(hardswish)
    @test acttest(relu)
    @test acttest(relu; max_value=1)
    @test acttest(relu; negative_slope=0.5)
    @test acttest(relu; threshold=1)
    @test acttest(selu)
    @test acttest(sigm)
    @test acttest(swish)
    @test acttest(tanh_)

    if CUDA.functional()
        @test acttest(elu; atype=CuArray)
        @test acttest(gelu; atype=CuArray)
        @test acttest(hardsigmoid; atype=CuArray)
        @test acttest(hardswish; atype=CuArray)
        @test acttest(relu; atype=CuArray)
        @test acttest(relu; atype=CuArray, max_value=1)
        @test acttest(relu; atype=CuArray, negative_slope=0.5)
        @test acttest(relu; atype=CuArray, threshold=1)
        @test acttest(selu; atype=CuArray)
        @test acttest(sigm; atype=CuArray)
        @test acttest(swish; atype=CuArray)
        @test acttest(tanh_; atype=CuArray)

        @test acttest(elu; atype=KnetArray)
        @test acttest(gelu; atype=KnetArray)
        @test acttest(hardsigmoid; atype=KnetArray)
        @test acttest(hardswish; atype=KnetArray)
        @test acttest(relu; atype=KnetArray)
        @test acttest(relu; atype=KnetArray, max_value=1)
        @test acttest(relu; atype=KnetArray, negative_slope=0.5)
        @test acttest(relu; atype=KnetArray, threshold=1)
        @test acttest(selu; atype=KnetArray)
        @test acttest(sigm; atype=KnetArray)
        @test acttest(swish; atype=KnetArray)
        @test acttest(tanh_; atype=KnetArray)
    end
end
