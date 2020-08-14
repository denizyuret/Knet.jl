using Test
using Knet.Ops20: logp, logsumexp, softmax, ∇softmax, logsoftmax, ∇logsoftmax
using Knet.Ops20_gpu: _cudnnSoftmaxForward, _cudnnSoftmaxBackward
using Knet.KnetArrays: KnetArray
using CUDA: CUDA, functional
using CUDA.CUDNN: CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_LOG

@testset "loss" begin
    for f in (logp, logsumexp)
        a = rand(10,10)
        @test gradcheck(f,a)
        @test gradcheck(f,a,kw=(:dims=>1,))
        @test gradcheck(f,a,kw=(:dims=>2,))
        if CUDA.functional()
            k = KnetArray(a)
            @test gradcheck(f,k)
            @test gradcheck(f,k,kw=(:dims=>1,))
            @test gradcheck(f,k,kw=(:dims=>2,))
            @test isapprox(f(a),f(k))
            @test isapprox(f(a,dims=1),f(k,dims=1))
            @test isapprox(f(a,dims=2),f(k,dims=2))
        end
        
        a = rand(10,10,10)
        @test gradcheck(f,a)
        @test gradcheck(f,a,kw=(:dims=>1,))
        @test gradcheck(f,a,kw=(:dims=>2,))
        @test gradcheck(f,a,kw=(:dims=>3,))
        @test gradcheck(f,a,kw=(:dims=>(1,2),))
        @test gradcheck(f,a,kw=(:dims=>(3,2),))
        @test gradcheck(f,a,kw=(:dims=>(1,3),))
        
        if CUDA.functional()
            k = KnetArray(a)
            @test gradcheck(f,k)
            @test isapprox(f(a),f(k))
            for d in [1,2,3]
                @test gradcheck(f,k,kw=(:dims=>d,))
                @test isapprox(f(a,dims=d),f(k,dims=d))
            end
            for dims in [(1,2), (1,3), (3,1), [1,3,2]]
                @test isapprox(f(a,dims=dims),f(k,dims=dims))
            end
        end

        a = rand(10,10, 10)
        for d in [1, 2, 3, (1,2), (1,3), [2,3], [1,2,3]]
            @test softmax(a, dims=d) ≈ exp.(logsoftmax(a, dims=d))
            @test all(sum(softmax(a, dims=d), dims=d) .≈ 1)
            if CUDA.functional()
                k = KnetArray(a)
                @test softmax(k, dims=d) ≈ exp.(logsoftmax(k, dims=d))
                @test all(Array(sum(softmax(k, dims=d), dims=d)) .≈ 1)
            end
        end
    end

    a = rand(10,10)
    indices = rand(1:10,10)
    @test gradcheck(nll, a, indices, kw=(:dims=>1,), args=1)
    @test gradcheck(nll, a, indices, kw=(:dims=>2,), args=1)
    if CUDA.functional()
        k = KnetArray(a)
        @test gradcheck(nll, k, indices, kw=(:dims=>1,), args=1)
        @test gradcheck(nll, k, indices, kw=(:dims=>2,), args=1)
        @test isapprox(nll(k, indices, dims=1), nll(a, indices, dims=1))
        @test isapprox(nll(k, indices, dims=2), nll(a, indices, dims=2))
    end
    @test gradcheck(logistic,a[:],a[:])
    @test gradcheck(bce,a[:],a[:])

    # Issue 439: highorder derivatives
    x = randn(3,4); y1 = softmax(x,dims=1); y2 = logsoftmax(x,dims=1); dy = randn(3,4)
    @test @gcheck softmax(Param(x),dims=1)
    @test @gcheck ∇softmax(Param(x),Param(y1),Param(dy),dims=1)
    @test @gcheck logsoftmax(Param(x),dims=1)
    @test @gcheck ∇logsoftmax(Param(x),Param(y2),Param(dy),dims=1)
    if CUDA.functional()
        x = KnetArray(x); y1 = KnetArray(y1); y2 = KnetArray(y2); dy = KnetArray(dy)
        @test isapprox(softmax(x,dims=1), _cudnnSoftmaxForward(x,algo=CUDNN_SOFTMAX_FAST))
        @test isapprox(softmax(x,dims=1), _cudnnSoftmaxForward(x,algo=CUDNN_SOFTMAX_ACCURATE))
        @test isapprox(logsoftmax(x,dims=1), _cudnnSoftmaxForward(x,algo=CUDNN_SOFTMAX_LOG))
        @test isapprox(∇softmax(x,y1,dy,dims=1), _cudnnSoftmaxBackward(y1,dy,algo=CUDNN_SOFTMAX_FAST))
        @test isapprox(∇softmax(x,y1,dy,dims=1), _cudnnSoftmaxBackward(y1,dy,algo=CUDNN_SOFTMAX_ACCURATE))
        @test isapprox(∇logsoftmax(x,y2,dy,dims=1), _cudnnSoftmaxBackward(y2,dy,algo=CUDNN_SOFTMAX_LOG))
        @test @gcheck _cudnnSoftmaxForward(Param(x),algo=CUDNN_SOFTMAX_FAST)
        @test @gcheck _cudnnSoftmaxForward(Param(x),algo=CUDNN_SOFTMAX_ACCURATE)
        @test @gcheck _cudnnSoftmaxForward(Param(x),algo=CUDNN_SOFTMAX_LOG)
        @test @gcheck _cudnnSoftmaxBackward(Param(y1),Param(dy),algo=CUDNN_SOFTMAX_FAST)
        @test @gcheck _cudnnSoftmaxBackward(Param(y1),Param(dy),algo=CUDNN_SOFTMAX_ACCURATE)
        @test @gcheck _cudnnSoftmaxBackward(Param(y2),Param(dy),algo=CUDNN_SOFTMAX_LOG)

        # Broken example from Alkan's notebook:
        f(w,x,y) = nll(w*x,y)
        ∇f = grad(f)
        ∇fj(w,x,y,j) = ∇f(w,x,y)[j]
        ∇∇fj = grad(∇fj)
        a = rand(10,10); b = rand(10,10); c = rand(1:10,10)
        A = KnetArray(a); B = KnetArray(b); C = c
        @test isapprox(f(a,b,c), f(A,B,C))
        @test isapprox(∇f(a,b,c), ∇f(A,B,C))
        i = 10; j = 20; d = 1e-4
        @test isapprox(∇∇fj(a,b,c,i), ∇∇fj(A,B,C,i))
    end
end
