include("header.jl")

@testset "loss" begin
    for f in (logp, logsumexp)
        a = rand(10,10)
        @test gradcheck(f,a)
        @test gradcheck(f,a,kw=(:dims=>1,))
        @test gradcheck(f,a,kw=(:dims=>2,))
        if gpu() >= 0
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
        
        if gpu() >= 0
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
            if gpu() > 0
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
    if gpu() >= 0
        k = KnetArray(a)
        @test gradcheck(nll, k, indices, kw=(:dims=>1,), args=1)
        @test gradcheck(nll, k, indices, kw=(:dims=>2,), args=1)
        @test isapprox(nll(k, indices, dims=1), nll(a, indices, dims=1))
        @test isapprox(nll(k, indices, dims=2), nll(a, indices, dims=2))
    end
    @test gradcheck(logistic,a[:],a[:])
    @test gradcheck(bce,a[:],a[:])

    # Issue 439: highorder derivatives
    using Knet.Ops20: _softmax, _softback, _logp, _logpback
    using Knet.KnetArrays: cudnnSoftmaxForward, cudnnSoftmaxBackward
    x = randn(3,4); y1 = _softmax(x,dims=1); y2 = _logp(x,dims=1); dy = randn(3,4)
    @test @gcheck _softmax(Param(x),dims=1)
    @test @gcheck _softback(Param(x),Param(y1),Param(dy),dims=1)
    @test @gcheck _logp(Param(x),dims=1)
    @test @gcheck _logpback(Param(x),Param(y2),Param(dy),dims=1)
    if gpu() >= 0
        x = KnetArray(x); y1 = KnetArray(y1); y2 = KnetArray(y2); dy = KnetArray(dy)
        @test isapprox(_softmax(x,dims=1), cudnnSoftmaxForward(x,algo=0))
        @test isapprox(_softmax(x,dims=1), cudnnSoftmaxForward(x,algo=1))
        @test isapprox(_logp(x,dims=1), cudnnSoftmaxForward(x,algo=2))
        @test isapprox(_softback(x,y1,dy,dims=1), cudnnSoftmaxBackward(y1,dy,algo=0))
        @test isapprox(_softback(x,y1,dy,dims=1), cudnnSoftmaxBackward(y1,dy,algo=1))
        @test isapprox(_logpback(x,y2,dy,dims=1), cudnnSoftmaxBackward(y2,dy,algo=2))
        @test @gcheck cudnnSoftmaxForward(Param(x),algo=0)
        @test @gcheck cudnnSoftmaxForward(Param(x),algo=1)
        @test @gcheck cudnnSoftmaxForward(Param(x),algo=2)
        @test @gcheck cudnnSoftmaxBackward(Param(y1),Param(dy),algo=0)
        @test @gcheck cudnnSoftmaxBackward(Param(y1),Param(dy),algo=1)
        @test @gcheck cudnnSoftmaxBackward(Param(y2),Param(dy),algo=2)

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
