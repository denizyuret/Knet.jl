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

    # nll tests
    N = 10
    as = Any[rand(N,N)]
    gpu() >= 0 && push!(as, KnetArray(as[1]))
    indices = rand(1:N,N)
    indices[1:2] = [1,2];
    ignore = [1, [1,], [1,2], (1,2)]
    mask = [indices .!= 1, indices .> 2]
    for (i,ai) in enumerate(as), d in 1:2, avg in (true,false)
        # gradcheck tests
        kw = (:dims => d,:average => avg)
        @test gradcheck(nll, ai, indices, kw=kw, args=1)
        for (ki,k) in enumerate(ignore)
            @test gradcheck(nll, ai, indices, kw=(:ignore => k, kw...), args=1)
        end
        for (ki,k) in enumerate(mask)
            @test gradcheck(nll, ai, indices, kw=(:mask => k, kw...), args=1)
        end

        # test different array types
        if length(as) > 1 && i == 1
            aj = as[end]
            @test isapprox(nll(ai, indices; kw...), nll(aj, indices; kw...))
        end

        # tests whether masking and averaging mechanism works or not
        @test isapprox(nll(ai, indices; kw..., ignore=-1),
                       nll(ai, indices; kw...))
        for (ki,k) in enumerate(mask)
            @test !isapprox(nll(ai, indices; kw..., mask=k),
                            nll(ai, indices; kw...))
            !avg && continue
            @test isless(nll(ai, indices; mask=k, dims=d, average=false),
                         nll(ai, indices; dims=d, average=false))
            @test isapprox(nll(ai, indices; mask=k, dims=d) * sum(k),
                           nll(ai, indices; mask=k, dims=d, average=false))
            @test isapprox(nll(ai, indices; dims=d) * length(indices),
                           nll(ai, indices; dims=d, average=false))
        end

        # tests for different masking mechanisms with different array types
        for (j,aj) in enumerate(as)
            i == 2 && j == 1 && continue
            @test isapprox(nll(ai, indices; kw..., ignore=ignore[1]),
                           nll(aj, indices; kw..., ignore=ignore[2]))
            @test isapprox(nll(ai, indices; kw..., ignore=ignore[3]),
                           nll(aj, indices; kw..., ignore=ignore[4]))
            @test isapprox(nll(aj, indices; kw..., ignore=ignore[1]),
                           nll(ai, indices; kw..., mask=mask[1]))
            @test isapprox(nll(aj, indices; kw..., ignore=ignore[3]),
                               nll(ai, indices; kw..., mask=mask[2]))
        end

        # tests for nll(model, data, [ignore]; kw...)
        #           nll(model, x, y, [ignore]; kw...)
        model, data  = identity, [(ai,indices)]
        x, y = first(data)
        desired = nll(ai, indices; kw...)
        @test isapprox(nll(model, data; kw...), desired)
        @test isapprox(nll(model, data; ignore=0, kw...), desired)
        @test isapprox(nll(model, x, y; kw...), desired)
        @test isapprox(nll(model, x, y; ignore=0, kw...), desired)
        @test isapprox(nll(model, x, y; kw..., mask=mask[1]),
                       nll(model, x, y; kw..., ignore=ignore[1]))
        for k in ignore
            @test isapprox(nll(model, data; ignore=k, kw...),
                           nll(ai, indices; ignore=k, kw...))
            @test isapprox(nll(model, x, y; ignore=k, kw...),
                           nll(ai, indices; ignore=k, kw...))
        end

        # tests for mask/ignore errors
        @test_throws ErrorException nll(ai, indices; kw...,
                                        mask=mask[1], ignore=0)
        @test_throws DimensionMismatch nll(ai, indices; kw...,
                                           mask=mask[1][1:end-1])

        # tests for old interface
        f(w, x; o...) = identity(x)
        w = nothing
        @test isapprox(nll(w, data, f;  kw..., ignore=ignore[1]),
                       nll(ai, indices; kw..., ignore=ignore[1]))
    end

    @test gradcheck(logistic,a[:],a[:])
    @test gradcheck(bce,a[:],a[:])

    # Issue 439: highorder derivatives
    using Knet: _softmax, _softback, _logp, _logpback, cudnnSoftmaxForward, cudnnSoftmaxBackward
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
