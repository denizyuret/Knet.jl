include("header.jl")
include("combinatorics.jl")
using Knet.KnetArrays: sumabs, sumabs2, minabs, maxabs, countnz, reduction_ops
using LinearAlgebra: norm

@testset "reduction" begin

    MIN_DIM  = 3
    MAX_DIM  = 5
    MIN_SIZE = 2
    TOL1 = 0.05

    function rand21(f,t,d...)
        if f==maximum || f==minimum || f==norm || f==sumabs2
            reshape(shuffle(t(0.01)*t[1:prod(d...)...]), d...)
            # elseif f==countnz || f==countnz2
            #     t(0.01)+rand(t,d...)
        elseif f==prod
            exp.(t(0.01)*randn(t,d...))
        else
            randn(t,d...)
        end
    end

    ### countnz is deprecated
    # countnz2(a::AbstractArray{T}; dims=:) where {T}=Array{T}(sum(a.!=0,dims=dims))
    # using AutoGrad
    # @zerograd countnz2(a,d...)

    reduction_fns = []
    for f in reduction_ops
        if isa(f,Tuple); f=f[2]; end
        if f == "countnz"; continue; end # deprecated
        push!(reduction_fns, eval(Meta.parse(f)))
    end

    #Knet.seed!(42)

    for f in reduction_fns
        for t in (Float32, Float64)
            for n in (1,(1,1),2,(2,1),(1,2),(2,2))
                #@show f,t,n
                ax = rand21(f,t,n)
                @test gradcheck(f, ax; rtol=TOL1)
                @test gradcheck(f, ax; kw=(:dims=>1,), rtol=TOL1)
                @test gradcheck(f, ax; kw=(:dims=>2,), rtol=TOL1)
                if gpu() >= 0
                    gx = KnetArray(ax)
                    @test gradcheck(f, gx; rtol=TOL1)
                    @test gradcheck(f, gx; kw=(:dims=>1,), rtol=TOL1)
                    @test gradcheck(f, gx; kw=(:dims=>2,), rtol=TOL1)
                    @test isapprox(f(ax),f(gx))
                    @test isapprox(f(ax,dims=1),Array(f(gx,dims=1)))
                    @test isapprox(f(ax,dims=2),Array(f(gx,dims=2)))
                end
            end
            # test for kernel bug with dims > 64K
            # gradcheck difficult to pass on large arrays due to numerical error
            if gpu() >= 0
                for n in ((10,100000),(100000,10))
                    #@show f,t,n
                    ax = rand21(f,t,n)
                    gx = KnetArray(ax)
                    @test isapprox(f(ax),f(gx))
                    @test isapprox(f(ax,dims=1),Array(f(gx,dims=1)))
                    @test isapprox(f(ax,dims=2),Array(f(gx,dims=2)))
                end
            end
        end
    end

    f = norm
    for t in (Float32, Float64)
        for n in (1,(1,1),2,(2,1),(1,2),(2,2))
            ax = rand21(f,t,n)
            for p in (0,1,2,Inf,-Inf,1/pi,-1/pi,0+pi,-pi)
                #@show f,t,n,p
                @test gradcheck(f, ax, p; rtol=TOL1, args=1)
                if gpu() >= 0
                    gx = KnetArray(ax)
                    @test gradcheck(f, gx, p; rtol=TOL1, args=1)
                    @test isapprox(f(ax,p), f(gx,p); rtol=1e-6)
                end
            end
        end
    end

    # countnz is deprecated

    # # 2-arg countnz is missing in base so we write custom tests for countnz
    # f = countnz
    # f2 = countnz2
    # for t in (Float32, Float64)
    #     for n in (1,(1,1),2,(2,1),(1,2),(2,2))
    #         #@show f,t,n
    #         ax = rand21(f,t,n)
    #         @test gradcheck(f, ax; rtol=TOL1)
    #         @test gradcheck(f2, ax; kw=(:dims=>1,), rtol=TOL1)
    #         @test gradcheck(f2, ax; kw=(:dims=>2,), rtol=TOL1)
    #         if gpu() >= 0
    #             gx = KnetArray(ax)
    #             @test gradcheck(f, gx; rtol=TOL1)
    #             @test gradcheck(f, gx; kw=(:dims=>1,), rtol=TOL1)
    #             @test gradcheck(f, gx; kw=(:dims=>2,), rtol=TOL1)
    #             @test isapprox(f(ax),f(gx))
    #             @test isapprox(f2(ax,dims=1),Array(f(gx,dims=1)))
    #             @test isapprox(f2(ax,dims=2),Array(f(gx,dims=2)))
    #         end
    #     end
    # end

    # all kind of reductions
    for f in (sum,) # reduction_fns takes too much time
        for t in (Float32, Float64)
            for dim = MIN_DIM:MAX_DIM
                # xsize = tuple(dim+MIN_SIZE-1:-1:MIN_SIZE...)
                xsize = ntuple(i->2,dim)
                ax = rand21(f,t,xsize)
                gx = nothing

                #@show f,t,dim,xsize
                @test gradcheck(f,ax; rtol=TOL1)
                if gpu() >= 0
                    gx = KnetArray(ax)
                    @test gradcheck(f, gx; rtol=TOL1)
                    @test isapprox(f(ax),f(gx))
                end

                # test all combinations
                for c in mapreduce(i->collect(combinas(1:dim,i)), vcat, 1:dim)
                    #@show f,t,dim,xsize,c
                    @test gradcheck(f, ax; kw=(:dims=>c,), rtol=TOL1)
                    if gpu() >= 0 && gx != nothing
                        @test gradcheck(f,gx; kw=(:dims=>c,), rtol=TOL1)
                        @test isapprox(f(ax,dims=c),Array(f(gx,dims=c)))
                    end
                end
            end
        end
    end
end

nothing

# Time	CUBLS32	CUBLS64	KN32	KN64	CPU32	CPU64	AF32	AF64
# sum	-	-	2.90	3.00	1.91	3.50	3.29	4.41
# prod	-	-	2.89	2.99	25.77	36.54	3.49	4.48
# max	3.77	3.88	2.92	3.01	6.25	8.00	3.53	4.43
# min	3.77	3.72	2.92	3.02	6.24	6.43	3.56	4.41
# asum	3.54	3.62	2.89	2.98	1.18	2.31	-	-
# nrmsq	5.45	9.30	2.88	2.98	3.35	3.13	5.51	7.41

# (*) BLK=128, THR=128 does best for kn.
# (*) what is the secret behind the cpu sum?
