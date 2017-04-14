include("header.jl")

rand21(f,t,d...)=rand(t,d...)*t(10)-t(5)

# This is missing from base
countnz2{T}(a::AbstractArray{T},region)=Array{T}(sum(a.!=0,region))
using AutoGrad
@zerograd countnz2(a,d...)

reduction_fns = Any[logsumexp]
for f in Knet.reduction_ops
    if isa(f,Tuple); f=f[2]; end
    if f == "countnz"; continue; end
    push!(reduction_fns, eval(parse(f)))
end

@testset "reduction" begin
    for f in reduction_fns
        for t in (Float32, Float64)
            for n in (1,(1,1),2,(2,1),(1,2),(2,2))
                @show f,t,n
                ax = rand21(f,t,n)
                @test gradcheck(f, ax)
                @test gradcheck(f, ax, 1)
                @test gradcheck(f, ax, 2)
                if gpu() >= 0
                    gx = KnetArray(ax)
                    @test gradcheck(f, gx)
                    @test gradcheck(f, gx, 1)
                    @test gradcheck(f, gx, 2)
                    @test isapprox(f(ax),f(gx))
                    @test isapprox(f(ax,1),Array(f(gx,1)))
                    @test isapprox(f(ax,2),Array(f(gx,2)))
                end
            end
            # test for kernel bug with dims > 64K
            # gradcheck difficult to pass on large arrays due to numerical error
            if gpu() >= 0
                for n in ((10,100000),(100000,10))
                    @show f,t,n
                    ax = rand21(f,t,n)
                    gx = KnetArray(ax)
                    @test isapprox(f(ax),f(gx))
                    @test isapprox(f(ax,1),Array(f(gx,1)))
                    @test isapprox(f(ax,2),Array(f(gx,2)))
                end
            end
        end
    end

    f = vecnorm
    for t in (Float32, Float64)
        for n in (1,(1,1),2,(2,1),(1,2),(2,2))
            ax = rand21(f,t,n)
            for p in (0,1,2,Inf,-Inf,1/pi,-1/pi,0+pi,-pi)
                # @show f,t,n,p
                @test gradcheck(f, ax, p)
                if gpu() >= 0
                    gx = KnetArray(ax)
                    @test gradcheck(f, gx, p)
                    @test isapprox(f(ax,p), f(gx,p); rtol=1e-6)
                end            
            end
        end
    end

    # 2-arg countnz is missing in base so we write custom tests for countnz
    f = countnz
    f2 = countnz2
    for t in (Float32, Float64)
        for n in (1,(1,1),2,(2,1),(1,2),(2,2))
            # @show f,t,n
            ax = rand21(f,t,n)
            @test gradcheck(f, ax)
            @test gradcheck(f2, ax, 1)
            @test gradcheck(f2, ax, 2)
            if gpu() >= 0
                gx = KnetArray(ax)
                @test gradcheck(f, gx)
                @test gradcheck(f, gx, 1)
                @test gradcheck(f, gx, 2)
                @test isapprox(f(ax),f(gx))
                @test isapprox(f2(ax,1),Array(f(gx,1)))
                @test isapprox(f2(ax,2),Array(f(gx,2)))
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
