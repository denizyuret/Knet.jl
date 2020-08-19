using Test, Statistics, Random
using CUDA: CUDA, functional
using Knet.KnetArrays: KnetArray
using AutoGrad: Param, @gcheck

# This is buggy in julia (missing dims arg) as of Sep 16, 2019 so fix it here:
# https://github.com/JuliaLang/julia/issues/33280
Statistics.stdm(A::AbstractArray, m; corrected::Bool=true, dims=:) =
    ((dims === :) ? sqrt.(varm(A, m; corrected=corrected)) : sqrt.(varm(A, m; corrected=corrected, dims=dims)))

if CUDA.functional(); @testset "statistics" begin
    for T in (Float32,Float64)
        a = randn(T,3,4)
        k = KnetArray(a)
        p = Param(k)
        @test mean(a) ≈ mean(k)
        @test @gcheck mean(p)
        @test mean(a,dims=1) ≈ mean(k,dims=1)
        @test @gcheck mean(p,dims=1)
        @test mean(a,dims=2) ≈ mean(k,dims=2)
        @test @gcheck mean(p,dims=2)
        @test mean(abs,a) ≈ mean(abs,k)
        @test @gcheck mean(abs,p)
        @test mean(abs2,a) ≈ mean(abs2,k)
        @test @gcheck mean(abs2,p)
        @test std(a) ≈ std(k)
        @test @gcheck std(p)
        @test std(a,dims=1) ≈ std(k,dims=1)
        @test @gcheck std(p,dims=1)
        @test std(a,dims=2) ≈ std(k,dims=2)
        @test @gcheck std(p,dims=2)
        @test stdm(a,mean(a)) ≈ stdm(k,mean(k))
        @test @gcheck stdm(p,mean(p))
        @test stdm(a,mean(a,dims=1),dims=1) ≈ stdm(k,mean(k,dims=1),dims=1)
        @test @gcheck stdm(p,mean(p,dims=1),dims=1)
        @test stdm(a,mean(a,dims=2),dims=2) ≈ stdm(k,mean(k,dims=2),dims=2)
        @test @gcheck stdm(p,mean(p,dims=2),dims=2)
        @test var(a) ≈ var(k)
        @test @gcheck var(p)
        @test var(a,dims=1) ≈ var(k,dims=1)
        @test @gcheck var(p,dims=1)
        @test var(a,dims=2) ≈ var(k,dims=2)
        @test @gcheck var(p,dims=2)
        @test varm(a,mean(a)) ≈ varm(k,mean(k))
        @test @gcheck varm(p,mean(p))
        @test varm(a,mean(a,dims=1),dims=1) ≈ varm(k,mean(k,dims=1),dims=1)
        @test @gcheck varm(p,mean(p,dims=1),dims=1)
        @test varm(a,mean(a,dims=2),dims=2) ≈ varm(k,mean(k,dims=2),dims=2)
        @test @gcheck varm(p,mean(p,dims=2),dims=2)
    end
end; end
