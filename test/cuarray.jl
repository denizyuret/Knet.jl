using Test
using CUDA: CUDA, functional
using Knet.KnetArrays: KnetArray
using AutoGrad: Param, @gcheck

if CUDA.functional(); @testset "cuarray" begin
    for nd in (1,2,3)
        sz = ntuple(i->8, nd)
        a0,b0 = rand(sz...),rand(sz...)
        a1,b1 = KnetArray(a0),KnetArray(b0)
        a2,b2 = Param(a0),Param(b0)
        a3,b3 = Param(a1),Param(b1)
        idx = ntuple(i->2:4, nd)
        @test getindex(a0,idx...) == getindex(a1,idx...)
        @test @gcheck getindex(a2,idx...)
        @test @gcheck getindex(a3,idx...)
        if nd == 1
            @test permutedims(a0) == permutedims(a1)
            @test @gcheck permutedims(a2)
            @test @gcheck permutedims(a3)
        elseif nd == 2
            @test permutedims(a0) == permutedims(a1)
            @test permutedims(a0,(2,1)) == permutedims(a1,(2,1))
            @test permutedims(a0,(1,2)) == permutedims(a1,(1,2))
            @test @gcheck permutedims(a2)
            @test @gcheck permutedims(a2,(2,1))
            @test @gcheck permutedims(a2,(1,2))
            @test @gcheck permutedims(a3)
            @test @gcheck permutedims(a3,(2,1))
            @test @gcheck permutedims(a3,(1,2))
        else
            @test permutedims(a0,(1,3,2)) == permutedims(a1,(1,3,2))
            @test @gcheck permutedims(a2,(1,3,2))
            @test @gcheck permutedims(a3,(1,3,2))
        end
        @test hcat(a0,b0) == hcat(a1,b1)
        @test vcat(a0,b0) == vcat(a1,b1)
        @test @gcheck hcat(a2,b2)
        @test @gcheck vcat(a2,b2)
        @test @gcheck hcat(a3,b3)
        @test @gcheck vcat(a3,b3)
        for i in 1:nd
            @test cat(a0,b0,dims=i) == cat(a1,b1,dims=i)
            @test @gcheck cat(a2,b2,dims=i)
            @test @gcheck cat(a3,b3,dims=i)
        end
        @test setindex!(a0,b0[idx...],idx...) == setindex!(a1,b1[idx...],idx...)

        # https://github.com/denizyuret/Knet.jl/issues/368
        @test argmax(a0) == argmax(a1)
        @test argmin(a0) == argmin(a1)
        @test findmax(a0) == findmax(a1)
        @test findmin(a0) == findmin(a1)
        for i in 1:nd
            @test argmax(a0,dims=i) == argmax(a1,dims=i)
            @test argmin(a0,dims=i) == argmin(a1,dims=i)
            @test findmax(a0,dims=i) == findmax(a1,dims=i)
            @test findmin(a0,dims=i) == findmin(a1,dims=i)
        end
    end
end; end
