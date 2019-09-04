include("header.jl")
using Knet: KnetMems, KnetPtr, gpuCount, blocksize, initKnetMems

function _testkptr(kptrs, navail, nfree)
    # sump = sum(p->p.nptr, values(mem.pools))
    # @show (mem.limit, mem.bytes, mem.avail, length(mem.pools), sump)
    # klens = kptrs == nothing ? [] : (p->p.len).(kptrs)
    # @show klens
    mem = KnetMems[gpu()+1]
    blocksizes = blocksize.(2 .^ (1:10))
    @test length(KnetMems) == gpuCount()
    @test length(mem.pools) == 10
    @test sort(collect(keys(mem.pools))) == blocksizes
    @test mem.limit >= mem.bytes
    @test mem.bytes == sum(blocksizes)
    @test mem.bfree == navail
    @test all(Bool[v.nptr==1 && length(v.free)==nfree for (k,v) in mem.pools])
    if nfree == 0
        @test (p->p.len).(kptrs) == blocksizes
    end
end

_testingkptr = false

if gpu() >= 0 && KnetMems === nothing
    initKnetMems()
    @testset "kptr:alloc"   begin; _testkptr(KnetPtr.(2 .^ (1:10)), 0, 0); end
    _testingkptr = true
end

GC.gc() # gc does not work reliably inside function, module, let, if, @testset etc.

if _testingkptr
    @testset "kptr:gc"      begin; _testkptr(nothing, sum(blocksize.(2 .^ (1:10))), 1); end
    @testset "kptr:realloc" begin; _testkptr(KnetPtr.(2 .^ (1:10)), 0, 0); end
    _testingkptr = false
end

nothing
