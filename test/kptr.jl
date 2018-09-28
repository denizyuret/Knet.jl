include("header.jl")
using Knet: KnetMems, KnetPtr, gpuCount, blocksize

if gpu() >= 0; let arraysizes = 2 .^ (1:10), blocksizes = blocksize.(arraysizes), kptrs = map(KnetPtr, arraysizes), mem = KnetMems[gpu()+1]
    (mem.limit, mem.bytes, mem.avail, length(mem.pools), sum(p->p.nptr, values(mem.pools)))
    (p->p.len).(kptrs)

    @testset "kptr" begin
        
        function testkptr(navail, nfree)
            @test length(KnetMems) == gpuCount()
            @test length(mem.pools) == 10
            @test sort(collect(keys(mem.pools))) == blocksizes
            @test mem.limit >= mem.bytes
            @test mem.bytes == sum(blocksizes)
            @test mem.avail == navail
            @test all(Bool[v.nptr==1 && length(v.free)==nfree for (k,v) in mem.pools])
            if nfree == 0
                @test (p->p.len).(kptrs) == blocksizes
            end
        end

        testkptr(0, 0)
        kptrs = nothing; GC.gc()
        testkptr(sum(blocksizes), 1)
        kptrs = map(KnetPtr, arraysizes)
        testkptr(0, 0)

    end
end; end

nothing
