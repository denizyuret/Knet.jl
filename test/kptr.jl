using Test
using CUDA: CUDA, devices, device, functional
using Knet.KnetArrays: KnetMems, KnetPtr, blocksize, initKnetMems, cuallocator
_cuallocator = cuallocator[]

# gc does not work reliably inside function, module, let, if, @testset etc. so some of this code is outside.

function _testkptr(kptrs, navail, nfree)
    # sump = sum(p->p.nptr, values(mem.pools))
    # @show (mem.limit, mem.bytes, mem.avail, length(mem.pools), sump)
    # klens = kptrs == nothing ? [] : (p->p.len).(kptrs)
    # @show klens
    mem = KnetMems[device().handle+1]
    blocksizes = blocksize.(2 .^ (1:10))
    @test length(KnetMems) == length(devices())
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

# Test the knet allocator, this can only be done before first alloc when KnetMems is nothing
_testingkptr = CUDA.functional() && (KnetMems === nothing)
if _testingkptr
    cuallocator[]=false
    initKnetMems()
    @testset "kptr:alloc"   begin; _testkptr(KnetPtr.(2 .^ (1:10)), 0, 0); end
end

GC.gc(true) 

if _testingkptr
    @testset "kptr:gc"      begin; _testkptr(nothing, sum(blocksize.(2 .^ (1:10))), 1); end
    @testset "kptr:realloc" begin; _testkptr(KnetPtr.(2 .^ (1:10)), 0, 0); end
end

# Test the cuda allocator.
if CUDA.functional()
    cuallocator[]=true
    dev = CUDA.device()
    usedmem() = CUDA.usage[dev][] - CUDA.cached_memory()
    used = usedmem()
    @testset "kptr:cuda" begin
        @test (p = KnetPtr(128); usedmem() == used + 128)
        p = nothing
    end
end

GC.gc(true)

if CUDA.functional()
    @testset "kptr:cudagc" begin
        @test usedmem() == used
    end
end

cuallocator[]=_cuallocator

nothing
