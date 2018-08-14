include("header.jl")
using Knet: KnetFree, KnetPtr, gpuCount

if gpu() >= 0
    _sizes = randperm(1000)[1:10]
    _ptrs = map(KnetPtr, _sizes)
    _kf = KnetFree[gpu()+2]
    @test length(_kf) == 10
    @test length(KnetFree) == gpuCount()+1
    @test sort(collect(keys(_kf))) == sort(_sizes)
    @test all(Bool[v.used==1 && isempty(v.free) for (k,v) in _kf])
    # gc doesn't work inside a testset
    _ptrs = nothing
    GC.gc()
    @test all(Bool[v.used==1 && length(v.free)==1 for (k,v) in _kf])
    _ptrs = map(KnetPtr, _sizes)
    @test all(Bool[v.used==1 && isempty(v.free) for (k,v) in _kf])
end

# Messes up gc if used with `if gpu()>=0`
# This is just for printing the name
@testset "kptr" begin
    @test true
end

nothing
