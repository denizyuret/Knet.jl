if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

using Knet
using Knet: KnetFree, KnetPtr, gpuCount

# Messes up gc if used with `if gpu()>=0`
# This is just for printing the name
@testset "kptr" begin
    @test true
end

if gpu() >= 0
    sizes = randperm(1000)[1:10]
    ptrs = map(KnetPtr, sizes)
    kf = KnetFree[gpu()+2]
    @test length(kf) == 10
    @test length(KnetFree) == gpuCount()+1
    @test sort(collect(keys(kf))) == sort(sizes)
    @test all(Bool[v.used==1 && isempty(v.free) for (k,v) in kf])
    # gc doesn't work inside a testset
    ptrs = nothing
    gc()
    @test all(Bool[v.used==1 && length(v.free)==1 for (k,v) in kf])
    ptrs = map(KnetPtr, sizes)
    @test all(Bool[v.used==1 && isempty(v.free) for (k,v) in kf])
end
