using Base.Test, Knet
using Knet: KnetFree, KnetPtr, gpuCount

sizes = randperm(1000)[1:10]
ptrs = map(KnetPtr, sizes)
kf = KnetFree[gpu()+2]

@testset "kptr" begin
    @test length(KnetFree) == gpuCount()+1
    @test length(kf) == 10
    @test sort(collect(keys(kf))) == sort(sizes)
    @test all(v.used==1 && isempty(v.free) for (k,v) in kf)
    ptrs = nothing
    gc()
    @test all(v.used==1 && length(v.free)==1 for (k,v) in kf)
    ptrs = map(KnetPtr, sizes)
    @test all(v.used==1 && isempty(v.free) for (k,v) in kf)
end
