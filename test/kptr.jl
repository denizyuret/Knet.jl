using Base.Test, Knet
using Knet: KnetFree, KnetPtr, gpuCount

if gpu() >= 0

kf = KnetFree[gpu()+2]
sizes = randperm(1000)[1:10]
ptrs = map(KnetPtr, sizes)      # things get messed up if this is inside the testset, multiple eval?

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

end # if gpu() >= 0
