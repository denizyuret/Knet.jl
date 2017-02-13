using Knet, Base.Test

if gpu() >= 0
    @testset "gpu" begin
        @test Knet.gpuCount() > 0
        @test Knet.cudaGetDeviceCount() > 0
        @test Knet.cudaGetDevice() >= 0
        @test Knet.cudaDeviceSynchronize() == nothing
        @test all(p->p>Knet.Cptr(0), (Knet.cublashandle(), Knet.cudnnhandle()))
        @test all(v->v>0, @show (Knet.cudaDriverVersion, Knet.cudaRuntimeVersion, Knet.cublasVersion, Knet.cudnnVersion))
        @test all(m->m>0, Knet.cudaGetMemInfo())
        @test all(m->m>0, Knet.nvmlDeviceGetMemoryInfo())
    end
end

nothing
