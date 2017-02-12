using Knet, Base.Test
using Knet: cudaDriverVersion, cudaRuntimeVersion, cublasVersion, cudnnVersion

@testset "gpu" begin
    if gpu() >= 0
        @test Knet.gpuCount() > 0
        @test Knet.cudaGetDeviceCount() > 0
        @test Knet.cudaGetDevice() >= 0
        @test Knet.cudaDeviceSynchronize() == nothing
        @test all(p->p>Knet.Cptr(0), (Knet.cublashandle, Knet.cudnnhandle))
        @test all(v->v>0, @show (cudaDriverVersion, cudaRuntimeVersion, cublasVersion, cudnnVersion))
        @test all(m->m>0, Knet.cudaGetMemInfo())
        @test all(m->m>0, Knet.nvmlDeviceGetMemoryInfo())
    else
        @test Knet.gpuCount() == 0
    end
end

nothing
