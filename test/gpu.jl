using Test,Libdl,Knet

if gpu() >= 0
    @testset "gpu" begin
        @test Knet.gpuCount() > 0
        @test Knet.cudaGetDeviceCount() > 0
        @test Knet.cudaGetDevice() >= 0
        @test Knet.cudaDeviceSynchronize() == nothing
        @test all(p->p>Knet.Cptr(0), (Knet.cublashandle(), Knet.cudnnhandle()))
        @test all(v->v>0, @show (Knet.cudaDriverVersion, Knet.cudaRuntimeVersion, Knet.cublasVersion, Knet.cudnnVersion))
        @test all(m->m>0, Knet.cudaMemGetInfo())
        if Knet.nvmlfound # Libdl.find_library(["libnvidia-ml"],[]) != ""
            @test all(m->m>0, Knet.nvmlDeviceGetMemoryInfo())
            @show Knet.nvmlDriverVersion, Knet.nvmlVersion
        end
    end
end

nothing
