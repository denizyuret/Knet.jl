using Test, Knet

@testset "gpu" begin

    @show Knet.gpuCount()
    @show Knet.gpu()
    @show Knet.tk
    @show Knet.libknet8
    @show Knet.cudartfound
    if Knet.cudartfound
        @show Knet.cudaRuntimeVersion
        @show Knet.cudaDriverVersion
        @show Knet.cudaGetDeviceCount()
        @show Knet.cudaGetDevice()
        @show Knet.cudaMemGetInfo()
        @show Knet.cudaDeviceSynchronize()
    end
    @show Knet.nvmlfound
    if Knet.nvmlfound
        @show Knet.nvmlDriverVersion
        @show Knet.nvmlVersion
        @show Knet.nvmlDeviceGetMemoryInfo()
    end
    @show Knet.cublashandle()
    if Knet.cublashandle() != nothing
        @show Knet.cublasVersion
    end
    @show Knet.cudnnhandle()
    if Knet.cudnnhandle() != nothing
        @show Knet.cudnnVersion
    end
    @show Knet.dir()
    @show readdir(Knet.dir("deps"))

    if gpu() >= 0
        @test Knet.gpuCount() > 0
        @test Knet.gpu() >= 0
        @test !isempty(Knet.tk)
        @test !isempty(Knet.libknet8)
        @test Knet.cudartfound
        @test Knet.cudaRuntimeVersion > 0
        @test Knet.cudaDriverVersion > 0
        @test Knet.cudaGetDeviceCount() > 0
        @test Knet.cudaGetDevice() >= 0
        @test all(m->m>0, Knet.cudaMemGetInfo())
        @test Knet.cudaDeviceSynchronize() == nothing
        if Knet.nvmlfound
            @test !isempty(Knet.nvmlDriverVersion)
            @test !isempty(Knet.nvmlVersion)
            @test all(m->m>0, Knet.nvmlDeviceGetMemoryInfo())
        end
        @test Knet.cublashandle() > Knet.Cptr(0)
        @test Knet.cublasVersion > 0
        @test Knet.cudnnhandle() > Knet.Cptr(0)
        @test Knet.cudnnVersion > 0
        @test isdir(Knet.dir())
        @test !isempty(readdir(Knet.dir("deps")))
    end
end

nothing
