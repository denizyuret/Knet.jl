using Test, Knet, CUDA, Pkg.Artifacts

@testset "gpu" begin
if CUDA.functional()

    CUDA.versioninfo()
    display(CUDA.device()) # Knet.gpu()
    @test CUDA.device().handle >= 0  # Knet.gpu() >= 0
    @show length(CUDA.devices()) # Knet.gpuCount()
    @test length(CUDA.devices()) > 0 # Knet.gpuCount() > 0
    @show CUDA.capability(CUDA.device())
    @show CUDA.warpsize(CUDA.device())
    @show CUDA.toolkit() # Knet.tk
    @show CUDA.version() # Knet.cudaDriverVersion
    @test CUDA.version() > v"0" # Knet.cudaDriverVersion > 0
    @show Mem.info() # Knet.cudaMemGetInfo()
    @test all(m->m>0, Mem.info()) # all(m->m>0, Knet.cudaMemGetInfo())
    @show CUDA.synchronize() # Knet.cudaDeviceSynchronize()
    @test CUDA.synchronize() === nothing # Knet.cudaDeviceSynchronize() == nothing
    if CUDA.has_nvml() #if Knet.nvmlfound
        @show NVML.driver_version() # Knet.nvmlDriverVersion
        @test NVML.driver_version() > v"0" # !isempty(Knet.nvmlDriverVersion)
        @show NVML.version() # Knet.nvmlVersion
        @test NVML.version() > v"0" #  !isempty(Knet.nvmlVersion)
        @show NVML.cuda_driver_version()
        nvmldev = NVML.Device(CUDA.uuid(CUDA.device()))
        @show NVML.memory_info(nvmldev) # Knet.nvmlDeviceGetMemoryInfo()
        @test all(m->m>0, NVML.memory_info(nvmldev)) # all(m->m>0, Knet.nvmlDeviceGetMemoryInfo())
    end
    @show CUBLAS.handle() # Knet.cublashandle()
    @test CUBLAS.handle() > Ptr{Nothing}(0) # Knet.cublashandle() > Knet.Cptr(0)
    @show CUBLAS.version() # Knet.cublasVersion
    @test CUBLAS.version() > v"0" # Knet.cublasVersion > 0
    @show CUDNN.handle() # Knet.cudnnhandle()
    @test CUDNN.handle() > Ptr{Nothing}(0) # Knet.cudnnhandle() > Knet.Cptr(0)
    @show CUDNN.version() # Knet.cudnnVersion
    @test CUDNN.version() > v"0" # Knet.cudnnVersion > 0
    @show Knet.LibKnet8.libknet8
    @test !isempty(Knet.LibKnet8.libknet8)
    @show readdir(artifact"libknet8") # readdir(Knet.dir("deps"))
    @test isdir(artifact"libknet8")

end # if CUDA.functional()
end # @testset "gpu" begin


nothing
