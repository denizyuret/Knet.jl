using Test, Knet, CUDA, Pkg.Artifacts

@testset "gpu" begin

    #@show Knet.gpu()
    #@show Knet.atype()
    @show Knet.dir()
    @show Knet.LibKnet8.libknet8
    @show readdir(artifact"libknet8") # readdir(Knet.dir("deps"))

if CUDA.functional()

    display(CUDA.device()); println() # Knet.gpu()
    @show length(CUDA.devices()) # Knet.gpuCount()
    @show CUDA.capability(CUDA.device())
    @show CUDA.warpsize(CUDA.device())
    @show CUDA.find_toolkit() # Knet.tk
    # @show Knet.cudartfound
    # if Knet.cudartfound
    # @show Knet.cudaRuntimeVersion
    @show CUDA.version() # Knet.cudaDriverVersion
    #@show Knet.cudaGetDeviceCount()
    #@show Knet.cudaGetDevice()
    @show Mem.info() # Knet.cudaMemGetInfo()
    @show CUDA.synchronize() # Knet.cudaDeviceSynchronize()
    #end
    #@show Knet.nvmlfound
    #if Knet.nvmlfound
    @show NVML.driver_version() # Knet.nvmlDriverVersion
    @show NVML.version() # Knet.nvmlVersion
    @show NVML.cuda_driver_version()
    nvmldev = NVML.Device(CUDA.device().handle)
    @show NVML.memory_info(nvmldev) # Knet.nvmlDeviceGetMemoryInfo()
    #end
    @show CUBLAS.handle() # Knet.cublashandle()
    #if Knet.cublashandle() != nothing
    @show CUBLAS.version() # Knet.cublasVersion
    #end
    @show CUDNN.handle() # Knet.cudnnhandle()
    #if Knet.cudnnhandle() != nothing
    @show CUDNN.version() # Knet.cudnnVersion
    #end

    @test length(CUDA.devices()) > 0 # Knet.gpuCount() > 0
    @test CUDA.device().handle >= 0  # Knet.gpu() >= 0
    # @test !isempty(Knet.tk)
    # @test Knet.cudartfound
    # @test Knet.cudaRuntimeVersion > 0
    @test CUDA.version() > v"0" # Knet.cudaDriverVersion > 0
    # @test Knet.cudaGetDeviceCount() > 0
    # @test Knet.cudaGetDevice() >= 0
    @test all(m->m>0, Mem.info()) # all(m->m>0, Knet.cudaMemGetInfo())
    @test CUDA.synchronize() === nothing # Knet.cudaDeviceSynchronize() == nothing
    #if Knet.nvmlfound
    @test NVML.driver_version() > v"0" # !isempty(Knet.nvmlDriverVersion)
    @test NVML.version() > v"0" #  !isempty(Knet.nvmlVersion)
    @test all(m->m>0, NVML.memory_info(nvmldev)) # all(m->m>0, Knet.nvmlDeviceGetMemoryInfo())
    #end
    @test CUBLAS.handle() > Ptr{Nothing}(0) # Knet.cublashandle() > Knet.Cptr(0)
    @test CUBLAS.version() > v"0" # Knet.cublasVersion > 0
    @test CUDNN.handle() > Ptr{Nothing}(0) # Knet.cudnnhandle() > Knet.Cptr(0)
    @test CUDNN.version() > v"0" # Knet.cudnnVersion > 0
    @test isdir(Knet.dir())
    @test isdir(artifact"libknet8")
    @test !isempty(Knet.LibKnet8.libknet8)
    # @test !isempty(readdir(Knet.dir("deps")))

end # if CUDA.functional()
end # @testset "gpu" begin


nothing
