using CUDAapi, CUDAdrv

# properties of the installation
toolkit_path = find_toolkit()
toolchain = find_toolchain(toolkit_path)

# properties of the device
dev = CuDevice(0)
cap = capability(dev)
arch = CUDAapi.shader(cap)

# if haskey(ENV,"CI")
#     Pkg.checkout("AutoGrad")
# end

cd(joinpath(dirname(@__DIR__), "src")) do
    flags = join(toolchain.flags, " ")
    run(`make NVCC=$(toolchain.nvcc) NVCCFLAGS="$flags --gpu-architecture $arch"`)
end

Base.compilecache("Knet")
