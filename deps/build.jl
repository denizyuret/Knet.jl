# using CUDAapi, CUDAdrv

NVCC = ""
NVCCFLAGS = ""

# properties of the installation
# toolkit_path = find_toolkit()
# toolchain = find_toolchain(toolkit_path)

if Pkg.installed("CUDAapi") != nothing
    eval(Expr(:using,:CUDAapi))
    try
        tk = find_toolkit()
        tc = find_toolchain(tk)
        NVCC = tc.cuda_compiler
        NVCCFLAGS *= " --compiler-bindir $(tc.host_compiler)"
    end
end

if NVCC == ""
    try success(`nvcc --version`)
        NVCC = "nvcc"
    end
end

# properties of the device
# dev = CuDevice(0)
# cap = capability(dev)
# arch = CUDAapi.shader(cap)

if Pkg.installed("CUDAdrv") != nothing
    eval(Expr(:using,:CUDAdrv))
    try
        dev = CuDevice(0)
        cap = capability(dev)
        arch = CUDAapi.shader(cap)
        NVCCFLAGS *= " --gpu-architecture $arch"
    end
end

# if haskey(ENV,"CI")
#     Pkg.checkout("AutoGrad")
# end

# cd(joinpath(dirname(@__DIR__), "src")) do
#     flags = join(toolchain.flags, " ")
#     run(`make NVCC=$(toolchain.nvcc) NVCCFLAGS="$flags --gpu-architecture $arch"`)
# end

cd(joinpath(dirname(@__DIR__), "src")) do
    global NVCC, NVCCFLAGS
    if NVCC==""
        warn("Cannot find nvcc, GPU support will not be available.")
        NVCC="nvcc" # to get make working
    end
    run(`make NVCC=$NVCC NVCCFLAGS=$NVCCFLAGS`)
end

Base.compilecache("Knet")
