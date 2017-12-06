# using CUDAapi, CUDAdrv

CXX = ""
CFLAGS = ""
NVCC = ""
NVCCFLAGS = "-O3 --use_fast_math -Wno-deprecated-gpu-targets"

if Pkg.installed("CUDAapi") != nothing
    eval(Expr(:using,:CUDAapi))
    try
        tk = find_toolkit()
        tc = find_toolchain(tk)
        NVCC = tc.cuda_compiler
        CXX = tc.host_compiler
        NVCCFLAGS *= " --compiler-bindir $CXX"
    end
end

if NVCC == ""
    try success(`nvcc --version`)
        NVCC = "nvcc"
    end
end

if CXX == ""
    if is_windows()
        try success(`cl.exe`)
            CXX = "cl.exe"
        end
    else
        try success(`gcc --version`)
            CXX = "gcc"
        end
    end
end

if is_windows()
    # https://docs.microsoft.com/en-us/cpp/build/reference/compiler-options
    CFLAGS = "/Ox /openmp /LD"
else
    CFLAGS = "-O3 -Wall -fPIC -fopenmp"
end

NVCCFLAGS *= "--compiler-options \"$CFLAGS\""

if Pkg.installed("CUDAdrv") != nothing
    eval(Expr(:using,:CUDAdrv))
    try
        dev = CuDevice(0)
        cap = capability(dev)
        arch = CUDAapi.shader(cap)
        NVCCFLAGS *= " --gpu-architecture $arch"
    end
end

function cudacomp(name)
    if !isfile("$name.jl")
        error("$name.jl not found")
    end
    # Note that these ignore the secondary dependents like broadcast.jl
    if !isfile("$name.cu") || mtime("$name.jl") > mtime("$name.cu")
        info("$name.jl")
        include("$name.jl")
    end
    if !isfile("$name.o") || mtime("$name.cu") > mtime("$name.o")
        info("$NVCC $NVCCFLAGS -c $name.cu -o $name.o")
        run(`$NVCC $NVCCFLAGS -c $name.cu -o $name.o`)
    end
end

cd(joinpath(dirname(@__DIR__), "src")) do
    for f in ("cuda1","cuda01","cuda11","cuda12","cuda13","cuda16","cuda17","cuda20","cuda21","cuda22")
        cudacomp(f)
    end
    if !isfile("conv.o") || mtime("conv.cpp") > mtime("conv.o")
        info("$NVCC $NVCCFLAGS -c conv.cpp -o conv.o")
        run(`$NVCC $NVCCFLAGS -c conv.cpp -o conv.o`)
    end
    info("$NVCC $NVCCFLAGS --shared -o libknet8.so cuda1.o cuda01.o cuda11.o cuda12.o cuda13.o cuda16.o cuda17.o cuda20.o cuda21.o cuda22.o conv.o")
    run(`$NVCC $NVCCFLAGS --shared -o libknet8.so cuda1.o cuda01.o cuda11.o cuda12.o cuda13.o cuda16.o cuda17.o cuda20.o cuda21.o cuda22.o conv.o`)
end

# Base.compilecache("Knet")


# if haskey(ENV,"CI")
#     Pkg.checkout("AutoGrad")
# end

# cd(joinpath(dirname(@__DIR__), "src")) do
#     flags = join(toolchain.flags, " ")
#     run(`make NVCC=$(toolchain.nvcc) NVCCFLAGS="$flags --gpu-architecture $arch"`)
# end

# if NVCC==""
#     warn("Cannot find nvcc, GPU support will not be available.")
#     NVCC="nvcc" # to get make working
# end
