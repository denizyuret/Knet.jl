using CUDAapi, Libdl

NVCC = CXX = ""
CFLAGS = Sys.iswindows() ? ["/Ox","/LD"] : ["-O3","-Wall","-fPIC"]
NVCCFLAGS = ["-O3","--use_fast_math","-Wno-deprecated-gpu-targets"]
const OBJEXT = Sys.iswindows() ? ".obj" : ".o"
const LIBKNET8 = "libknet8."*Libdl.dlext
const DLLEXPORT = Sys.iswindows() ? "__declspec(dllexport)" : "" # this needs to go before function declarations
inforun(cmd)=(@info(cmd);run(cmd))

# Try to find NVCC

try
    tk = CUDAapi.find_toolkit()
    tc = CUDAapi.find_toolchain(tk)
    global CXX = tc.host_compiler
    global NVCC = tc.cuda_compiler
    push!(NVCCFLAGS, "--compiler-bindir", CXX)
catch; end

push!(NVCCFLAGS,"--compiler-options",join(CFLAGS,' '))

# If CUDAdrv is available add architecture optimization flags
# Uncomment this for better compiler optimization
# We keep it commented to compile for multiple gpu types

# if NVCC != "" && Pkg.installed("CUDAdrv") != nothing
#     eval(Expr(:using,:CUDAdrv))
#     try
#         dev = CuDevice(0)
#         cap = capability(dev)
#         arch = CUDAapi.shader(cap)
#         push!(NVCCFLAGS,"--gpu-architecture",arch)
#     catch e
#         warn("CUDAdrv failed with $e")
#     end
# end


# Build scripts

function build_nvcc()
    SRC = [("cuda1","gamma","../src/unary"),
           ("cuda01","../src/binary"),
           ("cuda11","../src/binary","../src/unary"),
           ("cuda12","../src/binary"),
           ("cuda13","../src/binary"),
           ("cuda16","../src/binary"),
           ("cuda17","../src/binary"),
           ("cuda20","../src/reduction"),
           ("cuda21","../src/reduction"),
           ("cuda22","../src/reduction"),
           ]

    OBJ = []

    include("../src/ops.jl")

    for names in SRC
        for name in names
            if !isfile("$name.jl")
                error("$name.jl not found")
            end
        end
        name1 = names[1]
        if !isfile("$name1.cu") || any(d->(mtime("$d.jl") > mtime("$name1.cu")), names)
            @info("$name1.jl")
            include("$name1.jl")     # outputs name1.cu
        end
        obj = name1*OBJEXT
        if !isfile(obj) || mtime("$name1.cu") > mtime(obj)
            inforun(`$NVCC $NVCCFLAGS -c $name1.cu`)
        end
        push!(OBJ,obj)
    end

    if any(f->(mtime(f) > mtime(LIBKNET8)), OBJ)
        inforun(`$NVCC $NVCCFLAGS --shared -o $LIBKNET8 $OBJ`)
    end
end

function build()
    if NVCC != ""
        build_nvcc()
    else
        @warn("no compilers found, libknet8 will not be built.")
    end
end

build()

