using CUDAapi, Libdl

NVCC = nothing
CFLAGS = Sys.iswindows() ? ["/Ox","/LD"] : ["-O3","-Wall","-fPIC"]
NVCCFLAGS = ["-O3","--use_fast_math","-Wno-deprecated-gpu-targets"]
const OBJEXT = Sys.iswindows() ? ".obj" : ".o"
const LIBKNET8 = "libknet8."*Libdl.dlext
const DLLEXPORT = Sys.iswindows() ? "__declspec(dllexport)" : "" # this needs to go before function declarations
inforun(cmd)=(@info(cmd);run(cmd))

# Try to find NVCC

try
    tk = CUDAapi.find_toolkit()
    if isdefined(CUDAapi, :find_toolchain) # CUDAapi v1.x
        tc = CUDAapi.find_toolchain(tk)
        global NVCC = tc.cuda_compiler
        push!(NVCCFLAGS, "--compiler-bindir", tc.host_compiler)
    else                        # CUDAapi v2.x
        global NVCC = CUDAapi.find_cuda_binary("nvcc")
    end
catch; end

push!(NVCCFLAGS,"--compiler-options",join(CFLAGS,' '))

# If CUDAdrv is available add architecture optimization flags
# Uncomment this for better compiler optimization
# We keep it commented to compile for multiple gpu types

# if NVCC !== nothing && Pkg.installed("CUDAdrv") != nothing
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
    SRC = [("cuda1","gamma","../src/unary","../src/ops"),
           ("cuda01","../src/binary","../src/ops"),
           ("cuda11","../src/binary","../src/unary","../src/ops"),
           ("cuda12","../src/binary","../src/ops"),
           ("cuda13","../src/binary","../src/ops"),
           ("cuda16","../src/binary","../src/ops"),
           ("cuda17","../src/binary","../src/ops"),
           ("cuda20","../src/reduction","../src/ops"),
           ("cuda21","../src/reduction","../src/ops"),
           ("cuda22","../src/reduction","../src/ops"),
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
    if NVCC !== nothing
        build_nvcc()
    else
        @warn("no compilers found, libknet8 will not be built.")
    end
end

build()

