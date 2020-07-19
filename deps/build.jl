using CUDA, Libdl

NVCC = nothing
CFLAGS = Sys.iswindows() ? ["/Ox","/LD"] : ["-O3","-Wall","-fPIC"]
NVCCFLAGS = ["-O3","--use_fast_math","-Wno-deprecated-gpu-targets"]
const OBJEXT = Sys.iswindows() ? ".obj" : ".o"
const LIBKNET8 = "libknet8."*Libdl.dlext
const DLLEXPORT = Sys.iswindows() ? "__declspec(dllexport)" : "" # this needs to go before function declarations
inforun(cmd)=(@info(cmd);run(cmd))

# Try to find NVCC

try
    cuda_dirs = CUDA.find_toolkit()
    global NVCC = CUDA.find_cuda_binary("nvcc", cuda_dirs)
catch; end

push!(NVCCFLAGS,"--compiler-options",join(CFLAGS,' '))

# If CUDA is available add architecture optimization flags
# Uncomment this for better compiler optimization
# We keep it commented to compile for multiple gpu types

# if NVCC !== nothing && CUDA.functional()
#     dev = CuDevice(0)
#     cap = capability(dev)
#     arch = "sm_$(cap.major)$(cap.minor)"
#     push!(NVCCFLAGS,"--gpu-architecture",arch)
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

