using CUDA, Libdl, Tar, SHA

NVCC = nothing
CFLAGS = Sys.iswindows() ? ["/Ox","/LD"] : ["-O3","-Wall","-fPIC","-std=c++11"]
NVCCFLAGS = ["-O3","--use_fast_math","-Wno-deprecated-gpu-targets","--default-stream", "per-thread"]
const OBJEXT = Sys.iswindows() ? ".obj" : ".o"
const LIBKNET8 = "libknet8."*Libdl.dlext
const DLLEXPORT = Sys.iswindows() ? "__declspec(dllexport)" : "" # this needs to go before function declarations
inforun(cmd)=(@info(cmd);run(cmd))

# NVCC must be on the path:

try
    run(`nvcc -V`)
catch
    error("nvcc not found, libknet8 will not be built.")
end

NVCC = "nvcc"
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
    SRC = [("cuda1","gamma","../knetarrays/unary","ops"),
           ("cuda01","../knetarrays/binary","ops"),
           ("cuda11","../knetarrays/binary","../knetarrays/unary","ops"),
           ("cuda111","../knetarrays/binary","../knetarrays/unary","ops"),
           ("cuda12","../knetarrays/binary","ops"),
           ("cuda13","../knetarrays/binary","ops"),
           ("cuda16","../knetarrays/binary","ops"),
           ("cuda17","../knetarrays/binary","ops"),
           ("cuda20","../knetarrays/reduction","ops"),
           ("cuda21","../knetarrays/reduction","ops"),
           ("cuda22","../knetarrays/reduction","ops"),
           ("relu","ops"),
           ]

    OBJ = []

    include("ops.jl")

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
    @assert NVCC !== nothing "no compilers found, libknet8 will not be built."
    build_nvcc()
    if Sys.iswindows()
        run(`tar cf libknet8.tar libknet8.dll libknet8.lib libknet8.exp`)
    else
        run(`tar cf libknet8.tar $LIBKNET8`)
    end
    sha1 = Tar.tree_hash("libknet8.tar")
    run(`gzip libknet8.tar`)
    sha2 = open("libknet8.tar.gz") do f; bytes2hex(sha256(f)); end
    @info "git-tree-sha1 = \"$sha1\""
    @info "sha256 = \"$sha2\""
end

build()

