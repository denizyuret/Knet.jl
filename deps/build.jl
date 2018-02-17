using CUDAapi

NVCC = CXX = ""
CFLAGS = is_windows() ? ["/Ox","/LD"] : ["-O3","-Wall","-fPIC"]
NVCCFLAGS = ["-O3","--use_fast_math","-Wno-deprecated-gpu-targets"]
const OBJEXT = is_windows() ? ".obj" : ".o"
const LIBKNET8 = "libknet8."*Libdl.dlext
const DLLEXPORT = is_windows() ? "__declspec(dllexport)" : "" # this needs to go before function declarations
inforun(cmd)=(info(cmd);run(cmd))

# Try to find NVCC
try
    tk = CUDAapi.find_toolkit()
    tc = CUDAapi.find_toolchain(tk)
    CXX = tc.host_compiler
    NVCC = tc.cuda_compiler
    push!(NVCCFLAGS, "--compiler-bindir", CXX)
end

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


# In case there is no nvcc, find host_compiler to compile the cpu kernels

if CXX == ""
    try
        CXX,CXXVER = CUDAapi.find_host_compiler()
    end
end

# If openmp is available, use it:

if CXX != "" 
    cp("conv.cpp","foo.cpp",remove_destination=true)
    if is_windows()
        if success(`$CXX /openmp /c foo.cpp`)
            push!(CFLAGS, "/openmp")
        end
    else
        if success(`$CXX -fopenmp -c foo.cpp`)
            push!(CFLAGS, "-fopenmp")
        end
    end
end

push!(NVCCFLAGS,"--compiler-options",join(CFLAGS,' '))

# Build scripts

function build_nvcc()
    SRC = [("cuda1","../src/unary"),
           ("cuda01","../src/broadcast"),
           ("cuda11","../src/broadcast"),
           ("cuda12","../src/broadcast"),
           ("cuda13","../src/broadcast"),
           ("cuda16","../src/broadcast"),
           ("cuda17","../src/broadcast"),
           ("cuda20","../src/reduction"),
           ("cuda21","../src/reduction"),
           ("cuda22","../src/reduction")]

    CPP = ["conv"]

    OBJ = []

    for names in SRC
        for name in names
            if !isfile("$name.jl")
                error("$name.jl not found")
            end
        end
        name = names[1]
        if !isfile("$name.cu") || any(d->(mtime("$d.jl") > mtime("$name.cu")), names)
            info("$name.jl")
            include("$name.jl")     # outputs name.cu
        end
        obj = name*OBJEXT
        if !isfile(obj) || mtime("$name.cu") > mtime(obj)
            inforun(`$NVCC $NVCCFLAGS -c $name.cu`)
        end
        push!(OBJ,obj)
    end

    for name in CPP
        obj = name*OBJEXT
        if !isfile(obj) || mtime("$name.cpp") > mtime(obj)
            inforun(`$NVCC $NVCCFLAGS -c $name.cpp`)  # TODO: test on non-gpu machines
        end
        push!(OBJ,obj)
    end

    if any(f->(mtime(f) > mtime(LIBKNET8)), OBJ)
        inforun(`$NVCC $NVCCFLAGS --shared -o $LIBKNET8 $OBJ`)
    end
end

function build_cxx()
    SRC = ["conv"]
    OBJ = []
    for name in SRC
        obj = name*OBJEXT
        if !isfile(obj) || mtime("$name.cpp") > mtime(obj)
            inforun(`$CXX $CFLAGS -c $name.cpp`)  # TODO: test on non-gpu machines
        end
        push!(OBJ, obj)
    end
    if any(f->(mtime(f) > mtime(LIBKNET8)), OBJ)
        if is_windows()
            inforun(`$CXX $CFLAGS /LD /Fe:$LIBKNET8 $OBJ`)
        else
            inforun(`$CXX $CFLAGS --shared -o $LIBKNET8 $OBJ`)
        end
    end
end

function build()
    global CXX
    # OSX hack
    if CXX == "" && is_apple() && isfile("/usr/bin/clang")
        CXX = "/usr/bin/clang"
    end
    
    if NVCC != ""
        build_nvcc()
    elseif CXX != ""
        warn("nvcc not found, gpu kernels will not be compiled.")
        build_cxx()
    else
        warn("no compilers found, libknet8 will not be built.")
    end
    info("Compiling Knet cache.")
    Base.compilecache("Knet")
end

build()

