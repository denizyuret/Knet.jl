NVCC = CXX = ""
CFLAGS = is_windows() ? ["/Ox","/LD"] : ["-O3","-Wall","-fPIC"]
NVCCFLAGS = ["-O3","--use_fast_math","-Wno-deprecated-gpu-targets"]
const OBJEXT = is_windows() ? ".obj" : ".o"
const LIBKNET8 = "libknet8."*Libdl.dlext
const DLLEXPORT = is_windows() ? "__declspec(dllexport)" : "" # this needs to go before function declarations
inforun(cmd)=(info(cmd);run(cmd))

function build()
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


# Try to find NVCC

#if Pkg.installed("CUDAapi") != nothing # use this once CUDAapi is fixed
#    eval(Expr(:using,:CUDAapi))

# edit copy of CUDAapi here for now
using Compat
include("logging.jl")
include("compatibility.jl")
include("discovery.jl")
    try
        tk = find_toolkit()
        tc = find_toolchain(tk)
        NVCC = tc.cuda_compiler
        CXX = tc.host_compiler
        push!(NVCCFLAGS, "--compiler-bindir", CXX)
    end
#end

# CUDAapi checks path, but if no CUDAapi this acts as backup (no need for now)

# if NVCC == ""
#     try success(`nvcc --version`)
#         NVCC = "nvcc"
#     end
# end

if NVCC != "" && Pkg.installed("CUDAdrv") != nothing
    eval(Expr(:using,:CUDAdrv))
    try
        dev = CuDevice(0)
        cap = capability(dev)
        arch = CUDAapi.shader(cap)
        push!(NVCCFLAGS,"--gpu-architecture",arch)
    end
end

# In case there is no nvcc we can still compile the cpu kernels

if CXX == ""
    try
        # include("find_compiler.jl") # until CUDAapi is updated
        # CXX = find_compiler()
        CXX = find_host_compiler()
    end
end

if CXX != "" # test openmp
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

build()

