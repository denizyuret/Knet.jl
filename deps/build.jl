NVCC = CXX = ""
CFLAGS = is_windows() ? ["/Ox","/openmp","/LD"] : ["-O3","-Wall","-fPIC","-fopenmp"] # TODO: what if no openmp
NVCCFLAGS = ["-O3","--use_fast_math","-Wno-deprecated-gpu-targets","--compiler-options", join(CFLAGS,' ')]
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
            inforun(`$CXX $CFLAGS /LD /Fe:libknet8 $OBJ`)
        else
            inforun(`$CXX $CFLAGS --shared -o libknet8 $OBJ`)
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
        inforun(`$NVCC $NVCCFLAGS --shared -o libknet8 $OBJ`)
    end
end


# Try to find NVCC

if Pkg.installed("CUDAapi") != nothing
    eval(Expr(:using,:CUDAapi))
    try
        tk = find_toolkit()
        tc = find_toolchain(tk)
        NVCC = tc.cuda_compiler
        CXX = tc.host_compiler
        push!(NVCCFLAGS, "--compiler-bindir", CXX)
    end
end

# CUDAapi checks path, but if no CUDAapi this acts as backup

if NVCC == ""
    try success(`nvcc --version`)
        NVCC = "nvcc"
    end
end

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
        CXX = find_host_compiler()
    end
end



# OK let's compile

# TODO: get rid of Makefile, or just leave clean in it.
# TODO: get rid of cuda14?


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
