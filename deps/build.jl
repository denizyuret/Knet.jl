# Let's do some discovery first.

CXX = NVCC = ""

if is_windows()
    # https://docs.microsoft.com/en-us/cpp/build/reference/compiler-options
    CFLAGS = ["/Ox","/openmp","/LD"] # TODO: test this
else
    CFLAGS = ["-O3","-Wall","-fPIC","-fopenmp"] # TODO: what if no openmp
end

NVCCFLAGS = ["-O3","--use_fast_math","-Wno-deprecated-gpu-targets",
             "--compiler-options", join(CFLAGS,' ')]

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

if Pkg.installed("CUDAdrv") != nothing
    eval(Expr(:using,:CUDAdrv))
    try
        dev = CuDevice(0)
        cap = capability(dev)
        arch = CUDAapi.shader(cap)
        push!(NVCCFLAGS,"--gpu-architecture",arch)
    end
end

# In case CUDAapi fails, try to find executables the old fashioned way

if NVCC == ""
    try success(`nvcc --version`)
        NVCC = "nvcc"
    end
end

if CXX == ""
    if is_windows()
        try success(`cl.exe`)
            CXX = "cl.exe"      # TODO: test this
        end
    else
        try success(`gcc --version`)
            CXX = "gcc"         # TODO: there may be other compilers, check cudart
        end
    end
end

# OK let's compile

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

OBJ = map(x->x[1]*".o", SRC)

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
    if !isfile("$name.o") || mtime("$name.cu") > mtime("$name.o")
        info("$NVCC $NVCCFLAGS -c $name.cu -o $name.o")
        run(`$NVCC $NVCCFLAGS -c $name.cu -o $name.o`)
    end
end

if !isfile("conv.o") || mtime("conv.cpp") > mtime("conv.o")
    info("$NVCC $NVCCFLAGS -c conv.cpp -o conv.o") # TODO: use CXX if NVCC not available.
    run(`$NVCC $NVCCFLAGS -c conv.cpp -o conv.o`)  # test on non-gpu machines
end

push!(OBJ,"conv.o")             # TODO: is .o the suffix for windows?

if any(f->(mtime(f) > mtime("libknet8.so")), OBJ) # TODO: is .so the suffix for windows/osx?
    info("$NVCC $NVCCFLAGS --shared -o libknet8.so $OBJ")
    run(`$NVCC $NVCCFLAGS --shared -o libknet8.so $OBJ`)
end

Base.compilecache("Knet")

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
