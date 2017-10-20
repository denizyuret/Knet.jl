using CUDAapi

# properties of the installation
toolkit_path = find_toolkit()
toolchain = find_toolchain(toolkit_path)

# if haskey(ENV,"CI")
#     Pkg.checkout("AutoGrad")
# end

cd("../src") do
    flags = join(toolchain.flags, " ")
    run(`make NVCC=$(toolchain.nvcc) NVCCFLAGS="$flags"`)
end

Base.compilecache("Knet")
