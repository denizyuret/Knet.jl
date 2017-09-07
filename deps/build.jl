try success(`nvcc --version`)
    info("Compiling CUDA kernels.")
catch
    warn("Cannot find nvcc, GPU support will not be available.")
end
if haskey(ENV,"CI")
    Pkg.checkout("AutoGrad")
end
cd("../src") do
    run(`make`)
end
Base.compilecache("Knet")
