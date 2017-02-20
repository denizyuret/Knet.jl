try success(`nvcc --version`)
    info("Compiling CUDA kernels.")
catch
    warn("Cannot find nvcc, GPU support will not be available.")
end
cd("../src") do
    run(`make`)
end
Base.compilecache("Knet")
