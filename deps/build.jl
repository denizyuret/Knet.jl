try success(`nvcc --version`)
    cd("../src") do
        run(`make libknet8.so`)
    end
catch
    warn("CUDA not installed, GPU support will not be available.")
end
Base.compilecache("Knet")
