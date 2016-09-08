try success(`nvcc --version`)
    cd("../src") do
        run(`make libknet8.so`)
        Base.compilecache("Knet")
    end
catch
    warn("CUDA not installed, GPU support will not be available.")
end
