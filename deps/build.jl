try success(`nvcc --version`)
    cd("../src") do
        run(`make libknet.so`)
    end
catch
    warn("CUDA not installed, GPU support will not be available.")
end
