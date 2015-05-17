try success(`nvcc --version`)
    cd("../src") do
        run(`make libkunet.so`)
    end
catch
    warn("CUDA not installed, GPU support will not be available.")
end
