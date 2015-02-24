try success(`nvcc --version`)
    cd("../cuda") do
        run(`make libkunet.so`)
    end
catch
    warn("CUDA not installed, GPU support will not be available.")
end
