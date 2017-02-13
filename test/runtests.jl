                                  # osx ai (seconds)
@time include("kptr.jl")          # 0	1
@time include("gpu.jl")           # 0	1
@time include("distributions.jl") # 3	1
@time include("karray.jl")        # 0	9
@time include("linalg.jl")        # 16	10
@time include("update.jl")        # 24	19
@time include("reduction.jl")     # 41	24
@time include("broadcast.jl")     # 52	25
@time include("conv.jl")          # 60	29
@time include("unary.jl")         # 53	30
