                                  # osx ai (seconds)
@time include("kptr.jl")          # 0	1
@time include("gpu.jl")           # 0	1
@time include("distributions.jl") # 3	1
@time include("karray.jl")        # 0	13
@time include("linalg.jl")        # 16	14
@time include("update.jl")        # 24	27
@time include("reduction.jl")     # 41	33
@time include("broadcast.jl")     # 52	34
@time include("conv.jl")          # 60	37
@time include("unary.jl")         # 53	43
