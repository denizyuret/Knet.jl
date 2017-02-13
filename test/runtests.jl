#                                   tr  osx ai  ai4 tr4
@time include("kptr.jl")          # 0   0   1   1   0
@time include("gpu.jl")           # 0   0   1   1   0
@time include("distributions.jl") # 2   3   1   1   1
@time include("karray.jl")        # 0   0   9   5   0
@time include("linalg.jl")        # 9   17  10  6   6
@time include("update.jl")        # 77  18  19  17  20
@time include("broadcast.jl")     # 21  50  19  10  8
@time include("reduction.jl")     # 18  45  21  8   6
@time include("unary.jl")         # 31  54  30  4   3
@time include("conv.jl")          # 182 66  25  15  F
