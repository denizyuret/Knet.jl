#                                   ai  ai4 tr  tr4 aws osx os4
@time include("kptr.jl")          # 1   1   0   0   2   0   0
@time include("gpu.jl")           # 1   1   0   0   1   0   0
@time include("distributions.jl") # 1   1   2   1   2   3   2
@time include("karray.jl")        # 10  6   0   0   15  0   0
@time include("linalg.jl")        # 13  8   -   -   -   24  16
@time include("update.jl")        # 31  26  -   -   -   29  24
@time include("broadcast.jl")     # 19  9   19  7   29  31  15
@time include("reduction.jl")     # 21  8   16  6   31  33  16
@time include("unary.jl")         # 30  4   27  4   43  52  9
@time include("conv.jl")          # 28  16  57  13  37  58  29
