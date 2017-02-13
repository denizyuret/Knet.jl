#                                   tra osx ai  j4 (seconds)
@time include("kptr.jl")          # 1   0   1   1
@time include("gpu.jl")           # 0   0   1   1
@time include("distributions.jl") # 2   3   1   1
@time include("karray.jl")        # 0   0   9   5
@time include("linalg.jl")        # 0   17  10  6
@time include("update.jl")        # 60  18  19  17
@time include("broadcast.jl")     # 20  50  19  8
@time include("reduction.jl")     # 19  45  24  10
@time include("unary.jl")         # 27  54  30  4
@time include("conv.jl")          # 245 66  29  fail
