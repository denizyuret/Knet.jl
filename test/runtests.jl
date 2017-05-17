#                            commit e4d e4d 51b 51b 8.2 51b 51b
#                           machine ai  ai4 tr  tr4 aws osx os4
@time include("kptr.jl")          # 1   1   0   0   2   0   0
@time include("gpu.jl")           # 1   1   0   0   13  0   0
@time include("distributions.jl") # 1   1   2   1   2   3   2
@time include("update.jl")        # 29  27  142 31  103 38  23   :: this was 30 at ai?
@time include("karray.jl")        # 19  12  -   -   16   -   0   :: new indexing types
@time include("linalg.jl")        # 24  14  12  6   19   24  16
@time include("conv.jl")          # 22  12  71  63  34  44  16
@time include("broadcast.jl")     # 40  19  15  5   32  26  11
@time include("reduction.jl")     # 40  20  21  8   32  40  20
@time include("unary.jl")         # 42  6   29  5   44  52  10
