#                                   51b 51b 51b 51b 8.2 51b 51b
#                                   ai  ai4 tr  tr4 aws osx os4
@time include("kptr.jl")          # 1   1   0   0   2   0   0
@time include("gpu.jl")           # 1   1   0   0   13  0   0
@time include("distributions.jl") # 1   1   2   1   2   3   2
@time include("linalg.jl")        # 13  10  12  6   19   24  16
@time include("karray.jl")        # 17  11  -   -   16   -   0   :: new indexing types
@time include("broadcast.jl")     # 20  8   15  5   32  26  11
@time include("conv.jl")          # 23  13  71  63  34  44  16
@time include("reduction.jl")     # 25  10  21  8   32  40  20
@time include("unary.jl")         # 32  5   29  5   44  52  10
@time include("update.jl")        # 63  32  142 31  103 38  23   :: this was 30 at ai?
