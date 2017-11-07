#                            commit 8.3 8.3 6cb 6cb 8.2 6cb 6cb
#                           machine ai5 ai4 tr5 tr4 aws osx os4
@time include("kptr.jl")          #   1   1   0   0   2   0   0
@time include("gpu.jl")           #   1   1   0   0  13   0   0  0
@time include("distributions.jl") #   1   1   2   1   2   3   2  0
@time include("update.jl")        #  29  26 100  22 103  25  23  7/18  :: this was 30 at ai? slower if last? travis=100?
@time include("karray.jl")        #  19  12   -   -  16   -   0  5/236 :: new indexing types
@time include("linalg.jl")        #  24  14  22   7  19  33  19  0
@time include("conv.jl")          #  22  12  62  47  34  44  16  0
@time include("broadcast.jl")     #  34  19 491 119  32  53  25  0
@time include("unary.jl")         #  42   6  36   4  44  67  11  0
@time include("reduction.jl")     #  40  21  29  11  32  55  29  348/1260: sumabs issues, need to fix both in AG and KN
